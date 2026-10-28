// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Scalar Rust adaptation of fork_patches/marlin.py::_qpn_prepack from
// volta-nvfp4 commit b1a2181702e609dc92591343e46c63d1324ade43.
// The Apache-2.0 designation conservatively preserves the donor's license.
//! QPN2 fragment layout: [N/32 tile][K/16 group][32 lanes][8 code bytes].
//! Scale bytes have the same prefix with one byte per lane. Packing and
//! unpacking only permute packed nibbles and opaque scale bytes; they never
//! materialize a full unpacked code or floating-point weight plane.
//! Dequantization is a bounded CPU oracle, with no hardware qualification.

use crate::{invalid, quant, Error, Result};
use serde::{Deserialize, Serialize};

const K_ORDER: [usize; 16] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15];
const MAX_CPU_ELEMENTS: u64 = 1_048_576;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentRows {
    pub rows: u64,
    pub columns: u64,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_dequant: f64,
}

struct Shape {
    rows: usize,
    columns: usize,
    groups: usize,
    tiles: usize,
}

fn validate(
    rows: u64,
    columns: u64,
    codes_len: usize,
    scales_len: usize,
    global_dequant: f64,
) -> Result<Shape> {
    if rows == 0 || columns == 0 || !rows.is_multiple_of(32) || !columns.is_multiple_of(64) {
        return Err(invalid(
            "QPN2 requires positive rows divisible by 32 and columns divisible by 64",
        ));
    }
    let elements = rows
        .checked_mul(columns)
        .filter(|&n| n <= MAX_CPU_ELEMENTS)
        .ok_or_else(|| invalid("QPN2 matrix exceeds bounded CPU size limit"))?;
    if !global_dequant.is_finite() || global_dequant <= 0.0 {
        return Err(invalid(
            "QPN2 global dequant factor must be finite and positive",
        ));
    }
    if codes_len as u64 != elements / 2 || scales_len as u64 != elements / 16 {
        return Err(invalid("QPN2 packed plane length mismatch"));
    }
    // Positive dimensions and the element bound make these conversions exact.
    Ok(Shape {
        rows: rows as usize,
        columns: columns as usize,
        groups: (columns / 16) as usize,
        tiles: (rows / 32) as usize,
    })
}

fn row_for_lane(tile: usize, lane: usize) -> usize {
    tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + if lane & 16 != 0 { 4 } else { 0 }
}

fn nibble(codes: &[u8], index: usize) -> u8 {
    (codes[index / 2] >> (4 * (index % 2))) & 15
}

fn permute_into(p: &quant::PackedRows, shape: &Shape, codes: &mut [u8], scales: &mut [u8]) {
    for tile in 0..shape.tiles {
        for group in 0..shape.groups {
            for lane in 0..32 {
                let row = row_for_lane(tile, lane);
                let fragment = (tile * shape.groups + group) * 32 + lane;
                let source = row * shape.columns + group * 16;
                for (byte, pair) in K_ORDER.chunks_exact(2).enumerate() {
                    codes[fragment * 8 + byte] = nibble(&p.codes, source + pair[0])
                        | (nibble(&p.codes, source + pair[1]) << 4);
                }
                scales[fragment] = p.scales[row * shape.groups + group];
            }
        }
    }
}

/// Write one QPN2 plane into exact initialized destinations without a staging buffer.
/// Source and both destination lengths are checked before any byte is written.
pub(crate) fn pack_into(p: &quant::PackedRows, codes: &mut [u8], scales: &mut [u8]) -> Result<()> {
    let shape = validate(
        p.rows,
        p.columns,
        p.codes.len(),
        p.scales.len(),
        p.global_dequant,
    )?;
    if codes.len() != p.codes.len() || scales.len() != p.scales.len() {
        return Err(invalid("QPN2 destination plane length mismatch"));
    }
    permute_into(p, &shape, codes, scales);
    Ok(())
}

/// Permute row-major NVFP4 into the donor's exact QPN2 fragment layout.
/// All 256 scale-byte encodings are preserved; only the CPU oracle interprets them.
pub fn pack(p: &quant::PackedRows) -> Result<FragmentRows> {
    let shape = validate(
        p.rows,
        p.columns,
        p.codes.len(),
        p.scales.len(),
        p.global_dequant,
    )?;
    let mut codes = vec![0; p.codes.len()];
    let mut scales = vec![0; p.scales.len()];
    permute_into(p, &shape, &mut codes, &mut scales);
    Ok(FragmentRows {
        rows: p.rows,
        columns: p.columns,
        codes,
        scales,
        global_dequant: p.global_dequant,
    })
}

/// Invert the fragment permutation without decoding codes or scale values.
pub fn unpack(p: &FragmentRows) -> Result<quant::PackedRows> {
    let shape = validate(
        p.rows,
        p.columns,
        p.codes.len(),
        p.scales.len(),
        p.global_dequant,
    )?;
    let mut codes = vec![0; p.codes.len()];
    let mut scales = vec![0; p.scales.len()];
    for tile in 0..shape.tiles {
        for group in 0..shape.groups {
            for lane in 0..32 {
                let row = row_for_lane(tile, lane);
                let fragment = (tile * shape.groups + group) * 32 + lane;
                let target = row * shape.columns + group * 16;
                for (slot, &k) in K_ORDER.iter().enumerate() {
                    let value = nibble(&p.codes, fragment * 16 + slot);
                    let index = target + k;
                    // Each destination nibble is written once into a zeroed packed byte.
                    codes[index / 2] |= value << (4 * (index % 2));
                }
                scales[row * shape.groups + group] = p.scales[fragment];
            }
        }
    }
    Ok(quant::PackedRows {
        rows: p.rows,
        columns: p.columns,
        codes,
        scales,
        global_dequant: p.global_dequant,
    })
}

/// Decode fragments directly into the bounded CPU oracle output.
/// Unsigned finite E4M3 scales and finite decoded products follow quant::dequantize.
pub fn dequantize(p: &FragmentRows) -> Result<Vec<Vec<f64>>> {
    let shape = validate(
        p.rows,
        p.columns,
        p.codes.len(),
        p.scales.len(),
        p.global_dequant,
    )?;
    let mut out = vec![vec![0.0; shape.columns]; shape.rows];
    for tile in 0..shape.tiles {
        for group in 0..shape.groups {
            for lane in 0..32 {
                let row = row_for_lane(tile, lane);
                let fragment = (tile * shape.groups + group) * 32 + lane;
                let scale = quant::decode_scale(p.scales[fragment])?;
                for (slot, &k) in K_ORDER.iter().enumerate() {
                    let value = quant::decode_fp4(nibble(&p.codes, fragment * 16 + slot))?
                        * scale
                        * p.global_dequant;
                    if !value.is_finite() {
                        return Err(Error::new("NUMERICAL_OVERFLOW", "decoded value nonfinite"));
                    }
                    out[row][group * 16 + k] = value;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod destination_tests {
    use super::*;
    fn plane() -> quant::PackedRows {
        quant::PackedRows {
            rows: 32,
            columns: 64,
            codes: vec![0x22; 1024],
            scales: vec![0x38; 128],
            global_dequant: 1.0,
        }
    }
    #[test]
    fn exact_destination_lengths_are_checked_atomically() {
        let p = plane();
        for dc in [-1_i32, 0, 1] {
            for ds in [-1_i32, 0, 1] {
                if dc == 0 && ds == 0 {
                    continue;
                }
                let mut codes = vec![0xA5; (1024 + dc) as usize];
                let mut scales = vec![0x5A; (128 + ds) as usize];
                assert!(pack_into(&p, &mut codes, &mut scales).is_err());
                assert!(codes.iter().all(|&b| b == 0xA5));
                assert!(scales.iter().all(|&b| b == 0x5A));
            }
        }
    }
    #[test]
    fn invalid_source_does_not_change_either_destination() {
        for kind in 0..6 {
            let mut p = plane();
            match kind {
                0 => {
                    p.codes.pop();
                }
                1 => {
                    p.scales.push(0);
                }
                2 => p.rows = 0,
                3 => p.rows = u64::MAX - 31,
                4 => p.global_dequant = f64::NAN,
                5 => p.global_dequant = 0.0,
                _ => unreachable!(),
            }
            let mut codes = vec![0xA5; 1024];
            let mut scales = vec![0x5A; 128];
            assert!(pack_into(&p, &mut codes, &mut scales).is_err());
            assert!(codes.iter().all(|&b| b == 0xA5));
            assert!(scales.iter().all(|&b| b == 0x5A));
        }
    }
    #[test]
    fn direct_destinations_preserve_all_opaque_bytes() {
        for byte in 0..=255_u8 {
            let mut p = plane();
            p.codes.fill(byte);
            p.scales.fill(byte);
            let mut codes = vec![0xA5; 1024];
            let mut scales = vec![0x5A; 128];
            pack_into(&p, &mut codes, &mut scales).unwrap();
            let low = (byte & 15) * 17;
            let high = (byte >> 4) * 17;
            let wanted = [low, low, high, high, low, low, high, high];
            assert!(codes.chunks_exact(8).all(|chunk| chunk == wanted));
            assert!(scales.iter().all(|&b| b == byte));
        }
    }
}
