//! SC117 PLE n-gram row selection and IQ4_NL gather, CPU only.
//!
//! Adapted from MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
//! `include/strata/kernels/ngram.hpp` and `src/kernels/ngram.cpp`.
//! Caller must authenticate the checkpoint; file identity checks do not do so.
use crate::{completion::decode_fp16, invalid, Result};
use std::fs::File;

pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 160;
pub const ROW_BYTES: usize = 90;
pub const TABLE_ROWS: u64 = 320_001_536;
pub const VOCAB: i32 = 248_320;
pub const EOS: i32 = 248_044;
pub const NULL_TOKEN: i32 = -1;
const MULTIPLIERS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
const HEAD_VOCAB: [u64; HEADS] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];
const HEAD_OFFSET: [u64; HEADS] = [
    0,
    20_000_003,
    40_000_026,
    60_000_059,
    80_000_106,
    100_000_165,
    120_000_228,
    140_000_297,
    160_000_374,
    180_000_455,
    200_000_548,
    220_000_655,
    240_000_802,
    260_000_955,
    280_001_114,
    300_001_275,
];
const IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `previous` is oldest first. EOS or missing history cuts all older context.
pub fn rows(token: i32, previous: [i32; 2]) -> Result<[u32; HEADS]> {
    if !(0..VOCAB).contains(&token)
        || previous
            .iter()
            .any(|&x| x != NULL_TOKEN && !(0..VOCAB).contains(&x))
    {
        return Err(invalid("PLE token out of vocabulary"));
    }
    let mut context = [token as u64, 0, 0];
    let mut cut = false;
    for s in 1..3 {
        let t = if cut { NULL_TOKEN } else { previous[2 - s] };
        cut |= t == NULL_TOKEN || t == EOS;
        context[s] = if cut { EOS as u64 } else { t as u64 };
    }
    let mut result = [0u32; HEADS];
    let mut mixed = context[0].wrapping_mul(MULTIPLIERS[0]);
    mixed ^= context[1].wrapping_mul(MULTIPLIERS[1]);
    for h in 0..8 {
        result[h] = (HEAD_OFFSET[h] + mixed % HEAD_VOCAB[h]) as u32;
    }
    mixed ^= context[2].wrapping_mul(MULTIPLIERS[2]);
    for h in 8..HEADS {
        result[h] = (HEAD_OFFSET[h] + mixed % HEAD_VOCAB[h]) as u32;
    }
    Ok(result)
}

/// Decode one GGML IQ4_NL row: each byte holds an element and its +16 partner.
pub fn decode_row(bytes: &[u8]) -> Result<[f32; HEAD_DIM]> {
    if bytes.len() != ROW_BYTES {
        return Err(invalid("PLE row length"));
    }
    let mut out = [0f32; HEAD_DIM];
    for block in 0..5 {
        let start = block * 18;
        let scale = decode_fp16(u16::from_le_bytes([bytes[start], bytes[start + 1]]))? as f32;
        for j in 0..16 {
            let packed = bytes[start + 2 + j];
            out[block * 32 + j] = scale * f32::from(IQ4_NL[(packed & 15) as usize]);
            out[block * 32 + j + 16] = scale * f32::from(IQ4_NL[(packed >> 4) as usize]);
        }
    }
    Ok(out)
}

/// One open file descriptor pinned to the observed second-shard tensor span.
/// The caller authenticates file origin; changes to descriptor metadata are refused.
pub struct Table {
    file: File,
    identity: (u64, u64, u64, i64, i64, i64, i64),
    offset: u64,
}
#[cfg(unix)]
fn file_identity(file: &File) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let m = file.metadata().map_err(|e| invalid(e.to_string()))?;
    if !m.is_file() {
        return Err(invalid("PLE table must be regular"));
    }
    Ok((
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}
#[cfg(unix)]
impl Table {
    /// The offset and length must come from independently validated checkpoint metadata.
    pub fn open(file: File, expected_file_bytes: u64, offset: u64) -> Result<Self> {
        let identity = file_identity(&file)?;
        if identity.2 != expected_file_bytes
            || offset
                .checked_add(TABLE_ROWS * ROW_BYTES as u64)
                .is_none_or(|end| end > expected_file_bytes)
        {
            return Err(invalid("PLE tensor span or file size"));
        }
        Ok(Self {
            file,
            identity,
            offset,
        })
    }
    pub fn gather(&self, token: i32, previous: [i32; 2]) -> Result<([u32; HEADS], Vec<f32>)> {
        use std::os::unix::fs::FileExt;
        if file_identity(&self.file)? != self.identity {
            return Err(invalid("PLE table changed"));
        }
        let selected = rows(token, previous)?;
        let mut output = Vec::with_capacity(HEADS * HEAD_DIM);
        for &row in &selected {
            if u64::from(row) >= TABLE_ROWS {
                return Err(invalid("PLE row out of range"));
            }
            let offset = self
                .offset
                .checked_add(u64::from(row) * ROW_BYTES as u64)
                .ok_or_else(|| invalid("PLE row offset overflow"))?;
            let mut bytes = [0u8; ROW_BYTES];
            self.file
                .read_exact_at(&mut bytes, offset)
                .map_err(|e| invalid(e.to_string()))?;
            output.extend_from_slice(&decode_row(&bytes)?);
        }
        if file_identity(&self.file)? != self.identity {
            return Err(invalid("PLE table changed"));
        }
        Ok((selected, output))
    }
}
