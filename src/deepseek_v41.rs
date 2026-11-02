//! DeepSeek-V4.1-Flash CPU format/operator slice. No GPU kernel, full MoE layer or model
//! forward is implemented here. Semantics follow `inference/model.py` of
//! deepseek-ai/DeepSeek-V4.1-Flash at 314113f8a7dadb388215396ba5fbee2c610a04d6 (MIT,
//! see third_party/deepseek-v41-flash/PROVENANCE.md); no source is copied.
//!
//! The matrix owner keeps its planes private and immutable:
//! ```compile_fail
//! use voltafabric::deepseek_v41::Mxfp4Matrix;
//! let codes = vec![0u8; 16];
//! let m = Mxfp4Matrix::new(1, 32, codes, vec![127]).unwrap();
//! let _ = m.codes; // private field
//! ```
//! ```compile_fail
//! use voltafabric::deepseek_v41::Mxfp4Matrix;
//! let mut codes = vec![0u8; 16];
//! let m = Mxfp4Matrix::new(1, 32, codes, vec![127]).unwrap();
//! codes[0] = 0x11; // the constructor consumed the original buffer
//! ```
use crate::quant::decode_fp4;
use crate::{invalid, Error, Result};
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

pub mod r9_fp32;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
/// One F8_E8M0 scale per 32 consecutive K values (`fp4_block_size`).
pub const MXFP4_BLOCK: u64 = 32;
pub const MAX_MXFP4_DIM: u64 = 65_536;
pub const MAX_MXFP4_COMPRESSED_BYTES: u64 = 64 * 1024 * 1024;
/// Bound for one three-projection expert Safetensors artifact admitted by the CPU owner.
pub const MAX_MXFP4_EXPERT_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum work for the bounded CPU reference matvec. This covers one 2304x5120
/// V4.1 expert projection (11,796,480 MACs), but is not a production CPU fallback.
pub const MAX_MXFP4_REFERENCE_MACS: u64 = 16 * 1024 * 1024;
/// Total work cap for the three projections in one single-token CPU oracle.
pub const MAX_DEEPSEEK_EXPERT_REFERENCE_MACS: u64 = 48 * 1024 * 1024;
/// `swiglu_limit` of the official config.json.
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const MAX_ACTIVATION_LEN: usize = 1 << 20;
/// DeepSeek V4.1 uses one dynamically selected FP8 activation scale per 32 values.
pub const FP8_ACTIVATION_BLOCK: usize = 32;

/// Bounded CPU oracle output for the pinned FP8 activation quantizer. Byte planes
/// are immutable after construction; this is not a GPU input buffer or service path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fp8ActivationQuantized {
    values: Box<[u8]>,
    scale_codes: Box<[u8]>,
}

impl Fp8ActivationQuantized {
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn scale_codes(&self) -> &[u8] {
        &self.scale_codes
    }

    fn dequantized_values_f64(&self) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(self.values.len());
        for (block_index, block) in self.values.chunks_exact(FP8_ACTIVATION_BLOCK).enumerate() {
            let scale = decode_e8m0(self.scale_codes[block_index])?;
            for &code in block {
                let magnitude_code = code & 0x7f;
                if magnitude_code == 0x7f {
                    return Err(Error::new(
                        "INVALID_FP8_CODE",
                        "E4M3FN activation contains the NaN encoding",
                    ));
                }
                let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
                let value = sign * f64::from(e4m3fn_positive(magnitude_code)) * scale;
                if !value.is_finite() {
                    return Err(Error::new(
                        "NUMERICAL_OVERFLOW",
                        "dequantized E4M3FN activation is nonfinite",
                    ));
                }
                values.push(value);
            }
        }
        Ok(values)
    }
}

fn e4m3fn_positive(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        f32::from(8 + mantissa) * 2.0_f32.powi(i32::from(exponent) - 10)
    }
}

/// Round a finite value in [-448, 448] to E4M3FN, ties to an even encoding.
fn encode_e4m3fn(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }

    // E4M3FN's positive finite encodings are monotonically ordered 0x00..=0x7e.
    let mut low = 0u8;
    let mut high = 0x7eu8;
    while low < high {
        let mid = low + (high - low) / 2;
        if e4m3fn_positive(mid) < magnitude {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let upper = low;
    let upper_value = e4m3fn_positive(upper);
    let rounded = if upper_value == magnitude || upper == 0 {
        upper
    } else {
        let lower = upper - 1;
        let lower_value = e4m3fn_positive(lower);
        let lower_distance = f64::from(magnitude) - f64::from(lower_value);
        let upper_distance = f64::from(upper_value) - f64::from(magnitude);
        if lower_distance < upper_distance {
            lower
        } else if upper_distance < lower_distance || upper & 1 == 0 {
            upper
        } else {
            lower
        }
    };
    sign | rounded
}

/// Reproduce the pinned model's pre-GEMM dynamic FP8 quantizer on CPU.
///
/// Each complete 32-value group computes `amax=max(abs(x), 1e-4)` in FP32,
/// obtains `scale=2^ceil(log2(amax * (1/448)))` using the upstream IEEE-bit rule,
/// clamps `x/scale` to [-448,448], then rounds to E4M3FN. Scale bytes use UE8M0.
/// This is an operator-format oracle only: no GEMM, CUDA kernel, or inference path.
pub fn fp8_activation_quantize_reference(input: &[f32]) -> Result<Fp8ActivationQuantized> {
    if input.is_empty()
        || input.len() > MAX_ACTIVATION_LEN
        || !input.len().is_multiple_of(FP8_ACTIVATION_BLOCK)
    {
        return Err(invalid(format!(
            "FP8 activation length must be a nonzero multiple of 32 and <= {MAX_ACTIVATION_LEN}"
        )));
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err(Error::new(
            "NONFINITE_INPUT",
            "FP8 activation input contains NaN or Inf",
        ));
    }

    let mut values = Vec::with_capacity(input.len());
    let mut scale_codes = Vec::with_capacity(input.len() / FP8_ACTIVATION_BLOCK);
    let fp8_max_inverse = 1.0f32 / 448.0f32;
    for block in input.chunks_exact(FP8_ACTIVATION_BLOCK) {
        let amax = block.iter().fold(1.0e-4f32, |max, x| max.max(x.abs()));
        let scaled_amax = amax * fp8_max_inverse;
        let bits = scaled_amax.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
        // The 1e-4 floor and finite-f32 maximum derive a scale exponent in [-22,120].
        // This keeps every accepted scale a normal f32 and a valid finite UE8M0 code.
        if !(-22..=120).contains(&exponent) {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "FP8 activation scale exponent outside derived UE8M0 admission interval [-22,120]",
            ));
        }
        let scale_code = (exponent + 127) as u8;
        let scale = f32::from_bits(u32::from(scale_code) << 23);
        scale_codes.push(scale_code);
        for &x in block {
            let scaled = (x / scale).clamp(-448.0, 448.0);
            values.push(encode_e4m3fn(scaled));
        }
    }

    Ok(Fp8ActivationQuantized {
        values: values.into_boxed_slice(),
        scale_codes: scale_codes.into_boxed_slice(),
    })
}

/// Round one finite FP32 value to BF16 using round-to-nearest, ties-to-even.
/// The result is returned as an exactly representable FP32 value for CPU oracles.
pub fn cast_bf16_rne_reference(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(Error::new(
            "NONFINITE_INPUT",
            "BF16 reference cast input contains NaN or Inf",
        ));
    }
    let bits = value.to_bits();
    let tie_to_even_bias = 0x7fffu32 + ((bits >> 16) & 1);
    let rounded = f32::from_bits(bits.wrapping_add(tie_to_even_bias) & 0xffff_0000);
    if !rounded.is_finite() {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "BF16 reference cast produced a nonfinite value",
        ));
    }
    Ok(rounded)
}

/// Compose one token through a DeepSeek V4.1 SwiGLU expert from immutable compressed FP4 planes.
///
/// The input must already contain exact BF16 values represented as FP32. The reference
/// quantizes activations to E4M3FN/UE8M0 for w1/w3 and w2, follows the pinned gate/up
/// clamp and SwiGLU order, applies the optional routing coefficient once, and rounds
/// the down-projection input to BF16. Each FP4 matvec reads its packed code and scale
/// planes directly; no dense weights are materialized. Accumulation is an FP64 dyadic
/// CPU oracle, not the exact FP32 Tensor Core accumulation order and not a service path.
pub fn single_token_expert_forward_reference(
    w1: &Mxfp4Matrix,
    w3: &Mxfp4Matrix,
    w2: &Mxfp4Matrix,
    input_bf16: &[f32],
    routing_weight: Option<f32>,
) -> Result<Vec<f32>> {
    if w1.rows != w3.rows
        || w1.cols != w3.cols
        || w2.cols != w1.rows
        || input_bf16.len() as u64 != w1.cols
    {
        return Err(invalid(
            "single-token expert projection shapes or hidden input length do not compose",
        ));
    }
    if let Some(weight) = routing_weight {
        if !weight.is_finite() || weight < 0.0 {
            return Err(invalid(
                "routing weight must be finite and >= 0 (-0 allowed)",
            ));
        }
    }
    let total_macs = w1
        .rows
        .checked_mul(w1.cols)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(w2.rows.checked_mul(w2.cols)?))
        .ok_or_else(|| invalid("single-token expert work limit size overflow"))?;
    if total_macs > MAX_DEEPSEEK_EXPERT_REFERENCE_MACS {
        return Err(invalid(format!(
            "single-token expert work limit exceeded: {total_macs} MACs > {MAX_DEEPSEEK_EXPERT_REFERENCE_MACS}"
        )));
    }

    let mut input = Vec::with_capacity(input_bf16.len());
    for &value in input_bf16 {
        let rounded = cast_bf16_rne_reference(value)?;
        if rounded.to_bits() != value.to_bits() {
            return Err(invalid(
                "single-token expert input must already be exactly BF16-representable",
            ));
        }
        input.push(value);
    }
    let input_quantized = fp8_activation_quantize_reference(&input)?;
    let input_dequantized = input_quantized.dequantized_values_f64()?;

    let gate_f64 = w1.matvec_reference(&input_dequantized)?;
    let up_f64 = w3.matvec_reference(&input_dequantized)?;
    let gate = gate_f64
        .into_iter()
        .map(f64_to_f32_finite)
        .collect::<Result<Vec<_>>>()?;
    let up = up_f64
        .into_iter()
        .map(f64_to_f32_finite)
        .collect::<Result<Vec<_>>>()?;
    let pre_down = pre_down_activation(&gate, &up, routing_weight)?;
    let down_input = pre_down
        .iter()
        .copied()
        .map(cast_bf16_rne_reference)
        .collect::<Result<Vec<_>>>()?;
    let down_quantized = fp8_activation_quantize_reference(&down_input)?;
    let down_dequantized = down_quantized.dequantized_values_f64()?;
    w2.matvec_reference(&down_dequantized)?
        .into_iter()
        .map(f64_to_f32_finite)
        .collect()
}

fn f64_to_f32_finite(value: f64) -> Result<f32> {
    let rounded = value as f32;
    if !value.is_finite() || !rounded.is_finite() {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "FP64 reference projection cannot be represented as finite FP32",
        ));
    }
    Ok(rounded)
}

/// Decode an unsigned F8_E8M0 scale: `2^(b-127)` for 0..=254, 255 is NaN and refused.
pub fn decode_e8m0(b: u8) -> Result<f64> {
    if b == 255 {
        return Err(Error::new("MXFP4_SCALE_NAN", "E8M0 scale 255 encodes NaN"));
    }
    Ok(2.0_f64.powi(i32::from(b) - 127))
}

/// Native MXFP4 matrix, logical [rows, cols] row-major. `codes` holds [rows, cols/2]
/// bytes, even column in the low nibble; `scales` holds [rows, cols/32] E8M0 bytes.
#[derive(Debug)]
pub struct Mxfp4Matrix {
    rows: u64,
    cols: u64,
    codes: Mxfp4Plane,
    scales: Mxfp4Plane,
}

/// A matrix plane is either owned independently or an immutable view into one
/// expert Safetensors buffer. Shared planes keep packed weights resident once.
#[derive(Debug)]
enum Mxfp4Plane {
    Owned(Box<[u8]>),
    SharedArtifact {
        bytes: Arc<Vec<u8>>,
        range: Range<usize>,
    },
}

impl Mxfp4Plane {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::SharedArtifact { bytes, range } => &bytes[range.clone()],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mxfp4Summary {
    pub format: &'static str,
    pub rows: u64,
    pub cols: u64,
    pub block: u64,
    pub logical_elements: u64,
    pub code_bytes: u64,
    pub scale_bytes: u64,
    pub compressed_bytes: u64,
}

impl Mxfp4Matrix {
    /// Validates shape, size bound, plane lengths and every scale before taking ownership.
    pub fn new(rows: u64, cols: u64, codes: Vec<u8>, scales: Vec<u8>) -> Result<Self> {
        let (code_bytes, scale_bytes) = mxfp4_plane_bytes(rows, cols)?;
        if codes.len() as u64 != code_bytes {
            return Err(invalid("MXFP4 code plane length mismatch"));
        }
        if scales.len() as u64 != scale_bytes {
            return Err(invalid("MXFP4 scale plane length mismatch"));
        }
        if scales.contains(&255) {
            return Err(Error::new("MXFP4_SCALE_NAN", "E8M0 scale 255 encodes NaN"));
        }
        Ok(Self {
            rows,
            cols,
            codes: Mxfp4Plane::Owned(codes.into_boxed_slice()),
            scales: Mxfp4Plane::Owned(scales.into_boxed_slice()),
        })
    }

    fn from_safetensors_views(
        backing: Arc<Vec<u8>>,
        weight: TensorView<'_>,
        scale: TensorView<'_>,
    ) -> Result<Self> {
        if weight.dtype() != Dtype::I8 {
            return Err(invalid("MXFP4 weight tensor dtype must be I8"));
        }
        if scale.dtype() != Dtype::F8_E8M0 {
            return Err(invalid("MXFP4 scale tensor dtype must be F8_E8M0"));
        }
        if weight.shape().len() != 2 || scale.shape().len() != 2 {
            return Err(invalid("MXFP4 weight and scale tensors must be rank two"));
        }
        let rows = u64::try_from(weight.shape()[0])
            .map_err(|_| invalid("MXFP4 row dimension does not fit u64"))?;
        let packed_cols = u64::try_from(weight.shape()[1])
            .map_err(|_| invalid("MXFP4 packed column dimension does not fit u64"))?;
        let cols = packed_cols
            .checked_mul(2)
            .ok_or_else(|| invalid("MXFP4 logical column dimension overflow"))?;
        let (code_bytes, scale_bytes) = mxfp4_plane_bytes(rows, cols)?;
        let expected_scale_shape = [rows as usize, (cols / MXFP4_BLOCK) as usize];
        if scale.shape() != expected_scale_shape {
            return Err(invalid("MXFP4 scale tensor shape must be [rows, cols/32]"));
        }
        if weight.data().len() as u64 != code_bytes || scale.data().len() as u64 != scale_bytes {
            return Err(invalid("MXFP4 Safetensors plane byte length mismatch"));
        }
        if scale.data().contains(&255) {
            return Err(Error::new("MXFP4_SCALE_NAN", "E8M0 scale 255 encodes NaN"));
        }
        let code_range = artifact_range(backing.as_slice(), weight.data())?;
        let scale_range = artifact_range(backing.as_slice(), scale.data())?;
        Ok(Self {
            rows,
            cols,
            codes: Mxfp4Plane::SharedArtifact {
                bytes: Arc::clone(&backing),
                range: code_range,
            },
            scales: Mxfp4Plane::SharedArtifact {
                bytes: backing,
                range: scale_range,
            },
        })
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn cols(&self) -> u64 {
        self.cols
    }
    pub fn summary(&self) -> Mxfp4Summary {
        let code_bytes = self.codes.as_slice().len() as u64;
        let scale_bytes = self.scales.as_slice().len() as u64;
        Mxfp4Summary {
            format: "MXFP4_E2M1_E8M0_ROW_MAJOR_LOW_NIBBLE_FIRST",
            rows: self.rows,
            cols: self.cols,
            block: MXFP4_BLOCK,
            logical_elements: self.rows * self.cols,
            code_bytes,
            scale_bytes,
            compressed_bytes: code_bytes + scale_bytes,
        }
    }
    /// Immutable, already-admitted compressed planes. No copy or dequantization.
    pub fn compressed_planes(&self) -> (&[u8], &[u8]) {
        (self.codes.as_slice(), self.scales.as_slice())
    }
    /// (E2M1 nibble, E8M0 scale byte) stored for one logical coordinate.
    pub fn codes_at(&self, row: u64, col: u64) -> Result<(u8, u8)> {
        if row >= self.rows || col >= self.cols {
            return Err(invalid("MXFP4 coordinate outside the logical matrix"));
        }
        let byte = self.codes.as_slice()[(row * (self.cols / 2) + col / 2) as usize];
        let nibble = if col.is_multiple_of(2) {
            byte & 15
        } else {
            byte >> 4
        };
        let scale =
            self.scales.as_slice()[(row * (self.cols / MXFP4_BLOCK) + col / MXFP4_BLOCK) as usize];
        Ok((nibble, scale))
    }
    /// Exact dyadic f64 value of one logical element (signed zeros preserved).
    pub fn value(&self, row: u64, col: u64) -> Result<f64> {
        let (nibble, scale) = self.codes_at(row, col)?;
        Ok(decode_fp4(nibble)? * decode_e8m0(scale)?)
    }

    /// Evaluate a bounded, single-vector reference projection directly from the
    /// compressed planes. No dense weight copy is created. Products and additions
    /// occur in increasing K order as separate binary64 operations; this is a CPU
    /// format oracle only, not a service execution path or substitute for the GPU
    /// kernel.
    pub fn matvec_reference(&self, input: &[f64]) -> Result<Vec<f64>> {
        let macs = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| invalid("MXFP4 reference work limit size overflow"))?;
        if macs > MAX_MXFP4_REFERENCE_MACS {
            return Err(invalid(format!(
                "MXFP4 reference work limit exceeded: {macs} MACs > {MAX_MXFP4_REFERENCE_MACS}"
            )));
        }
        if input.len() as u64 != self.cols {
            return Err(invalid("MXFP4 matvec input length must equal cols"));
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Error::new(
                "NONFINITE_INPUT",
                "MXFP4 matvec input contains NaN or Inf",
            ));
        }

        let cols = self.cols as usize;
        let codes = self.codes.as_slice();
        let scales = self.scales.as_slice();
        let blocks_per_row = cols / MXFP4_BLOCK as usize;
        let mut output = Vec::with_capacity(self.rows as usize);
        for row in 0..self.rows as usize {
            let code_row = row * (cols / 2);
            let scale_row = row * blocks_per_row;
            let mut acc = 0.0f64;
            for block in 0..blocks_per_row {
                let scale = decode_e8m0(scales[scale_row + block])?;
                let block_col = block * MXFP4_BLOCK as usize;
                for lane in 0..MXFP4_BLOCK as usize {
                    let col = block_col + lane;
                    let packed = codes[code_row + col / 2];
                    let nibble = if col.is_multiple_of(2) {
                        packed & 15
                    } else {
                        packed >> 4
                    };
                    let weight = decode_fp4(nibble)? * scale;
                    let term = weight * input[col];
                    if !term.is_finite() {
                        return Err(Error::new(
                            "NUMERICAL_OVERFLOW",
                            "MXFP4 matvec product is nonfinite",
                        ));
                    }
                    acc += term;
                    if !acc.is_finite() {
                        return Err(Error::new(
                            "NUMERICAL_OVERFLOW",
                            "MXFP4 matvec accumulator is nonfinite",
                        ));
                    }
                }
            }
            output.push(acc);
        }
        Ok(output)
    }
}

const QPN2_K_ORDER: [usize; 16] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15];

struct ConvertedQpn2Plane {
    rows: usize,
    columns: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    global_dequant: f32,
}

fn half_positive_bits_to_f64(bits: u16) -> f64 {
    crate::quant::f16_positive_to_f64(bits)
}

fn round_f16_bits(value: f64) -> Option<u16> {
    if !value.is_finite() {
        return None;
    }
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let magnitude = value.abs() as f32;
    if !magnitude.is_finite() {
        return None;
    }
    crate::quant::f32_to_f16_bits_positive_rn(magnitude).map(|bits| sign | bits)
}

fn round_f16(value: f64) -> Option<f64> {
    round_f16_bits(value).map(|bits| {
        let sign = bits & 0x8000 != 0;
        let positive = half_positive_bits_to_f64(bits & 0x7fff);
        if sign {
            -positive
        } else {
            positive
        }
    })
}

fn decode_f16_bits(bits: u16) -> f64 {
    let sign = bits & 0x8000 != 0;
    let positive = half_positive_bits_to_f64(bits & 0x7fff);
    if sign {
        -positive
    } else {
        positive
    }
}

/// Decode the E4M3 byte using the same intermediate operations as
/// `fp8e4m3_to_half2`: reinterpret `(byte << 7)` as half, then half-multiply by 256.
fn qpn2_e4m3_half_value(byte: u8) -> Option<f64> {
    if byte & 0x80 != 0 || byte == 0x7f {
        return None;
    }
    let rebased = u16::from(byte & 0x7f) << 7;
    round_f16(decode_f16_bits(rebased) * 256.0)
}

fn qpn2_power_scale_code(exponent: i32) -> Option<u8> {
    match exponent {
        -9..=-7 => Some(1u8 << (exponent + 9)),
        -6..=7 => Some(((exponent + 7) as u8) << 3),
        8 => Some(0x78),
        _ => None,
    }
}

/// The upstream helper encodes an E2M1 code as `value * 2^-14` in half before
/// multiplying by `scale * half(gscale * 2^14)`. Check those ordered half
/// operations against the source `half(value * 2^e)` result for all 16 codes.
/// This accepts exact underflow when both representations produce the same half
/// value, and rejects scale-product overflow even when the ideal weight is finite.
fn validate_mxfp4_qpn2_half_equivalence(exponent: i32, shift: i32) -> Result<()> {
    if exponent > 1 {
        return Err(Error::new(
            "MXFP4_QPN2_FP16_DOMAIN",
            "scale * gscale * 2^14 overflows the SM70 half intermediate",
        ));
    }
    let global_dequant = 2.0f32.powi(shift);
    if !global_dequant.is_finite() || global_dequant <= 0.0 {
        return Err(Error::new(
            "MXFP4_QPN2_FP16_DOMAIN",
            "global factor is not a positive finite F32 power of two",
        ));
    }
    let gm_f32 = global_dequant * 16_384.0f32;
    let gm_bits = crate::quant::f32_to_f16_bits_positive_rn(gm_f32).ok_or_else(|| {
        Error::new(
            "MXFP4_QPN2_FP16_DOMAIN",
            "global factor intermediate overflows half",
        )
    })?;
    let gm = half_positive_bits_to_f64(gm_bits);
    let e4m3_code = qpn2_power_scale_code(exponent - shift).ok_or_else(|| {
        Error::new(
            "MXFP4_QPN2_SCALE_RANGE",
            "E8M0 power is not representable by one exact E4M3 power scale",
        )
    })?;
    let e4m3_half = qpn2_e4m3_half_value(e4m3_code).ok_or_else(|| {
        Error::new(
            "MXFP4_QPN2_SCALE_RANGE",
            "E4M3 power did not convert to a finite half value",
        )
    })?;
    let scale_product = round_f16(e4m3_half * gm).ok_or_else(|| {
        Error::new(
            "MXFP4_QPN2_FP16_DOMAIN",
            "E4M3 scale multiplied by the global factor overflows half",
        )
    })?;

    for code in 0..16u8 {
        let raw_bits = (u16::from(code & 8) << 12) | (u16::from(code & 7) << 9);
        let raw_half = decode_f16_bits(raw_bits);
        let actual = round_f16(raw_half * scale_product).ok_or_else(|| {
            Error::new(
                "MXFP4_QPN2_FP16_DOMAIN",
                "QPN2 half dequantization produced a nonfinite E2M1 value",
            )
        })?;
        let source = round_f16(decode_fp4(code)? * 2.0f64.powi(exponent)).ok_or_else(|| {
            Error::new(
                "MXFP4_QPN2_FP16_DOMAIN",
                "source E2M1/E8M0 value overflows half",
            )
        })?;
        if actual.to_bits() != source.to_bits() {
            return Err(Error::new(
                "MXFP4_QPN2_FP16_DOMAIN",
                "ordered QPN2 half decode differs from rounded E2M1/E8M0 source value",
            ));
        }
    }
    Ok(())
}

/// Select one exact global factor for the whole projection. QPN2's current resident
/// contract requires a uniform factor per projection, while E4M3 power scales span
/// exponents -9..=8. The shift is derived from the observed scale range, then every
/// distinct source scale is admitted by the ordered-half oracle above.
fn mxfp4_scale_shift(scales: &[u8]) -> Result<i32> {
    if scales.is_empty() {
        return Err(invalid("MXFP4 projection has no E8M0 scales"));
    }
    let mut min_exp = i32::MAX;
    let mut max_exp = i32::MIN;
    let mut unique = BTreeSet::new();
    for &code in scales {
        if code == 255 {
            return Err(Error::new("MXFP4_SCALE_NAN", "E8M0 scale 255 encodes NaN"));
        }
        let exponent = i32::from(code) - 127;
        min_exp = min_exp.min(exponent);
        max_exp = max_exp.max(exponent);
        unique.insert(exponent);
    }
    // global_dequant=2^shift must be normal F32; the CUDA `* 16384` result then
    // remains representable as F32 before its deliberate FP16 conversion.
    let lower = (max_exp - 8).max(-126);
    let upper = (min_exp + 9).min(1);
    if lower > upper {
        return Err(Error::new(
            "MXFP4_QPN2_SCALE_RANGE",
            "projection scale exponents have no common exact F32/E4M3 factor",
        ));
    }
    let shift = lower + (upper - lower) / 2;
    for exponent in unique {
        validate_mxfp4_qpn2_half_equivalence(exponent, shift)?;
    }
    Ok(shift)
}

fn checked_zeroed_bytes(len: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(|error| {
        Error::new(
            "MXFP4_QPN2_ALLOCATION",
            format!("could not reserve compressed QPN2 plane: {error}"),
        )
    })?;
    data.resize(len, 0);
    Ok(data)
}

fn qpn2_lane_row(tile: usize, lane: usize) -> usize {
    tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + if lane & 16 != 0 { 4 } else { 0 }
}

fn packed_nibble(codes: &[u8], index: usize) -> u8 {
    (codes[index / 2] >> (4 * (index % 2))) & 15
}

fn convert_mxfp4_plane(matrix: &Mxfp4Matrix) -> Result<ConvertedQpn2Plane> {
    let rows = usize::try_from(matrix.rows).map_err(|_| invalid("MXFP4 rows do not fit usize"))?;
    let columns =
        usize::try_from(matrix.cols).map_err(|_| invalid("MXFP4 columns do not fit usize"))?;
    if rows == 0 || !rows.is_multiple_of(32) || columns == 0 || !columns.is_multiple_of(64) {
        return Err(invalid(
            "QPN2 conversion requires rows divisible by 32 and columns divisible by 64",
        ));
    }
    let shift = mxfp4_scale_shift(matrix.scales.as_slice())?;
    let global_dequant = 2.0f32.powi(shift);
    let groups = columns / 16;
    let scale_len = rows
        .checked_mul(groups)
        .ok_or_else(|| invalid("QPN2 scale length overflow"))?;
    let code_len = rows
        .checked_mul(columns / 2)
        .ok_or_else(|| invalid("QPN2 code length overflow"))?;
    if matrix.codes.as_slice().len() != code_len
        || matrix.scales.as_slice().len() != rows * (columns / 32)
    {
        return Err(invalid("MXFP4 source plane length changed after admission"));
    }
    let mut codes = checked_zeroed_bytes(code_len)?;
    let mut scales = checked_zeroed_bytes(scale_len)?;
    let source_scales = matrix.scales.as_slice();
    let source_codes = matrix.codes.as_slice();
    for tile in 0..rows / 32 {
        for group in 0..groups {
            for lane in 0..32 {
                let row = qpn2_lane_row(tile, lane);
                let fragment = (tile * groups + group) * 32 + lane;
                let source_scale = source_scales[row * (columns / 32) + group / 2];
                let exponent = i32::from(source_scale) - 127;
                scales[fragment] = qpn2_power_scale_code(exponent - shift).ok_or_else(|| {
                    Error::new(
                        "MXFP4_QPN2_SCALE_RANGE",
                        "source scale escaped the admitted E4M3 range",
                    )
                })?;
                let source_start = row * columns + group * 16;
                for (byte, pair) in QPN2_K_ORDER.chunks_exact(2).enumerate() {
                    codes[fragment * 8 + byte] =
                        packed_nibble(source_codes, source_start + pair[0])
                            | (packed_nibble(source_codes, source_start + pair[1]) << 4);
                }
            }
        }
    }
    let factors = vec![global_dequant; rows / 32];
    crate::quant::validate_sm70_scale_domain(&scales, &factors, matrix.cols)?;
    Ok(ConvertedQpn2Plane {
        rows,
        columns,
        codes,
        scales,
        global_dequant,
    })
}

/// Identity domain for checkpoint expert names. Backbone and draft/MTP experts are
/// independent even when their numeric layer and expert IDs match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mxfp4ExpertNamespace {
    /// Main model experts under `layers.{layer}.ffn.experts.{expert}`.
    Backbone,
    /// Draft/MTP experts under `mtp.{layer}.ffn.experts.{expert}`.
    Mtp,
}

const MAX_PINNED_EXPERT_INDEX_BYTES: usize = 8 * 1024 * 1024;
const MAX_EXPERTS_PER_PAYLOAD_PLAN: usize = 6;
const PINNED_EXPERT_INDEX_SHA256: &str =
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp4TensorPayloadRange {
    pub expert: u32,
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    /// Absolute, half-open byte range in the original Safetensors file.
    pub file_offset_start: u64,
    pub file_offset_end: u64,
}

/// A metadata-only read plan for at most six experts from one pinned shard.
/// It names the exact six MXFP4 tensors per expert and their source byte ranges;
/// it does not fetch or authenticate those payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp4ExpertPayloadPlan {
    pub model_revision: String,
    pub namespace: Mxfp4ExpertNamespace,
    pub layer: u32,
    pub experts: Vec<u32>,
    pub source_shard: String,
    pub source_shard_bytes: u64,
    pub header_bytes: u64,
    pub data_start_bytes: u64,
    pub selected_payload_bytes: u64,
    pub tensors: Vec<Mxfp4TensorPayloadRange>,
}

/// One immutable, compressed DeepSeek expert. The Safetensors `Vec` is moved into a
/// shared owner once; each code/scale plane stores only a checked byte range into it.
/// This is a host-side resident format owner and CPU reference, not a service kernel.
#[derive(Debug)]
pub struct Mxfp4Expert {
    namespace: Mxfp4ExpertNamespace,
    layer: u32,
    expert: u32,
    artifact: Arc<Vec<u8>>,
    artifact_sha256: String,
    w1: Mxfp4Matrix,
    w3: Mxfp4Matrix,
    w2: Mxfp4Matrix,
    compressed_bytes: u64,
}

/// One immutable Safetensors shard holding complete compressed experts. Each expert's
/// six FP4/UE8M0 planes refer to the same backing allocation owned by this shard.
///
/// Loading validates the complete artifact once, including its caller-supplied byte
/// budget and digest. Expert lookup returns a borrowed owner and does not reparse the
/// header, rehash the shard or copy any weight plane.
#[derive(Debug)]
pub struct Mxfp4ExpertShard {
    artifact: Arc<Vec<u8>>,
    artifact_sha256: String,
    experts: BTreeMap<(Mxfp4ExpertNamespace, u32, u32), Mxfp4Expert>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertProjection {
    W1,
    W2,
    W3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertPlane {
    Weight,
    Scale,
}

#[derive(Debug, Clone, Copy)]
struct ParsedExpertTensorName {
    namespace: Mxfp4ExpertNamespace,
    layer: u32,
    expert: u32,
    projection: ExpertProjection,
    plane: ExpertPlane,
}

#[derive(Default)]
struct ExpertTensorViews<'a> {
    w1_weight: Option<TensorView<'a>>,
    w1_scale: Option<TensorView<'a>>,
    w2_weight: Option<TensorView<'a>>,
    w2_scale: Option<TensorView<'a>>,
    w3_weight: Option<TensorView<'a>>,
    w3_scale: Option<TensorView<'a>>,
}

impl Mxfp4ExpertShard {
    /// Read a shard with a caller-supplied byte budget before parsing it. At most
    /// `max_artifact_bytes + 1` source bytes are requested; the extra byte is a sentinel
    /// that distinguishes an exact fit from an over-budget input. Read and allocation
    /// failures are terminal and are never retried.
    pub fn from_bounded_reader<R: Read>(
        reader: R,
        expected_sha256: &str,
        max_artifact_bytes: usize,
    ) -> Result<Self> {
        let read_limit = max_artifact_bytes
            .checked_add(1)
            .ok_or_else(|| invalid("MXFP4 shard byte budget overflow"))?;
        if !crate::sha256::is_hex_digest(expected_sha256) {
            return Err(invalid(
                "expected MXFP4 shard SHA-256 must be lowercase hex",
            ));
        }

        let mut reader = reader;
        let mut artifact = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        while artifact.len() < read_limit {
            let requested = (read_limit - artifact.len()).min(chunk.len());
            artifact.try_reserve_exact(requested).map_err(|error| {
                Error::new(
                    "SHARD_BUFFER_ALLOCATION_ERROR",
                    format!("could not reserve bounded MXFP4 shard input: {error}"),
                )
            })?;
            let count = reader.read(&mut chunk[..requested]).map_err(|error| {
                Error::new(
                    "SHARD_READ_ERROR",
                    format!("failed while reading bounded MXFP4 shard: {error}"),
                )
            })?;
            if count == 0 {
                break;
            }
            if count > requested {
                return Err(Error::new(
                    "SHARD_READ_ERROR",
                    "reader returned more bytes than the supplied buffer length",
                ));
            }
            artifact.extend_from_slice(&chunk[..count]);
        }
        if artifact.len() > max_artifact_bytes {
            return Err(invalid(
                "MXFP4 shard artifact exceeds its caller-supplied byte budget",
            ));
        }
        Self::from_safetensors_shard_artifact(artifact, expected_sha256, max_artifact_bytes)
    }

    /// Move one immutable shard buffer into a shared owner and index all complete
    /// backbone `layers.{layer}.ffn.experts.{expert}.*` and draft/MTP
    /// `mtp.{layer}.ffn.experts.{expert}.*` groups. Namespace is part of identity.
    ///
    /// `max_artifact_bytes` is an explicit caller resource budget; no universal shard
    /// size is assumed. The digest is checked before parsing any tensor metadata.
    pub fn from_safetensors_shard_artifact(
        artifact: Vec<u8>,
        expected_sha256: &str,
        max_artifact_bytes: usize,
    ) -> Result<Self> {
        if artifact.is_empty() || max_artifact_bytes == 0 || artifact.len() > max_artifact_bytes {
            return Err(invalid(
                "MXFP4 shard artifact is empty or exceeds its caller-supplied byte budget",
            ));
        }
        if !crate::sha256::is_hex_digest(expected_sha256) {
            return Err(invalid(
                "expected MXFP4 shard SHA-256 must be lowercase hex",
            ));
        }
        let artifact_sha256 = crate::sha256::hex(&artifact);
        if artifact_sha256 != expected_sha256 {
            return Err(Error::new(
                "ARTIFACT_SHA256_MISMATCH",
                "MXFP4 shard artifact differs from the expected SHA-256",
            ));
        }

        let artifact = Arc::new(artifact);
        let tensors = SafeTensors::deserialize(artifact.as_slice())
            .map_err(|e| invalid(format!("invalid MXFP4 shard Safetensors: {e}")))?;
        let mut indexed =
            BTreeMap::<(Mxfp4ExpertNamespace, u32, u32), ExpertTensorViews<'_>>::new();
        for name in tensors.names() {
            let Some(ParsedExpertTensorName {
                namespace,
                layer,
                expert,
                projection,
                plane,
            }) = parse_expert_tensor_name(name)?
            else {
                continue;
            };
            let view = tensors
                .tensor(name)
                .map_err(|e| invalid(format!("invalid MXFP4 expert tensor: {e}")))?;
            let slots = indexed.entry((namespace, layer, expert)).or_default();
            let slot = match (projection, plane) {
                (ExpertProjection::W1, ExpertPlane::Weight) => &mut slots.w1_weight,
                (ExpertProjection::W1, ExpertPlane::Scale) => &mut slots.w1_scale,
                (ExpertProjection::W2, ExpertPlane::Weight) => &mut slots.w2_weight,
                (ExpertProjection::W2, ExpertPlane::Scale) => &mut slots.w2_scale,
                (ExpertProjection::W3, ExpertPlane::Weight) => &mut slots.w3_weight,
                (ExpertProjection::W3, ExpertPlane::Scale) => &mut slots.w3_scale,
            };
            if slot.replace(view).is_some() {
                return Err(invalid(format!(
                    "duplicate MXFP4 expert tensor identity after parsing: {name}"
                )));
            }
        }
        if indexed.is_empty() {
            return Err(invalid("MXFP4 shard contains no recognized expert tensors"));
        }

        let mut experts = BTreeMap::new();
        for ((namespace, layer, expert), slots) in indexed {
            let namespace_name = match namespace {
                Mxfp4ExpertNamespace::Backbone => "backbone",
                Mxfp4ExpertNamespace::Mtp => "MTP",
            };
            let missing = |name: &str| {
                invalid(format!(
                    "MXFP4 {namespace_name} expert {layer}/{expert} is missing tensor {name}"
                ))
            };
            let w1 = Mxfp4Matrix::from_safetensors_views(
                Arc::clone(&artifact),
                slots.w1_weight.ok_or_else(|| missing("w1.weight"))?,
                slots.w1_scale.ok_or_else(|| missing("w1.scale"))?,
            )?;
            let w2 = Mxfp4Matrix::from_safetensors_views(
                Arc::clone(&artifact),
                slots.w2_weight.ok_or_else(|| missing("w2.weight"))?,
                slots.w2_scale.ok_or_else(|| missing("w2.scale"))?,
            )?;
            let w3 = Mxfp4Matrix::from_safetensors_views(
                Arc::clone(&artifact),
                slots.w3_weight.ok_or_else(|| missing("w3.weight"))?,
                slots.w3_scale.ok_or_else(|| missing("w3.scale"))?,
            )?;
            if w1.rows != w3.rows || w1.cols != w3.cols || w2.cols != w1.rows || w2.rows != w1.cols
            {
                return Err(invalid(format!(
                    "MXFP4 {namespace_name} w1/w3/w2 projection shapes do not compose for expert {layer}/{expert}"
                )));
            }
            let compressed_bytes = w1
                .summary()
                .compressed_bytes
                .checked_add(w2.summary().compressed_bytes)
                .and_then(|n| n.checked_add(w3.summary().compressed_bytes))
                .ok_or_else(|| invalid("MXFP4 expert compressed byte count overflow"))?;
            experts.insert(
                (namespace, layer, expert),
                Mxfp4Expert {
                    namespace,
                    layer,
                    expert,
                    artifact: Arc::clone(&artifact),
                    artifact_sha256: artifact_sha256.clone(),
                    w1,
                    w3,
                    w2,
                    compressed_bytes,
                },
            );
        }

        Ok(Self {
            artifact,
            artifact_sha256,
            experts,
        })
    }

    /// Return one previously validated expert without reparsing or rescanning its scales.
    pub fn expert(&self, layer: u32, expert: u32) -> Result<&Mxfp4Expert> {
        self.expert_in(Mxfp4ExpertNamespace::Backbone, layer, expert)
    }

    /// Return one previously validated expert in an explicit checkpoint namespace.
    pub fn expert_in(
        &self,
        namespace: Mxfp4ExpertNamespace,
        layer: u32,
        expert: u32,
    ) -> Result<&Mxfp4Expert> {
        self.experts
            .get(&(namespace, layer, expert))
            .ok_or_else(|| {
                invalid(format!(
                    "MXFP4 {namespace:?} expert {layer}/{expert} is not in this shard"
                ))
            })
    }

    /// Count complete, validated expert identities in the shard.
    pub fn expert_count(&self) -> usize {
        self.experts.len()
    }

    /// Entire resident shard size, including header and non-expert tensors.
    pub fn backing_bytes(&self) -> usize {
        self.artifact.len()
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// Read-only bytes of the single immutable shard owner, for authenticated transfer.
    pub fn artifact_bytes(&self) -> &[u8] {
        self.artifact.as_slice()
    }

    pub(crate) fn shared_artifact_bytes(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.artifact)
    }
}

fn parse_expert_tensor_name(name: &str) -> Result<Option<ParsedExpertTensorName>> {
    if !name.contains(".ffn.experts.") {
        return Ok(None);
    }
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.len() != 7 || parts[2] != "ffn" || parts[3] != "experts" {
        return Err(invalid(format!(
            "malformed MXFP4 expert tensor name: {name}"
        )));
    }
    let namespace = match parts[0] {
        "layers" => Mxfp4ExpertNamespace::Backbone,
        "mtp" => Mxfp4ExpertNamespace::Mtp,
        _ => {
            return Err(invalid(format!(
                "unsupported MXFP4 expert namespace in tensor name: {name}"
            )))
        }
    };
    let layer = parts[1]
        .parse::<u32>()
        .map_err(|_| invalid(format!("invalid layer ID in MXFP4 tensor name: {name}")))?;
    let expert = parts[4]
        .parse::<u32>()
        .map_err(|_| invalid(format!("invalid expert ID in MXFP4 tensor name: {name}")))?;
    if parts[1] != layer.to_string() || parts[4] != expert.to_string() {
        return Err(invalid(format!(
            "MXFP4 layer and expert IDs must use canonical decimal spelling: {name}"
        )));
    }
    let projection = match parts[5] {
        "w1" => ExpertProjection::W1,
        "w2" => ExpertProjection::W2,
        "w3" => ExpertProjection::W3,
        _ => {
            return Err(invalid(format!(
                "unsupported MXFP4 projection tensor: {name}"
            )))
        }
    };
    let plane = match parts[6] {
        "weight" => ExpertPlane::Weight,
        "scale" => ExpertPlane::Scale,
        _ => return Err(invalid(format!("unsupported MXFP4 plane tensor: {name}"))),
    };
    Ok(Some(ParsedExpertTensorName {
        namespace,
        layer,
        expert,
        projection,
        plane,
    }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinnedExpertIndex {
    metadata: PinnedExpertIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinnedExpertIndexMetadata {
    total_size: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PinnedExpertPayloadRequest<'a> {
    pub index_bytes: &'a [u8],
    pub header_bytes: &'a [u8],
    pub expected_header_sha256: &'a str,
    pub header_shard_name: &'a str,
    pub source_shard_bytes: u64,
    pub max_source_shard_bytes: u64,
    pub namespace: Mxfp4ExpertNamespace,
    pub layer: u32,
    pub experts: &'a [u32],
}

fn pinned_shard_basename(name: &str) -> bool {
    let Some(stem) = name
        .strip_prefix("model-")
        .and_then(|value| value.strip_suffix(".safetensors"))
    else {
        return false;
    };
    let Some((ordinal, count)) = stem.split_once("-of-") else {
        return false;
    };
    ordinal.len() == 5
        && ordinal.bytes().all(|byte| byte.is_ascii_digit())
        && ordinal
            .parse::<u8>()
            .is_ok_and(|value| (1..=48).contains(&value))
        && count == "00048"
}

/// Build a bounded, metadata-only range plan from the exact pinned full-checkpoint
/// index and the matching Safetensors header. The caller must supply the independently
/// recorded header digest, shard identity, full file length, and resource ceiling.
/// No checkpoint payload is read here; use the returned ranges only with an admitted,
/// hash-checked source and keep the existing atomic refusal for absent resident experts.
pub fn plan_pinned_mxfp4_expert_payloads(
    request: PinnedExpertPayloadRequest<'_>,
) -> Result<Mxfp4ExpertPayloadPlan> {
    let PinnedExpertPayloadRequest {
        index_bytes,
        header_bytes,
        expected_header_sha256,
        header_shard_name,
        source_shard_bytes,
        max_source_shard_bytes,
        namespace,
        layer,
        experts,
    } = request;
    if index_bytes.is_empty() || index_bytes.len() > MAX_PINNED_EXPERT_INDEX_BYTES {
        return Err(invalid(
            "pinned expert index exceeds its 8 MiB metadata bound",
        ));
    }
    if crate::sha256::hex(index_bytes) != PINNED_EXPERT_INDEX_SHA256 {
        return Err(invalid(
            "DeepSeek expert index differs from the pinned revision",
        ));
    }
    if header_bytes.is_empty()
        || header_bytes.len() > MAX_PINNED_EXPERT_INDEX_BYTES
        || !crate::sha256::is_hex_digest(expected_header_sha256)
        || crate::sha256::hex(header_bytes) != expected_header_sha256
    {
        return Err(invalid(
            "pinned Safetensors header is empty, oversized, or has a wrong digest",
        ));
    }
    if !pinned_shard_basename(header_shard_name)
        || max_source_shard_bytes == 0
        || source_shard_bytes == 0
        || source_shard_bytes > max_source_shard_bytes
    {
        return Err(invalid(
            "DeepSeek expert shard identity or caller byte ceiling is invalid",
        ));
    }
    if experts.is_empty() || experts.len() > MAX_EXPERTS_PER_PAYLOAD_PLAN {
        return Err(invalid(
            "expert payload plan must contain between one and six experts",
        ));
    }
    let (max_layers, max_experts) = match namespace {
        Mxfp4ExpertNamespace::Backbone => (40, 384),
        Mxfp4ExpertNamespace::Mtp => (3, 128),
    };
    if layer >= max_layers {
        return Err(invalid(
            "expert payload plan layer is outside the pinned namespace",
        ));
    }
    let mut unique_experts = BTreeSet::new();
    if experts
        .iter()
        .any(|&expert| expert >= max_experts || !unique_experts.insert(expert))
    {
        return Err(invalid(
            "expert payload plan has duplicate or out-of-range expert IDs",
        ));
    }

    let index: PinnedExpertIndex = serde_json::from_slice(index_bytes)
        .map_err(|error| invalid(format!("invalid pinned DeepSeek expert index: {error}")))?;
    if index.weight_map.len() != 96_085 || index.metadata.total_size != 510_286_023_000 {
        return Err(invalid(
            "pinned DeepSeek expert index metadata differs from its revision",
        ));
    }
    let metadata: Metadata = serde_json::from_slice(header_bytes)
        .map_err(|error| invalid(format!("invalid pinned Safetensors header: {error}")))?;
    let header_len = u64::try_from(header_bytes.len())
        .map_err(|_| invalid("Safetensors header length does not fit u64"))?;
    let data_start_bytes = header_len
        .checked_add(8)
        .ok_or_else(|| invalid("Safetensors data-start offset overflow"))?;
    let data_len = u64::try_from(metadata.data_len())
        .map_err(|_| invalid("Safetensors data length does not fit u64"))?;
    if data_start_bytes
        .checked_add(data_len)
        .filter(|&length| length == source_shard_bytes)
        .is_none()
    {
        return Err(invalid(
            "Safetensors header does not cover the exact source shard length",
        ));
    }

    let root = match namespace {
        Mxfp4ExpertNamespace::Backbone => "layers",
        Mxfp4ExpertNamespace::Mtp => "mtp",
    };
    let mut source_shard: Option<String> = None;
    let mut tensors = Vec::with_capacity(experts.len() * 6);
    let mut selected_payload_bytes = 0_u64;
    for &expert in experts {
        let prefix = format!("{root}.{layer}.ffn.experts.{expert}");
        let expected = [
            (
                format!("{prefix}.w1.weight"),
                Dtype::I8,
                vec![2304, 2560],
                "I8",
            ),
            (
                format!("{prefix}.w1.scale"),
                Dtype::F8_E8M0,
                vec![2304, 160],
                "F8_E8M0",
            ),
            (
                format!("{prefix}.w2.weight"),
                Dtype::I8,
                vec![5120, 1152],
                "I8",
            ),
            (
                format!("{prefix}.w2.scale"),
                Dtype::F8_E8M0,
                vec![5120, 72],
                "F8_E8M0",
            ),
            (
                format!("{prefix}.w3.weight"),
                Dtype::I8,
                vec![2304, 2560],
                "I8",
            ),
            (
                format!("{prefix}.w3.scale"),
                Dtype::F8_E8M0,
                vec![2304, 160],
                "F8_E8M0",
            ),
        ];
        for (name, expected_dtype, expected_shape, dtype_label) in expected {
            let shard = index
                .weight_map
                .get(&name)
                .ok_or_else(|| invalid(format!("pinned index is missing expert tensor {name}")))?;
            if !pinned_shard_basename(shard)
                || source_shard
                    .as_ref()
                    .is_some_and(|expected| expected != shard)
            {
                return Err(invalid(
                    "selected expert tensor names do not share one safe pinned shard",
                ));
            }
            source_shard.get_or_insert_with(|| shard.clone());

            let info = metadata.info(&name).ok_or_else(|| {
                invalid(format!(
                    "selected tensor is absent from shard header: {name}"
                ))
            })?;
            if shard != header_shard_name
                || info.dtype != expected_dtype
                || info.shape != expected_shape
            {
                return Err(invalid(format!(
                    "pinned index/header tensor identity, dtype, or shape differs: {name}"
                )));
            }
            let relative_start = u64::try_from(info.data_offsets.0)
                .map_err(|_| invalid("Safetensors tensor offset does not fit u64"))?;
            let relative_end = u64::try_from(info.data_offsets.1)
                .map_err(|_| invalid("Safetensors tensor end does not fit u64"))?;
            let expected_len = expected_shape
                .iter()
                .try_fold(1_u64, |product, &dimension| {
                    product.checked_mul(u64::try_from(dimension).ok()?)
                });
            let Some(expected_len) = expected_len else {
                return Err(invalid("selected tensor byte length overflow"));
            };
            if relative_start >= relative_end
                || relative_end > data_len
                || relative_end - relative_start != expected_len
            {
                return Err(invalid(format!(
                    "selected tensor range is outside the bounded source shard: {name}"
                )));
            }
            selected_payload_bytes = selected_payload_bytes
                .checked_add(expected_len)
                .ok_or_else(|| invalid("selected expert payload byte count overflow"))?;
            tensors.push(Mxfp4TensorPayloadRange {
                expert,
                name,
                dtype: dtype_label,
                shape: expected_shape,
                file_offset_start: data_start_bytes
                    .checked_add(relative_start)
                    .ok_or_else(|| invalid("absolute tensor start offset overflow"))?,
                file_offset_end: data_start_bytes
                    .checked_add(relative_end)
                    .ok_or_else(|| invalid("absolute tensor end offset overflow"))?,
            });
        }
    }

    let mut ordered_ranges = tensors
        .iter()
        .map(|tensor| (tensor.file_offset_start, tensor.file_offset_end))
        .collect::<Vec<_>>();
    ordered_ranges.sort_unstable();
    if ordered_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(invalid(
            "selected expert tensor ranges overlap in the source shard",
        ));
    }

    let source_shard = source_shard.ok_or_else(|| invalid("expert payload plan is empty"))?;
    if source_shard != header_shard_name {
        return Err(invalid(
            "pinned index selected a different Safetensors shard",
        ));
    }
    Ok(Mxfp4ExpertPayloadPlan {
        model_revision: REVISION.to_owned(),
        namespace,
        layer,
        experts: experts.to_vec(),
        source_shard,
        source_shard_bytes,
        header_bytes: header_len,
        data_start_bytes,
        selected_payload_bytes,
        tensors,
    })
}

impl Mxfp4Expert {
    /// Load exactly one expert's six MXFP4 tensors without copying any plane out of the
    /// input buffer. The input Vec is moved into the owner, so callers should read the
    /// file directly into a Vec and transfer ownership here.
    pub fn from_safetensors_expert_artifact(
        artifact: Vec<u8>,
        layer: u32,
        expert: u32,
        expected_sha256: &str,
    ) -> Result<Self> {
        if artifact.is_empty()
            || u64::try_from(artifact.len())
                .map_err(|_| invalid("MXFP4 expert artifact size does not fit u64"))?
                > MAX_MXFP4_EXPERT_ARTIFACT_BYTES
        {
            return Err(invalid("MXFP4 expert artifact is empty or exceeds 64 MiB"));
        }
        if !crate::sha256::is_hex_digest(expected_sha256) {
            return Err(invalid(
                "expected MXFP4 expert SHA-256 must be lowercase hex",
            ));
        }
        let artifact_sha256 = crate::sha256::hex(&artifact);
        if artifact_sha256 != expected_sha256 {
            return Err(Error::new(
                "ARTIFACT_SHA256_MISMATCH",
                "MXFP4 expert artifact differs from the expected SHA-256",
            ));
        }
        let artifact = Arc::new(artifact);
        let tensors = SafeTensors::deserialize(artifact.as_slice())
            .map_err(|e| invalid(format!("invalid MXFP4 expert Safetensors: {e}")))?;
        let prefix = format!("layers.{layer}.ffn.experts.{expert}");
        let mut actual_names = tensors
            .names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut expected_names = [
            format!("{prefix}.w1.weight"),
            format!("{prefix}.w1.scale"),
            format!("{prefix}.w2.weight"),
            format!("{prefix}.w2.scale"),
            format!("{prefix}.w3.weight"),
            format!("{prefix}.w3.scale"),
        ];
        actual_names.sort_unstable();
        expected_names.sort_unstable();
        if actual_names != expected_names {
            return Err(invalid(
                "MXFP4 expert artifact must contain exactly the six expected w1/w2/w3 tensors",
            ));
        }

        let load = |projection: &str| -> Result<Mxfp4Matrix> {
            let weight = tensors
                .tensor(&format!("{prefix}.{projection}.weight"))
                .map_err(|e| invalid(format!("missing MXFP4 weight tensor: {e}")))?;
            let scale = tensors
                .tensor(&format!("{prefix}.{projection}.scale"))
                .map_err(|e| invalid(format!("missing MXFP4 scale tensor: {e}")))?;
            Mxfp4Matrix::from_safetensors_views(Arc::clone(&artifact), weight, scale)
        };
        let w1 = load("w1")?;
        let w3 = load("w3")?;
        let w2 = load("w2")?;
        if w1.rows != w3.rows || w1.cols != w3.cols || w2.cols != w1.rows || w2.rows != w1.cols {
            return Err(invalid("MXFP4 w1/w3/w2 projection shapes do not compose"));
        }
        let compressed_bytes = w1
            .summary()
            .compressed_bytes
            .checked_add(w3.summary().compressed_bytes)
            .and_then(|n| n.checked_add(w2.summary().compressed_bytes))
            .ok_or_else(|| invalid("MXFP4 expert compressed byte count overflow"))?;
        Ok(Self {
            namespace: Mxfp4ExpertNamespace::Backbone,
            layer,
            expert,
            artifact,
            artifact_sha256,
            w1,
            w3,
            w2,
            compressed_bytes,
        })
    }

    pub fn namespace(&self) -> Mxfp4ExpertNamespace {
        self.namespace
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn expert(&self) -> u32 {
        self.expert
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// Full resident Safetensors buffer, including its small metadata header.
    pub fn backing_bytes(&self) -> usize {
        self.artifact.len()
    }

    /// Sum of the packed FP4 codes and UE8M0 scale bytes for w1, w3 and w2.
    pub fn compressed_bytes(&self) -> u64 {
        self.compressed_bytes
    }

    pub fn w1(&self) -> &Mxfp4Matrix {
        &self.w1
    }

    pub fn w2(&self) -> &Mxfp4Matrix {
        &self.w2
    }

    pub fn w3(&self) -> &Mxfp4Matrix {
        &self.w3
    }

    /// Convert the three immutable compressed E2M1/E8M0 projections directly to
    /// Skinny's existing QPN2 resident buffers. Codes stay packed; scale bytes are
    /// expanded from one K32 byte to two K16 E4M3 power bytes. No dense weight plane
    /// or runtime fallback is created. Admission is per whole projection because the
    /// current resident ABI requires one common global factor for that projection.
    ///
    /// This proves CPU format/layout equivalence and the ordered SM70 half decode
    /// domain for admitted scales. It does not claim CUDA execution or model quality.
    pub fn to_qpn2_resident_stack(
        &self,
        rank: u32,
        local: u32,
        epoch: u64,
    ) -> Result<crate::resident::ResidentStack> {
        let gate = convert_mxfp4_plane(&self.w1)?;
        let up = convert_mxfp4_plane(&self.w3)?;
        let down = convert_mxfp4_plane(&self.w2)?;
        if gate.rows != up.rows
            || gate.columns != up.columns
            || down.columns != gate.rows
            || down.rows != gate.columns
        {
            return Err(invalid(
                "converted QPN2 DeepSeek projections do not compose",
            ));
        }

        let mut codes13 = checked_zeroed_bytes(
            gate.codes
                .len()
                .checked_add(up.codes.len())
                .ok_or_else(|| invalid("QPN2 w1/w3 code length overflow"))?,
        )?;
        let mut scales13 = checked_zeroed_bytes(
            gate.scales
                .len()
                .checked_add(up.scales.len())
                .ok_or_else(|| invalid("QPN2 w1/w3 scale length overflow"))?,
        )?;
        let mut gscales13 = Vec::new();
        let gate_tiles = gate.rows / 32;
        let up_tiles = up.rows / 32;
        gscales13
            .try_reserve_exact(gate_tiles + up_tiles)
            .map_err(|error| {
                Error::new(
                    "MXFP4_QPN2_ALLOCATION",
                    format!("could not reserve QPN2 w1/w3 factors: {error}"),
                )
            })?;
        let gate_codes_len = gate.codes.len();
        let gate_scales_len = gate.scales.len();
        codes13[..gate_codes_len].copy_from_slice(&gate.codes);
        codes13[gate_codes_len..].copy_from_slice(&up.codes);
        scales13[..gate_scales_len].copy_from_slice(&gate.scales);
        scales13[gate_scales_len..].copy_from_slice(&up.scales);
        gscales13.extend(std::iter::repeat_n(gate.global_dequant, gate_tiles));
        gscales13.extend(std::iter::repeat_n(up.global_dequant, up_tiles));
        let gscales2: Vec<f32> = std::iter::repeat_n(down.global_dequant, down.rows / 32).collect();

        let payload_bytes = [
            codes13.len() as u64,
            scales13.len() as u64,
            (gscales13.len() as u64)
                .checked_mul(4)
                .ok_or_else(|| invalid("QPN2 w1/w3 factor byte count overflow"))?,
            down.codes.len() as u64,
            down.scales.len() as u64,
            (gscales2.len() as u64)
                .checked_mul(4)
                .ok_or_else(|| invalid("QPN2 w2 factor byte count overflow"))?,
        ]
        .into_iter()
        .try_fold(0u64, |sum, value| sum.checked_add(value))
        .ok_or_else(|| invalid("QPN2 resident payload byte count overflow"))?;

        crate::resident::from_parts(crate::resident::Parts {
            model_revision: REVISION.to_owned(),
            layer: self.layer,
            rank,
            epoch,
            physical_experts: vec![(self.expert, local)],
            resident_order: vec![local],
            shape: [1, gate.rows * 2, gate.columns, down.rows, down.columns],
            payload_bytes,
            codes13,
            scales13,
            gscales13,
            codes2: down.codes,
            scales2: down.scales,
            gscales2,
        })
    }

    pub fn forward_reference(
        &self,
        input_bf16: &[f32],
        routing_weight: Option<f32>,
    ) -> Result<Vec<f32>> {
        single_token_expert_forward_reference(
            &self.w1,
            &self.w3,
            &self.w2,
            input_bf16,
            routing_weight,
        )
    }
}

/// Convert a Safetensors view back to a checked range in the original artifact. The
/// pointer arithmetic is address-only; all stored access is through safe slice bounds.
fn artifact_range(backing: &[u8], view: &[u8]) -> Result<Range<usize>> {
    let start = (view.as_ptr() as usize)
        .checked_sub(backing.as_ptr() as usize)
        .ok_or_else(|| invalid("Safetensors plane does not point into its artifact"))?;
    let end = start
        .checked_add(view.len())
        .filter(|&end| end <= backing.len())
        .ok_or_else(|| invalid("Safetensors plane extends outside its artifact"))?;
    let covered = backing
        .get(start..end)
        .ok_or_else(|| invalid("Safetensors plane range is outside its artifact"))?;
    if covered.as_ptr() != view.as_ptr() || covered.len() != view.len() {
        return Err(invalid(
            "Safetensors plane range does not preserve artifact identity",
        ));
    }
    Ok(start..end)
}

/// Code and scale plane byte counts, checked before any length comparison.
pub fn mxfp4_plane_bytes(rows: u64, cols: u64) -> Result<(u64, u64)> {
    if rows == 0 || cols == 0 || rows > MAX_MXFP4_DIM || cols > MAX_MXFP4_DIM {
        return Err(invalid("MXFP4 dimensions must be in 1..=65536"));
    }
    if !cols.is_multiple_of(MXFP4_BLOCK) {
        return Err(invalid("MXFP4 K must be a multiple of 32 (no padding)"));
    }
    let code_bytes = rows
        .checked_mul(cols / 2)
        .ok_or_else(|| invalid("MXFP4 size overflow"))?;
    let scale_bytes = rows
        .checked_mul(cols / MXFP4_BLOCK)
        .ok_or_else(|| invalid("MXFP4 size overflow"))?;
    if code_bytes
        .checked_add(scale_bytes)
        .is_none_or(|n| n > MAX_MXFP4_COMPRESSED_BYTES)
    {
        return Err(invalid("MXFP4 compressed matrix exceeds 64 MiB"));
    }
    Ok((code_bytes, scale_bytes))
}

/// Native FP32 SiLU in Torch CPU order, `g / (1 + exp(-g))`: for `g < -88.72` the f32
/// `exp` overflows to +Inf and the quotient underflows to exactly -0.
fn silu_f32(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

/// V4.1 `Expert.forward` up to (excluding) `x.to(dtype)`: clamp up to [-10, 10], clamp
/// gate from above only, `silu(gate) * up` in FP32, then the optional routing weight
/// once. Returns the pre-cast FP32 activation. No BF16 cast, activation quantisation,
/// down GEMM or routing.
pub fn pre_down_activation(gate: &[f32], up: &[f32], weight: Option<f32>) -> Result<Vec<f32>> {
    if gate.len() > MAX_ACTIVATION_LEN {
        return Err(invalid("activation length must be in 0..=1048576"));
    }
    if gate.len() != up.len() {
        return Err(invalid("gate and up must have the same shape"));
    }
    if gate.iter().chain(up).any(|x| !x.is_finite()) {
        return Err(Error::new("NONFINITE_INPUT", "gate/up contain NaN or Inf"));
    }
    if let Some(w) = weight {
        if !w.is_finite() || w < 0.0 {
            return Err(invalid(
                "routing weight must be finite and >= 0 (-0 allowed)",
            ));
        }
    }
    let mut out = Vec::with_capacity(gate.len());
    for (&g, &u) in gate.iter().zip(up) {
        let u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let g = g.min(SWIGLU_LIMIT);
        let mut x = silu_f32(g) * u;
        if let Some(w) = weight {
            x *= w;
        }
        if !x.is_finite() {
            return Err(Error::new("NUMERICAL_OVERFLOW", "activation nonfinite"));
        }
        out.push(x);
    }
    Ok(out)
}

#[cfg(test)]
mod a100_shared_artifact_tests {
    use super::{Mxfp4Expert, Mxfp4Plane};
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn all_six_planes_share_one_immutable_safetensors_allocation() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
        let expert = Mxfp4Expert::from_safetensors_expert_artifact(
            std::fs::read(path).unwrap(),
            0,
            0,
            "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b",
        )
        .unwrap();
        let planes = [
            &expert.w1.codes,
            &expert.w1.scales,
            &expert.w3.codes,
            &expert.w3.scales,
            &expert.w2.codes,
            &expert.w2.scales,
        ];
        for plane in planes {
            let Mxfp4Plane::SharedArtifact { bytes, range } = plane else {
                panic!("expert loader copied a Safetensors plane into owned storage");
            };
            assert!(Arc::ptr_eq(bytes, &expert.artifact));
            assert!(range.start < range.end);
            assert!(range.end <= expert.artifact.len());
        }
        assert_eq!(Arc::strong_count(&expert.artifact), 7);
        assert_eq!(expert.backing_bytes(), 18_801_312);
        assert_eq!(expert.compressed_bytes(), 18_800_640);
    }
}

#[cfg(test)]
mod a101_shared_shard_tests {
    use super::{Mxfp4ExpertShard, Mxfp4Plane};
    use safetensors::tensor::{serialize, SafeTensors};
    use std::sync::Arc;

    fn two_expert_shard() -> Vec<u8> {
        let original =
            include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
        let tensors = SafeTensors::deserialize(original).unwrap();
        let names = tensors.names();
        let mut entries = Vec::with_capacity(12);
        for expert in [1, 0] {
            for name in &names {
                let suffix = name
                    .strip_prefix("layers.0.ffn.experts.0.")
                    .expect("pinned fixture has an expert0 prefix");
                entries.push((
                    format!("layers.0.ffn.experts.{expert}.{suffix}"),
                    tensors.tensor(name).unwrap(),
                ));
            }
        }
        serialize(entries, None).unwrap()
    }

    #[test]
    fn every_projection_plane_for_multiple_experts_shares_one_backing_allocation() {
        let bytes = two_expert_shard();
        let digest = crate::sha256::hex(&bytes);
        let shard =
            Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
                .unwrap();
        assert_eq!(shard.expert_count(), 2);
        assert_eq!(Arc::strong_count(&shard.artifact), 15);
        for expert in shard.experts.values() {
            assert!(Arc::ptr_eq(&expert.artifact, &shard.artifact));
            let planes = [
                &expert.w1.codes,
                &expert.w1.scales,
                &expert.w2.codes,
                &expert.w2.scales,
                &expert.w3.codes,
                &expert.w3.scales,
            ];
            for plane in planes {
                let Mxfp4Plane::SharedArtifact { bytes, range } = plane else {
                    panic!("shard loader copied a tensor plane into owned storage");
                };
                assert!(Arc::ptr_eq(bytes, &shard.artifact));
                assert!(range.start < range.end);
                assert!(range.end <= shard.artifact.len());
            }
        }
    }
}

#[cfg(test)]
mod a103_official_shard_header_tests {
    use super::{
        parse_expert_tensor_name, ExpertPlane, ExpertProjection, Mxfp4ExpertNamespace,
        ParsedExpertTensorName,
    };
    use std::collections::BTreeMap;

    #[test]
    fn pinned_full_shard_header_matches_the_multi_expert_name_parser() {
        let bytes = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
        );
        assert_eq!(
            crate::sha256::hex(bytes),
            "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449"
        );
        let header: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let tensors = header.as_object().unwrap();
        assert_eq!(tensors.len(), 2334); // pinned source header has no __metadata__ key
        let mut expert_masks = BTreeMap::<(u32, u32), u8>::new();
        let mut expert_tensor_count = 0;
        let mut non_expert_tensor_count = 0;
        for (name, entry) in tensors {
            if name == "__metadata__" {
                continue;
            }
            let Some(ParsedExpertTensorName {
                namespace,
                layer,
                expert,
                projection,
                plane,
            }) = parse_expert_tensor_name(name).unwrap()
            else {
                assert!(
                    !name.contains(".ffn.experts."),
                    "unhandled expert key {name}"
                );
                non_expert_tensor_count += 1;
                continue;
            };
            assert_eq!(namespace, Mxfp4ExpertNamespace::Backbone, "{name}");
            let (bit, dtype, shape) = match (projection, plane) {
                (ExpertProjection::W1, ExpertPlane::Weight) => (1, "I8", [2304u64, 2560u64]),
                (ExpertProjection::W1, ExpertPlane::Scale) => (2, "F8_E8M0", [2304u64, 160u64]),
                (ExpertProjection::W2, ExpertPlane::Weight) => (4, "I8", [5120u64, 1152u64]),
                (ExpertProjection::W2, ExpertPlane::Scale) => (8, "F8_E8M0", [5120u64, 72u64]),
                (ExpertProjection::W3, ExpertPlane::Weight) => (16, "I8", [2304u64, 2560u64]),
                (ExpertProjection::W3, ExpertPlane::Scale) => (32, "F8_E8M0", [2304u64, 160u64]),
            };
            assert_eq!(entry["dtype"].as_str(), Some(dtype), "{name}");
            let actual_shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_u64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual_shape, shape, "{name}");
            let offsets = entry["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap();
            let end = offsets[1].as_u64().unwrap();
            assert_eq!(end - start, shape[0] * shape[1], "{name}");
            let mask = expert_masks.entry((layer, expert)).or_default();
            assert_eq!(*mask & bit, 0, "duplicate expert plane {name}");
            *mask |= bit;
            expert_tensor_count += 1;
        }
        assert_eq!(expert_tensor_count, 2304);
        assert_eq!(non_expert_tensor_count, 30);
        assert_eq!(expert_masks.len(), 384);
        assert!(expert_masks.keys().all(|&(layer, _)| layer == 0));
        assert_eq!(
            expert_masks
                .keys()
                .map(|&(_, expert)| expert)
                .collect::<Vec<_>>(),
            (0..384).collect::<Vec<_>>()
        );
        assert!(expert_masks.values().all(|&mask| mask == 0b11_1111));
    }
}

#[cfg(test)]
mod a104_official_mtp_shard_header_tests {
    use super::{
        parse_expert_tensor_name, ExpertPlane, ExpertProjection, Mxfp4ExpertNamespace,
        ParsedExpertTensorName,
    };
    use super::{Mxfp4ExpertShard, Mxfp4Plane};
    use safetensors::tensor::{serialize, SafeTensors};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn backbone_and_mtp_planes_share_the_single_immutable_shard_backing() {
        let fixture =
            include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
        let tensors = SafeTensors::deserialize(fixture).unwrap();
        let mut entries = Vec::with_capacity(12);
        for namespace in ["mtp", "layers"] {
            for name in tensors.names() {
                let suffix = name.strip_prefix("layers.0.ffn.experts.0.").unwrap();
                entries.push((
                    format!("{namespace}.0.ffn.experts.0.{suffix}"),
                    tensors.tensor(name).unwrap(),
                ));
            }
        }
        let artifact = serialize(entries, None).unwrap();
        let digest = crate::sha256::hex(&artifact);
        let shard = Mxfp4ExpertShard::from_safetensors_shard_artifact(
            artifact.clone(),
            &digest,
            artifact.len(),
        )
        .unwrap();
        let experts = [
            shard
                .expert_in(Mxfp4ExpertNamespace::Backbone, 0, 0)
                .unwrap(),
            shard.expert_in(Mxfp4ExpertNamespace::Mtp, 0, 0).unwrap(),
        ];
        assert_eq!(Arc::strong_count(&shard.artifact), 15);
        for expert in experts {
            assert!(Arc::ptr_eq(&expert.artifact, &shard.artifact));
            for plane in [
                &expert.w1.codes,
                &expert.w1.scales,
                &expert.w2.codes,
                &expert.w2.scales,
                &expert.w3.codes,
                &expert.w3.scales,
            ] {
                let Mxfp4Plane::SharedArtifact { bytes, range } = plane else {
                    panic!("namespace-aware loader copied a Safetensors plane");
                };
                assert!(Arc::ptr_eq(bytes, &shard.artifact));
                assert!(range.start < range.end);
                assert!(range.end <= shard.artifact.len());
            }
        }
    }

    #[test]
    fn all_pinned_mtp_headers_match_the_namespace_parser_and_official_schema() {
        let shards: [(&[u8], &str, u32, usize, usize); 3] = [
            (
                include_bytes!("../evidence/continuation_a104_mtp_namespace/model-00044-of-00048.safetensors.header.314113f8.json"),
                "bcfa221d7a325e0a910e262d782c0e97bd5e5b7cbde6c9b480bb88ab7c23a36e",
                0,
                801,
                33,
            ),
            (
                include_bytes!("../evidence/continuation_a104_mtp_namespace/model-00045-of-00048.safetensors.header.314113f8.json"),
                "c8f2cad66c6387793b335cbc10231661b61f2ec7a8c814bccb8d6ae418200675",
                1,
                798,
                30,
            ),
            (
                include_bytes!("../evidence/continuation_a104_mtp_namespace/model-00046-of-00048.safetensors.header.314113f8.json"),
                "1baad4e59ff44a15727f0c294d015da70a90502fd5d1e844c1c0939a8194793e",
                2,
                802,
                34,
            ),
        ];
        let expected_planes = [
            (
                ExpertProjection::W1,
                ExpertPlane::Weight,
                1,
                "I8",
                [2304, 2560],
            ),
            (
                ExpertProjection::W1,
                ExpertPlane::Scale,
                2,
                "F8_E8M0",
                [2304, 160],
            ),
            (
                ExpertProjection::W2,
                ExpertPlane::Weight,
                4,
                "I8",
                [5120, 1152],
            ),
            (
                ExpertProjection::W2,
                ExpertPlane::Scale,
                8,
                "F8_E8M0",
                [5120, 72],
            ),
            (
                ExpertProjection::W3,
                ExpertPlane::Weight,
                16,
                "I8",
                [2304, 2560],
            ),
            (
                ExpertProjection::W3,
                ExpertPlane::Scale,
                32,
                "F8_E8M0",
                [2304, 160],
            ),
        ];

        for (bytes, expected_sha256, expected_layer, expected_tensors, expected_non_experts) in
            shards
        {
            assert_eq!(crate::sha256::hex(bytes), expected_sha256);
            let header: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            let tensors = header.as_object().unwrap();
            assert_eq!(tensors.len(), expected_tensors);
            let mut masks = BTreeMap::<(u32, u32), u8>::new();
            let mut expert_tensors = 0;
            let mut non_expert_tensors = 0;
            for (name, entry) in tensors {
                if name == "__metadata__" {
                    continue;
                }
                let Some(ParsedExpertTensorName {
                    namespace,
                    layer,
                    expert,
                    projection,
                    plane,
                }) = parse_expert_tensor_name(name).unwrap()
                else {
                    assert!(
                        !name.contains(".ffn.experts."),
                        "unhandled MTP expert key {name}"
                    );
                    non_expert_tensors += 1;
                    continue;
                };
                assert_eq!(namespace, Mxfp4ExpertNamespace::Mtp, "{name}");
                assert_eq!(layer, expected_layer, "{name}");
                let (bit, dtype, shape) = expected_planes
                    .iter()
                    .find_map(|(expected_projection, expected_plane, bit, dtype, shape)| {
                        (*expected_projection == projection && *expected_plane == plane)
                            .then_some((*bit, *dtype, *shape))
                    })
                    .expect("projection plane is part of the pinned MXFP4 schema");
                assert_eq!(entry["dtype"].as_str(), Some(dtype), "{name}");
                let actual_shape = entry["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|dim| dim.as_u64().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(actual_shape, shape, "{name}");
                let offsets = entry["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap();
                let end = offsets[1].as_u64().unwrap();
                assert_eq!(end - start, shape[0] * shape[1], "{name}");
                let mask = masks.entry((layer, expert)).or_default();
                assert_eq!(*mask & bit, 0, "duplicate MTP expert plane {name}");
                *mask |= bit;
                expert_tensors += 1;
            }
            assert_eq!(expert_tensors, 768);
            assert_eq!(non_expert_tensors, expected_non_experts);
            assert_eq!(masks.len(), 128);
            assert_eq!(
                masks.keys().map(|&(_, expert)| expert).collect::<Vec<_>>(),
                (0..128).collect::<Vec<_>>()
            );
            assert!(masks.values().all(|&mask| mask == 0b11_1111));
        }
    }
}

#[cfg(test)]
mod a105_full_pinned_expert_index_tests {
    use super::{
        parse_expert_tensor_name, ExpertPlane, ExpertProjection, Mxfp4ExpertNamespace,
        ParsedExpertTensorName,
    };
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct OfficialIndex {
        weight_map: BTreeMap<String, String>,
    }

    #[test]
    fn pinned_full_index_expert_names_have_exact_backbone_and_mtp_coverage() {
        let bytes = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        assert_eq!(
            crate::sha256::hex(bytes),
            "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
        );
        let index: OfficialIndex = serde_json::from_slice(bytes).unwrap();
        assert_eq!(index.weight_map.len(), 96_085);

        let mut masks = BTreeMap::<(Mxfp4ExpertNamespace, u32, u32), u8>::new();
        let mut source_by_layer = BTreeMap::<(Mxfp4ExpertNamespace, u32), &str>::new();
        let mut expert_tensor_count = 0usize;
        for (name, source) in &index.weight_map {
            if !name.contains(".ffn.experts.") {
                continue;
            }
            let Some(ParsedExpertTensorName {
                namespace,
                layer,
                expert,
                projection,
                plane,
            }) = parse_expert_tensor_name(name).unwrap()
            else {
                panic!("official index expert key was ignored: {name}");
            };
            let bit = match (projection, plane) {
                (ExpertProjection::W1, ExpertPlane::Weight) => 1,
                (ExpertProjection::W1, ExpertPlane::Scale) => 2,
                (ExpertProjection::W2, ExpertPlane::Weight) => 4,
                (ExpertProjection::W2, ExpertPlane::Scale) => 8,
                (ExpertProjection::W3, ExpertPlane::Weight) => 16,
                (ExpertProjection::W3, ExpertPlane::Scale) => 32,
            };
            let mask = masks.entry((namespace, layer, expert)).or_default();
            assert_eq!(*mask & bit, 0, "duplicate official expert plane {name}");
            *mask |= bit;
            if let Some(previous_source) = source_by_layer.insert((namespace, layer), source) {
                assert_eq!(previous_source, source, "expert layer split across shards");
            }
            expert_tensor_count += 1;
        }

        let backbone = masks
            .iter()
            .filter(|((namespace, _, _), _)| *namespace == Mxfp4ExpertNamespace::Backbone)
            .collect::<Vec<_>>();
        let mtp = masks
            .iter()
            .filter(|((namespace, _, _), _)| *namespace == Mxfp4ExpertNamespace::Mtp)
            .collect::<Vec<_>>();
        assert_eq!(backbone.len(), 40 * 384);
        assert_eq!(mtp.len(), 3 * 128);
        assert_eq!(expert_tensor_count, 92_160 + 2_304);
        assert!(masks.values().all(|&mask| mask == 0b11_1111));
        assert_eq!(
            source_by_layer
                .keys()
                .filter(|(namespace, _)| *namespace == Mxfp4ExpertNamespace::Backbone)
                .count(),
            40
        );
        assert_eq!(
            source_by_layer
                .keys()
                .filter(|(namespace, _)| *namespace == Mxfp4ExpertNamespace::Mtp)
                .count(),
            3
        );
        for layer in 0..40 {
            assert_eq!(
                masks
                    .keys()
                    .filter(|(namespace, actual_layer, _)| {
                        *namespace == Mxfp4ExpertNamespace::Backbone && *actual_layer == layer
                    })
                    .map(|(_, _, expert)| *expert)
                    .collect::<Vec<_>>(),
                (0..384).collect::<Vec<_>>(),
                "backbone layer {layer} expert identity coverage"
            );
        }
        for layer in 0..3 {
            assert_eq!(
                masks
                    .keys()
                    .filter(|(namespace, actual_layer, _)| {
                        *namespace == Mxfp4ExpertNamespace::Mtp && *actual_layer == layer
                    })
                    .map(|(_, _, expert)| *expert)
                    .collect::<Vec<_>>(),
                (0..128).collect::<Vec<_>>(),
                "MTP layer {layer} expert identity coverage"
            );
        }
    }
}

#[cfg(test)]
mod a147_pinned_expert_payload_plan_tests {
    use super::{
        plan_pinned_mxfp4_expert_payloads, Mxfp4ExpertNamespace, PinnedExpertPayloadRequest,
        PINNED_EXPERT_INDEX_SHA256,
    };

    const HEADER_SHA256: &str = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449";
    const SHARD_BYTES: u64 = 7_389_759_032;
    const TOP6: [u32; 6] = [0, 383, 382, 381, 380, 379];

    fn plan(experts: &[u32], header_sha256: &str) -> crate::Result<super::Mxfp4ExpertPayloadPlan> {
        let index = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        let header = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
        );
        plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
            index_bytes: index,
            header_bytes: header,
            expected_header_sha256: header_sha256,
            header_shard_name: "model-00003-of-00048.safetensors",
            source_shard_bytes: SHARD_BYTES,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
            namespace: Mxfp4ExpertNamespace::Backbone,
            layer: 0,
            experts,
        })
    }

    #[test]
    fn top6_maps_exact_six_tensor_ranges_per_expert_to_one_official_shard() {
        let plan = plan(&TOP6, HEADER_SHA256).unwrap();
        assert_eq!(plan.model_revision, super::REVISION);
        assert_eq!(plan.namespace, Mxfp4ExpertNamespace::Backbone);
        assert_eq!(plan.layer, 0);
        assert_eq!(plan.experts, TOP6);
        assert_eq!(plan.source_shard, "model-00003-of-00048.safetensors");
        assert_eq!(plan.source_shard_bytes, SHARD_BYTES);
        assert_eq!(plan.header_bytes, 258_136);
        assert_eq!(plan.data_start_bytes, 258_144);
        assert_eq!(plan.tensors.len(), 36);
        assert_eq!(plan.selected_payload_bytes, 112_803_840);

        for &expert in &TOP6 {
            let tensors = plan
                .tensors
                .iter()
                .filter(|tensor| tensor.expert == expert)
                .collect::<Vec<_>>();
            assert_eq!(tensors.len(), 6);
            let prefix = format!("layers.0.ffn.experts.{expert}");
            assert_eq!(
                tensors
                    .iter()
                    .map(|tensor| tensor.name.clone())
                    .collect::<Vec<_>>(),
                vec![
                    format!("{prefix}.w1.weight"),
                    format!("{prefix}.w1.scale"),
                    format!("{prefix}.w2.weight"),
                    format!("{prefix}.w2.scale"),
                    format!("{prefix}.w3.weight"),
                    format!("{prefix}.w3.scale"),
                ]
            );
            assert!(tensors.iter().all(|tensor| {
                tensor.file_offset_start < tensor.file_offset_end
                    && tensor.file_offset_end <= SHARD_BYTES
            }));
        }
    }

    #[test]
    fn payload_plan_fails_closed_on_duplicate_ids_and_unpinned_metadata() {
        assert!(plan(&[0, 0], HEADER_SHA256).is_err());
        assert!(plan(&[384], HEADER_SHA256).is_err());
        assert!(plan(
            &[0],
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        .is_err());

        let index = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        assert_eq!(crate::sha256::hex(index), PINNED_EXPERT_INDEX_SHA256);
    }
}

#[cfg(test)]
mod a125_qpn2_conversion_tests {
    use super::*;
    use crate::quant;

    fn matrix_codes_at(codes: &[u8], row: usize, columns: usize, col: usize) -> u8 {
        packed_nibble(codes, row * columns + col)
    }

    fn qpn2_value(plane: &ConvertedQpn2Plane, row: usize, col: usize) -> f64 {
        let tile = row / 32;
        let r = row % 32;
        let lane = (r & 3) | (((r >> 3) & 3) << 2) | (((r >> 2) & 1) << 4);
        let group = col / 16;
        let logical_k = col % 16;
        let slot = QPN2_K_ORDER.iter().position(|&k| k == logical_k).unwrap();
        let fragment = (tile * (plane.columns / 16) + group) * 32 + lane;
        let byte = plane.codes[fragment * 8 + slot / 2];
        let code = if slot % 2 == 0 { byte & 15 } else { byte >> 4 };
        let factor = f64::from(plane.global_dequant);
        let scale = quant::decode_scale(plane.scales[fragment]).unwrap();
        quant::decode_fp4(code).unwrap() * scale * factor
    }

    #[test]
    fn ordered_kernel_half_steps_match_independent_fp4_dyadics_and_edges() {
        for code in 0..16u8 {
            let raw_bits = (u16::from(code & 8) << 12) | (u16::from(code & 7) << 9);
            assert_eq!(
                decode_f16_bits(raw_bits).to_bits(),
                (quant::decode_fp4(code).unwrap() * 2.0f64.powi(-14)).to_bits(),
                "dequant8_tm rebias for code {code:#x}"
            );
        }
        for exponent in [-127, -39, -38, -28, -27, -8, -3, 0, 1] {
            let code = (exponent + 127) as u8;
            let shift = mxfp4_scale_shift(&[code]).unwrap();
            validate_mxfp4_qpn2_half_equivalence(exponent, shift).unwrap();
        }
        assert!(mxfp4_scale_shift(&[129]).is_err());
        // The ideal E2M1 weight remains finite at e=2, but the exact CUDA
        // intermediate scale * gscale * 2^14 is 2^16 and overflows half.
        assert!(validate_mxfp4_qpn2_half_equivalence(2, -3).is_err());
        assert!(mxfp4_scale_shift(&[0, 128]).is_err());
    }

    #[test]
    fn exact_e4m3_power_codes_cover_normal_and_subnormal_scales() {
        for exponent in -9..=8 {
            let code = qpn2_power_scale_code(exponent).unwrap();
            let expected = 2.0f64.powi(exponent);
            assert_eq!(
                quant::decode_scale(code).unwrap().to_bits(),
                expected.to_bits()
            );
            assert_eq!(
                qpn2_e4m3_half_value(code).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        assert!(qpn2_power_scale_code(-10).is_none());
        assert!(qpn2_power_scale_code(9).is_none());
    }

    #[test]
    fn qpn2_conversion_preserves_every_code_coordinate_on_a_small_adversarial_plane() {
        let rows = 32usize;
        let columns = 64usize;
        let mut source_codes = vec![0u8; rows * columns / 2];
        for row in 0..rows {
            for col in 0..columns {
                let code = ((row * 3 + col * 5) & 15) as u8;
                let index = row * columns + col;
                let shift = 4 * (index % 2);
                source_codes[index / 2] =
                    (source_codes[index / 2] & !(15 << shift)) | (code << shift);
            }
        }
        let mut source_scales = vec![0u8; rows * columns / 32];
        for row in 0..rows {
            for group in 0..columns / 32 {
                let exponent = ((row + group) % 4) as i32 - 2;
                source_scales[row * (columns / 32) + group] = (exponent + 127) as u8;
            }
        }
        let matrix =
            Mxfp4Matrix::new(rows as u64, columns as u64, source_codes, source_scales).unwrap();
        let converted = convert_mxfp4_plane(&matrix).unwrap();
        for row in 0..rows {
            for col in 0..columns {
                assert_eq!(
                    qpn2_value(&converted, row, col).to_bits(),
                    matrix.value(row as u64, col as u64).unwrap().to_bits(),
                    "coordinate [{row},{col}]"
                );
                let tile = row / 32;
                let r = row % 32;
                let lane = (r & 3) | (((r >> 3) & 3) << 2) | (((r >> 2) & 1) << 4);
                let group = col / 16;
                let logical_k = col % 16;
                let slot = QPN2_K_ORDER.iter().position(|&k| k == logical_k).unwrap();
                let fragment = (tile * (columns / 16) + group) * 32 + lane;
                let byte = converted.codes[fragment * 8 + slot / 2];
                let converted_code = if slot % 2 == 0 { byte & 15 } else { byte >> 4 };
                assert_eq!(
                    matrix_codes_at(matrix.codes.as_slice(), row, columns, col),
                    converted_code,
                    "packed code coordinate [{row},{col}]"
                );
            }
        }
    }
}
