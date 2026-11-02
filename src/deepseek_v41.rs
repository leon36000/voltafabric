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
use safetensors::tensor::{serialize, Dtype, Metadata, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

pub mod r9_fp32;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
/// One F8_E8M0 scale per 32 consecutive K values (`fp4_block_size`).
pub const MXFP4_BLOCK: u64 = 32;
pub const MAX_MXFP4_DIM: u64 = 65_536;
pub const MAX_MXFP4_COMPRESSED_BYTES: u64 = 64 * 1024 * 1024;
/// Separate storage ceiling for the FP8 shared expert; its codes are byte-per-value,
/// not MXFP4 packed nibbles.
pub const MAX_FP8_COMPRESSED_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_FP8_DIM: u64 = 65_536;
/// Bound for one three-projection expert Safetensors artifact admitted by the CPU owner.
pub const MAX_MXFP4_EXPERT_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum work for the bounded CPU reference matvec. This covers one 2304x5120
/// V4.1 expert projection (11,796,480 MACs), but is not a production CPU fallback.
pub const MAX_MXFP4_REFERENCE_MACS: u64 = 16 * 1024 * 1024;
/// Total work cap for the three projections in one single-token CPU oracle.
pub const MAX_DEEPSEEK_EXPERT_REFERENCE_MACS: u64 = 48 * 1024 * 1024;
/// `swiglu_limit` of the official config.json.
pub const SWIGLU_LIMIT: f32 = 10.0;
/// V4.1 residual stream multiplicity from the pinned checkpoint configuration.
pub const PINNED_HC_MULT: usize = 4;
/// One 2304x5120 shared-expert projection is 11,796,480 MACs.
pub const MAX_FP8_REFERENCE_MACS: u64 = 16 * 1024 * 1024;
pub const MAX_SHARED_EXPERT_REFERENCE_MACS: u64 = 48 * 1024 * 1024;
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

    /// Reconstruct the pinned fake-quant/dequant activation boundary as exact
    /// BF16 values represented in FP32. This is used by the CPU output-projection
    /// reference; it is not a device buffer or a GEMM result.
    pub fn dequantized_bf16_f32(&self) -> Result<Vec<f32>> {
        self.dequantized_values_f64()?
            .into_iter()
            .map(|value| {
                let value = value as f32;
                if !value.is_finite() {
                    return Err(Error::new(
                        "NUMERICAL_OVERFLOW",
                        "dequantized FP8 activation does not fit finite FP32",
                    ));
                }
                cast_bf16_rne_reference(value)
            })
            .collect()
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

/// CPU reference for the pinned Hyper-Connections `hc_pre` operation.
///
/// `residual_copies` is copy-major `[hc_mult, hidden]`; `pre_mix` is the already
/// computed FP32 vector from `hc_split_sinkhorn`. The input copies are represented
/// as exact BF16 values, products/sum use explicit FP32 copy order, and the result
/// is rounded to BF16 once, matching `Block.hc_pre`. This does not compute the
/// learned projection or Sinkhorn coefficients, and it is not a GPU kernel.
pub fn deepseek_hc_pre_reference(
    residual_copies: &[f32],
    pre_mix: &[f32],
    hidden: usize,
) -> Result<Vec<f32>> {
    if pre_mix.len() != PINNED_HC_MULT
        || hidden == 0
        || hidden > MAX_ACTIVATION_LEN
        || residual_copies.len() != PINNED_HC_MULT.saturating_mul(hidden)
        || pre_mix.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "pinned Hyper-Connections pre-mix shape or coefficients are invalid",
        ));
    }
    for &value in residual_copies {
        if !value.is_finite() || cast_bf16_rne_reference(value)?.to_bits() != value.to_bits() {
            return Err(invalid(
                "Hyper-Connections residual copies must be exact finite BF16 values",
            ));
        }
    }

    let mut output = Vec::with_capacity(hidden);
    for column in 0..hidden {
        let mut sum = 0.0f32;
        for copy in 0..PINNED_HC_MULT {
            sum += pre_mix[copy] * residual_copies[copy * hidden + column];
        }
        output.push(cast_bf16_rne_reference(sum)?);
    }
    Ok(output)
}

/// Immutable FP8 E4M3FN weights with one E8M0 scale per 32 consecutive K values.
/// This is the distinct always-active shared-expert format in the pinned model;
/// it is not MXFP4 and does not use packed nibbles.
#[derive(Debug)]
pub struct Fp8E4m3Matrix {
    rows: u64,
    cols: u64,
    codes: Box<[u8]>,
    scales: Box<[u8]>,
}

impl Fp8E4m3Matrix {
    /// Check dimensions, K blocking, finite E4M3FN codes, E8M0 scales and the
    /// byte cap before taking ownership of either plane.
    pub fn new(rows: u64, cols: u64, codes: Vec<u8>, scales: Vec<u8>) -> Result<Self> {
        if rows == 0
            || cols == 0
            || rows > MAX_FP8_DIM
            || cols > MAX_FP8_DIM
            || !cols.is_multiple_of(MXFP4_BLOCK)
        {
            return Err(invalid("FP8 E4M3 matrix dimensions or K block are invalid"));
        }
        let code_bytes = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("FP8 E4M3 code byte count overflow"))?;
        let scale_bytes = rows
            .checked_mul(cols / MXFP4_BLOCK)
            .ok_or_else(|| invalid("FP8 E4M3 scale byte count overflow"))?;
        if code_bytes
            .checked_add(scale_bytes)
            .is_none_or(|bytes| bytes > MAX_FP8_COMPRESSED_BYTES)
            || u64::try_from(codes.len()).ok() != Some(code_bytes)
            || u64::try_from(scales.len()).ok() != Some(scale_bytes)
        {
            return Err(invalid("FP8 E4M3 plane lengths or byte cap are invalid"));
        }
        if scales.contains(&255) {
            return Err(Error::new("FP8_SCALE_NAN", "E8M0 scale 255 encodes NaN"));
        }
        if codes.iter().any(|code| code & 0x7f == 0x7f) {
            return Err(Error::new(
                "FP8_WEIGHT_NAN",
                "E4M3FN weight contains its NaN encoding",
            ));
        }
        Ok(Self {
            rows,
            cols,
            codes: codes.into_boxed_slice(),
            scales: scales.into_boxed_slice(),
        })
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn cols(&self) -> u64 {
        self.cols
    }

    /// Immutable resident FP8 code and E8M0 scale planes.
    pub fn compressed_planes(&self) -> (&[u8], &[u8]) {
        (&self.codes, &self.scales)
    }

    /// Bounded single-token reference GEMV from E4M3FN/E8M0 planes and the
    /// pinned dynamically quantized FP8 activation. Each 32-value dot is
    /// accumulated in f64 and multiplied by the two block scales exactly once.
    pub fn matvec_reference(&self, input: &Fp8ActivationQuantized) -> Result<Vec<f32>> {
        let macs = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| invalid("FP8 E4M3 reference work count overflow"))?;
        if macs > MAX_FP8_REFERENCE_MACS {
            return Err(invalid(format!(
                "FP8 E4M3 reference work limit exceeded: {macs} > {MAX_FP8_REFERENCE_MACS}"
            )));
        }
        if u64::try_from(input.values.len()).ok() != Some(self.cols)
            || input.scale_codes.len() as u64 != self.cols / MXFP4_BLOCK
        {
            return Err(invalid("FP8 E4M3 matvec activation shape differs from K"));
        }

        let cols =
            usize::try_from(self.cols).map_err(|_| invalid("FP8 E4M3 K does not fit usize"))?;
        let blocks = cols / MXFP4_BLOCK as usize;
        let mut output = Vec::with_capacity(self.rows as usize);
        for row in 0..self.rows as usize {
            let mut total = 0.0f64;
            for block in 0..blocks {
                let activation_scale = decode_e8m0(input.scale_codes[block])?;
                let weight_scale = decode_e8m0(self.scales[row * blocks + block])?;
                let mut dot = 0.0f64;
                let start = row * cols + block * MXFP4_BLOCK as usize;
                for offset in 0..MXFP4_BLOCK as usize {
                    let activation = decode_e4m3fn_f64(input.values[block * 32 + offset])?;
                    let weight = decode_e4m3fn_f64(self.codes[start + offset])?;
                    dot += activation * weight;
                }
                total += (dot * activation_scale) * weight_scale;
            }
            let value = total as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "FP8 E4M3 GEMV output is nonfinite",
                ));
            }
            output.push(value);
        }
        Ok(output)
    }
}

fn decode_e4m3fn_f64(code: u8) -> Result<f64> {
    let magnitude_code = code & 0x7f;
    if magnitude_code == 0x7f {
        return Err(Error::new("FP8_WEIGHT_NAN", "E4M3FN code encodes NaN"));
    }
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    Ok(sign * f64::from(e4m3fn_positive(magnitude_code)))
}

/// CPU reference for one always-active pinned DeepSeek shared SwiGLU expert.
/// Each projection is resident FP8 E4M3FN/E8M0; the routed MXFP4 K32 experts
/// remain a separate type and path. The return is raw FP32 shared output, with
/// no gate coefficient; callers add it once to the completed routed contribution.
#[derive(Debug)]
pub struct DeepSeekV41SharedExpert {
    w1: Fp8E4m3Matrix,
    w2: Fp8E4m3Matrix,
    w3: Fp8E4m3Matrix,
}

impl DeepSeekV41SharedExpert {
    pub fn new(w1: Fp8E4m3Matrix, w2: Fp8E4m3Matrix, w3: Fp8E4m3Matrix) -> Result<Self> {
        let w1_macs = w1
            .rows
            .checked_mul(w1.cols)
            .ok_or_else(|| invalid("shared FP8 W1 MAC count overflow"))?;
        let w2_macs = w2
            .rows
            .checked_mul(w2.cols)
            .ok_or_else(|| invalid("shared FP8 W2 MAC count overflow"))?;
        let total_macs = w1_macs
            .checked_mul(2)
            .and_then(|macs| macs.checked_add(w2_macs))
            .ok_or_else(|| invalid("shared FP8 expert MAC count overflow"))?;
        if w1.rows != w3.rows
            || w1.cols != w3.cols
            || w2.cols != w1.rows
            || w2.rows != w1.cols
            || w1_macs > MAX_FP8_REFERENCE_MACS
            || w2_macs > MAX_FP8_REFERENCE_MACS
            || total_macs > MAX_SHARED_EXPERT_REFERENCE_MACS
        {
            return Err(invalid(
                "pinned shared FP8 expert projection shapes or work bounds differ",
            ));
        }
        Ok(Self { w1, w2, w3 })
    }

    /// Execute the pinned `w1 -> clamp -> SiLU*up -> BF16 cast -> w2` sequence.
    /// `w3` is the up projection; output is deliberately unweighted.
    pub fn forward_reference(&self, input_bf16_f32: &[f32]) -> Result<Vec<f32>> {
        if input_bf16_f32.len() as u64 != self.w1.cols
            || input_bf16_f32.iter().any(|value| {
                !value.is_finite()
                    || cast_bf16_rne_reference(*value)
                        .map(|rounded| rounded.to_bits() != value.to_bits())
                        .unwrap_or(true)
            })
        {
            return Err(invalid(
                "shared FP8 expert input must be finite exact-BF16 values of width K",
            ));
        }
        let quantized_input = fp8_activation_quantize_reference(input_bf16_f32)?;
        let mut gate = self.w1.matvec_reference(&quantized_input)?;
        let mut up = self.w3.matvec_reference(&quantized_input)?;
        let mut down_input = Vec::with_capacity(gate.len());
        for (gate_value, up_value) in gate.iter_mut().zip(up.iter_mut()) {
            *gate_value = gate_value.min(SWIGLU_LIMIT);
            *up_value = up_value.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
            let silu = *gate_value / (1.0 + (-*gate_value).exp());
            down_input.push(cast_bf16_rne_reference(silu * *up_value)?);
        }
        let quantized_down_input = fp8_activation_quantize_reference(&down_input)?;
        self.w2.matvec_reference(&quantized_down_input)
    }
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
pub(crate) const PINNED_EXPERT_INDEX_SHA256: &str =
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

/// Inputs required to read exactly one route-selected expert for a bound rank.
/// Payload hashes are independent values supplied by the caller's admitted source
/// manifest; the pinned model index and header do not authenticate weight bytes.
#[derive(Debug, Clone, Copy)]
pub struct PinnedMxfp4RankExpertReadRequest<'a> {
    pub route: &'a crate::routing::RoutingProblem,
    pub attempt: u64,
    pub expert: u32,
    pub rank: u32,
    pub index_bytes: &'a [u8],
    pub header_shard_name: &'a str,
    pub expected_header_sha256: &'a str,
    pub expected_payload_sha256: &'a BTreeMap<String, String>,
    pub max_source_shard_bytes: u64,
    pub max_payload_bytes: u64,
}

/// Inputs for importing one route-selected expert from a pinned, indexed
/// SafeTensors checkpoint directory. The index and each of the six selected
/// tensor payloads are independently hash checked before a read receipt exists.
#[derive(Debug, Clone, Copy)]
pub struct PinnedMxfp4IndexedRankExpertReadRequest<'a> {
    pub route: &'a crate::routing::RoutingProblem,
    pub attempt: u64,
    pub expert: u32,
    pub rank: u32,
    pub index_bytes: &'a [u8],
    pub expected_header_sha256: &'a str,
    pub expected_payload_sha256: &'a BTreeMap<String, String>,
    pub max_source_shard_bytes: u64,
    pub max_payload_bytes: u64,
}

/// One expert's selected payloads, tied to the original route/attempt and exact
/// rank-local binding. Payloads are still separate compressed tensors; this is not
/// a constructed owner, Session LOAD, or proof of generation.
#[derive(Debug)]
pub struct PinnedMxfp4RankExpertRead {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    expert: u32,
    rank: u32,
    local: u32,
    plan: Mxfp4ExpertPayloadPlan,
    header_sha256: String,
    /// The first pass reads only the 8-byte Safetensors length and bounded header.
    header_validation_io_bytes: u64,
    /// Includes the header-only validation pass and `read_named_tensors` pass.
    total_io_bytes: u64,
    payloads: crate::checkpoint::NamedTensorImport,
}

impl PinnedMxfp4RankExpertRead {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn expert(&self) -> u32 {
        self.expert
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn local(&self) -> u32 {
        self.local
    }

    pub fn plan(&self) -> &Mxfp4ExpertPayloadPlan {
        &self.plan
    }

    pub fn header_sha256(&self) -> &str {
        &self.header_sha256
    }

    pub fn header_validation_io_bytes(&self) -> u64 {
        self.header_validation_io_bytes
    }

    pub fn total_io_bytes(&self) -> u64 {
        self.total_io_bytes
    }

    /// Consume this exact read receipt into the existing bounded CPU session owner.
    /// A minimal one-expert Safetensors buffer is assembled from only the six selected
    /// tensors; the multi-gigabyte source shard is never copied or retained. The caller
    /// supplies a peak-memory budget covering both the selected payloads and compact
    /// owner while serialization is in progress.
    pub fn into_session_resident(
        self,
        max_peak_bytes: u64,
    ) -> Result<PinnedMxfp4RankResidentExpert> {
        let top_k = crate::deepseek_v41_gate::TOP_K;
        let route_ids = self.route_experts.iter().copied().collect::<BTreeSet<_>>();
        if self.attempt == 0
            || self.stamp.model_revision != REVISION
            || self.plan.model_revision != REVISION
            || self.stamp.layer != self.plan.layer
            || self.plan.namespace != Mxfp4ExpertNamespace::Backbone
            || self.plan.experts.as_slice() != [self.expert]
            || self.route_experts.len() != top_k
            || route_ids.len() != top_k
            || !route_ids.contains(&self.expert)
            || self.plan.source_shard_bytes != self.payloads.file_bytes
            || self.plan.header_bytes != self.payloads.header_bytes
            || self.plan.data_start_bytes != self.plan.header_bytes.saturating_add(8)
            || self.plan.tensors.len() != 6
            || self.payloads.tensors.len() != 6
            || self.plan.selected_payload_bytes == 0
            || !crate::sha256::is_hex_digest(&self.header_sha256)
        {
            return Err(invalid(
                "DeepSeek rank read receipt identity or six-tensor inventory is invalid",
            ));
        }

        let prefix = format!("layers.{}.ffn.experts.{}", self.plan.layer, self.expert);
        let expected_names = [
            format!("{prefix}.w1.weight"),
            format!("{prefix}.w1.scale"),
            format!("{prefix}.w2.weight"),
            format!("{prefix}.w2.scale"),
            format!("{prefix}.w3.weight"),
            format!("{prefix}.w3.scale"),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let plan_names = self
            .plan
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        let payload_names = self
            .payloads
            .tensors
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if plan_names != expected_names || payload_names != expected_names {
            return Err(invalid(
                "DeepSeek rank owner must contain exactly the six selected expert tensor names",
            ));
        }

        let mut payload_bytes = 0u64;
        let mut payload_hashes = BTreeMap::new();
        let mut views = Vec::with_capacity(self.plan.tensors.len());
        for tensor in &self.plan.tensors {
            if tensor.expert != self.expert || tensor.file_offset_start >= tensor.file_offset_end {
                return Err(invalid(
                    "DeepSeek rank owner plan contains a foreign or empty tensor range",
                ));
            }
            let payload =
                self.payloads.tensors.get(&tensor.name).ok_or_else(|| {
                    invalid("DeepSeek selected payload disappeared before packing")
                })?;
            let expected_dtype = match tensor.dtype {
                "I8" => Dtype::I8,
                "F8_E8M0" => Dtype::F8_E8M0,
                _ => return Err(invalid("DeepSeek selected payload dtype is unsupported")),
            };
            let expected_len = tensor
                .file_offset_end
                .checked_sub(tensor.file_offset_start)
                .ok_or_else(|| invalid("DeepSeek selected payload range is reversed"))?;
            if payload.dtype != expected_dtype
                || payload.shape != tensor.shape
                || u64::try_from(payload.bytes.len()).ok() != Some(expected_len)
                || !crate::sha256::is_hex_digest(&payload.sha256)
                || crate::sha256::hex(&payload.bytes) != payload.sha256
            {
                return Err(invalid(format!(
                    "DeepSeek selected tensor failed its identity, length, or payload digest: {}",
                    tensor.name
                )));
            }
            payload_bytes = payload_bytes
                .checked_add(expected_len)
                .ok_or_else(|| invalid("DeepSeek resident payload size overflow"))?;
            payload_hashes.insert(tensor.name.clone(), payload.sha256.clone());
            let view = TensorView::new(payload.dtype, payload.shape.clone(), &payload.bytes)
                .map_err(|error| invalid(format!("invalid selected MXFP4 tensor view: {error}")))?;
            views.push((tensor.name.as_str(), view));
        }
        if payload_bytes != self.plan.selected_payload_bytes {
            return Err(invalid(
                "DeepSeek selected payload byte count differs from its checked plan",
            ));
        }

        let header_slack = crate::checkpoint::MAX_HEADER_BYTES
            .checked_add(8)
            .ok_or_else(|| invalid("Safetensors header budget overflow"))?;
        let packing_peak_bound_bytes = payload_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(header_slack))
            .ok_or_else(|| invalid("DeepSeek owner packing peak budget overflow"))?;
        let packed_owner_bound_bytes = payload_bytes
            .checked_add(header_slack)
            .ok_or_else(|| invalid("DeepSeek packed owner budget overflow"))?;
        if packing_peak_bound_bytes > max_peak_bytes
            || packed_owner_bound_bytes > MAX_MXFP4_EXPERT_ARTIFACT_BYTES
        {
            return Err(Error::new(
                "ARTIFACT_BUDGET",
                "DeepSeek compact expert owner exceeds the admitted memory bound",
            ));
        }

        let packed = serialize(views, None).map_err(|error| {
            invalid(format!("could not serialize compact MXFP4 owner: {error}"))
        })?;
        drop(self.payloads);
        let packed_artifact_bytes = u64::try_from(packed.len())
            .map_err(|_| invalid("compact MXFP4 owner size does not fit u64"))?;
        if packed_artifact_bytes > MAX_MXFP4_EXPERT_ARTIFACT_BYTES
            || packed_artifact_bytes < payload_bytes
        {
            return Err(invalid(
                "serialized compact MXFP4 owner violates its exact artifact bound",
            ));
        }

        let packed_artifact_sha256 = crate::sha256::hex(&packed);
        let binding = crate::session::mxfp4::Binding {
            artifact_sha256: packed_artifact_sha256.clone(),
            model_revision: self.stamp.model_revision.clone(),
            rank: self.rank,
            layer: self.stamp.layer,
            epoch: self.stamp.epoch,
            experts: vec![crate::session::mxfp4::Expert {
                expert: self.expert,
                local: self.local,
            }],
        };
        let owner = crate::session::mxfp4::Artifact::new(packed, binding)?;
        if owner.binding().rank != self.rank
            || owner.binding().experts.as_slice()
                != [crate::session::mxfp4::Expert {
                    expert: self.expert,
                    local: self.local,
                }]
        {
            return Err(invalid(
                "constructed MXFP4 session owner differs from the route/rank binding",
            ));
        }

        Ok(PinnedMxfp4RankResidentExpert {
            stamp: self.stamp,
            attempt: self.attempt,
            route_experts: self.route_experts,
            route_fingerprint: self.route_fingerprint,
            expert: self.expert,
            rank: self.rank,
            local: self.local,
            source_shard: self.plan.source_shard,
            source_header_sha256: self.header_sha256,
            payload_sha256: payload_hashes,
            packed_artifact_sha256,
            packed_artifact_bytes,
            packing_peak_bound_bytes,
            owner,
        })
    }
}

/// A one-expert host resident that retains full route identity beside the existing
/// session artifact binding (model/layer/rank/local/epoch). It has not been LOADed.
pub struct PinnedMxfp4RankResidentExpert {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    expert: u32,
    rank: u32,
    local: u32,
    source_shard: String,
    source_header_sha256: String,
    payload_sha256: BTreeMap<String, String>,
    packed_artifact_sha256: String,
    packed_artifact_bytes: u64,
    packing_peak_bound_bytes: u64,
    owner: crate::session::mxfp4::Artifact,
}

impl PinnedMxfp4RankResidentExpert {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn expert(&self) -> u32 {
        self.expert
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn local(&self) -> u32 {
        self.local
    }

    pub fn source_shard(&self) -> &str {
        &self.source_shard
    }

    pub fn source_header_sha256(&self) -> &str {
        &self.source_header_sha256
    }

    pub fn payload_sha256(&self) -> &BTreeMap<String, String> {
        &self.payload_sha256
    }

    pub fn packed_artifact_sha256(&self) -> &str {
        &self.packed_artifact_sha256
    }

    pub fn packed_artifact_bytes(&self) -> u64 {
        self.packed_artifact_bytes
    }

    pub fn packing_peak_bound_bytes(&self) -> u64 {
        self.packing_peak_bound_bytes
    }

    pub fn owner(&self) -> &crate::session::mxfp4::Artifact {
        &self.owner
    }

    /// Transfer this single, already-built owner into an existing CPU MXFP4
    /// Session and return its load receipt with the full route identity retained.
    /// This consumes the receipt so callers cannot silently replay a failed or
    /// ambiguous LOAD through the same value. It does not authorize EXEC: a batch
    /// still has to preflight every selected expert and rank atomically first.
    pub fn load_into_session(
        self,
        session: &mut crate::session::Session,
        preflight: &PinnedMxfp4RoutePreflight,
        timeout: std::time::Duration,
    ) -> Result<PinnedMxfp4SessionLoadReceipt> {
        let binding = self.owner.binding();
        let selected = crate::session::mxfp4::Expert {
            expert: self.expert,
            local: self.local,
        };
        let route_ids = self.route_experts.iter().copied().collect::<BTreeSet<_>>();
        if self.attempt == 0
            || self.stamp.model_revision != REVISION
            || self.stamp.layer != binding.layer
            || self.stamp.epoch != binding.epoch
            || binding.model_revision != self.stamp.model_revision
            || binding.rank != self.rank
            || binding.experts.as_slice() != [selected]
            || self.route_experts.len() != crate::deepseek_v41_gate::TOP_K
            || route_ids.len() != self.route_experts.len()
            || !route_ids.contains(&self.expert)
            || !crate::sha256::is_hex_digest(&self.route_fingerprint)
            || !crate::sha256::is_hex_digest(&self.source_header_sha256)
            || self.payload_sha256.len() != 6
            || self
                .payload_sha256
                .values()
                .any(|digest| !crate::sha256::is_hex_digest(digest))
            || self.owner.bytes().len() as u64 != self.packed_artifact_bytes
            || crate::sha256::hex(self.owner.bytes()) != self.packed_artifact_sha256
            || binding.artifact_sha256 != self.packed_artifact_sha256
            || preflight.stamp != self.stamp
            || preflight.attempt != self.attempt
            || preflight.route_experts != self.route_experts
            || preflight.route_fingerprint != self.route_fingerprint
            || !preflight.owners.iter().any(|owner| {
                owner.expert == self.expert
                    && owner.rank == self.rank
                    && owner.local == self.local
                    && owner.artifact_sha256 == self.packed_artifact_sha256
                    && owner.artifact_bytes == self.packed_artifact_bytes
            })
        {
            return Err(invalid(
                "DeepSeek resident owner lacks a matching complete-route preflight",
            ));
        }

        let session_load = session.load_mxfp4(&self.owner, self.stamp.epoch, timeout)?;
        if session_load.rank != self.rank
            || session_load.epoch != self.stamp.epoch
            || session_load.artifact_sha256 != self.packed_artifact_sha256
        {
            return Err(Error::new(
                "SESSION_PROTOCOL",
                "CPU Session LOAD receipt differs from the pinned DeepSeek owner",
            ));
        }

        Ok(PinnedMxfp4SessionLoadReceipt {
            stamp: self.stamp,
            attempt: self.attempt,
            route_experts: self.route_experts,
            route_fingerprint: self.route_fingerprint,
            expert: self.expert,
            rank: self.rank,
            local: self.local,
            source_shard: self.source_shard,
            source_header_sha256: self.source_header_sha256,
            payload_sha256: self.payload_sha256,
            packed_artifact_sha256: self.packed_artifact_sha256,
            session_load,
        })
    }
}

/// One LOAD receipt per rank artifact, regardless of how many selected experts it
/// contains. The member list is in the Session binding's physical-local order.
#[derive(Debug)]
pub struct PinnedMxfp4RouteRankGroupLoadReceipt {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    rank: u32,
    members: Vec<PinnedMxfp4RouteOwnerBinding>,
    packed_artifact_sha256: String,
    session_load: crate::session::LoadReceipt,
}

impl PinnedMxfp4RouteRankGroupLoadReceipt {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn members(&self) -> &[PinnedMxfp4RouteOwnerBinding] {
        &self.members
    }

    pub fn packed_artifact_sha256(&self) -> &str {
        &self.packed_artifact_sha256
    }

    pub fn session_load(&self) -> &crate::session::LoadReceipt {
        &self.session_load
    }
}

/// One route-selected rank-local Safetensors owner. A rank may contain several of the
/// selected experts, but the combined artifact remains subject to the existing 64-MiB
/// Session owner limit and exact `resident_order` contract.
pub struct PinnedMxfp4RankResidentGroup {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    rank: u32,
    members: Vec<PinnedMxfp4RouteOwnerBinding>,
    source_shards: BTreeMap<u32, (String, String)>,
    payload_sha256: BTreeMap<String, String>,
    packed_artifact_sha256: String,
    packed_artifact_bytes: u64,
    packing_peak_bound_bytes: u64,
    owner: crate::session::mxfp4::Artifact,
}

impl PinnedMxfp4RankResidentGroup {
    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn members(&self) -> &[PinnedMxfp4RouteOwnerBinding] {
        &self.members
    }

    pub fn packed_artifact_sha256(&self) -> &str {
        &self.packed_artifact_sha256
    }

    pub fn packed_artifact_bytes(&self) -> u64 {
        self.packed_artifact_bytes
    }

    /// Per-expert (source shard basename, verified source header digest).
    pub fn source_shards(&self) -> &BTreeMap<u32, (String, String)> {
        &self.source_shards
    }

    pub fn payload_sha256(&self) -> &BTreeMap<String, String> {
        &self.payload_sha256
    }

    pub fn packing_peak_bound_bytes(&self) -> u64 {
        self.packing_peak_bound_bytes
    }

    /// One rank receives one artifact containing exactly the selected local owners.
    /// The input route preflight is retained so the later EXEC can still be tied to the
    /// full donor order, coefficients and original per-expert read receipts.
    pub fn load_into_session(
        self,
        session: &mut crate::session::Session,
        preflight: &PinnedMxfp4RoutePreflight,
        timeout: std::time::Duration,
    ) -> Result<PinnedMxfp4RouteRankGroupLoadReceipt> {
        let binding = self.owner.binding();
        let expected_members = preflight
            .owners
            .iter()
            .filter(|owner| owner.rank == self.rank)
            .cloned()
            .collect::<Vec<_>>();
        let mut expected_members_by_local = expected_members;
        expected_members_by_local.sort_by_key(|owner| owner.local);
        if self.attempt != preflight.attempt
            || self.stamp != preflight.stamp
            || self.route_experts != preflight.route_experts
            || self.route_fingerprint != preflight.route_fingerprint
            || self.members != expected_members_by_local
            || self.members.is_empty()
            || session.rank != self.rank
            || binding.rank != self.rank
            || binding.model_revision != self.stamp.model_revision
            || binding.layer != self.stamp.layer
            || binding.epoch != self.stamp.epoch
            || binding.experts
                != self
                    .members
                    .iter()
                    .map(|owner| crate::session::mxfp4::Expert {
                        expert: owner.expert,
                        local: owner.local,
                    })
                    .collect::<Vec<_>>()
            || self.owner.bytes().len() as u64 != self.packed_artifact_bytes
            || crate::sha256::hex(self.owner.bytes()) != self.packed_artifact_sha256
            || binding.artifact_sha256 != self.packed_artifact_sha256
        {
            return Err(invalid(
                "DeepSeek rank-group owner differs from its route preflight or Session rank",
            ));
        }

        let session_load = session.load_mxfp4(&self.owner, self.stamp.epoch, timeout)?;
        if session_load.rank != self.rank
            || session_load.epoch != self.stamp.epoch
            || session_load.artifact_sha256 != self.packed_artifact_sha256
        {
            return Err(Error::new(
                "SESSION_PROTOCOL",
                "CPU Session rank-group LOAD receipt differs from its bound artifact",
            ));
        }
        Ok(PinnedMxfp4RouteRankGroupLoadReceipt {
            stamp: self.stamp,
            attempt: self.attempt,
            route_experts: self.route_experts,
            route_fingerprint: self.route_fingerprint,
            rank: self.rank,
            members: self.members,
            packed_artifact_sha256: self.packed_artifact_sha256,
            session_load,
        })
    }
}

/// Merge the single-expert selective-read receipts for each selected rank into the
/// exact multi-expert artifacts accepted by Session. If a rank's route projection
/// includes additional resident locals, this adaptation refuses; it never forges a
/// shorter binding than the dispatch describes.
pub fn group_pinned_mxfp4_route_residents(
    route: &crate::routing::RoutingProblem,
    preflight: PinnedMxfp4RoutePreflight,
    residents: Vec<PinnedMxfp4RankResidentExpert>,
    max_peak_bytes: u64,
) -> Result<(PinnedMxfp4RoutePreflight, Vec<PinnedMxfp4RankResidentGroup>)> {
    if max_peak_bytes == 0 {
        return Err(Error::new(
            "ARTIFACT_BUDGET",
            "DeepSeek rank-group packing requires a nonzero aggregate peak budget",
        ));
    }
    let verified = preflight_pinned_mxfp4_route_residents(
        route,
        preflight.attempt,
        &residents,
        preflight.total_artifact_bytes,
    )?;
    if verified != preflight {
        return Err(invalid(
            "DeepSeek rank-group residents differ from the consumed complete-route preflight",
        ));
    }
    let dispatch = crate::routing::dispatch(route)?;
    let mut by_expert = residents
        .into_iter()
        .map(|resident| (resident.expert, resident))
        .collect::<BTreeMap<_, _>>();
    if by_expert.len() != preflight.owners.len() {
        return Err(invalid(
            "DeepSeek rank-group packing received duplicate or missing route residents",
        ));
    }

    let mut grouped = Vec::with_capacity(dispatch.ranks.len());
    let mut grouped_bytes = 0u64;
    for rank_dispatch in &dispatch.ranks {
        let mut members = preflight
            .owners
            .iter()
            .filter(|owner| owner.rank == rank_dispatch.rank)
            .cloned()
            .collect::<Vec<_>>();
        members.sort_by_key(|owner| owner.local);
        if members.is_empty()
            || members.iter().map(|owner| owner.local).collect::<Vec<_>>()
                != rank_dispatch.resident_order
        {
            return Err(invalid(
                "DeepSeek rank-group must contain exactly the selected locals in Session resident order",
            ));
        }
        let mut source_residents = members
            .iter()
            .map(|owner| {
                by_expert
                    .remove(&owner.expert)
                    .ok_or_else(|| invalid("DeepSeek rank-group resident disappeared"))
            })
            .collect::<Result<Vec<_>>>()?;
        if source_residents.is_empty() {
            return Err(invalid("DeepSeek rank-group has no resident owners"));
        }
        if source_residents.iter().any(|resident| {
            resident.rank != rank_dispatch.rank
                || resident.stamp != preflight.stamp
                || resident.attempt != preflight.attempt
                || resident.route_experts != preflight.route_experts
                || resident.route_fingerprint != preflight.route_fingerprint
        }) {
            return Err(invalid(
                "DeepSeek rank-group identity differs across selected owners",
            ));
        }
        let source_shards = source_residents
            .iter()
            .map(|resident| {
                (
                    resident.expert,
                    (
                        resident.source_shard.clone(),
                        resident.source_header_sha256.clone(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let (
            owner,
            payload_sha256,
            packed_artifact_sha256,
            packed_artifact_bytes,
            packing_peak_bound_bytes,
        ) = if source_residents.len() == 1 {
            let resident = source_residents
                .pop()
                .ok_or_else(|| invalid("DeepSeek singleton rank owner disappeared"))?;
            (
                resident.owner,
                resident.payload_sha256,
                resident.packed_artifact_sha256,
                resident.packed_artifact_bytes,
                resident.packing_peak_bound_bytes,
            )
        } else {
            let source_artifact_bytes =
                source_residents.iter().try_fold(0u64, |sum, resident| {
                    sum.checked_add(resident.packed_artifact_bytes)
                        .ok_or_else(|| invalid("DeepSeek rank-group source byte count overflow"))
                })?;
            let mut buffers = BTreeMap::<String, (Dtype, Vec<usize>, Vec<u8>, String)>::new();
            let mut payload_bytes = 0u64;
            for resident in &source_residents {
                let tensors = SafeTensors::deserialize(resident.owner.bytes())
                    .map_err(|error| invalid(format!("invalid source resident owner: {error}")))?;
                let prefix = format!(
                    "layers.{}.ffn.experts.{}",
                    route.stamp.layer, resident.expert
                );
                let expected_names = [
                    format!("{prefix}.w1.weight"),
                    format!("{prefix}.w1.scale"),
                    format!("{prefix}.w2.weight"),
                    format!("{prefix}.w2.scale"),
                    format!("{prefix}.w3.weight"),
                    format!("{prefix}.w3.scale"),
                ]
                .into_iter()
                .collect::<BTreeSet<_>>();
                if tensors
                    .names()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>()
                    != expected_names
                    || resident.payload_sha256.len() != expected_names.len()
                {
                    return Err(invalid(
                            "DeepSeek rank-group input does not contain exactly six selected tensors per expert",
                        ));
                }
                for name in expected_names {
                    let view = tensors
                        .tensor(&name)
                        .map_err(|error| invalid(format!("invalid rank-group tensor: {error}")))?;
                    let bytes = view.data().to_vec();
                    let digest = crate::sha256::hex(&bytes);
                    if resident.payload_sha256.get(&name) != Some(&digest)
                        || buffers
                            .insert(name, (view.dtype(), view.shape().to_vec(), bytes, digest))
                            .is_some()
                    {
                        return Err(invalid(
                            "DeepSeek rank-group payload hash or tensor identity differs",
                        ));
                    }
                    payload_bytes = payload_bytes
                        .checked_add(view.data().len() as u64)
                        .ok_or_else(|| invalid("DeepSeek rank-group payload size overflow"))?;
                }
            }
            let header_slack = crate::checkpoint::MAX_HEADER_BYTES
                .checked_add(8)
                .ok_or_else(|| invalid("DeepSeek rank-group header bound overflow"))?;
            if payload_bytes > MAX_MXFP4_EXPERT_ARTIFACT_BYTES.saturating_sub(header_slack) {
                return Err(Error::new(
                    "ARTIFACT_BUDGET",
                    "DeepSeek grouped tensor payload cannot fit the 64-MiB Session artifact bound",
                ));
            }
            let packing_peak_bound_bytes = source_artifact_bytes
                .checked_add(payload_bytes.checked_mul(2).ok_or_else(|| {
                    invalid("DeepSeek rank-group packing peak multiplication overflow")
                })?)
                .and_then(|bytes| bytes.checked_add(header_slack.checked_mul(2)?))
                .ok_or_else(|| invalid("DeepSeek rank-group packing peak overflow"))?;
            if packing_peak_bound_bytes > max_peak_bytes {
                return Err(Error::new(
                    "ARTIFACT_BUDGET",
                    "DeepSeek rank-group owner packing exceeds its aggregate peak budget",
                ));
            }
            let views = buffers
                .iter()
                .map(|(name, (dtype, shape, bytes, _))| {
                    TensorView::new(*dtype, shape.clone(), bytes)
                        .map(|view| (name.as_str(), view))
                        .map_err(|error| invalid(format!("invalid grouped tensor view: {error}")))
                })
                .collect::<Result<Vec<_>>>()?;
            let packed = serialize(views, None).map_err(|error| {
                invalid(format!("could not serialize grouped MXFP4 owner: {error}"))
            })?;
            let packed_artifact_bytes = u64::try_from(packed.len())
                .map_err(|_| invalid("grouped MXFP4 owner size does not fit u64"))?;
            if packed_artifact_bytes > MAX_MXFP4_EXPERT_ARTIFACT_BYTES
                || packed_artifact_bytes < payload_bytes
            {
                return Err(Error::new(
                    "ARTIFACT_BUDGET",
                    "DeepSeek grouped owner exceeds the existing 64-MiB Session artifact bound",
                ));
            }
            let packed_artifact_sha256 = crate::sha256::hex(&packed);
            let binding = crate::session::mxfp4::Binding {
                artifact_sha256: packed_artifact_sha256.clone(),
                model_revision: route.stamp.model_revision.clone(),
                rank: rank_dispatch.rank,
                layer: route.stamp.layer,
                epoch: route.stamp.epoch,
                experts: members
                    .iter()
                    .map(|member| crate::session::mxfp4::Expert {
                        expert: member.expert,
                        local: member.local,
                    })
                    .collect(),
            };
            let owner = crate::session::mxfp4::Artifact::new(packed, binding)?;
            let payload_sha256 = buffers
                .into_iter()
                .map(|(name, (_, _, _, digest))| (name, digest))
                .collect();
            (
                owner,
                payload_sha256,
                packed_artifact_sha256,
                packed_artifact_bytes,
                packing_peak_bound_bytes,
            )
        };

        grouped_bytes = grouped_bytes
            .checked_add(packed_artifact_bytes)
            .ok_or_else(|| invalid("DeepSeek grouped artifact byte count overflow"))?;
        if grouped_bytes > preflight.total_artifact_bytes {
            return Err(Error::new(
                "ARTIFACT_BUDGET",
                "DeepSeek rank-group artifacts exceed the preflighted aggregate owner budget",
            ));
        }
        grouped.push(PinnedMxfp4RankResidentGroup {
            stamp: preflight.stamp.clone(),
            attempt: preflight.attempt,
            route_experts: preflight.route_experts.clone(),
            route_fingerprint: preflight.route_fingerprint.clone(),
            rank: rank_dispatch.rank,
            members,
            source_shards,
            payload_sha256,
            packed_artifact_sha256,
            packed_artifact_bytes,
            packing_peak_bound_bytes,
            owner,
        });
    }
    if !by_expert.is_empty() {
        return Err(invalid(
            "DeepSeek route rank grouping left an unselected resident owner",
        ));
    }
    Ok((preflight, grouped))
}

/// The exact single-token top-6 owner binding admitted before any rank worker starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedMxfp4RouteOwnerBinding {
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
}

/// Immutable proof that every expert selected by one identified route is resident
/// exactly once at the rank/local selected by VoltaFabric dispatch. Construct this
/// before connecting workers; the proof includes the exact route coefficients.
#[derive(Debug, PartialEq, Eq)]
pub struct PinnedMxfp4RoutePreflight {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    owners: Vec<PinnedMxfp4RouteOwnerBinding>,
    total_artifact_bytes: u64,
}

impl PinnedMxfp4RoutePreflight {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    /// Entries are in donor route order; coefficients are committed by the digest.
    pub fn owners(&self) -> &[PinnedMxfp4RouteOwnerBinding] {
        &self.owners
    }

    pub fn total_artifact_bytes(&self) -> u64 {
        self.total_artifact_bytes
    }
}

/// Preflight a complete route-bound resident set before opening a worker or sending
/// LOAD. It rejects incomplete/duplicate experts, changed coefficients, foreign rank
/// bindings, bad artifact inventories/digests, and insufficient aggregate byte budget.
pub fn preflight_pinned_mxfp4_route_residents(
    route: &crate::routing::RoutingProblem,
    attempt: u64,
    residents: &[PinnedMxfp4RankResidentExpert],
    max_total_artifact_bytes: u64,
) -> Result<PinnedMxfp4RoutePreflight> {
    let route_fingerprint = pinned_top6_route_fingerprint(route, attempt)?;
    let top_k = crate::deepseek_v41_gate::TOP_K;
    let dispatch = crate::routing::dispatch(route)?;
    if dispatch.slots.len() != top_k {
        return Err(invalid(
            "DeepSeek route preflight dispatch differs from its top-6 route",
        ));
    }
    let mut expected = BTreeMap::new();
    let mut expected_local_bindings = BTreeSet::new();
    for slot in dispatch.slots {
        if !expected_local_bindings.insert((slot.rank, slot.local)) {
            return Err(invalid(
                "DeepSeek route preflight maps multiple selected experts to one rank-local owner",
            ));
        }
        if expected
            .insert(slot.expert, (slot.rank, slot.local))
            .is_some()
        {
            return Err(invalid(
                "DeepSeek route preflight has a duplicate dispatched expert",
            ));
        }
    }
    if residents.len() != top_k || max_total_artifact_bytes == 0 {
        return Err(invalid(
            "DeepSeek route preflight requires exactly six owners and a nonzero aggregate budget",
        ));
    }
    let route_ids = route.experts.iter().copied().collect::<BTreeSet<_>>();
    if expected.len() != top_k || route_ids.len() != top_k {
        return Err(invalid(
            "DeepSeek route preflight does not identify six unique dispatched experts",
        ));
    }

    let mut by_expert = BTreeMap::new();
    let mut total_artifact_bytes = 0u64;
    for resident in residents {
        if resident.stamp != route.stamp
            || resident.attempt != attempt
            || resident.route_experts != route.experts
            || resident.route_fingerprint != route_fingerprint
            || !route_ids.contains(&resident.expert)
            || !crate::sha256::is_hex_digest(&resident.route_fingerprint)
            || !crate::sha256::is_hex_digest(&resident.source_header_sha256)
            || !pinned_shard_basename(&resident.source_shard)
            || resident.payload_sha256.len() != 6
            || resident
                .payload_sha256
                .values()
                .any(|digest| !crate::sha256::is_hex_digest(digest))
            || resident.packed_artifact_bytes == 0
            || resident.packed_artifact_bytes > MAX_MXFP4_EXPERT_ARTIFACT_BYTES
            || u64::try_from(resident.owner.bytes().len()).ok()
                != Some(resident.packed_artifact_bytes)
            || crate::sha256::hex(resident.owner.bytes()) != resident.packed_artifact_sha256
        {
            return Err(invalid(
                "DeepSeek route resident identity, digest inventory, or artifact bytes differ",
            ));
        }
        let Some(&(expected_rank, expected_local)) = expected.get(&resident.expert) else {
            return Err(invalid(
                "DeepSeek route resident contains an unselected expert",
            ));
        };
        let binding = resident.owner.binding();
        let selected = crate::session::mxfp4::Expert {
            expert: resident.expert,
            local: resident.local,
        };
        if resident.rank != expected_rank
            || resident.local != expected_local
            || binding.rank != expected_rank
            || binding.layer != route.stamp.layer
            || binding.epoch != route.stamp.epoch
            || binding.model_revision != route.stamp.model_revision
            || binding.experts.as_slice() != [selected]
            || binding.artifact_sha256 != resident.packed_artifact_sha256
        {
            return Err(invalid(
                "DeepSeek route resident does not match its selected rank/local binding",
            ));
        }
        let prefix = format!(
            "layers.{}.ffn.experts.{}",
            route.stamp.layer, resident.expert
        );
        let expected_payload_names = [
            format!("{prefix}.w1.weight"),
            format!("{prefix}.w1.scale"),
            format!("{prefix}.w2.weight"),
            format!("{prefix}.w2.scale"),
            format!("{prefix}.w3.weight"),
            format!("{prefix}.w3.scale"),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        if resident
            .payload_sha256
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected_payload_names
        {
            return Err(invalid(
                "DeepSeek route resident payload inventory is not its exact six tensors",
            ));
        }
        if by_expert
            .insert(
                resident.expert,
                PinnedMxfp4RouteOwnerBinding {
                    expert: resident.expert,
                    rank: resident.rank,
                    local: resident.local,
                    artifact_sha256: resident.packed_artifact_sha256.clone(),
                    artifact_bytes: resident.packed_artifact_bytes,
                },
            )
            .is_some()
        {
            return Err(invalid(
                "DeepSeek route preflight contains a duplicate expert owner",
            ));
        }
        total_artifact_bytes = total_artifact_bytes
            .checked_add(resident.packed_artifact_bytes)
            .ok_or_else(|| invalid("DeepSeek route artifact byte count overflow"))?;
    }
    let hard_limit = MAX_MXFP4_EXPERT_ARTIFACT_BYTES
        .checked_mul(top_k as u64)
        .ok_or_else(|| invalid("DeepSeek route artifact hard limit overflow"))?;
    if by_expert.keys().copied().collect::<BTreeSet<_>>() != route_ids {
        return Err(invalid(
            "DeepSeek route preflight is missing one or more selected experts",
        ));
    }
    if total_artifact_bytes > max_total_artifact_bytes || total_artifact_bytes > hard_limit {
        return Err(Error::new(
            "ARTIFACT_BUDGET",
            "DeepSeek selected route owners exceed the admitted aggregate artifact budget",
        ));
    }
    let owners = route
        .experts
        .iter()
        .map(|expert| {
            by_expert
                .remove(expert)
                .ok_or_else(|| invalid("DeepSeek route owner disappeared after preflight"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PinnedMxfp4RoutePreflight {
        stamp: route.stamp.clone(),
        attempt,
        route_experts: route.experts.clone(),
        route_fingerprint,
        owners,
        total_artifact_bytes,
    })
}

/// Successful LOAD receipt that retains identity not carried by the legacy
/// Session wire header. This records one owner only; it is not a route completion.
#[derive(Debug)]
pub struct PinnedMxfp4SessionLoadReceipt {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    expert: u32,
    rank: u32,
    local: u32,
    source_shard: String,
    source_header_sha256: String,
    payload_sha256: BTreeMap<String, String>,
    packed_artifact_sha256: String,
    session_load: crate::session::LoadReceipt,
}

impl PinnedMxfp4SessionLoadReceipt {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn expert(&self) -> u32 {
        self.expert
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn local(&self) -> u32 {
        self.local
    }

    pub fn source_shard(&self) -> &str {
        &self.source_shard
    }

    pub fn source_header_sha256(&self) -> &str {
        &self.source_header_sha256
    }

    pub fn payload_sha256(&self) -> &BTreeMap<String, String> {
        &self.payload_sha256
    }

    pub fn packed_artifact_sha256(&self) -> &str {
        &self.packed_artifact_sha256
    }

    pub fn session_load(&self) -> &crate::session::LoadReceipt {
        &self.session_load
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinnedMxfp4RouteExecutionState {
    Completed,
    Aborted,
    Unknown,
}

/// One-shot result for one single-token top-6 route. Output is present only after all
/// six rank receipts are Completed and the existing FP32-preweighted Collector finishes.
#[derive(Debug, Serialize, Deserialize)]
pub struct PinnedMxfp4RouteExecutionReceipt {
    state: PinnedMxfp4RouteExecutionState,
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_experts: Vec<u32>,
    route_fingerprint: String,
    exec_receipts: Vec<crate::session::ExecReceipt>,
    output_bits: Option<Vec<Vec<u32>>>,
    error: Option<Error>,
}

impl PinnedMxfp4RouteExecutionReceipt {
    pub fn state(&self) -> PinnedMxfp4RouteExecutionState {
        self.state
    }

    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.route_experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn exec_receipts(&self) -> &[crate::session::ExecReceipt] {
        &self.exec_receipts
    }

    pub fn output_bits(&self) -> Option<&[Vec<u32>]> {
        self.output_bits.as_deref()
    }

    pub fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }
}

/// Execute one preflighted six-expert token through the existing CPU MXFP4 Sessions
/// and FP32-preweighted Collector. The preflight and LOAD receipts are consumed so this
/// adapter cannot accidentally reuse the same values for a retry. Any ambiguous EXEC
/// is returned as `Unknown` with its receipts; it never resends or exposes partial output.
///
/// Preflight errors return `Err` before any EXEC request. After the first EXEC attempt,
/// the function returns a terminal receipt, including failures, to preserve uncertainty.
pub fn execute_pinned_mxfp4_route(
    preflight: PinnedMxfp4RoutePreflight,
    problem: &crate::routing::RoutingProblem,
    attempt: u64,
    inputs: Vec<crate::session::mxfp4::TokenInput>,
    load_receipts: Vec<PinnedMxfp4SessionLoadReceipt>,
    sessions: &mut BTreeMap<u32, crate::session::Session>,
    timeout: std::time::Duration,
) -> Result<PinnedMxfp4RouteExecutionReceipt> {
    let top_k = crate::deepseek_v41_gate::TOP_K;
    if timeout.is_zero()
        || attempt == 0
        || attempt != preflight.attempt
        || problem.stamp != preflight.stamp
        || problem.experts != preflight.route_experts
        || preflight.route_fingerprint != pinned_top6_route_fingerprint(problem, attempt)?
        || problem.routes.len() != 1
        || preflight.owners.len() != top_k
        || load_receipts.len() != top_k
        || inputs.len() != 1
        || inputs[0].token != 0
        || inputs[0].value_bits.len() != problem.hidden
    {
        return Err(invalid(
            "DeepSeek EXEC requires its exact one-token route, six LOAD receipts, and nonzero deadline",
        ));
    }
    for &bits in &inputs[0].value_bits {
        let value = f32::from_bits(bits);
        if crate::deepseek_v41::cast_bf16_rne_reference(value)?.to_bits() != bits {
            return Err(invalid(
                "DeepSeek EXEC input must be finite and exactly BF16 before any rank is sent",
            ));
        }
    }

    let dispatch = crate::routing::dispatch(problem)?;
    if dispatch.slots.len() != top_k || dispatch.ranks.len() != top_k {
        return Err(invalid(
            "DeepSeek EXEC requires exactly one distinct participating rank per route expert",
        ));
    }
    let expected_ranks = preflight
        .owners
        .iter()
        .map(|owner| owner.rank)
        .collect::<BTreeSet<_>>();
    let dispatch_ranks = dispatch
        .ranks
        .iter()
        .map(|rank| rank.rank)
        .collect::<BTreeSet<_>>();
    if expected_ranks.len() != top_k
        || dispatch_ranks != expected_ranks
        || dispatch
            .ranks
            .iter()
            .any(|rank| rank.slots.len() != 1 || rank.slots[0].slot.token != 0)
        || sessions.len() != top_k
        || sessions.keys().copied().collect::<BTreeSet<_>>() != expected_ranks
    {
        return Err(invalid(
            "DeepSeek EXEC Session set does not match all six distinct route ranks",
        ));
    }

    let mut loads_by_expert = BTreeMap::new();
    for receipt in load_receipts {
        if receipt.stamp != preflight.stamp
            || receipt.attempt != preflight.attempt
            || receipt.route_experts != preflight.route_experts
            || receipt.route_fingerprint != preflight.route_fingerprint
            || receipt.session_load.rank != receipt.rank
            || receipt.session_load.epoch != preflight.stamp.epoch
            || receipt.session_load.artifact_sha256 != receipt.packed_artifact_sha256
            || loads_by_expert.insert(receipt.expert, receipt).is_some()
        {
            return Err(invalid(
                "DeepSeek EXEC LOAD receipt is stale, duplicated, or has a foreign artifact",
            ));
        }
    }
    for owner in &preflight.owners {
        let receipt = loads_by_expert
            .remove(&owner.expert)
            .ok_or_else(|| invalid("DeepSeek EXEC is missing one route owner's LOAD receipt"))?;
        if receipt.rank != owner.rank
            || receipt.local != owner.local
            || receipt.packed_artifact_sha256 != owner.artifact_sha256
            || receipt.session_load.artifact_sha256 != owner.artifact_sha256
        {
            return Err(invalid(
                "DeepSeek EXEC LOAD receipt does not match the complete route preflight",
            ));
        }
    }
    if !loads_by_expert.is_empty() {
        return Err(invalid("DeepSeek EXEC received an unselected LOAD receipt"));
    }
    for (&rank, session) in sessions.iter() {
        if session.rank != rank {
            return Err(invalid(
                "DeepSeek EXEC Session map key differs from HELLO rank",
            ));
        }
    }

    let mut collector = crate::preweighted::Collector::new(problem)?;
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| invalid("DeepSeek EXEC timeout overflows Instant"))?;
    // Validate every live worker's immutable loaded binding and all activations before
    // the first rank receives bytes. start_preweighted_fp32_until repeats this check.
    for rank in &dispatch.ranks {
        sessions
            .get_mut(&rank.rank)
            .ok_or_else(|| invalid("DeepSeek EXEC rank Session disappeared"))?
            .preflight_preweighted_fp32_until(
                problem,
                &collector,
                attempt,
                &inputs,
                &crate::process::Fault::None,
                deadline,
            )?;
    }

    let mut pending = Vec::with_capacity(top_k);
    let mut start_error = None;
    for rank in &dispatch.ranks {
        let session = sessions
            .get_mut(&rank.rank)
            .expect("the complete Session set was preflighted");
        match session.start_preweighted_fp32_until(
            problem,
            &collector,
            attempt,
            inputs.clone(),
            crate::process::Fault::None,
            deadline,
        ) {
            Ok(work) => pending.push((rank.rank, work)),
            Err(error) => {
                start_error = Some(error);
                break;
            }
        }
    }

    let mut exec_receipts = Vec::with_capacity(pending.len());
    let mut result_error = start_error;
    for (rank, work) in pending {
        let receipt = sessions
            .get_mut(&rank)
            .expect("rank Session remains in the preflighted set")
            .finish_preweighted_fp32(work, &mut collector, deadline);
        if result_error.is_none() {
            result_error = receipt.error.clone();
        }
        exec_receipts.push(receipt);
    }

    let all_completed = exec_receipts.len() == top_k
        && exec_receipts.iter().all(|receipt| {
            receipt.state == crate::process::ReceiptState::Completed && receipt.error.is_none()
        })
        && result_error.is_none();
    let (state, output_bits, error) = if all_completed {
        match collector.finish() {
            Ok(output) if output.len() == 1 => (
                PinnedMxfp4RouteExecutionState::Completed,
                Some(
                    output
                        .into_iter()
                        .map(|row| row.into_iter().map(f32::to_bits).collect())
                        .collect(),
                ),
                None,
            ),
            Ok(_) => (
                PinnedMxfp4RouteExecutionState::Aborted,
                None,
                Some(invalid(
                    "DeepSeek EXEC collector returned a non-single-token result",
                )),
            ),
            Err(error) => (PinnedMxfp4RouteExecutionState::Aborted, None, Some(error)),
        }
    } else {
        collector.expire();
        let unknown = exec_receipts.iter().any(|receipt| {
            matches!(
                receipt.state,
                crate::process::ReceiptState::Unknown | crate::process::ReceiptState::Started
            )
        });
        if result_error.is_none() {
            result_error = exec_receipts
                .iter()
                .find_map(|receipt| receipt.error.clone());
        }
        if result_error.is_none() {
            result_error = Some(Error::new(
                "BATCH_INCOMPLETE",
                "DeepSeek route EXEC did not complete every rank slot; output withheld",
            ));
        }
        (
            if unknown {
                PinnedMxfp4RouteExecutionState::Unknown
            } else {
                PinnedMxfp4RouteExecutionState::Aborted
            },
            None,
            result_error,
        )
    };
    Ok(PinnedMxfp4RouteExecutionReceipt {
        state,
        stamp: preflight.stamp,
        attempt: preflight.attempt,
        route_experts: preflight.route_experts,
        route_fingerprint: preflight.route_fingerprint,
        exec_receipts,
        output_bits,
        error,
    })
}

/// Execute one complete top-6 route when multiple selected experts are co-located on
/// a rank. One already-loaded Session and one group LOAD receipt are required for each
/// participating rank. Every rank is preflighted before the first EXEC; the existing
/// FP32-preweighted Collector applies each route coefficient exactly once.
pub fn execute_pinned_mxfp4_route_grouped(
    preflight: PinnedMxfp4RoutePreflight,
    problem: &crate::routing::RoutingProblem,
    attempt: u64,
    inputs: Vec<crate::session::mxfp4::TokenInput>,
    load_receipts: Vec<PinnedMxfp4RouteRankGroupLoadReceipt>,
    sessions: &mut BTreeMap<u32, crate::session::Session>,
    timeout: std::time::Duration,
) -> Result<PinnedMxfp4RouteExecutionReceipt> {
    let top_k = crate::deepseek_v41_gate::TOP_K;
    if timeout.is_zero()
        || attempt == 0
        || attempt != preflight.attempt
        || problem.stamp != preflight.stamp
        || problem.experts != preflight.route_experts
        || preflight.route_fingerprint != pinned_top6_route_fingerprint(problem, attempt)?
        || problem.routes.len() != 1
        || preflight.owners.len() != top_k
        || inputs.len() != 1
        || inputs[0].token != 0
        || inputs[0].value_bits.len() != problem.hidden
    {
        return Err(invalid(
            "DeepSeek grouped EXEC requires its exact one-token route, complete owner preflight, and nonzero deadline",
        ));
    }
    for &bits in &inputs[0].value_bits {
        let value = f32::from_bits(bits);
        if crate::deepseek_v41::cast_bf16_rne_reference(value)?.to_bits() != bits {
            return Err(invalid(
                "DeepSeek grouped EXEC input must be finite and exactly BF16 before any rank is sent",
            ));
        }
    }

    let dispatch = crate::routing::dispatch(problem)?;
    if dispatch.slots.len() != top_k || dispatch.ranks.is_empty() {
        return Err(invalid(
            "DeepSeek grouped EXEC requires all six selected route slots",
        ));
    }
    let expected_ranks = preflight
        .owners
        .iter()
        .map(|owner| owner.rank)
        .collect::<BTreeSet<_>>();
    let dispatch_ranks = dispatch
        .ranks
        .iter()
        .map(|rank| rank.rank)
        .collect::<BTreeSet<_>>();
    if expected_ranks != dispatch_ranks
        || sessions.len() != expected_ranks.len()
        || sessions.keys().copied().collect::<BTreeSet<_>>() != expected_ranks
    {
        return Err(invalid(
            "DeepSeek grouped EXEC Session set does not match all participating route ranks",
        ));
    }

    let mut expected_members = BTreeMap::<u32, Vec<PinnedMxfp4RouteOwnerBinding>>::new();
    for owner in &preflight.owners {
        expected_members
            .entry(owner.rank)
            .or_default()
            .push(owner.clone());
    }
    for members in expected_members.values_mut() {
        members.sort_by_key(|owner| owner.local);
    }
    for rank in &dispatch.ranks {
        let members = expected_members
            .get(&rank.rank)
            .ok_or_else(|| invalid("DeepSeek grouped EXEC dispatch rank has no owners"))?;
        let locals = members.iter().map(|owner| owner.local).collect::<Vec<_>>();
        if rank.resident_order != locals
            || rank.slots.len() != members.len()
            || rank.slots.iter().any(|slot| slot.slot.token != 0)
        {
            return Err(invalid(
                "DeepSeek grouped EXEC selected locals do not match the exact Session resident order",
            ));
        }
    }

    let mut loads_by_rank = BTreeMap::new();
    for receipt in load_receipts {
        if receipt.stamp != preflight.stamp
            || receipt.attempt != preflight.attempt
            || receipt.route_experts != preflight.route_experts
            || receipt.route_fingerprint != preflight.route_fingerprint
            || receipt.session_load.rank != receipt.rank
            || receipt.session_load.epoch != preflight.stamp.epoch
            || receipt.session_load.artifact_sha256 != receipt.packed_artifact_sha256
            || expected_members.get(&receipt.rank) != Some(&receipt.members)
            || loads_by_rank.insert(receipt.rank, receipt).is_some()
        {
            return Err(invalid(
                "DeepSeek grouped EXEC LOAD receipt is stale, duplicated, or has a foreign rank owner set",
            ));
        }
    }
    if loads_by_rank.keys().copied().collect::<BTreeSet<_>>() != expected_ranks {
        return Err(invalid(
            "DeepSeek grouped EXEC is missing a participating rank's LOAD receipt",
        ));
    }
    for (&rank, session) in sessions.iter() {
        if session.rank != rank
            || loads_by_rank
                .get(&rank)
                .is_none_or(|receipt| receipt.session_load.rank != session.rank)
        {
            return Err(invalid(
                "DeepSeek grouped EXEC Session identity differs from its rank LOAD receipt",
            ));
        }
    }

    let mut collector = crate::preweighted::Collector::new(problem)?;
    let deadline = std::time::Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| invalid("DeepSeek grouped EXEC timeout overflows Instant"))?;
    // Validate every live Session and activation before any rank can observe EXEC.
    for rank in &dispatch.ranks {
        sessions
            .get_mut(&rank.rank)
            .ok_or_else(|| invalid("DeepSeek grouped EXEC rank Session disappeared"))?
            .preflight_preweighted_fp32_until(
                problem,
                &collector,
                attempt,
                &inputs,
                &crate::process::Fault::None,
                deadline,
            )?;
    }

    let mut pending = Vec::with_capacity(dispatch.ranks.len());
    let mut start_error = None;
    for rank in &dispatch.ranks {
        let session = sessions
            .get_mut(&rank.rank)
            .expect("the complete grouped Session set was preflighted");
        match session.start_preweighted_fp32_until(
            problem,
            &collector,
            attempt,
            inputs.clone(),
            crate::process::Fault::None,
            deadline,
        ) {
            Ok(work) => pending.push((rank.rank, work)),
            Err(error) => {
                start_error = Some(error);
                break;
            }
        }
    }

    let mut exec_receipts = Vec::with_capacity(pending.len());
    let mut result_error = start_error;
    for (rank, work) in pending {
        let receipt = sessions
            .get_mut(&rank)
            .expect("rank Session remains in the preflighted grouped set")
            .finish_preweighted_fp32(work, &mut collector, deadline);
        if result_error.is_none() {
            result_error = receipt.error.clone();
        }
        exec_receipts.push(receipt);
    }

    let all_completed = exec_receipts.len() == dispatch.ranks.len()
        && exec_receipts.iter().all(|receipt| {
            receipt.state == crate::process::ReceiptState::Completed && receipt.error.is_none()
        })
        && result_error.is_none();
    let (state, output_bits, error) = if all_completed {
        match collector.finish() {
            Ok(output) if output.len() == 1 => (
                PinnedMxfp4RouteExecutionState::Completed,
                Some(
                    output
                        .into_iter()
                        .map(|row| row.into_iter().map(f32::to_bits).collect())
                        .collect(),
                ),
                None,
            ),
            Ok(_) => (
                PinnedMxfp4RouteExecutionState::Aborted,
                None,
                Some(invalid(
                    "DeepSeek grouped EXEC collector returned a non-single-token result",
                )),
            ),
            Err(error) => (PinnedMxfp4RouteExecutionState::Aborted, None, Some(error)),
        }
    } else {
        collector.expire();
        let unknown = exec_receipts.iter().any(|receipt| {
            matches!(
                receipt.state,
                crate::process::ReceiptState::Unknown | crate::process::ReceiptState::Started
            )
        });
        if result_error.is_none() {
            result_error = exec_receipts
                .iter()
                .find_map(|receipt| receipt.error.clone());
        }
        if result_error.is_none() {
            result_error = Some(Error::new(
                "BATCH_INCOMPLETE",
                "DeepSeek grouped EXEC did not complete every rank slot; output withheld",
            ));
        }
        (
            if unknown {
                PinnedMxfp4RouteExecutionState::Unknown
            } else {
                PinnedMxfp4RouteExecutionState::Aborted
            },
            None,
            result_error,
        )
    };
    Ok(PinnedMxfp4RouteExecutionReceipt {
        state,
        stamp: preflight.stamp,
        attempt: preflight.attempt,
        route_experts: preflight.route_experts,
        route_fingerprint: preflight.route_fingerprint,
        exec_receipts,
        output_bits,
        error,
    })
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

fn read_pinned_safetensors_header<R: Read + Seek>(
    reader: &mut R,
    max_source_shard_bytes: u64,
) -> Result<(u64, Vec<u8>)> {
    let file_bytes = reader
        .seek(SeekFrom::End(0))
        .map_err(|error| Error::new("CHECKPOINT_IO", error.to_string()))?;
    if file_bytes == 0 || max_source_shard_bytes == 0 || file_bytes > max_source_shard_bytes {
        return Err(invalid(
            "DeepSeek expert source shard is empty or exceeds its caller byte ceiling",
        ));
    }
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| Error::new("CHECKPOINT_IO", error.to_string()))?;
    let mut length = [0_u8; 8];
    reader
        .read_exact(&mut length)
        .map_err(|error| Error::new("CHECKPOINT_IO", error.to_string()))?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > crate::checkpoint::MAX_HEADER_BYTES {
        return Err(invalid(
            "DeepSeek expert Safetensors header exceeds its 8 MiB bound or is empty",
        ));
    }
    let data_start = header_len
        .checked_add(8)
        .ok_or_else(|| invalid("DeepSeek expert header offset overflow"))?;
    if data_start > file_bytes {
        return Err(invalid("DeepSeek expert source has a truncated header"));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| invalid("DeepSeek expert header length does not fit memory size"))?;
    let mut header = Vec::new();
    header
        .try_reserve_exact(header_len)
        .map_err(|error| Error::new("ALLOCATION_FAILED", error.to_string()))?;
    header.resize(header_len, 0);
    reader
        .read_exact(&mut header)
        .map_err(|error| Error::new("CHECKPOINT_IO", error.to_string()))?;
    Ok((file_bytes, header))
}

/// Hash the complete one-token route identity and its selected rank/local owners.
/// Routing weights are encoded as their exact IEEE-754 bit patterns: no sorting,
/// rounding, or renormalization can silently change the route a resident belongs to.
pub(crate) fn pinned_top6_route_fingerprint(
    route: &crate::routing::RoutingProblem,
    attempt: u64,
) -> Result<String> {
    crate::routing::validate(route)?;
    let top_k = crate::deepseek_v41_gate::TOP_K;
    if attempt == 0
        || route.stamp.model_revision != REVISION
        || route.hidden != crate::deepseek_v41_gate::HIDDEN_SIZE
        || route.routes.len() != 1
        || route.routes[0].len() != top_k
        || route.experts.len() != top_k
        || route.experts
            != route.routes[0]
                .iter()
                .map(|choice| choice.expert)
                .collect::<Vec<_>>()
    {
        return Err(invalid(
            "DeepSeek route fingerprint requires one identified top-6 route",
        ));
    }
    let dispatch = crate::routing::dispatch(route)?;
    if dispatch.slots.len() != top_k {
        return Err(invalid(
            "DeepSeek route fingerprint dispatch does not contain six slots",
        ));
    }

    let mut bytes = Vec::with_capacity(512);
    bytes.extend_from_slice(b"VoltaFabric\0DeepSeek-MXFP4-Top6-Route\0v1\0");
    for value in [
        route.stamp.run_id.as_str(),
        route.stamp.model_revision.as_str(),
        route.stamp.plan_id.as_str(),
    ] {
        let length = u32::try_from(value.len())
            .map_err(|_| invalid("DeepSeek route identity string is too long"))?;
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    bytes.extend_from_slice(&route.stamp.layer.to_be_bytes());
    bytes.extend_from_slice(&route.stamp.batch.to_be_bytes());
    bytes.extend_from_slice(&route.stamp.epoch.to_be_bytes());
    bytes.extend_from_slice(&attempt.to_be_bytes());
    bytes.extend_from_slice(
        &u64::try_from(route.hidden)
            .map_err(|_| invalid("DeepSeek route hidden width does not fit u64"))?
            .to_be_bytes(),
    );
    for (slot_index, (choice, selected)) in route.routes[0].iter().zip(&dispatch.slots).enumerate()
    {
        if selected.token != 0
            || selected.slot != slot_index
            || selected.expert != choice.expert
            || selected.weight.to_bits() != choice.weight.to_bits()
        {
            return Err(invalid(
                "DeepSeek route fingerprint dispatch changed donor slot order or coefficient",
            ));
        }
        bytes.extend_from_slice(&choice.expert.to_be_bytes());
        bytes.extend_from_slice(&choice.weight.to_bits().to_be_bytes());
        bytes.extend_from_slice(&selected.rank.to_be_bytes());
        bytes.extend_from_slice(&selected.local.to_be_bytes());
    }
    Ok(crate::sha256::hex(&bytes))
}

/// Read one gate-selected MXFP4 expert for one registered rank. The route is validated
/// without changing its expert order or coefficients; all six expected tensor hashes
/// must be supplied before any weight range is read. The existing checkpoint reader
/// performs the bounded selected-range reads and withholds the import on any hash
/// failure. The source must remain quiescent during this operation.
pub fn read_pinned_mxfp4_rank_expert<R: Read + Seek>(
    reader: &mut R,
    request: PinnedMxfp4RankExpertReadRequest<'_>,
) -> Result<PinnedMxfp4RankExpertRead> {
    let PinnedMxfp4RankExpertReadRequest {
        route,
        attempt,
        expert,
        rank,
        index_bytes,
        header_shard_name,
        expected_header_sha256,
        expected_payload_sha256,
        max_source_shard_bytes,
        max_payload_bytes,
    } = request;
    crate::routing::validate(route)?;
    if attempt == 0
        || route.stamp.model_revision != REVISION
        || route.hidden != crate::deepseek_v41_gate::HIDDEN_SIZE
        || route.routes.len() != 1
        || route.routes[0].len() != crate::deepseek_v41_gate::TOP_K
        || route.experts.len() != crate::deepseek_v41_gate::TOP_K
        || route.experts
            != route.routes[0]
                .iter()
                .map(|choice| choice.expert)
                .collect::<Vec<_>>()
        || !route.experts.contains(&expert)
    {
        return Err(invalid(
            "DeepSeek expert payload request does not match one identified top-6 route",
        ));
    }
    let route_fingerprint = pinned_top6_route_fingerprint(route, attempt)?;
    let local = route
        .registry
        .copies
        .iter()
        .find(|copy| copy.expert == expert && copy.rank == rank)
        .map(|copy| copy.local)
        .ok_or_else(|| invalid("selected DeepSeek expert has no binding on the requested rank"))?;
    if !route
        .resident_order
        .get(&rank)
        .is_some_and(|order| order.contains(&local))
    {
        return Err(invalid(
            "selected DeepSeek expert local binding is absent from the rank order",
        ));
    }
    if index_bytes.is_empty()
        || index_bytes.len() > MAX_PINNED_EXPERT_INDEX_BYTES
        || crate::sha256::hex(index_bytes) != PINNED_EXPERT_INDEX_SHA256
        || !pinned_shard_basename(header_shard_name)
        || !crate::sha256::is_hex_digest(expected_header_sha256)
        || max_source_shard_bytes == 0
        || max_payload_bytes == 0
        || max_payload_bytes > crate::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES
    {
        return Err(invalid(
            "DeepSeek selective read identity, pinned metadata, or byte ceiling is invalid",
        ));
    }

    let (source_shard_bytes, header) =
        read_pinned_safetensors_header(reader, max_source_shard_bytes)?;
    let header_sha256 = crate::sha256::hex(&header);
    if header_sha256 != expected_header_sha256 {
        return Err(invalid(
            "DeepSeek selective read source header SHA-256 differs",
        ));
    }
    let data_start_bytes = u64::try_from(header.len())
        .ok()
        .and_then(|length| length.checked_add(8))
        .ok_or_else(|| invalid("DeepSeek selective read header offset overflow"))?;
    let plan = plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
        index_bytes,
        header_bytes: &header,
        expected_header_sha256,
        header_shard_name,
        source_shard_bytes,
        max_source_shard_bytes,
        namespace: Mxfp4ExpertNamespace::Backbone,
        layer: route.stamp.layer,
        experts: &[expert],
    })?;
    if plan.selected_payload_bytes > max_payload_bytes
        || plan.tensors.len() != 6
        || expected_payload_sha256.len() != plan.tensors.len()
    {
        return Err(invalid(
            "DeepSeek one-expert payload selection exceeds its cap or lacks six hashes",
        ));
    }

    let mut selected = Vec::with_capacity(plan.tensors.len());
    for tensor in &plan.tensors {
        let Some(expected_sha256) = expected_payload_sha256.get(&tensor.name) else {
            return Err(invalid(format!(
                "DeepSeek expert payload digest is missing: {}",
                tensor.name
            )));
        };
        if !crate::sha256::is_hex_digest(expected_sha256) {
            return Err(invalid(format!(
                "DeepSeek expert payload digest is invalid: {}",
                tensor.name
            )));
        }
        let dtype = match tensor.dtype {
            "I8" => Dtype::I8,
            "F8_E8M0" => Dtype::F8_E8M0,
            _ => return Err(invalid("DeepSeek expert payload dtype is unsupported")),
        };
        let bytes = tensor
            .file_offset_end
            .checked_sub(tensor.file_offset_start)
            .ok_or_else(|| invalid("DeepSeek expert payload byte range is reversed"))?;
        selected.push(crate::checkpoint::NamedTensorSpec {
            name: tensor.name.clone(),
            dtype,
            shape: tensor.shape.clone(),
            bytes,
            sha256: expected_sha256.clone(),
        });
    }
    if expected_payload_sha256
        .keys()
        .any(|name| !plan.tensors.iter().any(|tensor| &tensor.name == name))
    {
        return Err(invalid(
            "DeepSeek expert payload digest map contains an unselected tensor",
        ));
    }

    let header_validation_io_bytes = data_start_bytes;
    let payloads = crate::checkpoint::read_named_tensors(reader, &selected, max_payload_bytes)?;
    if payloads.file_bytes != plan.source_shard_bytes
        || payloads.header_bytes != plan.header_bytes
        || payloads.tensors.len() != plan.tensors.len()
    {
        return Err(invalid(
            "DeepSeek selective reader result differs from the checked range plan",
        ));
    }
    let (final_file_bytes, final_header) =
        read_pinned_safetensors_header(reader, max_source_shard_bytes)?;
    if final_file_bytes != source_shard_bytes || crate::sha256::hex(&final_header) != header_sha256
    {
        return Err(Error::new(
            "CHECKPOINT_CHANGED",
            "DeepSeek Safetensors source size or header changed during selective read",
        ));
    }
    let total_io_bytes = header_validation_io_bytes
        .checked_add(payloads.io_bytes)
        .and_then(|bytes| bytes.checked_add(data_start_bytes))
        .ok_or_else(|| invalid("DeepSeek selective read I/O byte count overflow"))?;
    Ok(PinnedMxfp4RankExpertRead {
        stamp: route.stamp.clone(),
        attempt,
        route_experts: route.experts.clone(),
        route_fingerprint,
        expert,
        rank,
        local,
        plan,
        header_sha256,
        header_validation_io_bytes,
        total_io_bytes,
        payloads,
    })
}

/// Read one route-selected DeepSeek expert through the Hugging Face indexed
/// shard map, then return the same receipt type as the bounded single-shard
/// reader. The pinned V4.1 index maps each expert layer to one source shard;
/// the generic index reader is still used for the actual selected-tensor read
/// and fails closed if the six tensors stop sharing that shard.
pub fn read_pinned_mxfp4_rank_expert_indexed(
    root: &Path,
    request: PinnedMxfp4IndexedRankExpertReadRequest<'_>,
) -> Result<PinnedMxfp4RankExpertRead> {
    let PinnedMxfp4IndexedRankExpertReadRequest {
        route,
        attempt,
        expert,
        rank,
        index_bytes,
        expected_header_sha256,
        expected_payload_sha256,
        max_source_shard_bytes,
        max_payload_bytes,
    } = request;
    crate::routing::validate(route)?;
    if attempt == 0
        || route.stamp.model_revision != REVISION
        || route.hidden != crate::deepseek_v41_gate::HIDDEN_SIZE
        || route.routes.len() != 1
        || route.routes[0].len() != crate::deepseek_v41_gate::TOP_K
        || route.experts.len() != crate::deepseek_v41_gate::TOP_K
        || route.experts
            != route.routes[0]
                .iter()
                .map(|choice| choice.expert)
                .collect::<Vec<_>>()
        || !route.experts.contains(&expert)
    {
        return Err(invalid(
            "DeepSeek indexed payload request does not match one identified top-6 route",
        ));
    }
    let route_fingerprint = pinned_top6_route_fingerprint(route, attempt)?;
    let local = route
        .registry
        .copies
        .iter()
        .find(|copy| copy.expert == expert && copy.rank == rank)
        .map(|copy| copy.local)
        .ok_or_else(|| invalid("selected DeepSeek expert has no binding on the requested rank"))?;
    if !route
        .resident_order
        .get(&rank)
        .is_some_and(|order| order.contains(&local))
    {
        return Err(invalid(
            "selected DeepSeek expert local binding is absent from the rank order",
        ));
    }
    if index_bytes.is_empty()
        || index_bytes.len() > MAX_PINNED_EXPERT_INDEX_BYTES
        || crate::sha256::hex(index_bytes) != PINNED_EXPERT_INDEX_SHA256
        || !crate::sha256::is_hex_digest(expected_header_sha256)
        || max_source_shard_bytes == 0
        || max_payload_bytes == 0
        || max_payload_bytes > crate::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES
    {
        return Err(invalid(
            "DeepSeek indexed read identity, pinned index, or byte ceiling is invalid",
        ));
    }

    let prefix = format!("layers.{}.ffn.experts.{expert}", route.stamp.layer);
    let tensor_names = [
        format!("{prefix}.w1.weight"),
        format!("{prefix}.w1.scale"),
        format!("{prefix}.w2.weight"),
        format!("{prefix}.w2.scale"),
        format!("{prefix}.w3.weight"),
        format!("{prefix}.w3.scale"),
    ];
    let expected_names = tensor_names.iter().cloned().collect::<BTreeSet<_>>();
    if expected_payload_sha256.len() != tensor_names.len()
        || expected_payload_sha256
            .keys()
            .any(|name| !expected_names.contains(name))
        || tensor_names.iter().any(|name| {
            expected_payload_sha256
                .get(name)
                .is_none_or(|digest| !crate::sha256::is_hex_digest(digest))
        })
    {
        return Err(invalid(
            "DeepSeek indexed read requires exactly six selected payload hashes",
        ));
    }

    let indexed_plan = crate::checkpoint::plan_indexed_safetensors_shards(
        index_bytes,
        PINNED_EXPERT_INDEX_SHA256,
        &tensor_names,
    )?;
    if indexed_plan.tensor_count != 96_085
        || indexed_plan.total_size_bytes != Some(510_286_023_000)
        || indexed_plan.tensors_by_shard.len() != 1
    {
        return Err(invalid(
            "pinned DeepSeek index metadata or one-layer shard mapping differs",
        ));
    }
    let (header_shard_name, selected_names) = indexed_plan
        .tensors_by_shard
        .iter()
        .next()
        .ok_or_else(|| invalid("pinned DeepSeek index selected no source shard"))?;
    if selected_names.len() != tensor_names.len()
        || selected_names.iter().collect::<BTreeSet<_>>()
            != tensor_names.iter().collect::<BTreeSet<_>>()
    {
        return Err(invalid(
            "pinned DeepSeek shard plan differs from the exact six selected tensors",
        ));
    }

    let header = crate::checkpoint::read_indexed_safetensors_shard_header(
        root,
        header_shard_name,
        max_source_shard_bytes,
    )?;
    if header.sha256 != expected_header_sha256 {
        return Err(invalid(
            "DeepSeek indexed shard header differs from the pinned digest",
        ));
    }
    let plan = plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
        index_bytes,
        header_bytes: &header.header_bytes,
        expected_header_sha256: &header.sha256,
        header_shard_name,
        source_shard_bytes: header.file_bytes,
        max_source_shard_bytes,
        namespace: Mxfp4ExpertNamespace::Backbone,
        layer: route.stamp.layer,
        experts: &[expert],
    })?;
    if plan.data_start_bytes != header.data_start_bytes
        || plan.selected_payload_bytes > max_payload_bytes
        || plan.tensors.len() != tensor_names.len()
    {
        return Err(invalid(
            "DeepSeek indexed shard header differs from the pinned expert plan",
        ));
    }

    let selected = plan
        .tensors
        .iter()
        .map(|tensor| {
            let dtype = match tensor.dtype {
                "I8" => Dtype::I8,
                "F8_E8M0" => Dtype::F8_E8M0,
                _ => return Err(invalid("DeepSeek indexed tensor dtype is unsupported")),
            };
            Ok(crate::checkpoint::NamedTensorSpec {
                name: tensor.name.clone(),
                dtype,
                shape: tensor.shape.clone(),
                bytes: tensor
                    .file_offset_end
                    .checked_sub(tensor.file_offset_start)
                    .ok_or_else(|| invalid("DeepSeek indexed tensor range is reversed"))?,
                sha256: expected_payload_sha256[&tensor.name].clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indexed_import = crate::checkpoint::read_indexed_named_tensors(
        root,
        index_bytes,
        PINNED_EXPERT_INDEX_SHA256,
        &selected,
        max_payload_bytes,
    )?;
    if indexed_import.plan != indexed_plan
        || indexed_import.file_bytes != header.file_bytes
        || indexed_import.header_bytes
            != u64::try_from(header.header_bytes.len())
                .map_err(|_| invalid("DeepSeek indexed header size does not fit u64"))?
        || indexed_import.tensors.len() != tensor_names.len()
    {
        return Err(invalid(
            "indexed Safetensors import differs from the pinned DeepSeek read plan",
        ));
    }
    let final_header = crate::checkpoint::read_indexed_safetensors_shard_header(
        root,
        header_shard_name,
        max_source_shard_bytes,
    )?;
    if final_header.file_bytes != header.file_bytes || final_header.sha256 != header.sha256 {
        return Err(Error::new(
            "CHECKPOINT_CHANGED",
            "DeepSeek indexed shard size or header changed during selective read",
        ));
    }
    let header_io_bytes = header.data_start_bytes;
    let total_io_bytes = header_io_bytes
        .checked_add(indexed_import.io_bytes)
        .and_then(|bytes| bytes.checked_add(final_header.data_start_bytes))
        .ok_or_else(|| invalid("DeepSeek indexed reader I/O byte count overflow"))?;
    Ok(PinnedMxfp4RankExpertRead {
        stamp: route.stamp.clone(),
        attempt,
        route_experts: route.experts.clone(),
        route_fingerprint,
        expert,
        rank,
        local,
        plan,
        header_sha256: header.sha256,
        header_validation_io_bytes: header_io_bytes,
        total_io_bytes,
        payloads: crate::checkpoint::NamedTensorImport {
            file_bytes: indexed_import.file_bytes,
            header_bytes: indexed_import.header_bytes,
            io_bytes: indexed_import.io_bytes,
            tensors: indexed_import.tensors,
        },
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
mod a148_pinned_expert_selective_read_tests {
    use super::{
        plan_pinned_mxfp4_expert_payloads, preflight_pinned_mxfp4_route_residents,
        read_pinned_mxfp4_rank_expert, read_pinned_mxfp4_rank_expert_indexed, Mxfp4ExpertNamespace,
        PinnedExpertPayloadRequest, PinnedMxfp4IndexedRankExpertReadRequest,
        PinnedMxfp4RankExpertReadRequest, PinnedMxfp4RankResidentExpert,
        PINNED_EXPERT_INDEX_SHA256, REVISION,
    };
    use crate::moe::{Choice, Registry, Replica, Stamp};
    use std::{
        collections::BTreeMap,
        fs::{self, File},
        io::{self, Read, Seek, SeekFrom, Write},
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    const HEADER_SHA256: &str = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449";
    const SHARD_BYTES: u64 = 7_389_759_032;
    const TOP6: [u32; 6] = [0, 383, 382, 381, 380, 379];
    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "voltafabric-r722-pinned-index-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct SparseShard {
        file_bytes: u64,
        position: u64,
        segments: Vec<(u64, Vec<u8>)>,
        read_spans: Vec<(u64, u64)>,
    }

    impl Read for SparseShard {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let available = self.file_bytes.saturating_sub(self.position);
            let count = output
                .len()
                .min(usize::try_from(available).unwrap_or(usize::MAX));
            if count == 0 {
                return Ok(0);
            }
            let start = self.position;
            let end = start + count as u64;
            output[..count].fill(0);
            for (segment_start, bytes) in &self.segments {
                let segment_end = *segment_start + bytes.len() as u64;
                let overlap_start = start.max(*segment_start);
                let overlap_end = end.min(segment_end);
                if overlap_start < overlap_end {
                    let output_start = usize::try_from(overlap_start - start)
                        .map_err(|error| io::Error::other(error.to_string()))?;
                    let output_end = usize::try_from(overlap_end - start)
                        .map_err(|error| io::Error::other(error.to_string()))?;
                    let input_start = usize::try_from(overlap_start - *segment_start)
                        .map_err(|error| io::Error::other(error.to_string()))?;
                    let input_end = input_start + (output_end - output_start);
                    output[output_start..output_end]
                        .copy_from_slice(&bytes[input_start..input_end]);
                }
            }
            self.position = end;
            self.read_spans.push((start, end));
            Ok(count)
        }
    }

    impl Seek for SparseShard {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
            let next = match from {
                SeekFrom::Start(position) => i128::from(position),
                SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
                SeekFrom::End(delta) => i128::from(self.file_bytes) + i128::from(delta),
            };
            if !(0..=i128::from(u64::MAX)).contains(&next) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "seek out of range",
                ));
            }
            self.position = next as u64;
            Ok(self.position)
        }
    }

    fn route_problem() -> crate::routing::RoutingProblem {
        let experts = TOP6.to_vec();
        let epoch = 13;
        crate::routing::RoutingProblem {
            hidden: crate::deepseek_v41_gate::HIDDEN_SIZE,
            experts: experts.clone(),
            routes: vec![experts
                .iter()
                .map(|&expert| Choice {
                    expert,
                    weight: 0.25,
                })
                .collect()],
            registry: Registry {
                epoch,
                copies: experts
                    .iter()
                    .enumerate()
                    .map(|(local, &expert)| Replica {
                        expert,
                        rank: 4,
                        local: local as u32,
                    })
                    .collect(),
            },
            stamp: Stamp {
                run_id: "r710-selective-read".to_owned(),
                model_revision: REVISION.to_owned(),
                plan_id: "r710-route-rank-plan".to_owned(),
                layer: 0,
                batch: 7,
                epoch,
            },
            preferred_owners: BTreeMap::new(),
            resident_order: BTreeMap::from([(4, (0..TOP6.len() as u32).collect())]),
        }
    }

    fn distributed_route_problem() -> crate::routing::RoutingProblem {
        let mut route = route_problem();
        route.registry.copies = TOP6
            .iter()
            .enumerate()
            .map(|(index, &expert)| Replica {
                expert,
                rank: 4 + index as u32,
                local: 0,
            })
            .collect();
        route.resident_order = (0..TOP6.len())
            .map(|index| (4 + index as u32, vec![0]))
            .collect();
        route
    }

    fn fixture(
        expert: u32,
    ) -> (
        super::Mxfp4ExpertPayloadPlan,
        BTreeMap<String, String>,
        SparseShard,
    ) {
        let index = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        let header = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
        );
        assert_eq!(crate::sha256::hex(index), PINNED_EXPERT_INDEX_SHA256);
        let plan = plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
            index_bytes: index,
            header_bytes: header,
            expected_header_sha256: HEADER_SHA256,
            header_shard_name: "model-00003-of-00048.safetensors",
            source_shard_bytes: SHARD_BYTES,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
            namespace: Mxfp4ExpertNamespace::Backbone,
            layer: 0,
            experts: &[expert],
        })
        .unwrap();
        let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
        prefix.extend_from_slice(header);
        let mut segments = vec![(0, prefix)];
        let mut hashes = BTreeMap::new();
        for (slot, tensor) in plan.tensors.iter().enumerate() {
            let length =
                usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
            let payload = vec![u8::try_from(slot + 1).unwrap(); length];
            hashes.insert(tensor.name.clone(), crate::sha256::hex(&payload));
            segments.push((tensor.file_offset_start, payload));
        }
        (
            plan,
            hashes,
            SparseShard {
                file_bytes: SHARD_BYTES,
                position: 0,
                segments,
                read_spans: Vec::new(),
            },
        )
    }

    fn read_request<'a>(
        route: &'a crate::routing::RoutingProblem,
        expert: u32,
        hashes: &'a BTreeMap<String, String>,
    ) -> PinnedMxfp4RankExpertReadRequest<'a> {
        read_request_for_rank(route, expert, 4, hashes)
    }

    fn read_request_for_rank<'a>(
        route: &'a crate::routing::RoutingProblem,
        expert: u32,
        rank: u32,
        hashes: &'a BTreeMap<String, String>,
    ) -> PinnedMxfp4RankExpertReadRequest<'a> {
        PinnedMxfp4RankExpertReadRequest {
            route,
            attempt: 29,
            expert,
            rank,
            index_bytes: include_bytes!(
                "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
            ),
            header_shard_name: "model-00003-of-00048.safetensors",
            expected_header_sha256: HEADER_SHA256,
            expected_payload_sha256: hashes,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
            max_payload_bytes: crate::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES,
        }
    }

    fn resident_for_route_expert(
        route: &crate::routing::RoutingProblem,
        expert: u32,
        rank: u32,
    ) -> PinnedMxfp4RankResidentExpert {
        let (_, hashes, mut source) = fixture(expert);
        let read = read_pinned_mxfp4_rank_expert(
            &mut source,
            read_request_for_rank(route, expert, rank, &hashes),
        )
        .unwrap();
        let peak =
            read.plan.selected_payload_bytes * 2 + crate::checkpoint::MAX_HEADER_BYTES as u64 + 8;
        read.into_session_resident(peak).unwrap()
    }

    #[test]
    fn route_rank_read_uses_only_one_expert_ranges_and_checks_each_payload_hash() {
        let route = route_problem();
        let (expected_plan, hashes, mut source) = fixture(TOP6[0]);
        let read =
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes))
                .unwrap();

        assert_eq!(read.stamp, route.stamp);
        assert_eq!(read.attempt, 29);
        assert_eq!(read.route_experts, TOP6);
        assert_eq!((read.expert, read.rank, read.local), (TOP6[0], 4, 0));
        assert_eq!(read.plan, expected_plan);
        assert_eq!(read.header_sha256, HEADER_SHA256);
        assert_eq!(read.payloads.file_bytes, SHARD_BYTES);
        assert_eq!(read.payloads.tensors.len(), 6);
        assert_eq!(
            read.total_io_bytes,
            read.header_validation_io_bytes * 2 + read.payloads.io_bytes
        );
        for tensor in &read.plan.tensors {
            assert_eq!(
                read.payloads.tensors[&tensor.name].sha256,
                hashes[&tensor.name]
            );
        }

        let mut expected = read
            .plan
            .tensors
            .iter()
            .map(|tensor| (tensor.file_offset_start, tensor.file_offset_end))
            .collect::<Vec<_>>();
        expected.sort_unstable();
        let mut actual = source
            .read_spans
            .iter()
            .copied()
            .filter(|(start, _)| *start >= read.plan.data_start_bytes)
            .collect::<Vec<_>>();
        actual.sort_unstable();
        assert_eq!(actual, expected, "unselected shard ranges were read");
    }

    #[test]
    fn one_rank_read_becomes_one_compact_resident_session_owner() {
        let route = route_problem();
        let (expected_plan, hashes, mut source) = fixture(TOP6[0]);
        let read =
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes))
                .unwrap();
        let payload_bytes = expected_plan.selected_payload_bytes;
        let peak_bound = payload_bytes * 2 + crate::checkpoint::MAX_HEADER_BYTES as u64 + 8;
        let expected_w1 = expected_plan
            .tensors
            .iter()
            .find(|tensor| tensor.name.ends_with(".w1.weight"))
            .unwrap()
            .shape
            .clone();
        let resident = read.into_session_resident(peak_bound).unwrap();

        assert_eq!(resident.stamp(), &route.stamp);
        assert_eq!(resident.attempt(), 29);
        assert_eq!(resident.route_experts(), TOP6);
        assert_eq!(
            (resident.expert(), resident.rank(), resident.local()),
            (TOP6[0], 4, 0)
        );
        assert_eq!(resident.source_header_sha256(), HEADER_SHA256);
        assert_eq!(resident.payload_sha256().len(), 6);
        assert_eq!(resident.packing_peak_bound_bytes(), peak_bound);
        assert!(resident.packed_artifact_bytes() < SHARD_BYTES);
        assert_eq!(
            resident.owner().binding().model_revision,
            REVISION.to_owned()
        );
        assert_eq!(resident.owner().binding().rank, 4);
        assert_eq!(resident.owner().binding().layer, 0);
        assert_eq!(resident.owner().binding().epoch, route.stamp.epoch);
        assert_eq!(
            resident.owner().binding().experts,
            vec![crate::session::mxfp4::Expert {
                expert: TOP6[0],
                local: 0,
            }]
        );
        assert_eq!(
            resident.owner().bytes().len() as u64,
            resident.packed_artifact_bytes()
        );
        let expert = resident.owner().expert(TOP6[0]).unwrap();
        assert_eq!(expert.namespace(), Mxfp4ExpertNamespace::Backbone);
        assert_eq!(expert.layer(), 0);
        assert_eq!(expert.expert(), TOP6[0]);
        assert_eq!(expert.w1().rows() as usize, expected_w1[0]);
        assert_eq!(expert.w1().cols() as usize, expected_w1[1] * 2);
    }

    #[test]
    fn session_owner_refuses_insufficient_peak_budget_or_changed_payloads() {
        let route = route_problem();
        let (_, hashes, mut source) = fixture(TOP6[0]);
        let read =
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes))
                .unwrap();
        let required =
            read.plan.selected_payload_bytes * 2 + crate::checkpoint::MAX_HEADER_BYTES as u64 + 8;
        assert!(read.into_session_resident(required - 1).is_err());

        let (_, hashes, mut source) = fixture(TOP6[0]);
        let mut read =
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes))
                .unwrap();
        let tensor = read.plan.tensors[0].name.clone();
        read.payloads.tensors.get_mut(&tensor).unwrap().bytes[0] ^= 1;
        assert!(read.into_session_resident(64 * 1024 * 1024).is_err());

        let (_, hashes, mut source) = fixture(TOP6[0]);
        let mut read =
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes))
                .unwrap();
        read.payloads.tensors.pop_first();
        assert!(read.into_session_resident(64 * 1024 * 1024).is_err());
    }

    #[test]
    fn route_preflight_requires_all_six_exact_coefficients_and_rank_bindings() {
        let route = distributed_route_problem();
        let residents = TOP6
            .iter()
            .enumerate()
            .map(|(index, &expert)| resident_for_route_expert(&route, expert, 4 + index as u32))
            .collect::<Vec<_>>();
        let total = residents
            .iter()
            .map(PinnedMxfp4RankResidentExpert::packed_artifact_bytes)
            .sum::<u64>();

        assert!(
            preflight_pinned_mxfp4_route_residents(&route, 29, &residents[..5], total).is_err()
        );
        assert!(preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total - 1).is_err());
        let mut changed_coefficients = route.clone();
        changed_coefficients.routes[0][2].weight =
            f64::from_bits(changed_coefficients.routes[0][2].weight.to_bits() + 1);
        assert!(preflight_pinned_mxfp4_route_residents(
            &changed_coefficients,
            29,
            &residents,
            total,
        )
        .is_err());
        let mut changed_binding = route.clone();
        changed_binding.registry.copies[0].rank = 10;
        changed_binding.resident_order.remove(&4);
        changed_binding.resident_order.insert(10, vec![0]);
        assert!(
            preflight_pinned_mxfp4_route_residents(&changed_binding, 29, &residents, total,)
                .is_err()
        );

        let preflight =
            preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total).unwrap();
        assert_eq!(preflight.stamp(), &route.stamp);
        assert_eq!(preflight.attempt(), 29);
        assert_eq!(preflight.route_experts(), TOP6);
        assert!(crate::sha256::is_hex_digest(preflight.route_fingerprint()));
        assert_eq!(preflight.total_artifact_bytes(), total);
        assert_eq!(
            preflight
                .owners()
                .iter()
                .map(|owner| (owner.expert, owner.rank, owner.local))
                .collect::<Vec<_>>(),
            TOP6.iter()
                .enumerate()
                .map(|(index, &expert)| (expert, 4 + index as u32, 0))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_payload_hash_or_nonroute_expert_refuses_before_weight_reads() {
        let route = route_problem();
        let (_, mut hashes, mut source) = fixture(TOP6[0]);
        hashes.pop_first();
        assert!(
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes),)
                .is_err()
        );
        assert!(source
            .read_spans
            .iter()
            .all(|(start, _)| { *start < 258_144 }));

        let (_, hashes, mut source) = fixture(TOP6[0]);
        assert!(
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, 17, &hashes),).is_err()
        );
        assert!(source.read_spans.is_empty());
    }

    #[test]
    fn digest_mismatch_returns_no_imported_partial_tensors() {
        let route = route_problem();
        let (_, mut hashes, mut source) = fixture(TOP6[0]);
        let first = hashes.keys().next().unwrap().clone();
        hashes.insert(first, "f".repeat(64));
        assert!(
            read_pinned_mxfp4_rank_expert(&mut source, read_request(&route, TOP6[0], &hashes),)
                .is_err()
        );
        assert!(source.read_spans.iter().any(|(start, _)| *start >= 258_144));
    }

    #[test]
    fn indexed_file_reader_returns_the_existing_route_resident_type() {
        let route = route_problem();
        let (expected_plan, hashes, _) = fixture(TOP6[0]);
        let index = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        let header = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
        );
        let temp = TempDirectory::new();
        let shard_path = temp.path().join("model-00003-of-00048.safetensors");
        let mut shard = File::create(&shard_path).unwrap();
        shard.set_len(SHARD_BYTES).unwrap();
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        shard.write_all(header).unwrap();
        for (slot, tensor) in expected_plan.tensors.iter().enumerate() {
            let length =
                usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
            let payload = vec![u8::try_from(slot + 1).unwrap(); length];
            shard
                .seek(SeekFrom::Start(tensor.file_offset_start))
                .unwrap();
            shard.write_all(&payload).unwrap();
        }
        drop(shard);

        let read = read_pinned_mxfp4_rank_expert_indexed(
            temp.path(),
            PinnedMxfp4IndexedRankExpertReadRequest {
                route: &route,
                attempt: 29,
                expert: TOP6[0],
                rank: 4,
                index_bytes: index,
                expected_header_sha256: HEADER_SHA256,
                expected_payload_sha256: &hashes,
                max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
                max_payload_bytes: crate::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES,
            },
        )
        .unwrap();
        assert_eq!(read.plan, expected_plan);
        assert_eq!(read.stamp, route.stamp);
        assert_eq!(read.route_experts, TOP6);
        assert_eq!((read.expert, read.rank, read.local), (TOP6[0], 4, 0));
        assert_eq!(read.header_sha256, HEADER_SHA256);
        assert_eq!(read.payloads.file_bytes, SHARD_BYTES);
        assert_eq!(read.payloads.tensors.len(), 6);

        let peak = expected_plan.selected_payload_bytes * 2
            + crate::checkpoint::MAX_HEADER_BYTES as u64
            + 8;
        let resident = read.into_session_resident(peak).unwrap();
        assert_eq!(resident.stamp(), &route.stamp);
        assert_eq!(resident.route_experts(), TOP6);
        assert_eq!(
            (resident.expert(), resident.rank(), resident.local()),
            (0, 4, 0)
        );
        assert_eq!(resident.source_header_sha256(), HEADER_SHA256);
        assert_eq!(resident.payload_sha256().len(), 6);
        assert_eq!(resident.owner().binding().experts[0].expert, TOP6[0]);
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
