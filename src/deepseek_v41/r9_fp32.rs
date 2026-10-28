//! Bounded CPU producer for the R9 laboratory's explicit binary32 warp order.
//! This is independent of the existing FP64 format oracle and is never a GPU fallback.
//! The 32 lane partials and shuffle tree mirror the preserved R9/A116 contract,
//! not an assertion about an upstream Tensor Core GEMM's reduction order.
use super::{
    cast_bf16_rne_reference, e4m3fn_positive, fp8_activation_quantize_reference,
    pre_down_activation, Fp8ActivationQuantized, Mxfp4Expert, Mxfp4Matrix,
    MAX_DEEPSEEK_EXPERT_REFERENCE_MACS, MAX_MXFP4_REFERENCE_MACS,
};
use crate::{invalid, Error, Result};

fn finite(x: f32) -> Result<f32> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "R9 binary32 projection has a nonfinite intermediate",
        ))
    }
}

fn scale(code: u8) -> Result<f32> {
    match code {
        // UE8M0 0 is 2^-127, a valid binary32 subnormal, not zero.
        0 => Ok(f32::from_bits(0x0040_0000)),
        255 => Err(invalid("UE8M0 NaN scale")),
        _ => Ok(f32::from_bits(u32::from(code) << 23)),
    }
}

/// Separate RN32 multiply then add for each K32 block in each lane, followed by
/// delta=16,8,4,2,1 shuffle additions. No FMA, FP64 sum, or dense weight plane.
pub fn project(matrix: &Mxfp4Matrix, input: &Fp8ActivationQuantized) -> Result<Vec<f32>> {
    let macs = matrix.rows.checked_mul(matrix.cols);
    if macs.is_none_or(|n| n > MAX_MXFP4_REFERENCE_MACS)
        || input.values.len() as u64 != matrix.cols
        || input.scale_codes.len() * 32 != input.values.len()
    {
        return Err(invalid("R9 projection shape or CPU work budget differs"));
    }
    let mut activations = Vec::with_capacity(input.values.len());
    for (index, &code) in input.values.iter().enumerate() {
        if code & 0x7f == 0x7f {
            return Err(invalid("E4M3FN NaN activation"));
        }
        let value = e4m3fn_positive(code & 0x7f);
        let value = if code & 0x80 == 0 { value } else { -value };
        activations.push(finite(value * scale(input.scale_codes[index / 32])?)?);
    }
    let (codes, scales) = matrix.compressed_planes();
    let mut result = Vec::with_capacity(matrix.rows as usize);
    for row in 0..matrix.rows {
        let mut partials = [0.0_f32; 32];
        for (lane, partial) in partials.iter_mut().enumerate() {
            for block in 0..matrix.cols / 32 {
                let col = block * 32 + lane as u64;
                let byte = codes[(row * (matrix.cols / 2) + col / 2) as usize];
                let nibble = if col.is_multiple_of(2) {
                    byte & 15
                } else {
                    byte >> 4
                };
                let code = scales[(row * (matrix.cols / 32) + col / 32) as usize];
                let weight = finite(super::decode_fp4(nibble)? as f32 * scale(code)?)?;
                let term = finite(weight * activations[col as usize])?;
                *partial = finite(*partial + term)?;
            }
        }
        for delta in [16, 8, 4, 2, 1] {
            let before = partials;
            for lane in 0..32 - delta {
                partials[lane] = finite(before[lane] + before[lane + delta])?;
            }
        }
        result.push(partials[0]);
    }
    Ok(result)
}

/// The original coefficient is applied before BF16 and the second activation pack.
/// Input bits must already represent BF16 exactly. The compressed owner is borrowed.
pub fn forward(expert: &Mxfp4Expert, input: &[f32], weight: f32) -> Result<Vec<f32>> {
    let macs = expert
        .w1
        .rows
        .checked_mul(expert.w1.cols)
        .and_then(|n| n.checked_mul(3));
    if macs.is_none_or(|n| n > MAX_DEEPSEEK_EXPERT_REFERENCE_MACS)
        || input.len() as u64 != expert.w1.cols
    {
        return Err(invalid("R9 expert shape or CPU work budget differs"));
    }
    if !weight.is_finite() || weight < 0.0 {
        return Err(invalid(
            "R9 routing coefficient must be finite and nonnegative",
        ));
    }
    for &value in input {
        if cast_bf16_rne_reference(value)?.to_bits() != value.to_bits() {
            return Err(invalid("R9 input must already be exactly BF16"));
        }
    }
    let quantized = fp8_activation_quantize_reference(input)?;
    let gate = project(&expert.w1, &quantized)?;
    let up = project(&expert.w3, &quantized)?;
    let down_input = pre_down_activation(&gate, &up, Some(weight))?
        .into_iter()
        .map(cast_bf16_rne_reference)
        .collect::<Result<Vec<_>>>()?;
    project(&expert.w2, &fp8_activation_quantize_reference(&down_input)?)
}
