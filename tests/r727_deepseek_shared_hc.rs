use serde_json::Value;
use voltafabric::deepseek_v41::{
    cast_bf16_rne_reference, deepseek_hc_pre_reference, fp8_activation_quantize_reference,
    DeepSeekV41SharedExpert, Fp8E4m3Matrix, MAX_FP8_COMPRESSED_BYTES, PINNED_HC_MULT, SWIGLU_LIMIT,
};

const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 32;

fn exact_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fffu32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn decode_e4m3fn(code: u8) -> f64 {
    let magnitude = code & 0x7f;
    assert_ne!(magnitude, 0x7f);
    let exponent = (magnitude >> 3) & 15;
    let mantissa = magnitude & 7;
    let positive = if exponent == 0 {
        f64::from(mantissa) * 2.0_f64.powi(-9)
    } else {
        f64::from(8 + mantissa) * 2.0_f64.powi(i32::from(exponent) - 10)
    };
    if code & 0x80 == 0 {
        positive
    } else {
        -positive
    }
}

fn decode_e8m0(code: u8) -> f64 {
    assert_ne!(code, 255);
    2.0_f64.powi(i32::from(code) - 127)
}

fn scalar_matvec(
    rows: usize,
    cols: usize,
    weights: &[u8],
    weight_scales: &[u8],
    activation: &voltafabric::deepseek_v41::Fp8ActivationQuantized,
) -> Vec<f32> {
    let blocks = cols / 32;
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(weight_scales.len(), rows * blocks);
    let mut outputs = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut total = 0.0f64;
        for block in 0..blocks {
            let mut dot = 0.0f64;
            for offset in 0..32 {
                let k = block * 32 + offset;
                dot +=
                    decode_e4m3fn(activation.values()[k]) * decode_e4m3fn(weights[row * cols + k]);
            }
            total += dot
                * decode_e8m0(activation.scale_codes()[block])
                * decode_e8m0(weight_scales[row * blocks + block]);
        }
        outputs.push(total as f32);
    }
    outputs
}

fn scalar_shared_expert(
    input: &[f32],
    w1: &[u8],
    w1_scales: &[u8],
    w2: &[u8],
    w2_scales: &[u8],
    w3: &[u8],
    w3_scales: &[u8],
) -> Vec<f32> {
    let q_input = fp8_activation_quantize_reference(input).unwrap();
    let mut gate = scalar_matvec(INTERMEDIATE, HIDDEN, w1, w1_scales, &q_input);
    let mut up = scalar_matvec(INTERMEDIATE, HIDDEN, w3, w3_scales, &q_input);
    let mut down_input = Vec::with_capacity(INTERMEDIATE);
    for (gate, up) in gate.iter_mut().zip(up.iter_mut()) {
        *gate = gate.min(SWIGLU_LIMIT);
        *up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let gate64 = f64::from(*gate);
        let silu64 = gate64 / (1.0 + (-gate64).exp());
        down_input.push(exact_bf16((silu64 * f64::from(*up)) as f32));
    }
    let q_down = fp8_activation_quantize_reference(&down_input).unwrap();
    scalar_matvec(HIDDEN, INTERMEDIATE, w2, w2_scales, &q_down)
}

#[test]
fn pinned_shared_expert_format_is_fp8_e4m3_e8m0_and_co_located_by_layer() {
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let header: Value = serde_json::from_slice(header).unwrap();
    for projection in ["w1", "w3"] {
        let prefix = format!("layers.0.ffn.shared_experts.{projection}");
        assert_eq!(header[format!("{prefix}.weight")]["dtype"], "F8_E4M3");
        assert_eq!(
            header[format!("{prefix}.weight")]["shape"],
            serde_json::json!([2304, 5120])
        );
        assert_eq!(header[format!("{prefix}.scale")]["dtype"], "F8_E8M0");
        assert_eq!(
            header[format!("{prefix}.scale")]["shape"],
            serde_json::json!([72, 160])
        );
    }
    assert_eq!(
        header["layers.0.ffn.shared_experts.w2.weight"]["dtype"],
        "F8_E4M3"
    );
    assert_eq!(
        header["layers.0.ffn.shared_experts.w2.weight"]["shape"],
        serde_json::json!([5120, 2304])
    );
    assert_eq!(
        header["layers.0.ffn.shared_experts.w2.scale"]["dtype"],
        "F8_E8M0"
    );
    assert_eq!(
        header["layers.0.ffn.shared_experts.w2.scale"]["shape"],
        serde_json::json!([160, 72])
    );

    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    assert_eq!(
        voltafabric::sha256::hex(index),
        "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
    );
    let index: Value = serde_json::from_slice(index).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    for layer in 0..40 {
        let mut shard = None;
        for name in [
            format!("layers.{layer}.ffn.gate.weight"),
            format!("layers.{layer}.ffn_norm.weight"),
            format!("layers.{layer}.ffn.shared_experts.w1.weight"),
            format!("layers.{layer}.ffn.shared_experts.w1.scale"),
            format!("layers.{layer}.ffn.shared_experts.w2.weight"),
            format!("layers.{layer}.ffn.shared_experts.w2.scale"),
            format!("layers.{layer}.ffn.shared_experts.w3.weight"),
            format!("layers.{layer}.ffn.shared_experts.w3.scale"),
        ] {
            let actual = weight_map[&name].as_str().unwrap();
            if let Some(expected) = shard {
                assert_eq!(actual, expected, "layer {layer}: {name}");
            } else {
                shard = Some(actual);
            }
        }
    }
}

#[test]
fn hyperconnection_pre_matches_independent_fp64_reference_and_returns_bf16() {
    let hidden = HIDDEN;
    let residual: Vec<f32> = (0..PINNED_HC_MULT)
        .flat_map(|copy| {
            (0..hidden).map(move |column| {
                exact_bf16(((copy as i32 - 2) * 8 + (column % 17) as i32 - 8) as f32 * 0.125)
            })
        })
        .collect();
    let mix = [0.125f32, 0.25, 0.25, 0.375];
    let actual = deepseek_hc_pre_reference(&residual, &mix, hidden).unwrap();
    let mut expected = Vec::with_capacity(hidden);
    for column in 0..hidden {
        let sum = (0..PINNED_HC_MULT)
            .map(|copy| f64::from(mix[copy]) * f64::from(residual[copy * hidden + column]))
            .sum::<f64>();
        expected.push(exact_bf16(sum as f32));
    }
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(actual
        .iter()
        .all(|value| cast_bf16_rne_reference(*value).unwrap().to_bits() == value.to_bits()));
}

#[test]
fn shared_fp8_expert_runs_three_distinct_e4m3_projections_without_route_weight() {
    let w1_codes = vec![0x38; INTERMEDIATE * HIDDEN];
    let w1_scales = vec![127; INTERMEDIATE * (HIDDEN / 32)];
    let w3_codes = vec![0x38; INTERMEDIATE * HIDDEN];
    let w3_scales = vec![127; INTERMEDIATE * (HIDDEN / 32)];
    let w2_codes: Vec<u8> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| {
            if (index / INTERMEDIATE).is_multiple_of(2) {
                0x38
            } else {
                0xb8
            }
        })
        .collect();
    let w2_scales = vec![127; HIDDEN * (INTERMEDIATE / 32)];
    let expert = DeepSeekV41SharedExpert::new(
        Fp8E4m3Matrix::new(
            INTERMEDIATE as u64,
            HIDDEN as u64,
            w1_codes.clone(),
            w1_scales.clone(),
        )
        .unwrap(),
        Fp8E4m3Matrix::new(
            HIDDEN as u64,
            INTERMEDIATE as u64,
            w2_codes.clone(),
            w2_scales.clone(),
        )
        .unwrap(),
        Fp8E4m3Matrix::new(
            INTERMEDIATE as u64,
            HIDDEN as u64,
            w3_codes.clone(),
            w3_scales.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let input = vec![1.0f32; HIDDEN];
    let actual = expert.forward_reference(&input).unwrap();
    let expected = scalar_shared_expert(
        &input, &w1_codes, &w1_scales, &w2_codes, &w2_scales, &w3_codes, &w3_scales,
    );
    assert_eq!(actual.len(), HIDDEN);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_eq!(
        actual, expected,
        "independent blockwise FP64 shared-expert oracle"
    );
    assert!(actual.iter().any(|value| *value > 0.0));
    assert!(actual.iter().any(|value| *value < 0.0));
    assert!(MAX_FP8_COMPRESSED_BYTES >= (w1_codes.len() + w1_scales.len()) as u64);
}

#[test]
fn fp8_planes_and_hc_inputs_fail_closed_on_invalid_formats() {
    assert!(Fp8E4m3Matrix::new(1, 31, vec![0x38; 31], vec![127]).is_err());
    assert!(Fp8E4m3Matrix::new(1, 32, vec![0x38; 32], vec![255]).is_err());
    let mut nan_codes = vec![0x38; 32];
    nan_codes[7] = 0x7f;
    assert!(Fp8E4m3Matrix::new(1, 32, nan_codes, vec![127]).is_err());
    let bad_mix = [0.25f32, 0.25, f32::NAN, 0.25];
    assert!(deepseek_hc_pre_reference(&vec![1.0; 4 * 32], &bad_mix, 32).is_err());
    assert!(deepseek_hc_pre_reference(&vec![0.1; 4 * 32], &[0.25; 4], 32).is_err());
}
