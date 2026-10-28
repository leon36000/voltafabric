//! Characterization and CPU oracle tests for DeepSeek's pre-FP4 GEMM activation quantizer.
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};
use voltafabric::deepseek_v41::fp8_activation_quantize_reference;

fn invoke(input: &Value) -> (i32, Value) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["deepseek-v41-ops", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    (
        output.status.code().unwrap(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

#[test]
fn fp8_activation_reference_uses_ue8m0_group_scales_and_e4m3fn_bytes() {
    let mut input = vec![0.0f32; 64];
    input[0] = 448.0;
    input[1] = -448.0;
    input[2] = 0.5;
    input[3] = 1.0;
    input[4] = -0.0;
    input[32] = 0.5;
    input[33] = -0.0;
    let input_bits = input
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let (exit_code, output) = invoke(&json!({
        "op": "fp8-act-quant-reference",
        "input_bits": input_bits,
    }));

    assert_eq!(exit_code, 0, "{output}");
    let result = &output["result"];
    assert_eq!(result["op"], "fp8-act-quant-reference");
    assert_eq!(result["scale_codes"], json!([127, 118]));
    assert_eq!(result["e4m3_bytes"][0], 126);
    assert_eq!(result["e4m3_bytes"][1], 254);
    assert_eq!(result["e4m3_bytes"][2], 48);
    assert_eq!(result["e4m3_bytes"][3], 56);
    assert_eq!(result["e4m3_bytes"][4], 128);
    assert_eq!(result["e4m3_bytes"][32], 120);
    assert_eq!(result["e4m3_bytes"][33], 128);
    assert_eq!(result["capability"]["gpu_executed"], false);
    assert_eq!(result["capability"]["cuda_executed"], false);
}

#[test]
fn zero_floor_signed_zero_and_extreme_finite_inputs_are_preserved() {
    let mut input = vec![0.0f32; 96];
    input[1] = -0.0;
    input[32] = f32::MAX;
    input[33] = -f32::MAX;
    input[64] = f32::from_bits(1); // least positive FP32 subnormal
    input[65] = -f32::from_bits(1);
    let quantized = fp8_activation_quantize_reference(&input).unwrap();
    assert_eq!(quantized.scale_codes(), &[105, 247, 105]);
    assert_eq!(quantized.values()[1], 0x80);
    assert_eq!(quantized.values()[32], 0x78); // 256 after dividing by 2^120
    assert_eq!(quantized.values()[33], 0xf8);
    assert_eq!(quantized.values()[64], 0x00);
    assert_eq!(quantized.values()[65], 0x80);
}

#[test]
fn rejects_empty_misaligned_oversized_and_nonfinite_inputs() {
    let empty = fp8_activation_quantize_reference(&[]).unwrap_err();
    assert_eq!(empty.code, "INVALID_INPUT");
    let misaligned = fp8_activation_quantize_reference(&[0.0; 31]).unwrap_err();
    assert_eq!(misaligned.code, "INVALID_INPUT");
    let oversized = vec![0.0; (1 << 20) + 32];
    let oversized = fp8_activation_quantize_reference(&oversized).unwrap_err();
    assert_eq!(oversized.code, "INVALID_INPUT");

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = [0.0; 32];
        input[17] = invalid;
        let error = fp8_activation_quantize_reference(&input).unwrap_err();
        assert_eq!(error.code, "NONFINITE_INPUT");
    }
}
