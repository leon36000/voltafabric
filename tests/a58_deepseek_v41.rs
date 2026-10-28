//! A58: DeepSeek-V4.1-Flash native MXFP4 matrix owner and pre-down clamped SwiGLU.
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};
use voltafabric::deepseek_v41::{
    decode_e8m0, mxfp4_plane_bytes, pre_down_activation, Mxfp4Matrix, SWIGLU_LIMIT,
};

/// Independent E2M1 from its bit fields (sign | 2-bit exponent bias 1 | 1-bit mantissa).
fn e2m1_bits(n: u8) -> f64 {
    let (s, e, m) = ((n >> 3) & 1, (n >> 1) & 3, n & 1);
    let mag = if e == 0 {
        f64::from(m) * 0.5
    } else {
        (1.0 + f64::from(m) * 0.5) * 2.0_f64.powi(i32::from(e) - 1)
    };
    if s == 1 {
        -mag
    } else {
        mag
    }
}

fn vf(input: &Value) -> (i32, Value) {
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
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap(),
        serde_json::from_slice(&out.stdout).unwrap(),
    )
}

fn err_code(m: std::result::Result<Mxfp4Matrix, voltafabric::Error>) -> String {
    m.unwrap_err().code
}

#[test]
fn all_16_nibbles_times_all_255_valid_scales_match_bit_expectation() {
    // Row r uses scale code r; columns c cycle through nibbles c % 16.
    let rows = 255u64;
    let codes: Vec<u8> = (0..rows)
        .flat_map(|_| (0..16u8).map(|j| ((2 * j) % 16) | (((2 * j + 1) % 16) << 4)))
        .collect();
    let scales: Vec<u8> = (0..=254u8).collect();
    let m = Mxfp4Matrix::new(rows, 32, codes, scales).unwrap();
    for r in 0..rows {
        for c in 0..32 {
            let want = e2m1_bits((c % 16) as u8) * 2.0_f64.powi(r as i32 - 127);
            let got = m.value(r, c).unwrap();
            assert_eq!(got.to_bits(), want.to_bits(), "row {r} col {c}");
        }
    }
    // Code 0 at scale 2^-127 is a valid +0; nibble 8 is -0 at every scale.
    assert_eq!(m.value(0, 0).unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(m.value(0, 8).unwrap().to_bits(), (-0.0f64).to_bits());
    assert_eq!(m.value(0, 1).unwrap(), 2.0_f64.powi(-128));
    assert_eq!(m.value(254, 7).unwrap(), 6.0 * 2.0_f64.powi(127));
}

#[test]
fn pinned_official_config_matches_mxfp4_contract_and_projection_work_bound() {
    let config: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/deepseek_v41_config_314113f8.json"
    ))
    .unwrap();
    assert_eq!(config["model_type"], "deepseek_v41");
    assert_eq!(config["quantization_config"]["quant_method"], "fp8");
    assert_eq!(config["quantization_config"]["expert_dtype"], "fp4");
    assert_eq!(config["quantization_config"]["scale_fmt"], "ue8m0");
    assert_eq!(
        config["quantization_config"]["weight_block_size"],
        json!([32, 32])
    );

    let hidden = config["text_config"]["hidden_size"].as_u64().unwrap();
    let intermediate = config["text_config"]["moe_intermediate_size"]
        .as_u64()
        .unwrap();
    assert_eq!(hidden, 5120);
    assert_eq!(intermediate, 2304);
    assert_eq!(config["text_config"]["n_routed_experts"], 384);
    assert_eq!(config["text_config"]["num_experts_per_tok"], 6);

    // Official w1/w3 and w2 projection shapes both fit the bounded CPU reference.
    let projection_macs = hidden * intermediate;
    assert_eq!(projection_macs, 11_796_480);
    assert!(projection_macs <= voltafabric::deepseek_v41::MAX_MXFP4_REFERENCE_MACS);
    assert_eq!(
        mxfp4_plane_bytes(intermediate, hidden).unwrap(),
        (projection_macs / 2, projection_macs / 32)
    );
    assert_eq!(
        mxfp4_plane_bytes(hidden, intermediate).unwrap(),
        (projection_macs / 2, projection_macs / 32)
    );
}

#[test]
fn pinned_real_layer0_expert0_runs_from_compressed_safetensors_planes() {
    let artifact = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let golden: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/deepseek_v41_layer0_expert0_w1_fraction_samples.json"
    ))
    .unwrap();
    assert_eq!(
        golden["artifact_sha256"],
        "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b"
    );
    let safe = safetensors::SafeTensors::deserialize(artifact).unwrap();
    let specs = [
        (
            1u8,
            2304usize,
            5120usize,
            [2304usize, 2560usize],
            [2304usize, 160usize],
        ),
        (
            2u8,
            5120usize,
            2304usize,
            [5120usize, 1152usize],
            [5120usize, 72usize],
        ),
        (
            3u8,
            2304usize,
            5120usize,
            [2304usize, 2560usize],
            [2304usize, 160usize],
        ),
    ];

    for (projection, rows, cols, code_shape, scale_shape) in specs {
        let prefix = format!("layers.0.ffn.experts.0.w{projection}");
        let weight = safe.tensor(&format!("{prefix}.weight")).unwrap();
        let scale = safe.tensor(&format!("{prefix}.scale")).unwrap();
        assert_eq!(format!("{:?}", weight.dtype()), "I8");
        assert_eq!(format!("{:?}", scale.dtype()), "F8_E8M0");
        assert_eq!(weight.shape(), code_shape);
        assert_eq!(scale.shape(), scale_shape);
        assert_eq!(
            mxfp4_plane_bytes(rows as u64, cols as u64).unwrap(),
            (weight.data().len() as u64, scale.data().len() as u64)
        );

        let matrix = Mxfp4Matrix::new(
            rows as u64,
            cols as u64,
            weight.data().to_vec(),
            scale.data().to_vec(),
        )
        .unwrap();

        if projection == 1 {
            let input: Vec<f64> = (0..cols)
                .map(|col| match col % 7 {
                    0 => 0.25,
                    1 => -0.5,
                    2 => 1.0,
                    3 => -2.0,
                    4 => 0.0,
                    5 => 0.125,
                    _ => -1.0,
                })
                .collect();
            let output = matrix.matvec_reference(&input).unwrap();
            assert_eq!(output.len(), rows);
            assert_eq!(golden["rows"], json!([0, 1, 17, 1024, 2047, 2303]));

            // Independent E2M1 bit-field decode over the checkpoint bytes. Sample
            // multiple rows, including both edges; the operator covers all 2304 rows.
            for row in [0usize, 1, 17, 1024, 2047, 2303] {
                let mut expected = 0.0f64;
                for (col, &input_value) in input.iter().enumerate() {
                    let packed = weight.data()[row * (cols / 2) + col / 2];
                    let nibble = if col % 2 == 0 {
                        packed & 15
                    } else {
                        packed >> 4
                    };
                    let scale_code = scale.data()[row * (cols / 32) + col / 32];
                    expected +=
                        e2m1_bits(nibble) * 2.0f64.powi(i32::from(scale_code) - 127) * input_value;
                }
                assert_eq!(output[row].to_bits(), expected.to_bits(), "row {row}");
                assert_eq!(
                    format!("{:016x}", output[row].to_bits()),
                    golden["output_f64_bits"][row.to_string()].as_str().unwrap(),
                    "Python Fraction oracle row {row}"
                );
            }
        }
    }
}

#[test]
fn compressed_mxfp4_matvec_preserves_row_order_and_each_32_lane_scale() {
    let rows = 2u64;
    let cols = 64u64;
    let codes: Vec<u8> = (0..rows * cols / 2)
        .map(|i| {
            let lo = ((i * 3 + 1) % 16) as u8;
            let hi = ((i * 7 + 2) % 16) as u8;
            lo | (hi << 4)
        })
        .collect();
    let scales = vec![0u8, 128, 126, 254];
    let input: Vec<f64> = (0..cols)
        .map(|i| match i % 5 {
            0 => 0.5,
            1 => -2.0,
            2 => 3.0,
            3 => -0.25,
            _ => 1.0,
        })
        .collect();

    // Independent bit-field oracle over the original packed buffers. All terms are dyadic;
    // the fixture is deliberately small enough that each expected f64 step is exact.
    let mut expected = vec![0.0f64; rows as usize];
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            let packed = codes[row * (cols as usize / 2) + col / 2];
            let nibble = if col % 2 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            let scale = scales[row * (cols as usize / 32) + col / 32];
            expected[row] += e2m1_bits(nibble) * 2.0f64.powi(i32::from(scale) - 127) * input[col];
        }
    }

    let payload = json!({
        "op":"mxfp4-matvec",
        "rows":rows,
        "cols":cols,
        "codes":codes,
        "scales":scales,
        "input_bits":input.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
    });
    let (code, out) = vf(&payload);
    assert_eq!(code, 0, "{out}");
    let result = &out["result"];
    assert_eq!(result["capability"]["scope"], "CPU_FORMAT_OPERATOR_ONLY");
    assert_eq!(result["capability"]["gpu_executed"], false);
    assert_eq!(result["capability"]["full_expert_forward"], false);
    let got: Vec<String> = serde_json::from_value(result["output_f64_bits"].clone()).unwrap();
    assert_eq!(
        got,
        expected
            .iter()
            .map(|x| format!("{:016x}", x.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn compressed_mxfp4_matvec_preserves_binary64_subnormal_results_and_underflow() {
    let request = json!({
        "op":"mxfp4-matvec",
        "rows":1,
        "cols":32,
        "codes":vec![0x11u8; 16], // +0.5 in every lane
        "scales":[0],            // 2^-127, hence each weight is 2^-128
        "input_bits":vec![2.0f64.powi(-900).to_bits(); 32],
    });
    let (code, out) = vf(&request);
    assert_eq!(code, 0, "{out}");
    assert_eq!(
        out["result"]["output_f64_bits"][0],
        format!("{:016x}", 2.0f64.powi(-1023).to_bits())
    );

    let underflow = json!({
        "op":"mxfp4-matvec",
        "rows":1,
        "cols":32,
        "codes":vec![0x11u8; 16],
        "scales":[0],
        "input_bits":vec![1u64; 32], // least positive binary64 subnormal
    });
    let (code, out) = vf(&underflow);
    assert_eq!(code, 0, "{out}");
    assert_eq!(out["result"]["output_f64_bits"][0], "0000000000000000");
}

#[test]
fn compressed_mxfp4_matvec_refuses_bad_input_and_overflow() {
    let base = json!({
        "op":"mxfp4-matvec",
        "rows":1,
        "cols":32,
        "codes":vec![0x11u8; 16],
        "scales":[127],
        "input_bits":vec![1.0f64.to_bits(); 32],
    });
    for (payload, expected) in [
        (json!({"input_bits": []}), "INVALID_INPUT"),
        (
            json!({"input_bits":vec![f64::NAN.to_bits(); 32]}),
            "NONFINITE_INPUT",
        ),
        (
            json!({"input_bits":vec![f64::INFINITY.to_bits(); 32]}),
            "NONFINITE_INPUT",
        ),
        (
            json!({"input_bits":vec![f64::NEG_INFINITY.to_bits(); 32]}),
            "NONFINITE_INPUT",
        ),
    ] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(payload.as_object().unwrap().clone());
        let (code, out) = vf(&request);
        assert_eq!(code, 2, "{out}");
        assert_eq!(out["error"]["code"], expected, "{out}");
    }

    let overflow = json!({
        "op":"mxfp4-matvec",
        "rows":1,
        "cols":32,
        "codes":vec![0x11u8; 16],
        "scales":[254],
        "input_bits":vec![f64::MAX.to_bits(); 32],
    });
    let (code, out) = vf(&overflow);
    assert_eq!(code, 2, "{out}");
    assert_eq!(out["error"]["code"], "NUMERICAL_OVERFLOW");

    // Each product is finite; the ordered sum overflows on its second term.
    let accumulation_overflow = json!({
        "op":"mxfp4-matvec",
        "rows":1,
        "cols":32,
        "codes":vec![0x22u8; 16],
        "scales":[127],
        "input_bits":vec![f64::MAX.to_bits(); 32],
    });
    let (code, out) = vf(&accumulation_overflow);
    assert_eq!(code, 2, "{out}");
    assert_eq!(out["error"]["code"], "NUMERICAL_OVERFLOW");
}

#[test]
fn reference_matvec_rejects_work_above_its_bound_before_computation() {
    // 4097 * 4096 exceeds the 16,777,216-MAC reference budget by one row.
    // The compressed planes stay under the existing 64 MiB artifact limit.
    let rows = 4097u64;
    let cols = 4096u64;
    let codes = vec![0u8; (rows * cols / 2) as usize];
    let scales = vec![127u8; (rows * cols / 32) as usize];
    let matrix = Mxfp4Matrix::new(rows, cols, codes, scales).unwrap();
    let error = matrix
        .matvec_reference(&vec![1.0; cols as usize])
        .unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    assert!(error.message.contains("reference work limit"), "{error}");
}

#[test]
fn e8m0_is_unsigned_power_of_two_and_255_is_nan() {
    for b in 0..=254u8 {
        assert_eq!(decode_e8m0(b).unwrap(), 2.0_f64.powi(i32::from(b) - 127));
    }
    assert_eq!(decode_e8m0(127).unwrap(), 1.0);
    assert_eq!(decode_e8m0(255).unwrap_err().code, "MXFP4_SCALE_NAN");
    // Not the E4M3 decoder: E4M3 0x38 is 1.0, E8M0 0x38 is 2^-71.
    assert_eq!(decode_e8m0(0x38).unwrap(), 2.0_f64.powi(-71));
}

#[test]
fn low_nibble_is_first_k_and_groups_rows_are_separate() {
    // Row 0: bytes 0x10 0x32 ... ; group 1 (cols 32..63) scale 2^1; row 1 scale 2^-1.
    let mut codes = vec![0u8; 64];
    codes[0] = 0x10;
    codes[1] = 0x32;
    codes[16] = 0x7f; // col 32 = nibble 15 (-6), col 33 = nibble 7 (6)
    codes[32] = 0x21; // row 1 col 0 = nibble 1, col 1 = nibble 2
    let m = Mxfp4Matrix::new(2, 64, codes, vec![127, 128, 126, 127]).unwrap();
    assert_eq!(m.value(0, 0).unwrap(), 0.0);
    assert_eq!(m.value(0, 1).unwrap(), 0.5);
    assert_eq!(m.value(0, 2).unwrap(), 1.0);
    assert_eq!(m.value(0, 3).unwrap(), 1.5);
    assert_eq!(m.value(0, 31).unwrap(), 0.0);
    assert_eq!(m.value(0, 32).unwrap(), -12.0);
    assert_eq!(m.value(0, 33).unwrap(), 12.0);
    assert_eq!(m.value(1, 0).unwrap(), 0.25);
    assert_eq!(m.value(1, 1).unwrap(), 0.5);
    let s = m.summary();
    assert_eq!(
        (
            s.code_bytes,
            s.scale_bytes,
            s.compressed_bytes,
            s.logical_elements
        ),
        (64, 4, 68, 128)
    );
}

#[test]
fn scale_255_rejected_even_when_codes_are_zero() {
    assert_eq!(
        err_code(Mxfp4Matrix::new(1, 32, vec![0; 16], vec![255])),
        "MXFP4_SCALE_NAN"
    );
    assert_eq!(
        err_code(Mxfp4Matrix::new(2, 64, vec![0; 64], vec![1, 2, 3, 255])),
        "MXFP4_SCALE_NAN"
    );
}

#[test]
fn reordered_short_overlong_planes_and_shapes_are_refused() {
    let ok = || (vec![0u8; 32], vec![127u8; 2]);
    let (c, s) = ok();
    assert!(Mxfp4Matrix::new(2, 32, c, s).is_ok());
    // Planes swapped, short and overlong by one byte.
    let (c, s) = ok();
    assert_eq!(err_code(Mxfp4Matrix::new(2, 32, s, c)), "INVALID_INPUT");
    for (dc, ds) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
        let c = vec![0u8; (32 + dc) as usize];
        let s = vec![127u8; (2 + ds) as usize];
        assert!(Mxfp4Matrix::new(2, 32, c, s).is_err(), "{dc} {ds}");
    }
    // Transposed shape with the same byte count, padding, zero dims.
    assert!(Mxfp4Matrix::new(32, 2, vec![0; 32], vec![127; 2]).is_err());
    assert!(Mxfp4Matrix::new(1, 48, vec![0; 24], vec![127; 2]).is_err());
    assert!(Mxfp4Matrix::new(0, 32, vec![], vec![]).is_err());
    assert!(Mxfp4Matrix::new(1, 0, vec![], vec![]).is_err());
}

#[test]
fn size_and_overflow_guards_fire_before_length_checks() {
    for (r, c) in [
        (u64::MAX, 32),
        (1, u64::MAX - 31),
        (65_537, 32),
        (1, 65_568),
        (65_536, 65_536),
    ] {
        let e = mxfp4_plane_bytes(r, c).unwrap_err();
        assert_eq!(e.code, "INVALID_INPUT");
        assert!(!e.message.contains("length mismatch"), "{r} {c} {e}");
        assert!(!err_code(Mxfp4Matrix::new(r, c, vec![], vec![])).is_empty());
    }
    // A real V4.1 expert w1/w3 [2304, 5120] and w2 [5120, 2304] fit the bound.
    assert_eq!(
        mxfp4_plane_bytes(2304, 5120).unwrap(),
        (2304 * 2560, 2304 * 160)
    );
    assert_eq!(
        mxfp4_plane_bytes(5120, 2304).unwrap(),
        (5120 * 1152, 5120 * 72)
    );
}

#[test]
fn coordinates_are_bounded() {
    let m = Mxfp4Matrix::new(1, 32, vec![0x10; 16], vec![127]).unwrap();
    assert!(m.value(1, 0).is_err());
    assert!(m.value(0, 32).is_err());
    assert!(m.value(u64::MAX, u64::MAX).is_err());
}

#[test]
fn ownership_snapshot_is_independent_of_caller_copies() {
    let mut codes = vec![0x21u8; 16];
    let mut scales = vec![127u8];
    let m = Mxfp4Matrix::new(1, 32, codes.clone(), scales.clone()).unwrap();
    codes.fill(0xff);
    scales[0] = 200;
    assert_eq!(m.value(0, 0).unwrap(), 0.5);
    assert_eq!(m.value(0, 1).unwrap(), 1.0);
}

fn act(g: f32, u: f32, w: Option<f32>) -> f32 {
    pre_down_activation(&[g], &[u], w).unwrap()[0]
}

fn silu64(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

fn close(got: f32, want: f64) -> bool {
    let w = want as f32;
    let ulp = (got.to_bits() as i64 - w.to_bits() as i64).abs();
    (got.is_sign_negative() == w.is_sign_negative() && ulp <= 8)
        || (f64::from(got) - want).abs() <= 1e-7
        || (f64::from(got) - want).abs() <= 1e-6 * want.abs()
}

#[test]
fn gate_is_clamped_only_from_above_and_up_on_both_sides() {
    // gate = -20 discriminates a symmetric clamp (silu(-10) ~ -4.5e-4 vs ~ -4.1e-8).
    let x = act(-20.0, 1.0, None);
    assert!(close(x, silu64(-20.0)), "{x}");
    assert!(x > -1e-7 && x < 0.0);
    assert!(close(act(20.0, 1.0, None), silu64(10.0)));
    assert!(close(act(1.0, 20.0, None), silu64(1.0) * 10.0));
    assert!(close(act(1.0, -20.0, None), silu64(1.0) * -10.0));
    assert!(close(act(-1.0, -3.0, None), silu64(-1.0) * -3.0));
    // Boundaries and next f32 neighbours.
    let above = f32::from_bits(SWIGLU_LIMIT.to_bits() + 1);
    let below = f32::from_bits(SWIGLU_LIMIT.to_bits() - 1);
    assert_eq!(act(above, 1.0, None), act(10.0, 1.0, None));
    assert_ne!(act(below, 1.0, None), act(10.0, 1.0, None));
    assert_eq!(act(1.0, above, None), act(1.0, 10.0, None));
    assert_eq!(act(1.0, -above, None), act(1.0, -10.0, None));
    assert!(close(act(1.0, below, None), silu64(1.0) * f64::from(below)));
    assert!(close(
        act(1.0, -below, None),
        -silu64(1.0) * f64::from(below)
    ));
    let gneg = f32::from_bits((-10.0f32).to_bits() + 1); // -10.000001, not clamped
    assert!(close(act(gneg, 1.0, None), silu64(f64::from(gneg))));
}

#[test]
fn signed_zeros_and_extremes() {
    assert_eq!(act(0.0, 5.0, None).to_bits(), 0.0f32.to_bits());
    assert_eq!(act(0.0, -5.0, None).to_bits(), (-0.0f32).to_bits());
    assert_eq!(act(-0.0, 5.0, None).to_bits(), (-0.0f32).to_bits());
    assert_eq!(act(-5.0, 0.0, None).to_bits(), (-0.0f32).to_bits());
    assert_eq!(act(5.0, -0.0, None).to_bits(), (-0.0f32).to_bits());
    // Stable extreme negative silu: tiny negative or -0, never NaN.
    for g in [f32::MIN, -1e30, -200.0, -104.0, -88.0] {
        let x = act(g, 1.0, None);
        assert!(
            x.is_finite() && x.is_sign_negative() && x > -1e-30,
            "{g} {x}"
        );
    }
    assert!(close(act(f32::MAX, f32::MAX, None), silu64(10.0) * 10.0));
}

#[test]
fn native_fp32_silu_underflows_to_exact_negative_zero() {
    // Torch 2.10 CPU F.silu: exp(-g) overflows in FP32, so these are exactly -0. A
    // sign-split f64 logistic instead returns tiny negative subnormals/normals.
    for g in [-105.0, -100.0, -95.0, -90.0, -89.0, -88.75] {
        assert_eq!(act(g, 1.0, None).to_bits(), 0x8000_0000, "{g}");
    }
    assert_ne!(act(-88.5, 1.0, None).to_bits(), 0x8000_0000);
}

#[test]
fn routing_weight_is_applied_once_before_cast() {
    let base = act(2.0, 3.0, None);
    assert_eq!(act(2.0, 3.0, Some(1.0)), base);
    assert_eq!(act(2.0, 3.0, Some(1.5)), base * 1.5);
    assert_eq!(act(2.0, 3.0, Some(0.0)).to_bits(), 0.0f32.to_bits());
    assert_eq!(act(2.0, -3.0, Some(0.0)).to_bits(), (-0.0f32).to_bits());
    // -0 is a finite coefficient >= 0, accepted like route Choice / Expert.forward.
    assert_eq!(act(2.0, 3.0, Some(-0.0)).to_bits(), (-0.0f32).to_bits());
    assert_eq!(act(2.0, -3.0, Some(-0.0)).to_bits(), 0.0f32.to_bits());
    for w in [-1.0, -f32::MIN_POSITIVE, f32::NAN, f32::INFINITY] {
        assert_eq!(
            pre_down_activation(&[1.0], &[1.0], Some(w))
                .unwrap_err()
                .code,
            "INVALID_INPUT",
            "{w}"
        );
    }
    // Finite extreme weight overflows FP32 -> refused, no hidden clamp.
    assert_eq!(
        pre_down_activation(&[10.0], &[10.0], Some(f32::MAX))
            .unwrap_err()
            .code,
        "NUMERICAL_OVERFLOW"
    );
    assert!(pre_down_activation(&[10.0], &[10.0], Some(1e36)).is_ok());
}

#[test]
fn nonfinite_and_shape_refusals_before_clamps() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // A clamp would turn +-Inf into a finite boundary; it must be refused first.
        assert_eq!(
            pre_down_activation(&[bad], &[1.0], None).unwrap_err().code,
            "NONFINITE_INPUT"
        );
        assert_eq!(
            pre_down_activation(&[1.0], &[bad], None).unwrap_err().code,
            "NONFINITE_INPUT"
        );
    }
    assert!(pre_down_activation(&[], &[], None).unwrap().is_empty());
    assert!(pre_down_activation(&[], &[], Some(0.5)).unwrap().is_empty());
    assert!(pre_down_activation(&[], &[1.0], None).is_err());
    assert!(pre_down_activation(&[1.0, 2.0], &[1.0], None).is_err());
}

#[test]
fn cli_reports_exact_values_scope_and_refusals() {
    let (code, out) = vf(&json!({"op":"mxfp4-values","rows":1,"cols":32,
        "codes":[0x10,0x32,0x98,0,0,0,0,0,0,0,0,0,0,0,0,0],"scales":[0],
        "positions":[[0,0],[0,1],[0,3],[0,4],[0,5]]}));
    assert_eq!(code, 0, "{out}");
    let r = &out["result"];
    assert_eq!(r["capability"]["scope"], "CPU_FORMAT_OPERATOR_ONLY");
    assert_eq!(r["capability"]["gpu_executed"], false);
    assert_eq!(r["capability"]["full_model"], false);
    let bits: Vec<_> = r["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["f64_bits"].as_str().unwrap().to_owned())
        .collect();
    let want = [
        0.0,
        2f64.powi(-128),
        1.5 * 2f64.powi(-127),
        -0.0,
        -(2f64.powi(-128)),
    ];
    for (b, w) in bits.iter().zip(want) {
        assert_eq!(*b, format!("{:016x}", w.to_bits()));
    }
    assert_eq!(r["summary"]["compressed_bytes"], 17);
    for bad in [
        json!({"op":"mxfp4-values","rows":1,"cols":32,"codes":vec![0u8;16],"scales":[255],"positions":[[0,0]]}),
        json!({"op":"mxfp4-values","rows":1,"cols":32,"codes":vec![0u8;16],"scales":[1],"positions":[[0,32]]}),
        json!({"op":"mxfp4-values","rows":1,"cols":32,"codes":vec![0u8;16],"scales":[1],"positions":[]}),
        json!({"op":"mxfp4-values","rows":1,"cols":32,"codes":vec![0u8;16],"scales":[1],
            "positions":vec![[0,0];4097]}),
        json!({"op":"mxfp4-values","rows":1,"cols":32,"codes":vec![0u8;16],"scales":[1],
            "positions":[[0,0]],"global_scale":1.0}),
        json!({"op":"pre-down-activation","gate_bits":[0x7fc00000u32],"up_bits":[0]}),
        json!({"op":"pre-down-activation","gate_bits":[0],"up_bits":[0],
            "routing_weight_bits":0xbf80_0000u32}),
    ] {
        let (code, out) = vf(&bad);
        assert_eq!(code, 2, "{out}");
        assert_eq!(out["ok"], false);
    }
    // Finite nonnegative -0 is valid; retain the negative -1 refusal above.
    let (code, out) = vf(&json!({"op":"pre-down-activation",
        "gate_bits":[2.0f32.to_bits()],"up_bits":[1.0f32.to_bits()],
        "routing_weight_bits":0x8000_0000u32}));
    assert_eq!(code, 0, "{out}");
    assert_eq!(out["result"]["activation_bits"], json!([0x8000_0000u32]));
    let (code, out) = vf(&json!({"op":"pre-down-activation", "gate_bits":[], "up_bits":[]}));
    assert_eq!(code, 0, "{out}");
    assert_eq!(out["result"]["activation_bits"], json!([]));
    let (code, out) = vf(&json!({"op":"pre-down-activation",
        "gate_bits":[(-20.0f32).to_bits(), 20.0f32.to_bits()],
        "up_bits":[1.0f32.to_bits(), 20.0f32.to_bits()],"routing_weight_bits":1.5f32.to_bits()}));
    assert_eq!(code, 0, "{out}");
    let got: Vec<u32> = serde_json::from_value(out["result"]["activation_bits"].clone()).unwrap();
    let lib = pre_down_activation(&[-20.0, 20.0], &[1.0, 20.0], Some(1.5)).unwrap();
    assert_eq!(got, lib.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    assert_eq!(out["result"]["bf16_cast"], false);
}
