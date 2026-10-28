use voltafabric::deepseek_v41::{
    fp8_activation_quantize_reference, r9_fp32, Mxfp4ExpertShard, Mxfp4Matrix,
};

fn rounding_witness() -> Mxfp4Matrix {
    let mut codes = vec![0; 48];
    for col in [0, 32, 64] {
        codes[col / 2] = 2; // low nibble E2M1 = +1
    }
    Mxfp4Matrix::new(1, 96, codes, vec![127, 103, 103]).unwrap()
}

#[test]
fn scale_zero_is_a_subnormal_and_nonfinite_intermediates_fail() {
    let mut codes = vec![0; 16];
    codes[0] = 2;
    let m = Mxfp4Matrix::new(1, 32, codes.clone(), vec![0]).unwrap();
    let x = fp8_activation_quantize_reference(&[1.0; 32]).unwrap();
    assert_eq!(r9_fp32::project(&m, &x).unwrap()[0].to_bits(), 0x0040_0000);
    codes[0] = 7;
    let m = Mxfp4Matrix::new(1, 32, codes, vec![254]).unwrap();
    assert_eq!(
        r9_fp32::project(&m, &x).unwrap_err().code,
        "NUMERICAL_OVERFLOW"
    );
}

#[test]
fn shuffle_tree_differs_from_a_sequential_column_sum() {
    let mut codes = vec![0; 32];
    codes[0] = 5 << 4; // lane 1, block 0: +3
    codes[16] = 2; // lane 0, block 1: +2^64
    codes[17] = 10; // lane 2, block 1: -2^64
    let m = Mxfp4Matrix::new(1, 64, codes, vec![127, 191]).unwrap();
    let x = fp8_activation_quantize_reference(&[1.0; 64]).unwrap();
    assert_eq!(r9_fp32::project(&m, &x).unwrap()[0], 3.0);
    assert_eq!(m.matvec_reference(&[1.0; 64]).unwrap()[0], 0.0);
}

#[test]
fn complete_tiny_experts_match_all_independent_torch_golden_bits() {
    let golden: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../fixtures/preweighted_mxfp4_tiny_golden.json"
    ))
    .unwrap();
    let bytes = include_bytes!("../fixtures/preweighted_mxfp4_tiny.safetensors");
    let shard = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        bytes.to_vec(),
        golden["weights_sha256"].as_str().unwrap(),
        65536,
    )
    .unwrap();
    assert_eq!(golden["case_count"], 45);
    for case in golden["cases"].as_array().unwrap() {
        let input_name = case["input"].as_str().unwrap();
        let input: Vec<f32> = golden["inputs"][input_name]["bf16_as_f32_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()))
            .collect();
        let expert = shard
            .expert(0, case["expert"].as_u64().unwrap() as u32)
            .unwrap();
        let result =
            r9_fp32::forward(expert, &input, case["coefficient"].as_f64().unwrap() as f32).unwrap();
        let expected: Vec<u32> = case["output_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            .collect();
        assert_eq!(
            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected,
            "expert={} input={} coefficient={}",
            case["expert"],
            input_name,
            case["coefficient"]
        );
    }
}

#[test]
fn projection_preserves_the_r9_lane_rounding_cutpoints() {
    // Lane zero sees 1, 2^-24, 2^-24: each small add ties down in binary32.
    // An FP64 sum rounded once instead gives 1 + 2^-23.
    let matrix = rounding_witness();
    let input = fp8_activation_quantize_reference(&[1.0; 96]).unwrap();
    let output = r9_fp32::project(&matrix, &input).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f80_0000);
    let fp64 = matrix.matvec_reference(&[1.0; 96]).unwrap();
    assert_eq!((fp64[0] as f32).to_bits(), 0x3f80_0001);
}

#[test]
fn compressed_plane_access_preserves_nibbles_rows_scales_and_row_permutations() {
    // All inputs dequantize exactly to one; these dyadic sums are exact in FP32.
    let mut packed = vec![0; 3 * 32];
    packed[0] = 2 | (9 << 4); // +1, -0.5 at scale 1
    packed[16] = 4; // +2 at scale 2
    packed[31] = 10 << 4; // -1 at scale 2 => row 0 = 2.5
    packed[32 + 15] = 3 << 4; // +1.5 at scale 2^-7
    packed[32 + 31] = 11; // -1.5 at scale 4 => row 1 = -5.98828125
    packed[64..].fill(0x22); // 32 * 1 + 32 * 0.5 => row 2 = 48
    let scales = [127, 128, 120, 129, 127, 126];
    let matrix = Mxfp4Matrix::new(3, 64, packed.clone(), scales.to_vec()).unwrap();
    let input = fp8_activation_quantize_reference(&[1.0; 64]).unwrap();
    let expected = [2.5_f32, (-1533.0_f32 / 256.0_f32), 48.0_f32].map(f32::to_bits);
    let actual = r9_fp32::project(&matrix, &input).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected
    );
    let order = [2, 0, 1];
    let permuted_codes = order
        .iter()
        .flat_map(|&i| packed[i * 32..(i + 1) * 32].iter().copied())
        .collect();
    let permuted_scales = order
        .iter()
        .flat_map(|&i| scales[i * 2..(i + 1) * 2].iter().copied())
        .collect();
    let permuted = Mxfp4Matrix::new(3, 64, permuted_codes, permuted_scales).unwrap();
    assert_eq!(
        r9_fp32::project(&permuted, &input)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        order.map(|i| expected[i])
    );
    let short = fp8_activation_quantize_reference(&[1.0; 32]).unwrap();
    assert_eq!(
        r9_fp32::project(&matrix, &short).unwrap_err().code,
        "INVALID_INPUT"
    );
}
