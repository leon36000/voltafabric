use voltafabric::quant::*;
#[test]
fn fp4_all_codes() {
    let positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (n, &x) in positive.iter().enumerate() {
        assert_eq!(decode_fp4(n as u8).unwrap(), x);
        assert_eq!(decode_fp4(n as u8 + 8).unwrap(), -x);
    }
    assert!(decode_fp4(16).is_err());
}
#[test]
fn fp8_boundaries() {
    assert_eq!(decode_scale(0).unwrap(), 0.0);
    assert_eq!(decode_scale(1).unwrap(), 2.0_f64.powi(-9));
    assert_eq!(decode_scale(0x38).unwrap(), 1.0);
    assert_eq!(decode_scale(0x7e).unwrap(), 448.0);
    assert!(decode_scale(0x7f).is_err());
    assert!(decode_scale(0x80).is_err());
}
#[test]
fn dequant_preserves_row_padding() {
    let p = PackedRows {
        rows: 2,
        columns: 3,
        codes: vec![0x21, 0x03, 0, 0, 0, 0, 0, 0, 0x45, 0x06, 0, 0, 0, 0, 0, 0],
        scales: vec![0x38, 0x40],
        global_dequant: 2.0,
    };
    assert_eq!(
        dequantize(&p).unwrap(),
        vec![vec![1.0, 2.0, 3.0], vec![12.0, 8.0, 16.0]]
    );
}
#[test]
fn invalid_packed_length() {
    let p = PackedRows {
        rows: 1,
        columns: 16,
        codes: vec![0; 7],
        scales: vec![0x38],
        global_dequant: 1.0,
    };
    assert!(dequantize(&p).is_err());
}
#[test]
fn invalid_global_scale() {
    for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let p = PackedRows {
            rows: 1,
            columns: 16,
            codes: vec![0; 8],
            scales: vec![0x38],
            global_dequant: v,
        };
        assert!(dequantize(&p).is_err());
    }
}
#[test]
fn zero_block_scale_is_valid() {
    let p = PackedRows {
        rows: 1,
        columns: 16,
        codes: vec![0x77; 8],
        scales: vec![0],
        global_dequant: 1.0,
    };
    assert!(dequantize(&p).unwrap()[0].iter().all(|&x| x == 0.0));
}

// A40 guard sweep: one clause per case; with the clause disabled each was either
// accepted (oversized plane, extra scale byte, non-finite decoded value) or refused
// only by a later check with another reason.
fn plane(rows: u64, columns: u64, global_dequant: f64) -> PackedRows {
    PackedRows {
        rows,
        columns,
        codes: vec![0x11; (rows * columns / 2) as usize],
        scales: vec![0x38; (rows * columns / 16) as usize],
        global_dequant,
    }
}
#[test]
fn dequantize_refuses_each_contract_violation_for_its_own_reason() {
    let bound = "invalid global dequant factor or CPU size limit";
    assert_eq!(
        dequantize(&plane(32, 64, f64::NAN)).unwrap_err().message,
        bound
    );
    assert_eq!(
        dequantize(&plane(2048, 1024, 0.5)).unwrap_err().message,
        bound
    );
    let mut extra = plane(32, 64, 0.5);
    extra.scales.push(0);
    assert_eq!(
        dequantize(&extra).unwrap_err().message,
        "packed plane length mismatch"
    );
    // 6 * 448 * 1e306 overflows binary64.
    let mut huge = plane(32, 64, 1e306);
    huge.codes.fill(0x77);
    huge.scales.fill(126);
    assert_eq!(dequantize(&huge).unwrap_err().code, "NUMERICAL_OVERFLOW");
}
#[test]
fn sm70_global_scales_are_refused_at_the_first_failing_step() {
    let scales = [0x38; 128];
    let first = "global scale must be positive and finite";
    for g in [f32::NAN, -1.0] {
        let error = validate_sm70_scale_domain(&scales, &[g], 64).unwrap_err();
        assert_eq!(error.message, first, "{g}");
    }
    // Finite, but g * 16384 overflows binary32.
    let error = validate_sm70_scale_domain(&scales, &[3e34], 64).unwrap_err();
    assert_eq!(error.message, "global scale intermediate is nonfinite");
}
