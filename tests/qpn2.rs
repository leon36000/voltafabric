use voltafabric::{qpn2, quant};

fn linear(rows: u64, columns: u64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: (0..rows * columns / 2)
            .map(|i| ((i * 73 + i / 19) % 256) as u8)
            .collect(),
        scales: (0..rows * columns / 16).map(|i| (i % 256) as u8).collect(),
        global_dequant: 1.25,
    }
}

fn same_linear(actual: &quant::PackedRows, expected: &quant::PackedRows) {
    assert_eq!(actual.rows, expected.rows);
    assert_eq!(actual.columns, expected.columns);
    assert_eq!(actual.codes, expected.codes);
    assert_eq!(actual.scales, expected.scales);
    assert_eq!(
        actual.global_dequant.to_bits(),
        expected.global_dequant.to_bits()
    );
}

#[test]
fn donor_ast_tinytrace_golden() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qpn2_golden.json")).unwrap();
    let p: quant::PackedRows = serde_json::from_value(golden["packed"].clone()).unwrap();
    let expected: qpn2::FragmentRows = serde_json::from_value(golden["fragment"].clone()).unwrap();
    let actual = qpn2::pack(&p).unwrap();
    assert_eq!(actual.codes, expected.codes);
    assert_eq!(actual.scales, expected.scales);
    assert_eq!(actual.rows, expected.rows);
    assert_eq!(actual.columns, expected.columns);
    assert_eq!(
        actual.global_dequant.to_bits(),
        expected.global_dequant.to_bits()
    );
    same_linear(&qpn2::unpack(&expected).unwrap(), &p);
    assert_eq!(
        qpn2::dequantize(&expected).unwrap(),
        quant::dequantize(&p).unwrap()
    );
}

#[test]
fn roundtrip_multiple_tiles_groups_all_bytes_and_nibbles() {
    for (rows, columns) in [(32, 64), (64, 128), (96, 192)] {
        let p = linear(rows, columns);
        let f = qpn2::pack(&p).unwrap();
        same_linear(&qpn2::unpack(&f).unwrap(), &p);
        assert_eq!(
            qpn2::pack(&qpn2::unpack(&f).unwrap()).unwrap().codes,
            f.codes
        );
    }
    for byte in 0..=255 {
        let mut p = linear(32, 64);
        p.codes.fill(byte);
        p.scales.fill(byte);
        let f = qpn2::pack(&p).unwrap();
        let low = (byte & 15) * 17;
        let high = (byte >> 4) * 17;
        let expected = [low, low, high, high, low, low, high, high];
        assert!(f.codes.chunks_exact(8).all(|chunk| chunk == expected));
        assert!(f.scales.iter().all(|&b| b == byte));
        same_linear(&qpn2::unpack(&f).unwrap(), &p);
    }
}

#[test]
fn one_nibble_impulses_reach_exact_fragments() {
    // Coordinate-forward oracle, independent of implementation's lane-to-row traversal.
    // row = 8*quad + 4*half + thread, lane = 4*quad + 16*half + thread.
    let k_to_slot = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];
    for row in [0_usize, 4, 7, 8, 15, 16, 23, 31, 32, 63] {
        for (k, &slot) in k_to_slot.iter().enumerate() {
            let mut p = linear(64, 128);
            p.codes.fill(0);
            let col = 7 * 16 + k;
            p.codes[row * 64 + col / 2] = 13 << (4 * (col % 2));
            let f = qpn2::pack(&p).unwrap();
            let lane = (row % 32 / 8) * 4 + (row % 8 / 4) * 16 + row % 4;
            let index = ((row / 32 * 8 + 7) * 32 + lane) * 8 + slot / 2;
            let mut expected = vec![0; p.codes.len()];
            expected[index] = 13 << (4 * (slot % 2));
            assert_eq!(f.codes, expected, "row={row}, k={k}");
        }
    }
}

#[test]
fn rejects_invalid_shapes_and_bounded_cpu_overflow() {
    for (rows, columns) in [
        (0, 64),
        (32, 0),
        (1, 64),
        (31, 64),
        (33, 64),
        (32, 16),
        (32, 63),
        (32, 65),
        (32768, 64),
        (u64::MAX - 31, 64),
    ] {
        let p = quant::PackedRows {
            rows,
            columns,
            codes: vec![],
            scales: vec![],
            global_dequant: 1.0,
        };
        assert!(qpn2::pack(&p).is_err(), "{rows}x{columns}");
        let f = qpn2::FragmentRows {
            rows,
            columns,
            codes: vec![],
            scales: vec![],
            global_dequant: 1.0,
        };
        assert!(qpn2::unpack(&f).is_err(), "{rows}x{columns}");
        assert!(qpn2::dequantize(&f).is_err(), "{rows}x{columns}");
    }
}

#[test]
fn accepts_exact_cpu_bound() {
    let p = linear(1024, 1024);
    let f = qpn2::pack(&p).unwrap();
    same_linear(&qpn2::unpack(&f).unwrap(), &p);
}

#[test]
fn rejects_truncated_and_extra_planes() {
    for plane in 0..2 {
        for extra in [false, true] {
            let mut p = linear(32, 64);
            let data = if plane == 0 {
                &mut p.codes
            } else {
                &mut p.scales
            };
            if extra {
                data.push(0);
            } else {
                data.pop();
            }
            assert!(qpn2::pack(&p).is_err());
            let mut f = qpn2::pack(&linear(32, 64)).unwrap();
            let data = if plane == 0 {
                &mut f.codes
            } else {
                &mut f.scales
            };
            if extra {
                data.push(0);
            } else {
                data.pop();
            }
            assert!(qpn2::unpack(&f).is_err());
            assert!(qpn2::dequantize(&f).is_err());
        }
    }
}

#[test]
fn rejects_nonpositive_nonfinite_global() {
    for global in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = linear(32, 64);
        p.global_dequant = global;
        assert!(qpn2::pack(&p).is_err());
        let mut f = qpn2::pack(&linear(32, 64)).unwrap();
        f.global_dequant = global;
        assert!(qpn2::unpack(&f).is_err());
        assert!(qpn2::dequantize(&f).is_err());
    }
}

#[test]
fn dequant_oracle_all_finite_scale_encodings_and_signed_zero() {
    let mut p = linear(64, 128);
    p.scales = (0..p.scales.len()).map(|i| (i % 127) as u8).collect();
    let f = qpn2::pack(&p).unwrap();
    let actual = qpn2::dequantize(&f).unwrap();
    let expected = quant::dequantize(&p).unwrap();
    for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert_eq!(a.to_bits(), e.to_bits());
    }
}

#[test]
fn scales_are_opaque_for_permutations_but_checked_by_cpu_oracle() {
    for byte in [127, 128, 255] {
        let mut p = linear(32, 64);
        p.scales.fill(0x38);
        p.scales[93] = byte;
        let f = qpn2::pack(&p).unwrap();
        same_linear(&qpn2::unpack(&f).unwrap(), &p);
        assert!(qpn2::dequantize(&f).is_err());
    }
}

#[test]
fn oracle_rejects_numeric_overflow() {
    let mut p = linear(32, 64);
    p.codes.fill(0x77);
    p.scales.fill(0x7e);
    p.global_dequant = f64::MAX;
    let f = qpn2::pack(&p).unwrap();
    assert_eq!(qpn2::dequantize(&f).unwrap_err().code, "NUMERICAL_OVERFLOW");
}

#[test]
fn fragment_serde_rejects_unknown_fields() {
    let f = qpn2::pack(&linear(32, 64)).unwrap();
    let mut value = serde_json::to_value(f).unwrap();
    value["unexpected"] = serde_json::json!(1);
    assert!(serde_json::from_value::<qpn2::FragmentRows>(value).is_err());
}

#[test]
fn misaligned_shapes_are_refused_even_with_consistent_planes() {
    // A40 guard sweep: the shape test above uses empty planes, so the length check
    // refused every misaligned shape; with matching planes only alignment can.
    for (rows, columns) in [(48, 64), (32, 96)] {
        let error = qpn2::pack(&linear(rows, columns)).unwrap_err();
        assert!(
            error
                .message
                .contains("rows divisible by 32 and columns divisible by 64"),
            "{rows}x{columns}: {error}"
        );
    }
}

#[test]
fn oversized_plane_is_refused_by_the_cpu_size_bound() {
    // A41 reason sweep: reached before, but only through is_err().
    let error = qpn2::pack(&linear(32768, 64)).unwrap_err();
    assert_eq!(error.message, "QPN2 matrix exceeds bounded CPU size limit");
}
