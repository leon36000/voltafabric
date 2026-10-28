use voltafabric::quant::validate_sm70_scale_domain;

fn tile(scale: u8, k: u64) -> Vec<u8> {
    vec![scale; (k as usize / 16) * 32]
}

fn code(result: voltafabric::Result<()>) -> Option<String> {
    result.err().map(|error| error.code)
}

#[test]
fn rejects_intermediate_half_overflow_even_when_final_product_is_finite() {
    assert_eq!(
        code(validate_sm70_scale_domain(&tile(0x38, 64), &[4.0], 64)),
        Some("SM70_SCALE_RANGE".into())
    );
    assert_eq!(
        code(validate_sm70_scale_domain(&tile(0x7e, 64), &[0.5], 64)),
        Some("SM70_SCALE_RANGE".into())
    );
}

#[test]
fn accepts_binary16_boundary_and_rejects_rne_overflow_tie() {
    let gscale_max = 65_504.0_f32 / 16_384.0;
    assert_eq!(
        validate_sm70_scale_domain(&tile(0x38, 64), &[gscale_max], 64),
        Ok(())
    );

    let gscale_tie = 146.25_f32 / 16_384.0;
    assert_eq!(
        code(validate_sm70_scale_domain(
            &tile(0x7e, 64),
            &[gscale_tie],
            64
        )),
        Some("SM70_SCALE_RANGE".into())
    );

    for neighbor in [
        f32::from_bits(gscale_tie.to_bits() - 1),
        f32::from_bits(gscale_tie.to_bits() + 1),
    ] {
        assert_eq!(
            code(validate_sm70_scale_domain(&tile(0x7e, 64), &[neighbor], 64)),
            Some("SM70_SCALE_RANGE".into())
        );
    }

    let gscale_below = 146.125_f32 / 16_384.0;
    assert_eq!(
        validate_sm70_scale_domain(&tile(0x7e, 64), &[gscale_below], 64),
        Ok(())
    );
}

#[test]
fn zero_scale_and_tiny_global_scale_remain_valid() {
    assert_eq!(validate_sm70_scale_domain(&tile(0, 64), &[1.0], 64), Ok(()));
    assert_eq!(
        validate_sm70_scale_domain(&tile(0x7e, 64), &[f32::from_bits(1)], 64),
        Ok(())
    );
}

#[test]
fn evaluates_each_contiguous_tile_against_its_own_global_scale() {
    let mut scales = tile(0x7e, 64);
    scales.extend(tile(0x38, 64));
    assert_eq!(
        code(validate_sm70_scale_domain(&scales, &[0.5, 1.0], 64)),
        Some("SM70_SCALE_RANGE".into())
    );

    let mut scales = tile(0x38, 64);
    scales.extend(tile(0x38, 64));
    assert_eq!(validate_sm70_scale_domain(&scales, &[0.5, 1.0], 64), Ok(()));
}

#[test]
fn checks_dimensions_and_lengths_before_scanning() {
    assert_eq!(
        code(validate_sm70_scale_domain(&[], &[1.0], 64)),
        Some("INVALID_INPUT".into())
    );
    assert_eq!(
        code(validate_sm70_scale_domain(&[], &[1.0], 63)),
        Some("INVALID_INPUT".into())
    );
    assert_eq!(
        code(validate_sm70_scale_domain(&tile(0x7f, 64), &[1.0], 64)),
        Some("INVALID_INPUT".into())
    );
}
