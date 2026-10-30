use voltafabric::sc117_ple_post::WIDTH;
use voltafabric::sc117_ple_weights::project;
#[test]
fn bf16_projection_checks_layout_and_nonfinite() {
    let mut input = vec![0.0; WIDTH];
    input[0] = 2.0;
    input[1] = -3.0;
    let mut weights = vec![0u8; WIDTH * 4];
    weights[0..2].copy_from_slice(&0x3f80u16.to_le_bytes());
    weights[2..4].copy_from_slice(&0x4000u16.to_le_bytes());
    weights[WIDTH * 2..WIDTH * 2 + 2].copy_from_slice(&0xbf80u16.to_le_bytes());
    assert_eq!(project(&weights, &input, 2).unwrap(), vec![-4.0, -2.0]);
    weights[0..2].copy_from_slice(&0x7f80u16.to_le_bytes());
    assert!(project(&weights, &input, 2).is_err());
    assert!(project(&weights, &input[..WIDTH - 1], 2).is_err());
}

#[test]
fn activations_round_bf16_ties_even_and_refuse_overflow() {
    let mut input = vec![0.0; WIDTH];
    let mut weights = vec![0u8; WIDTH * 2];
    input[0] = f32::from_bits(0x3f80_8000);
    input[1] = f32::from_bits(0x3f81_8000);
    weights[..2].copy_from_slice(&0x3f80u16.to_le_bytes());
    weights[2..4].copy_from_slice(&0x3f80u16.to_le_bytes());
    assert_eq!(
        project(&weights, &input, 1).unwrap()[0],
        1.0 + f32::from_bits(0x3f82_0000)
    );
    input[0] = f32::MAX;
    assert!(project(&weights, &input, 1).is_err());
}
