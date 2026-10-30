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
