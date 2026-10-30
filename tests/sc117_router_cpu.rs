use voltafabric::sc117_router_cpu::{route, HIDDEN, WEIGHT_BYTES};
#[test]
fn ties_are_stable_and_weights_are_normalized() {
    let weights = vec![0_u8; WEIGHT_BYTES];
    let input = vec![0_f32; HIDDEN];
    let selected = route(&weights, &input, 10).unwrap();
    assert_eq!(
        selected.iter().map(|x| x.expert).collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
    assert!(selected.iter().all(|c| (c.weight - 0.1).abs() < 1e-7));
}
#[test]
fn selected_rows_follow_bf16_values_and_refuse_invalids() {
    let mut weights = vec![0_u8; WEIGHT_BYTES];
    for (expert, bits) in [(3, 0x4000_u16), (7, 0x3f80_u16), (9, 0xbf80_u16)] {
        let p = expert * HIDDEN * 2;
        weights[p..p + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let mut input = vec![0_f32; HIDDEN];
    input[0] = 1.0;
    let out = route(&weights, &input, 2).unwrap();
    assert_eq!((out[0].expert, out[1].expert), (3, 7));
    assert!((f64::from(out[0].weight) - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-6);
    input[0] = f32::NAN;
    assert!(route(&weights, &input, 2).is_err());
    input[0] = 1.0;
    weights[0..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(route(&weights, &input, 2).is_err());
}
