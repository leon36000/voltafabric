use voltafabric::deepseek_v41_gate::{RouterBiases, TokenKind, ROUTED_EXPERTS, ROUTE_SCALE};

// Generated independently by research/R701/r700_gate_decimal_oracle.py using
// Python Decimal at 60 digits; it imports neither the Rust gate nor donor code.
#[test]
fn fp32_gate_route_matches_independent_decimal_fixture_oracle() {
    let logits: Vec<f32> = (0..ROUTED_EXPERTS).map(|i| i as f32).collect();
    let mut text_bias = vec![0.0; ROUTED_EXPERTS];
    text_bias[0] = 1_000.0;
    let gate = RouterBiases::new(text_bias, vec![0.0; ROUTED_EXPERTS], ROUTE_SCALE).unwrap();

    let route = gate.route(&logits, TokenKind::Text).unwrap();
    let expected = [
        (0, 0.012_687_707_376_015_25_f64),
        (383, 0.298_242_691_616_205_54_f64),
        (382, 0.297_853_086_366_122_95_f64),
        (381, 0.297_462_970_826_91_f64),
        (380, 0.297_072_342_988_230_93_f64),
        (379, 0.296_681_200_826_515_33_f64),
    ];

    assert_eq!(route.len(), expected.len());
    for (actual, (expert, coefficient)) in route.iter().zip(expected) {
        assert_eq!(actual.expert, expert);
        assert!((actual.weight - coefficient).abs() < 2.0e-7);
    }
}
