use voltafabric::deepseek_v41_gate::{RouterBiases, TokenKind, ROUTED_EXPERTS, ROUTE_SCALE, TOP_K};

fn biases(text: Vec<f32>, image: Vec<f32>) -> RouterBiases {
    RouterBiases::new(text, image, ROUTE_SCALE).expect("valid pinned gate biases")
}

fn expected_score(logit: f64) -> f64 {
    logit.exp().ln_1p().sqrt()
}

#[test]
fn selected_bias_changes_membership_but_coefficients_use_unbiased_scores() {
    let logits: Vec<f32> = (0..ROUTED_EXPERTS).map(|i| i as f32).collect();
    let mut text_bias = vec![0.0; ROUTED_EXPERTS];
    let mut image_bias = vec![0.0; ROUTED_EXPERTS];
    text_bias[0] = 1_000.0;
    image_bias[1] = 1_000.0;
    let gate = biases(text_bias, image_bias);

    let text = gate.route(&logits, TokenKind::Text).unwrap();
    let image = gate.route(&logits, TokenKind::Image).unwrap();
    assert_eq!(text.len(), TOP_K);
    assert_eq!(image.len(), TOP_K);
    assert_eq!(
        text.iter().map(|x| x.expert).collect::<Vec<_>>(),
        [0, 383, 382, 381, 380, 379]
    );
    assert_eq!(
        image.iter().map(|x| x.expert).collect::<Vec<_>>(),
        [1, 383, 382, 381, 380, 379]
    );

    let text_raw_scores = [0.0, 383.0, 382.0, 381.0, 380.0, 379.0].map(expected_score);
    let expected_sum = text_raw_scores.iter().sum::<f64>();
    for (choice, raw_score) in text.iter().zip(text_raw_scores) {
        let expected = raw_score / expected_sum * f64::from(ROUTE_SCALE);
        assert!((choice.weight - expected).abs() < 2.0e-7);
    }
    let actual_sum = text.iter().map(|choice| choice.weight).sum::<f64>();
    assert!((actual_sum - f64::from(ROUTE_SCALE)).abs() < 2.0e-7);
}

#[test]
fn caller_supplied_route_scale_is_applied_once() {
    let logits: Vec<f32> = (0..ROUTED_EXPERTS).map(|i| i as f32).collect();
    let gate =
        RouterBiases::new(vec![0.0; ROUTED_EXPERTS], vec![0.0; ROUTED_EXPERTS], 2.25).unwrap();

    let route = gate.route(&logits, TokenKind::Text).unwrap();
    let total = route.iter().map(|choice| choice.weight).sum::<f64>();
    assert!((total - 2.25).abs() < 3.0e-7);
    assert_eq!(route.len(), TOP_K);
}

#[test]
fn topk_is_sorted_and_rejects_ties_that_make_routing_backend_dependent() {
    let logits: Vec<f32> = (0..ROUTED_EXPERTS).map(|i| i as f32).collect();
    let mut text_bias = vec![0.0; ROUTED_EXPERTS];
    let image_bias = vec![0.0; ROUTED_EXPERTS];
    text_bias[200] = 10_000.0;
    let route = biases(text_bias, image_bias)
        .route(&logits, TokenKind::Text)
        .unwrap();
    assert_eq!(
        route.iter().map(|x| x.expert).collect::<Vec<_>>(),
        [200, 383, 382, 381, 380, 379]
    );
    assert!(route.iter().all(|choice| choice.weight.is_finite()));

    let equal_logits = vec![0.0; ROUTED_EXPERTS];
    let equal_bias = biases(vec![0.0; ROUTED_EXPERTS], vec![0.0; ROUTED_EXPERTS]);
    assert!(equal_bias.route(&equal_logits, TokenKind::Text).is_err());
}

#[test]
fn gate_rejects_wrong_shapes_nonfinite_values_and_missing_model_inputs() {
    assert!(RouterBiases::new(
        vec![0.0; ROUTED_EXPERTS - 1],
        vec![0.0; ROUTED_EXPERTS],
        ROUTE_SCALE,
    )
    .is_err());
    let mut invalid_bias = vec![0.0; ROUTED_EXPERTS];
    invalid_bias[7] = f32::NAN;
    assert!(RouterBiases::new(vec![0.0; ROUTED_EXPERTS], invalid_bias, ROUTE_SCALE).is_err());
    assert!(RouterBiases::new(
        vec![0.0; ROUTED_EXPERTS],
        vec![0.0; ROUTED_EXPERTS],
        f32::NAN,
    )
    .is_err());

    let gate = biases(vec![0.0; ROUTED_EXPERTS], vec![0.0; ROUTED_EXPERTS]);
    assert!(gate
        .route(&vec![0.0; ROUTED_EXPERTS - 1], TokenKind::Text)
        .is_err());
    let mut invalid_logits = vec![0.0; ROUTED_EXPERTS];
    invalid_logits[9] = f32::INFINITY;
    assert!(gate.route(&invalid_logits, TokenKind::Text).is_err());
}

#[test]
fn negative_extreme_logits_remain_finite_and_coefficients_are_not_renormalized_twice() {
    let mut logits = vec![-100.0; ROUTED_EXPERTS];
    let mut text_bias = vec![0.0; ROUTED_EXPERTS];
    for (expert, value) in [0.0, -1.0, -2.0, -3.0, -4.0, -5.0].into_iter().enumerate() {
        logits[expert] = value;
        text_bias[expert] = 100.0 - expert as f32;
    }
    let gate = biases(text_bias, vec![0.0; ROUTED_EXPERTS]);
    let route = gate.route(&logits, TokenKind::Text).unwrap();
    assert_eq!(route.len(), TOP_K);
    assert!(route
        .iter()
        .all(|choice| choice.weight.is_finite() && choice.weight >= 0.0));
    let sum = route.iter().map(|choice| choice.weight).sum::<f64>();
    assert!((sum - f64::from(ROUTE_SCALE)).abs() < 2.0e-7);
}
