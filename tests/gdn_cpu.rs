use voltafabric::gdn_cpu::{ConvHistory, Identity};
fn id() -> Identity {
    Identity {
        run_id: "test".into(),
        model_revision: "model".into(),
        layer: 0,
        epoch: 1,
    }
}
#[test]
fn convolution_history_and_normalization_are_transactional() {
    let mut s = ConvHistory::new(id(), 3).unwrap();
    let weights = vec![1.0; 12];
    let out = s
        .step(&id(), &[1.0, 2.0, 3.0], &weights, 1, 1, 1, 1e-6)
        .unwrap();
    assert!((out.q[0] - 1.0).abs() < 1e-5);
    assert!((out.k[0] - 1.0).abs() < 1e-5);
    assert!((out.v[0] - (3.0_f32 / (1.0 + (-3.0_f32).exp()))).abs() < 1e-6);
    assert_eq!(s.history(), &[0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    let before = s.history().to_vec();
    let mut wrong = id();
    wrong.epoch = 2;
    assert!(s
        .step(&wrong, &[1.0, 2.0, 3.0], &weights, 1, 1, 1, 1e-6)
        .is_err());
    assert!(s
        .step(&id(), &[1.0, f32::NAN, 3.0], &weights, 1, 1, 1, 1e-6)
        .is_err());
    let mut huge = weights.clone();
    huge[3] = f32::MAX;
    assert!(s
        .step(&id(), &[f32::MAX, 2.0, 3.0], &huge, 1, 1, 1, 1e-6)
        .is_err());
    assert_eq!(s.history(), before);
    let next = s
        .step(&id(), &[2.0, 3.0, 4.0], &weights, 1, 1, 1, 1e-6)
        .unwrap();
    assert!((next.v[0] - (7.0_f32 / (1.0 + (-7.0_f32).exp()))).abs() < 1e-6);
}
