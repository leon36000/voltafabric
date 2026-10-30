use voltafabric::gdn_cpu::{ConvHistory, GdnState, Identity, Prepared, RecurrentState};
fn id() -> Identity {
    Identity {
        run_id: "test".into(),
        model_revision: "model".into(),
        layer: 0,
        epoch: 1,
    }
}

#[test]
fn token_transaction_commits_both_states_or_neither() {
    let mut joined = GdnState::new(id(), 1, 1, 1).unwrap();
    let weights = vec![1.0; 12];
    let first = joined
        .step(&id(), &[1.0, 2.0, 3.0], &weights, 1e-6, &[0.0], &[0.5])
        .unwrap();
    assert_eq!(
        joined.history(),
        &[0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]
    );
    assert!(first[0] > 0.0);
    let h = joined.history().to_vec();
    let s = joined.state().to_vec();
    // Convolution succeeds on the candidate, then recurrence rejects invalid beta.
    assert!(joined
        .step(&id(), &[2.0, 3.0, 4.0], &weights, 1e-6, &[0.0], &[1.1])
        .is_err());
    assert_eq!(joined.history(), h);
    assert_eq!(joined.state(), s);
    let mut wrong = id();
    wrong.epoch += 1;
    assert!(joined
        .step(&wrong, &[2.0, 3.0, 4.0], &weights, 1e-6, &[0.0], &[0.5])
        .is_err());
    assert_eq!(joined.history(), h);
    assert_eq!(joined.state(), s);
    let second = joined
        .step(&id(), &[2.0, 3.0, 4.0], &weights, 1e-6, &[0.0], &[0.5])
        .unwrap();
    assert!(second[0] > first[0]);
    assert_eq!(
        joined.history(),
        &[0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 3.0, 4.0]
    );
    assert_ne!(joined.state(), s);
}

#[test]
fn recurrence_uses_modulo_heads_and_preserves_state_on_failure() {
    let mut state = RecurrentState::new(id(), 2, 4, 1).unwrap();
    let input = Prepared {
        q: vec![1.0, 1.0],
        k: vec![1.0, 2.0],
        v: vec![1.0; 4],
    };
    assert_eq!(
        state.step(&id(), &input, &[0.0; 4], &[0.5; 4]).unwrap(),
        vec![0.5, 1.0, 0.5, 1.0]
    );
    assert_eq!(
        state.step(&id(), &input, &[0.0; 4], &[0.5; 4]).unwrap(),
        vec![0.75, 0.0, 0.75, 0.0]
    );
    let before = state.state().to_vec();
    let mut wrong = id();
    wrong.epoch += 1;
    assert!(state.step(&wrong, &input, &[0.0; 4], &[0.5; 4]).is_err());
    assert!(state.step(&id(), &input, &[1000.0; 4], &[0.5; 4]).is_err());
    assert!(state.step(&id(), &input, &[0.0; 4], &[1.1; 4]).is_err());
    assert_eq!(state.state(), before);
}

#[test]
fn recurrence_matches_independent_two_step_scalar_oracle() {
    let mut state = RecurrentState::new(id(), 2, 4, 2).unwrap();
    let input = Prepared {
        q: vec![0.6, 0.8, 0.6, 0.8],
        k: vec![0.8, 0.6, 0.0, 1.0],
        v: vec![1., 2., 3., 4., 5., 6., 7., 8.],
    };
    let gate = [0.8_f32.ln(), 0.7_f32.ln(), 0.6_f32.ln(), 0.5_f32.ln()];
    let beta = [0.25, 0.5, 0.75, 1.0];
    let expected = [
        [
            0.16970563, 0.33941126, 0.84852815, 1.1313709, 2.5455844, 3.0547013, 3.959798,
            4.5254836,
        ],
        [
            0.271529, 0.543058, 1.145513, 1.5273507, 2.927422, 3.5129066, 3.959798, 4.5254836,
        ],
    ];
    for oracle in expected {
        let actual = state.step(&id(), &input, &gate, &beta).unwrap();
        for (a, b) in actual.iter().zip(oracle) {
            assert!((a - b).abs() < 1e-5);
        }
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
