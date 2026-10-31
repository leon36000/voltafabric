use voltafabric::gdn_cpu::{
    output_norm, project_gates, ConvHistory, GdnState, Identity, Prepared, RecurrentState,
};
fn id() -> Identity {
    Identity {
        run_id: "test".into(),
        model_revision: "model".into(),
        layer: 0,
        epoch: 1,
    }
}

#[test]
fn provisional_gdn_batch_matches_sequential_and_commits_both_states() {
    let mut sequential = GdnState::new(id(), 1, 1, 1).unwrap();
    let mut batched = GdnState::new(id(), 1, 1, 1).unwrap();
    let kernel = vec![1.0; 12];
    let inputs = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
    let expected: Vec<_> = inputs
        .iter()
        .map(|x| {
            sequential
                .step(&id(), x, &kernel, 1e-6, &[0.0], &[0.5])
                .unwrap()
        })
        .collect();
    let mut tx = batched.begin_batch(3).unwrap();
    for (i, input) in inputs.iter().enumerate() {
        assert_eq!(
            tx.prepare_next(&id(), input, &kernel, 1e-6, &[0.0], &[0.5])
                .unwrap(),
            expected[i]
        );
    }
    assert_eq!(tx.commit().unwrap(), expected);
    assert_eq!(batched.history(), sequential.history());
    assert_eq!(batched.state(), sequential.state());
}

#[test]
fn failed_incomplete_and_dropped_gdn_batches_leave_both_states_unchanged() {
    let mut state = GdnState::new(id(), 1, 1, 1).unwrap();
    let kernel = vec![1.0; 12];
    let before_history = state.history().to_vec();
    let before_state = state.state().to_vec();
    {
        let mut tx = state.begin_batch(2).unwrap();
        tx.prepare_next(&id(), &[1., 2., 3.], &kernel, 1e-6, &[0.], &[0.5])
            .unwrap();
        assert!(tx
            .prepare_next(&id(), &[2., 3., 4.], &kernel, 1e-6, &[0.], &[1.1])
            .is_err());
        assert!(tx
            .prepare_next(&id(), &[2., 3., 4.], &kernel, 1e-6, &[0.], &[0.5])
            .is_err());
        assert!(tx.commit().is_err());
    }
    assert_eq!(state.history(), before_history);
    assert_eq!(state.state(), before_state);
    {
        let mut tx = state.begin_batch(2).unwrap();
        tx.prepare_next(&id(), &[1., 2., 3.], &kernel, 1e-6, &[0.], &[0.5])
            .unwrap();
        assert!(tx.commit().is_err());
    }
    assert_eq!(state.history(), before_history);
    assert_eq!(state.state(), before_state);
    {
        let mut tx = state.begin_batch(1).unwrap();
        tx.prepare_next(&id(), &[1., 2., 3.], &kernel, 1e-6, &[0.], &[0.5])
            .unwrap();
    }
    assert_eq!(state.history(), before_history);
    assert_eq!(state.state(), before_state);
    assert!(state.begin_batch(0).is_err());
    assert!(state.begin_batch(129).is_err());
}

#[test]
fn output_norm_is_per_head_and_sigmoid_gated() {
    let y = output_norm(&[3.0, 4.0, 0.0, 5.0], &[0.0; 4], &[1.0, 2.0], 1e-6).unwrap();
    let inv = (12.5_f64 + 1e-6).sqrt().recip() as f32;
    for (actual, expected) in y.iter().zip([1.5 * inv, 4.0 * inv, 0.0, 5.0 * inv]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert!(output_norm(&[1.0, f32::NAN], &[0.0; 2], &[1.0; 2], 1e-6).is_err());
    assert!(output_norm(&[1.0], &[0.0], &[1.0, 1.0], 1e-6).is_err());
}

#[test]
fn gate_projection_checks_bf16_layout_and_invalid_data() {
    let alpha = [0x80, 0x3f, 0x00, 0x40, 0x80, 0xbf, 0x80, 0x3f];
    let beta = [0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00];
    let (gate, b) = project_gates(&[1.0, 2.0], &alpha, &beta, &[0.0, 0.0], &[-1.0, -2.0]).unwrap();
    assert!((gate[0] + 5.0_f32.exp().ln_1p()).abs() < 1e-6);
    assert!((gate[1] + 2.0 * 1.0_f32.exp().ln_1p()).abs() < 1e-6);
    assert!((b[0] - 0.5).abs() < 1e-6 && (b[1] - 0.7310586).abs() < 1e-6);
    assert!(project_gates(&[1.0], &alpha, &beta, &[0.0, 0.0], &[-1.0, -2.0]).is_err());
    let mut bad = alpha;
    bad[0] = 0x80;
    bad[1] = 0x7f;
    assert!(project_gates(&[1.0, 2.0], &bad, &beta, &[0.0, 0.0], &[-1.0, -2.0]).is_err());
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
