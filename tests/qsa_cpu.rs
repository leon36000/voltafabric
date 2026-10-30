use voltafabric::{
    gdn_cpu::Identity,
    qsa_cpu::{State, MAX_IDENTITY_SELECTION},
};
fn identity() -> Identity {
    Identity {
        run_id: "test".into(),
        model_revision: "model".into(),
        layer: 3,
        epoch: 4,
    }
}
#[test]
fn gqa_two_steps_and_atomic_rejection() {
    let id = identity();
    let mut s = State::new(id.clone(), 4, 2, 2, 2).unwrap();
    let q = vec![0.; 8];
    let k = vec![0.; 4];
    let gate = vec![0.; 8];
    let first = s.prepare(&id, 0, &q, &k, &[2., 4., 6., 8.], &gate).unwrap();
    let stale = s.prepare(&id, 0, &q, &k, &[2., 4., 6., 8.], &gate).unwrap();
    assert_eq!(s.len(), 0);
    assert_eq!(
        s.commit(first).unwrap(),
        vec![1., 2., 1., 2., 3., 4., 3., 4.]
    );
    assert!(s.commit(stale).is_err());
    assert_eq!(s.len(), 1);
    let other = Identity {
        epoch: 5,
        ..id.clone()
    };
    assert!(s
        .prepare(&other, 1, &q, &k, &[4., 8., 10., 12.], &gate)
        .is_err());
    assert!(s
        .prepare(&id, 1, &q, &k, &[f32::NAN, 0., 0., 0.], &gate)
        .is_err());
    assert_eq!(s.len(), 1);
    let second = s
        .prepare(&id, 1, &q, &k, &[4., 8., 10., 12.], &gate)
        .unwrap();
    assert_eq!(
        s.commit(second).unwrap(),
        vec![1.5, 3., 1.5, 3., 4., 5., 4., 5.]
    );
    assert!(s.prepare(&id, 2, &q, &k, &[1.; 4], &gate).is_err());
    assert_eq!(s.len(), 2);
}
#[test]
fn different_states_reject_each_others_candidates() {
    let id = identity();
    let a = State::new(id.clone(), 2, 1, 2, 1).unwrap();
    let mut b = State::new(id.clone(), 2, 1, 2, 1).unwrap();
    let p = a
        .prepare(&id, 0, &[0.; 4], &[0.; 2], &[1.; 2], &[0.; 4])
        .unwrap();
    assert!(b.commit(p).is_err());
    assert!(b.is_empty());
    assert!(State::new(id, 2, 1, 2, MAX_IDENTITY_SELECTION + 1).is_err());
}
#[test]
fn two_cell_softmax_uses_causal_cache_and_scores() {
    let id = identity();
    let mut s = State::new(id.clone(), 2, 1, 1, 2).unwrap();
    let first = s
        .prepare(&id, 0, &[1., -1.], &[1.], &[2.], &[0., 0.])
        .unwrap();
    s.commit(first).unwrap();
    let p = s
        .prepare(&id, 1, &[1., -1.], &[-1.], &[6.], &[0., 0.])
        .unwrap();
    let out = p.output();
    let e = std::f64::consts::E;
    let expected0 = (2.0 * e + 6.0 / e) / (e + 1.0 / e) * 0.5;
    let expected1 = (2.0 / e + 6.0 * e) / (e + 1.0 / e) * 0.5;
    assert!((f64::from(out[0]) - expected0).abs() < 1e-6);
    assert!((f64::from(out[1]) - expected1).abs() < 1e-6);
    assert_eq!(s.len(), 1);
    s.commit(p).unwrap();
    assert_eq!(s.len(), 2);
}

#[test]
fn third_position_uses_all_three_causal_cells() {
    let id = identity();
    let mut state = State::new(id.clone(), 1, 1, 1, 3).unwrap();
    for (position, key, value) in [(0, -1.0, 2.0), (1, 0.0, 4.0)] {
        let candidate = state
            .prepare(&id, position, &[1.0], &[key], &[value], &[0.0])
            .unwrap();
        state.commit(candidate).unwrap();
    }
    let candidate = state
        .prepare(&id, 2, &[1.0], &[1.0], &[8.0], &[0.0])
        .unwrap();
    let e = std::f64::consts::E;
    let expected = (2.0 / e + 4.0 + 8.0 * e) / (1.0 / e + 1.0 + e) / 2.0;
    assert!((f64::from(candidate.output()[0]) - expected).abs() < 1e-6);
    assert_eq!(state.len(), 2);
    state.commit(candidate).unwrap();
    assert_eq!(state.len(), 3);
    assert!(state
        .prepare(&id, 3, &[1.0], &[0.0], &[0.0], &[0.0])
        .is_err());
}

#[test]
fn fifth_position_uses_all_causal_cells_and_rejects_overflow() {
    let id = identity();
    let mut state = State::new(id.clone(), 1, 1, 1, 5).unwrap();
    let keys = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let values = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
    for position in 0..5 {
        let candidate = state
            .prepare(
                &id,
                position,
                &[1.0],
                &[keys[position] as f32],
                &[values[position] as f32],
                &[0.0],
            )
            .unwrap();
        let numerator: f64 = (0..=position).map(|i| keys[i].exp() * values[i]).sum();
        let denominator: f64 = (0..=position).map(|i| keys[i].exp()).sum();
        assert!((f64::from(candidate.output()[0]) - numerator / denominator / 2.0).abs() < 1e-6);
        assert_eq!(state.len(), position);
        state.commit(candidate).unwrap();
    }
    assert_eq!(state.len(), 5);
    assert!(state
        .prepare(&id, 5, &[1.0], &[0.0], &[0.0], &[0.0])
        .is_err());
    assert_eq!(state.len(), 5);
}
