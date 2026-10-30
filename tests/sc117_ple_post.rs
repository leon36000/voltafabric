use voltafabric::sc117_ple_post::{self as p, Inputs, CHANNELS, HISTORY, WIDTH};

#[test]
fn projected_zero_key_and_dilated_history() {
    let key = vec![0.0; CHANNELS];
    let hidden = vec![0.0; CHANNELS];
    let value = vec![2.0; WIDTH];
    let weights = vec![1.0; CHANNELS];
    let mut kernel = vec![0.0; CHANNELS * 4];
    let mut history = vec![0.0; CHANNELS * HISTORY];
    for c in 0..CHANNELS {
        kernel[c * 4] = 1.0;
        history[c * HISTORY] = 2.0;
        history[c * HISTORY + 1] = 3.0;
    }
    let output = p::evaluate(Inputs {
        key: &key,
        value: &value,
        hidden: &hidden,
        norm_key: &weights,
        norm_query: &weights,
        norm_conv: &weights,
        kernel: &kernel,
        history: &history,
    })
    .unwrap();
    let expected = 1.0 + 2.0 / (1.0 + (-2.0f32).exp());
    assert!((output.residual[0] - expected).abs() < 1e-6);
    assert!((output.normalized[0] - 1.0 / (1.0f32 + 1e-6).sqrt()).abs() < 1e-6);
    assert_eq!(output.next_history[0], 3.0);
    assert_eq!(output.next_history[HISTORY - 1], output.normalized[0]);
    assert_eq!(history[0], 2.0);
}
#[test]
fn invalid_inputs_refused() {
    let zeros = vec![0.0; CHANNELS];
    let value = vec![0.0; WIDTH];
    let kernel = vec![0.0; CHANNELS * 4];
    let mut history = vec![0.0; CHANNELS * HISTORY];
    history[0] = f32::NAN;
    assert!(p::evaluate(Inputs {
        key: &zeros,
        value: &value,
        hidden: &zeros,
        norm_key: &zeros,
        norm_query: &zeros,
        norm_conv: &zeros,
        kernel: &kernel,
        history: &history
    })
    .is_err());
}

#[test]
fn independent_scalar_oracle_checks_signed_gates_and_tap_order() {
    // Expected values computed independently with Python math from the equations.
    let mut key = vec![0.0; CHANNELS];
    let mut hidden = vec![0.0; CHANNELS];
    let value = vec![0.75; WIDTH];
    let nk = vec![1.1; CHANNELS];
    let nq = vec![0.9; CHANNELS];
    let nc = vec![1.2; CHANNELS];
    let mut kernel = vec![0.0; CHANNELS * 4];
    let mut history = vec![0.0; CHANNELS * HISTORY];
    for c in 0..CHANNELS {
        let s = c / WIDTH;
        key[c] = s as f32 - 1.5;
        hidden[c] = 0.2 + s as f32 * 0.3;
        kernel[c * 4..c * 4 + 4].copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        history[c * HISTORY] = 0.25;
        history[c * HISTORY + 3] = 0.5;
        history[c * HISTORY + 6] = 0.75;
    }
    let o = p::evaluate(Inputs {
        key: &key,
        value: &value,
        hidden: &hidden,
        norm_key: &nk,
        norm_query: &nq,
        norm_conv: &nc,
        kernel: &kernel,
        history: &history,
    })
    .unwrap();
    let residual = [0.593_173_5, 0.893_169_06, 2.127_341_7, 2.427_342];
    let normalized = [0.641_407_43, 0.641_393_3, 1.199_999, 1.199_999];
    for s in 0..4 {
        assert!((o.residual[s * WIDTH] - residual[s]).abs() < 3e-5);
        assert!((o.normalized[s * WIDTH] - normalized[s]).abs() < 3e-5);
    }
}
