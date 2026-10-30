use voltafabric::shared_expert_cpu::{activation, combine, scalar_gate};
#[test]
fn shared_math_and_fail_closed() {
    let w = [0x80, 0x3f, 0x80, 0xbf];
    assert!((scalar_gate(&[2.0, 1.0], &w).unwrap() - 0.7310586).abs() < 1e-6);
    let m = activation(&[0.0, 1.0], &[2.0, 2.0]).unwrap();
    assert_eq!(m[0], 0.0);
    assert!((m[1] - 1.4621172).abs() < 1e-6);
    assert_eq!(
        combine(&[0.5, -1.0], &[0.25, 0.5]).unwrap(),
        vec![0.75, -0.5]
    );
    assert!(scalar_gate(&[f32::NAN, 1.0], &w).is_err());
    assert!(activation(&[f32::MAX], &[f32::MAX]).is_err());
    assert!(combine(&[f64::MAX], &[1.0]).is_err());
}
