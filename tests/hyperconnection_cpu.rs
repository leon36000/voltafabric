use voltafabric::hyperconnection_cpu::read;
fn b16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
#[test]
fn symmetric_zero_projection_and_stream_injection() {
    let r = [1.0, 2.0, 1.0, 2.0];
    let g = [1.0; 4];
    let down = b16(&[0.0; 8]);
    let up = b16(&[0.0; 8]);
    let inj = b16(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let o = read(&r, &g, &down, &up, &inj, 2, 2, 2, 1e-5).unwrap();
    let d = (2.5_f64 + 1e-5_f64).sqrt();
    assert!((f64::from(o.mixed[0]) - 0.5 / d).abs() < 1e-6);
    assert!((f64::from(o.mixed[1]) - 1.0 / d).abs() < 1e-6);
    let bf = |x: f32| {
        let bits = x.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    };
    assert_eq!(o.inject, vec![bf((1.0 / d) as f32), bf((2.0 / d) as f32)]);
}
#[test]
fn rejects_shape_nonfinite_and_overflow() {
    let d = b16(&[0.0; 8]);
    let u = b16(&[0.0; 8]);
    let i = b16(&[0.0; 8]);
    assert!(read(&[1.0; 4], &[1.0; 4], &d[..2], &u, &i, 2, 2, 2, 1e-5).is_err());
    assert!(read(&[f32::NAN; 4], &[1.0; 4], &d, &u, &i, 2, 2, 2, 1e-5).is_err());
    assert!(read(&[f32::MAX; 4], &[f32::MAX; 4], &d, &u, &i, 2, 2, 2, 1e-5).is_err());
}
