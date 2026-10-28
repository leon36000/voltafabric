use voltafabric::{deepseek_v41::decode_e8m0,quant::decode_fp4};
fn main() {
    let mut matching_target = 0u32;
    let mut matching_control = 0u32;
    let mut best = f64::INFINITY;
    for s in 0u8..=254 {
        let scale = decode_e8m0(s).expect("finite E8M0 scale");
        for n in 0u8..=15 {
            let weight = decode_fp4(n).expect("E2M1 nibble") * scale;
            if weight.to_bits() == (5.0f64/16.0).to_bits() { matching_target += 1; }
            if weight.to_bits() == (1.0f64/16.0).to_bits() { matching_control += 1; }
            best = best.min((weight - 5.0/16.0).abs());
        }
    }
    assert!(decode_e8m0(255).is_err());
    assert_eq!(matching_target, 0);
    assert!(matching_control > 0);
    assert_eq!(best.to_bits(), (1.0f64/16.0).to_bits());
    println!("{{\"pairs_tested\":4080,\"matches_5_over_16\":{},\"matches_1_over_16\":{},\"best_absolute_error_numerator\":1,\"denominator\":16,\"reserved_scale_rejected\":true}}",matching_target,matching_control);
}
