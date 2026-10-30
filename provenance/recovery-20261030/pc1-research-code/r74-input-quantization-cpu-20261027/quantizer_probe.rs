use serde_json::{json, Value};
use std::hint::black_box;
use std::time::Instant;
use voltafabric::deepseek_v41::{fp8_activation_quantize_reference, MAX_ACTIVATION_LEN};

fn cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0);
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}
fn resolution(id: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(unsafe { libc::clock_getres(id, &mut ts) }, 0);
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}
fn fp_environment() -> Value {
    #[cfg(target_arch = "x86_64")]
    {
        let mut mxcsr = 0u32;
        unsafe {
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        }
        assert_eq!(
            mxcsr & ((3 << 13) | (1 << 15) | (1 << 6)),
            0,
            "RN/FTZ/DAZ contract"
        );
        let sub = black_box(f32::from_bits(1)) * black_box(1.0f32);
        assert_eq!(sub.to_bits(), 1);
        json!({"arch":"x86_64", "mxcsr":mxcsr, "rounding":"nearest_even", "ftz":false,"daz":false,"subnormal_mul_bits":sub.to_bits()})
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        panic!("FP environment implementation required for this architecture")
    }
}
fn floats(row: &Value) -> Vec<f32> {
    row["bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f32::from_bits(u32::try_from(v.as_u64().unwrap()).unwrap()))
        .collect()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u8::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}
fn check_shape_limit() -> Value {
    // Public admission, not a CUDA status test. Inputs allocated outside timings.
    let max = vec![0.0f32; MAX_ACTIVATION_LEN];
    let q = fp8_activation_quantize_reference(&max).unwrap();
    assert_eq!(q.values().len(), MAX_ACTIVATION_LEN);
    assert_eq!(q.scale_codes().len(), MAX_ACTIVATION_LEN / 32);
    assert!(q.values().iter().all(|&b| b == 0));
    let too_long = vec![0.0f32; MAX_ACTIVATION_LEN + 1];
    let err = fp8_activation_quantize_reference(&too_long).unwrap_err();
    assert_eq!(err.code, "INVALID_INPUT");
    let too_long_aligned = vec![0.0f32; MAX_ACTIVATION_LEN + 32];
    assert_eq!(
        fp8_activation_quantize_reference(&too_long_aligned)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );
    json!({"max_len":MAX_ACTIVATION_LEN,"max_accepted":true,"max_plus_one_refused":true,"max_plus_32_refused":true})
}
fn verify(corpus: &Value) {
    let env = fp_environment();
    let mut rows = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for row in corpus["cases"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        assert!(ids.insert(id));
        let x = floats(row);
        match fp8_activation_quantize_reference(&x) {
            Ok(q) => {
                assert!(
                    row["expected"]["error"].is_null(),
                    "unexpected success: {id}"
                );
                assert_eq!(q.values(), bytes(&row["expected"]["codes"]), "codes: {id}");
                assert_eq!(
                    q.scale_codes(),
                    bytes(&row["expected"]["scales"]),
                    "scales: {id}"
                );
                rows.push(json!({"id":id,"codes":q.values(),"scales":q.scale_codes()}));
            }
            Err(e) => {
                let class = match e.code.as_str() {
                    "INVALID_INPUT" => "shape",
                    "NONFINITE_INPUT" => "nonfinite",
                    "NUMERICAL_OVERFLOW" => "exponent",
                    _ => panic!("unknown error {}", e.code),
                };
                assert_eq!(
                    row["expected"]["error"].as_str().unwrap(),
                    class,
                    "error: {id}"
                );
                rows.push(json!({"id":id,"error":class,"public_code":e.code}));
            }
        }
    }
    println!(
        "{}",
        json!({"kind":"actual_public_rust_quantizer_exactness","fp_environment":env,"cases":rows,"shape_limit":check_shape_limit(),"gpu":false})
    );
}
fn benchmark(corpus: &Value) {
    let env = fp_environment();
    let cases = corpus["cases"].as_array().unwrap();
    let ids = corpus["timing_ids"].as_array().unwrap();
    assert_eq!(ids.len(), 8);
    let inputs: Vec<Vec<f32>> = ids
        .iter()
        .map(|id| floats(cases.iter().find(|r| r["id"] == *id).unwrap()))
        .collect();
    let expected: Vec<(Vec<u8>, Vec<u8>)> = ids
        .iter()
        .map(|id| {
            let r = cases.iter().find(|r| r["id"] == *id).unwrap();
            (
                bytes(&r["expected"]["codes"]),
                bytes(&r["expected"]["scales"]),
            )
        })
        .collect();
    assert!(inputs.iter().all(|x| x.len() == 5120));
    assert_eq!(
        inputs
            .iter()
            .map(|x| x.as_ptr())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        8
    );
    let mut groups = Vec::new();
    // Exactly one untimed warmup, fixed before the measured nine groups.
    let warm = fp8_activation_quantize_reference(black_box(&inputs[0])).unwrap();
    assert_eq!(warm.values(), expected[0].0);
    assert_eq!(warm.scale_codes(), expected[0].1);
    drop(warm);
    let mut empty = Vec::new();
    for group in 0..9 {
        let mut control = Vec::with_capacity(256);
        let w = Instant::now();
        let c = cpu_ns();
        for i in 0..256 {
            control.push(black_box(inputs[i % 8].as_slice()));
        }
        let dc = cpu_ns() - c;
        let dw = w.elapsed().as_nanos() as u64;
        assert_eq!(black_box(control).len(), 256);
        empty.push(json!({"wall_ns":dw,"cpu_ns":dc}));
        let mut output = Vec::with_capacity(256);
        let w = Instant::now();
        let c = cpu_ns();
        for i in 0..256 {
            output.push(fp8_activation_quantize_reference(black_box(&inputs[i % 8])).unwrap());
        }
        let dc = cpu_ns() - c;
        let dw = w.elapsed().as_nanos() as u64;
        // All allocations in the quantizer measured; retained output verification/drop outside timing.
        for (i, q) in output.iter().enumerate() {
            assert_eq!(q.values(), expected[i % 8].0);
            assert_eq!(q.scale_codes(), expected[i % 8].1);
        }
        groups.push(json!({"group":group,"vectors":256,"wall_ns":dw,"cpu_ns":dc,"bytes_per_input":20480,"compact_bytes_per_input":5280,"all_values_verified_after_timing":true}));
        drop(output);
    }
    println!(
        "{}",
        json!({"kind":"unchanged_public_rust_quantizer_cpu_cost","fp_environment":env,"groups":groups,"empty_controls":empty,"warmup_vectors":1,"clock_resolution_ns":{"monotonic":resolution(libc::CLOCK_MONOTONIC),"process_cpu":resolution(libc::CLOCK_PROCESS_CPUTIME_ID)},"allocations_included":true,"drop_outside_timing":true,"packing_not_measured":true,"gpu":false})
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3);
    let raw = std::fs::read(&args[2]).unwrap();
    let corpus: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(corpus["schema_version"], 1);
    match args[1].as_str() {
        "verify" => verify(&corpus),
        "bench" => benchmark(&corpus),
        _ => panic!("bad mode"),
    }
}
