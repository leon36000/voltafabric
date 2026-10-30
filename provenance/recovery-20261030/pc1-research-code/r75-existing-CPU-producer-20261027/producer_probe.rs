use serde_json::{json, Value};
use std::hint::black_box;
use std::time::Instant;
use voltafabric::deepseek_v41::{
    fp8_activation_quantize_reference, Fp8ActivationQuantized, MAX_ACTIVATION_LEN,
};

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
type CompactFn = unsafe extern "C" fn(*const f32, usize, *mut u8, usize) -> i32;
fn compact(f: CompactFn, x: &[f32]) -> (i32, Vec<u8>) {
    let required = x.len() + x.len() / 32;
    // Caller allocation, canary initialization and the producer's staged allocation/copy
    // are included when this whole function is timed. No packing cost is hidden.
    let mut data = vec![0xa5u8; required + 32];
    let rc = unsafe { f(x.as_ptr(), x.len(), data.as_mut_ptr().add(16), required) };
    (rc, data)
}
fn guards(data: &[u8], n: usize) {
    assert!(data[..16].iter().all(|&b| b == 0xa5));
    assert!(data[16 + n + n / 32..].iter().all(|&b| b == 0xa5));
}
fn candidate_limits(f: CompactFn) -> Value {
    let x = vec![0.0; MAX_ACTIVATION_LEN];
    let q = fp8_activation_quantize_reference(&x).unwrap();
    let (rc, data) = compact(f, &x);
    assert_eq!(rc, 0);
    guards(&data, x.len());
    assert_eq!(&data[16..16 + x.len()], q.values());
    assert_eq!(&data[16 + x.len()..data.len() - 16], q.scale_codes());
    assert!(q.values().iter().all(|&b| b == 0));
    assert!(q.scale_codes().iter().all(|&b| b == 105));
    for n in [MAX_ACTIVATION_LEN + 1, MAX_ACTIVATION_LEN + 32] {
        let x = vec![0.0; n];
        let (rc, data) = compact(f, &x);
        assert_eq!(rc, 1);
        assert!(data.iter().all(|&b| b == 0xa5));
        assert_eq!(
            fp8_activation_quantize_reference(&x).unwrap_err().code,
            "INVALID_INPUT"
        );
    }
    json!({"max_len":MAX_ACTIVATION_LEN,"max_exact":true,"max_plus_one_refused_untouched":true,"max_plus_32_refused_untouched":true})
}
fn verify(corpus: &Value, f: CompactFn) {
    let env = fp_environment();
    let mut rows = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for row in corpus["cases"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        assert!(ids.insert(id));
        let x = floats(row);
        let (rc, data) = compact(f, &x);
        guards(&data, x.len());
        match fp8_activation_quantize_reference(&x) {
            Ok(q) => {
                assert!(
                    row["expected"]["error"].is_null(),
                    "unexpected success: {id}"
                );
                let codes = bytes(&row["expected"]["codes"]);
                let scales = bytes(&row["expected"]["scales"]);
                assert_eq!(q.values(), codes, "reference codes: {id}");
                assert_eq!(q.scale_codes(), scales, "reference scales: {id}");
                assert_eq!(rc, 0, "candidate status: {id}");
                assert_eq!(&data[16..16 + x.len()], q.values(), "candidate codes: {id}");
                assert_eq!(
                    &data[16 + x.len()..data.len() - 16],
                    q.scale_codes(),
                    "candidate scales: {id}"
                );
                rows.push(json!({"id":id,"status":rc,"codes":q.values(),"scales":q.scale_codes(),"candidate_codes":&data[16..16+x.len()],"candidate_scales":&data[16+x.len()..data.len()-16],"guards_untouched":true}));
            }
            Err(e) => {
                let (class, status) = match e.code.as_str() {
                    "INVALID_INPUT" => ("shape", 1),
                    "NONFINITE_INPUT" => ("nonfinite", 6),
                    "NUMERICAL_OVERFLOW" => panic!("unexpected scale overflow: C INTERNAL_ERROR is not an equivalent public status"),
                    _ => panic!("unexpected public error"),
                };
                assert_eq!(row["expected"]["error"].as_str().unwrap(), class);
                assert_eq!(rc, status);
                assert!(
                    data.iter().all(|&b| b == 0xa5),
                    "refusal wrote caller output: {id}"
                );
                rows.push(json!({"id":id,"error":class,"public_code":e.code,"candidate_status":rc,"entire_caller_untouched":true}));
            }
        }
    }
    println!(
        "{}",
        json!({"kind":"existing_R5_CPU_C_ABI_vs_public_reference","fp_environment":env,"cases":rows,"limits":candidate_limits(f),"gpu":false,"producer_fn_address":f as usize})
    );
}
enum Kept {
    Reference(Fp8ActivationQuantized),
    Compact(Vec<u8>),
}
fn benchmark(corpus: &Value, f: CompactFn) {
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
    let reference = black_box(
        fp8_activation_quantize_reference
            as fn(&[f32]) -> voltafabric::Result<Fp8ActivationQuantized>,
    );
    fn check(q: &Kept, expected: &(Vec<u8>, Vec<u8>)) {
        match q {
            Kept::Reference(q) => {
                assert_eq!(q.values(), expected.0);
                assert_eq!(q.scale_codes(), expected.1);
            }
            Kept::Compact(data) => {
                guards(data, 5120);
                assert_eq!(&data[16..5136], expected.0);
                assert_eq!(&data[5136..data.len() - 16], expected.1);
            }
        }
    }
    // Exactly one declared warmup per variant, no tuning or sample replacement.
    let q = Kept::Reference(reference(black_box(&inputs[0])).unwrap());
    check(&q, &expected[0]);
    drop(q);
    let (rc, data) = compact(f, black_box(&inputs[0]));
    assert_eq!(rc, 0);
    let q = Kept::Compact(data);
    check(&q, &expected[0]);
    drop(q);
    let mut samples = Vec::new();
    let mut controls = Vec::new();
    let pairs = [
        ["reference", "compact"],
        ["compact", "reference"],
        ["compact", "reference"],
        ["reference", "compact"],
    ];
    for (pair, order) in pairs.iter().enumerate() {
        for (position, variant) in order.iter().enumerate() {
            let mut control = Vec::with_capacity(256);
            let w = Instant::now();
            let c = cpu_ns();
            for i in 0..256 {
                control.push(black_box(inputs[i % 8].as_slice()));
            }
            let dc = cpu_ns() - c;
            let dw = w.elapsed().as_nanos() as u64;
            assert_eq!(black_box(control).len(), 256);
            controls.push(json!({"pair":pair,"position":position,"wall_ns":dw,"cpu_ns":dc}));
            let mut output = Vec::with_capacity(256);
            let w = Instant::now();
            let c = cpu_ns();
            for i in 0..256 {
                let q = if *variant == "reference" {
                    Kept::Reference(reference(black_box(&inputs[i % 8])).unwrap())
                } else {
                    let (rc, data) = compact(f, black_box(&inputs[i % 8]));
                    assert_eq!(rc, 0);
                    Kept::Compact(data)
                };
                output.push(q);
            }
            let dc = cpu_ns() - c;
            let dw = w.elapsed().as_nanos() as u64;
            for (i, q) in output.iter().enumerate() {
                check(q, &expected[i % 8]);
            }
            samples.push(json!({"pair":pair,"position":position,"variant":variant,"vectors":256,"wall_ns":dw,"cpu_ns":dc,"all_values_verified_after_timing":true,"allocations_included":true}));
            drop(output);
        }
    }
    println!(
        "{}",
        json!({"kind":"existing_R5_transaction_producer_CPU_cost_comparison","fp_environment":env,"samples":samples,"empty_controls":controls,"clock_resolution_ns":{"monotonic":resolution(libc::CLOCK_MONOTONIC),"process_cpu":resolution(libc::CLOCK_PROCESS_CPUTIME_ID)},"warmup_per_variant":1,"inputs":8,"hidden":5120,"producer_fn_address":f as usize,"compact_packing_copy_and_caller_allocation_included":true,"candidate_canary_initialization_included":true,"candidate_staged_drop_inside_call":true,"returned_outputs_drop_outside_timing":true,"reference_no_extra_flattening":true,"gpu":false})
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3);
    let raw = std::fs::read(&args[2]).unwrap();
    let corpus: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(corpus["schema_version"], 1);
    let producer: CompactFn = black_box(vf_lab_quantizer_bridge::vf_lab_quantize);
    match args[1].as_str() {
        "verify" => verify(&corpus, producer),
        "bench" => benchmark(&corpus, producer),
        _ => panic!("bad mode"),
    }
}
