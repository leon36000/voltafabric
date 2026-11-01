#![allow(unsafe_code)]

use std::{
    collections::BTreeMap,
    ffi::{c_char, c_void, CStr},
    fs::{self, File},
    io::Read,
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    time::Duration,
};
use voltafabric::{
    gguf_expert_slices::{ExpertCheckpointSlice, SelectedExpertSliceReader},
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::GroupedMatrixBinding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    session::{Endpoint, Session},
    sha256,
};

const HIDDEN: usize = 32;
const RANK: u32 = 7;
const EPOCH: u64 = 9;
const MODEL: &str = "r659-qwen2moe-cpu-fixture";
const TIMEOUT: Duration = Duration::from_secs(10);
const TENSORS: [&str; 3] = [
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_down_exps.weight",
];
const RESIDENTS: [(u32, u32); 2] = [(40, 0), (3, 1)];
const CHECKPOINT_SLICES: [ExpertCheckpointSlice; 2] = [
    ExpertCheckpointSlice {
        resident_expert: 40,
        resident_local: 0,
        checkpoint_expert: 1,
    },
    ExpertCheckpointSlice {
        resident_expert: 3,
        resident_local: 1,
        checkpoint_expert: 0,
    },
];

struct R688Context {
    gate: Vec<Session>,
    up: Vec<Session>,
    down: Vec<Session>,
    ledger: NativeBatchLedger,
    next_batch: u64,
}

fn selected_digest(file: &mut File, tensor: &str) -> Result<(String, u64), String> {
    let span = vf_ggml_native_ffi::gguf_tensor_span(file, tensor)
        .map_err(|e| format!("GGUF span {tensor}: {e}"))?;
    if span.ggml_type != 8 || span.dimensions != [32, 32, 4, 1] || span.bytes == 0 {
        return Err(format!("unexpected pinned Q8_0 shape for {tensor}"));
    }
    let mut reader = SelectedExpertSliceReader::new(file, span.offset, span.bytes, 4, &[1, 0])
        .map_err(|e| format!("selected Q8_0 reader {tensor}: {e}"))?;
    let mut payload = Vec::with_capacity((span.bytes / 2) as usize);
    reader
        .read_to_end(&mut payload)
        .map_err(|e| format!("read selected Q8_0 payload {tensor}: {e}"))?;
    let selected_bytes = u64::try_from(payload.len()).map_err(|e| e.to_string())?;
    Ok((sha256::hex(&payload), selected_bytes))
}

fn load_projection(source: &Path, worker: &str, tensor: &str) -> Result<Session, String> {
    let mut file = File::open(source).map_err(|e| format!("open source GGUF: {e}"))?;
    let (digest, selected_bytes) = selected_digest(&mut file, tensor)?;
    let binding = GroupedMatrixBinding {
        model_revision: MODEL.to_owned(),
        layer: 0,
        tensor: tensor.to_owned(),
        ggml_type: 8,
        rank: RANK,
        epoch: EPOCH,
        rows: HIDDEN,
        columns: HIDDEN,
        residents: RESIDENTS.to_vec(),
        sha256: digest,
    };
    let endpoint = Endpoint {
        rank: RANK,
        argv: vec![worker.to_owned()],
    };
    let mut session = Session::connect(&endpoint, TIMEOUT).map_err(|e| e.to_string())?;
    session
        .load_grouped_matrix_gguf_expert_slices(
            &binding,
            source,
            tensor,
            &CHECKPOINT_SLICES,
            TIMEOUT,
        )
        .map_err(|e| format!("resident GGUF expert-slice load {tensor}: {e}"))?;
    if session.traffic.weight_bytes_sent != selected_bytes {
        return Err(format!(
            "rank-local loader byte count differs for {tensor}: sent {}, expected {}",
            session.traffic.weight_bytes_sent, selected_bytes
        ));
    }
    eprintln!(
        "R688 rust_loader tensor={tensor} rank={RANK} resident_order=[40,3] checkpoint_order=[1,0] bytes={selected_bytes}"
    );
    Ok(session)
}

fn route_problem(batch: u64) -> RoutingProblem {
    RoutingProblem {
        hidden: HIDDEN,
        experts: vec![3, 40],
        routes: vec![vec![
            Choice {
                expert: 40,
                weight: 0.25,
            },
            Choice {
                expert: 3,
                weight: 0.75,
            },
        ]],
        registry: Registry {
            epoch: EPOCH,
            copies: vec![
                Replica {
                    expert: 40,
                    rank: RANK,
                    local: 0,
                },
                Replica {
                    expert: 3,
                    rank: RANK,
                    local: 1,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r688-same-process".into(),
            model_revision: MODEL.into(),
            plan_id: "r688-r659-rank-local-plan".into(),
            layer: 0,
            batch,
            epoch: EPOCH,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(RANK, vec![0, 1])]),
    }
}

fn make_context(source: &Path, worker: &str, ledger_root: &Path) -> Result<R688Context, String> {
    if !source.is_file() || worker.trim().is_empty() {
        return Err("rank-local source or worker executable missing".into());
    }
    for tensor in TENSORS {
        let file = File::open(source).map_err(|e| format!("open source GGUF: {e}"))?;
        vf_ggml_native_ffi::gguf_tensor_span(&file, tensor)
            .map_err(|e| format!("pinned tensor {tensor}: {e}"))?;
    }
    fs::create_dir_all(ledger_root).map_err(|e| format!("create R688 ledger root: {e}"))?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let ledger_path = ledger_root.join(format!("r688-local-ledger-{}-{nonce}", std::process::id()));
    fs::create_dir(&ledger_path).map_err(|e| format!("create R688 ledger: {e}"))?;
    let ledger = NativeBatchLedger::open(&ledger_path).map_err(|e| e.to_string())?;
    Ok(R688Context {
        gate: vec![load_projection(source, worker, TENSORS[0])?],
        up: vec![load_projection(source, worker, TENSORS[1])?],
        down: vec![load_projection(source, worker, TENSORS[2])?],
        ledger,
        next_batch: 1,
    })
}

#[no_mangle]
pub extern "C" fn vf_r688_open(
    source: *const c_char,
    worker: *const c_char,
    ledger_root: *const c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || worker.is_null() || ledger_root.is_null() {
            return Err("null R688 bridge argument".to_owned());
        }
        let source = unsafe { CStr::from_ptr(source) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let worker = unsafe { CStr::from_ptr(worker) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let ledger_root = unsafe { CStr::from_ptr(ledger_root) }
            .to_str()
            .map_err(|e| e.to_string())?;
        make_context(Path::new(source), worker, Path::new(ledger_root))
    }));
    match result {
        Ok(Ok(context)) => Box::into_raw(Box::new(context)).cast(),
        Ok(Err(error)) => {
            eprintln!("R688 Rust loader refused: {error}");
            std::ptr::null_mut()
        }
        Err(_) => {
            eprintln!("R688 Rust loader panicked; context not published");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn vf_r688_run_layer(
    opaque: *mut c_void,
    output: *mut f64,
    output_len: usize,
) -> i32 {
    if opaque.is_null() || output.is_null() || output_len != HIDDEN {
        return 1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let context = &mut *opaque.cast::<R688Context>();
        let batch = context.next_batch;
        context.next_batch = context
            .next_batch
            .checked_add(1)
            .ok_or_else(|| "R688 batch identity overflow".to_owned())?;
        let problem = route_problem(batch);
        let mut input = vec![0.0_f32; HIDDEN];
        input[0] = 1.0;
        let down = Session::exec_qwen2moe_ffn_batch(
            &mut context.gate,
            &mut context.up,
            &mut context.down,
            &problem,
            &[input],
            &context.ledger,
            TIMEOUT,
            None,
        )
        .map_err(|e| format!("VoltaFabric resident Qwen2MoE layer: {e}"))?;
        let rows = voltafabric::session::collect_qwen2moe_ffn_output(&problem, &down)
            .map_err(|e| format!("VoltaFabric Qwen2MoE collector: {e}"))?;
        if rows.len() != 1 || rows[0].len() != HIDDEN {
            return Err("VoltaFabric Qwen2MoE collector returned an unexpected shape".into());
        }
        let target = std::slice::from_raw_parts_mut(output, HIDDEN);
        target.copy_from_slice(&rows[0]);
        Ok::<(), String>(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            eprintln!("R688 VoltaFabric layer refused atomically: {error}");
            2
        }
        Err(_) => {
            eprintln!("R688 VoltaFabric layer panicked; output was not published");
            3
        }
    }
}

#[no_mangle]
pub extern "C" fn vf_r688_close(opaque: *mut c_void) {
    if !opaque.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(opaque.cast::<R688Context>()));
        }));
    }
}
