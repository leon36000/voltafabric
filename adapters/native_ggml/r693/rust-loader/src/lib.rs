#![allow(unsafe_code)]

use std::{
    collections::BTreeMap,
    ffi::{c_char, c_void, CStr},
    fs::File,
    io::Read,
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    time::Duration,
};
use voltafabric::{
    gguf_expert_slices::{ExpertCheckpointSlice, SelectedExpertSliceReader},
    moe::{Registry, Replica, Stamp},
    native_ggml_cpu::GroupedMatrixBinding,
    native_ledger::NativeBatchLedger,
    routing::{qwen2moe_routes_from_donor, Qwen2MoEDonorSelection},
    session::{Endpoint, Session},
    sha256,
};

const HIDDEN: usize = 32;
const LAYERS: u32 = 2;
const RANK: u32 = 7;
const EPOCH: u64 = 1;
const RUN: &str = "r693-public-decode-v10";
const MODEL: &str = "qwen2moe-synthetic-32x41";
const PLAN: &str = "r693-routed-layer-plan-v10";
const TIMEOUT: Duration = Duration::from_secs(10);
const PROJECTIONS: [&str; 3] = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
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

struct LayerSessions {
    gate: Vec<Session>,
    up: Vec<Session>,
    down: Vec<Session>,
}

struct R693Context {
    layers: Vec<LayerSessions>,
    ledger: NativeBatchLedger,
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

fn tensor_name(layer: u32, projection: &str) -> String {
    format!("blk.{layer}.{projection}")
}

fn load_projection(
    source: &Path,
    worker: &str,
    layer: u32,
    projection: &str,
) -> Result<Session, String> {
    let tensor = tensor_name(layer, projection);
    let mut file = File::open(source).map_err(|e| format!("open source GGUF: {e}"))?;
    let (digest, selected_bytes) = selected_digest(&mut file, &tensor)?;
    let binding = GroupedMatrixBinding {
        model_revision: MODEL.to_owned(),
        layer,
        tensor: tensor.clone(),
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
            &tensor,
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
        "R693 rust_loader tensor={tensor} rank={RANK} resident_order=[40,3] checkpoint_order=[1,0] bytes={selected_bytes}"
    );
    Ok(session)
}

fn make_context(source: &Path, worker: &str, ledger_root: &Path) -> Result<R693Context, String> {
    if !source.is_file() || worker.trim().is_empty() {
        return Err("rank-local source or worker executable missing".into());
    }
    for layer in 0..LAYERS {
        for projection in PROJECTIONS {
            let tensor = tensor_name(layer, projection);
            let file = File::open(source).map_err(|e| format!("open source GGUF: {e}"))?;
            vf_ggml_native_ffi::gguf_tensor_span(&file, &tensor)
                .map_err(|e| format!("pinned tensor {tensor}: {e}"))?;
        }
    }
    let ledger = NativeBatchLedger::open(ledger_root).map_err(|e| e.to_string())?;
    let mut layers = Vec::with_capacity(LAYERS as usize);
    for layer in 0..LAYERS {
        layers.push(LayerSessions {
            gate: vec![load_projection(source, worker, layer, PROJECTIONS[0])?],
            up: vec![load_projection(source, worker, layer, PROJECTIONS[1])?],
            down: vec![load_projection(source, worker, layer, PROJECTIONS[2])?],
        });
    }
    Ok(R693Context { layers, ledger })
}

#[no_mangle]
/// Opens the R693 local CPU Session adapter.
///
/// # Safety
/// Each argument must point to a NUL-terminated UTF-8 string valid for the
/// duration of this call. The strings must not be mutated concurrently.
pub unsafe extern "C" fn vf_r693_open(
    source: *const c_char,
    worker: *const c_char,
    ledger_root: *const c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || worker.is_null() || ledger_root.is_null() {
            return Err("null R693 bridge argument".to_owned());
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
            eprintln!("R693 Rust loader refused: {error}");
            std::ptr::null_mut()
        }
        Err(_) => {
            eprintln!("R693 Rust loader panicked; context not published");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
/// Runs one donor-routed Qwen2MoE FFN batch and atomically publishes its output.
///
/// # Safety
/// The identity pointers must be NUL-terminated UTF-8 strings. The expert,
/// weight, activation, and output pointers must reference arrays of the lengths
/// supplied here; `output` must be writable and must not alias any input array.
/// All referenced memory must remain valid for the duration of this call.
pub unsafe extern "C" fn vf_r693_run_routed_layer(
    opaque: *mut c_void,
    run_id: *const c_char,
    model_revision: *const c_char,
    plan_id: *const c_char,
    layer: u32,
    epoch: u64,
    batch: u64,
    attempt: u64,
    hidden: usize,
    slots: usize,
    tokens: usize,
    expert_ids: *const i32,
    expert_ids_len: usize,
    route_weights: *const f32,
    route_weights_len: usize,
    activations: *const f32,
    activations_len: usize,
    output: *mut f64,
    output_len: usize,
) -> i32 {
    if opaque.is_null()
        || run_id.is_null()
        || model_revision.is_null()
        || plan_id.is_null()
        || expert_ids.is_null()
        || route_weights.is_null()
        || activations.is_null()
        || output.is_null()
        || hidden != HIDDEN
        || slots != 2
        || tokens != 1
        || layer >= LAYERS
        || epoch != EPOCH
        || batch == 0
        || attempt == 0
        || expert_ids_len != slots * tokens
        || route_weights_len != slots * tokens
        || activations_len != hidden * tokens
        || output_len != hidden * tokens
    {
        return 1;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let context = &mut *opaque.cast::<R693Context>();
        let run_id = CStr::from_ptr(run_id).to_str().map_err(|e| e.to_string())?;
        let model_revision = CStr::from_ptr(model_revision)
            .to_str()
            .map_err(|e| e.to_string())?;
        let plan_id = CStr::from_ptr(plan_id)
            .to_str()
            .map_err(|e| e.to_string())?;
        let selected_experts = std::slice::from_raw_parts(expert_ids, expert_ids_len);
        let final_weights = std::slice::from_raw_parts(route_weights, route_weights_len);
        let activation_values = std::slice::from_raw_parts(activations, activations_len);
        if run_id != RUN
            || model_revision != MODEL
            || plan_id != PLAN
            || selected_experts
                .iter()
                .any(|&expert| expert != 3 && expert != 40)
            || activation_values.iter().any(|value| !value.is_finite())
        {
            return Err(
                "VoltaFabric route identity or input is outside the pinned fixture".to_owned(),
            );
        }

        let selection = Qwen2MoEDonorSelection {
            stamp: Stamp {
                run_id: run_id.to_owned(),
                model_revision: model_revision.to_owned(),
                plan_id: plan_id.to_owned(),
                layer,
                batch,
                epoch,
            },
            attempt,
            tokens,
            top_k: slots,
            selected_experts: selected_experts.to_vec(),
            final_weights: final_weights.to_vec(),
        };
        let problem = qwen2moe_routes_from_donor(
            selection,
            hidden,
            vec![3, 40],
            Registry {
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
            BTreeMap::new(),
            BTreeMap::from([(RANK, vec![0, 1])]),
        )
        .map_err(|e| format!("validate donor route batch: {e}"))?;
        let token_inputs = (0..tokens)
            .map(|token| {
                let start = token * hidden;
                activation_values[start..start + hidden].to_vec()
            })
            .collect::<Vec<_>>();
        let layer_sessions = context
            .layers
            .get_mut(layer as usize)
            .ok_or_else(|| "Qwen2MoE layer is outside resident Session set".to_owned())?;
        let down = Session::exec_qwen2moe_ffn_from_donor_routes(
            &mut layer_sessions.gate,
            &mut layer_sessions.up,
            &mut layer_sessions.down,
            &problem,
            &token_inputs,
            &context.ledger,
            TIMEOUT,
            None,
        )
        .map_err(|e| format!("VoltaFabric donor-routed Qwen2MoE FFN: {e}"))?;
        let rows = voltafabric::session::collect_qwen2moe_ffn_output_preweighted_fp32(
            problem.problem(),
            &down,
            problem.attempt(),
        )
        .map_err(|e| format!("VoltaFabric routed Qwen2MoE FP32 Collector: {e}"))?;
        if rows.len() != tokens
            || rows
                .iter()
                .any(|row| row.len() != hidden || row.iter().any(|v| !v.is_finite()))
        {
            return Err("VoltaFabric routed Collector returned an invalid batch".into());
        }
        let staged = rows
            .into_iter()
            .flatten()
            .map(f64::from)
            .collect::<Vec<_>>();
        if staged.len() != output_len || staged.iter().any(|value| !value.is_finite()) {
            return Err("VoltaFabric routed Collector output is incomplete".into());
        }
        std::slice::from_raw_parts_mut(output, output_len).copy_from_slice(&staged);
        Ok::<(), String>(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            eprintln!("R693 VoltaFabric routed layer refused atomically: {error}");
            2
        }
        Err(_) => {
            eprintln!("R693 VoltaFabric routed layer panicked; output was not published");
            3
        }
    }
}

#[no_mangle]
/// Closes a context previously returned by `vf_r693_open`.
///
/// # Safety
/// `opaque` must be null or a live context returned by `vf_r693_open` that has
/// not already been closed.
pub unsafe extern "C" fn vf_r693_close(opaque: *mut c_void) {
    if !opaque.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(opaque.cast::<R693Context>()));
        }));
    }
}
