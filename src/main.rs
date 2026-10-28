//! Bounded JSON CLI. All implemented execution is CPU-only.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    io::{self, Read, Write},
    path::Path,
};
use voltafabric::{
    checkpoint, completion, deepseek_v41, launch, memory, moe, pipeline, planner, preweighted,
    process, qpn2, quant, resident, resident_artifact, routing, schedule, session, traffic, Error,
    Result,
};
const MAX_INPUT: usize = 16 * 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentExportInput {
    source: ResidentExportSource,
    output: std::path::PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentExportSource {
    kind: String,
    request: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekMxfp4ExpertExportRequest {
    artifact_path: std::path::PathBuf,
    expected_sha256: String,
    layer: u32,
    expert: u32,
    rank: u32,
    local: u32,
    epoch: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutInput {
    rows: u64,
    columns: u64,
    alignment: u64,
    global_scale_bytes: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrafficInput {
    hidden: u64,
    counts: Vec<u64>,
    wire: traffic::WireFormat,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectInput {
    problem: moe::Problem,
    completions: Vec<moe::Completion>,
    #[serde(default)]
    expired: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectRoutingInput {
    problem: routing::RoutingProblem,
    completions: Vec<moe::Completion>,
    #[serde(default)]
    expired: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectPreweightedFp32Input {
    problem: routing::RoutingProblem,
    completions: Vec<preweighted::Completion>,
    #[serde(default)]
    expired: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentBindInput {
    weights: resident::ResidentInput,
    dispatch: routing::RankDispatch,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentArtifactMergeInput {
    artifacts: Vec<ResidentArtifactMergeSource>,
    output: std::path::PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidentArtifactMergeSource {
    path: std::path::PathBuf,
    expected_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchInput {
    weights: resident::ResidentInput,
    dispatch: routing::RankDispatch,
    expected_stamp: moe::Stamp,
    options: launch::Options,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputInput {
    launch: LaunchInput,
    output: completion::RawOutput,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFp16OutputInput {
    launch: LaunchInput,
    output: RawFp16OutputMetadata,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFp16CompressedSeriesInput {
    series: session::CompressedSeriesInput,
    rows: usize,
    columns: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFp16OutputMetadata {
    stamp: moe::Stamp,
    rank: u32,
    rows: usize,
    columns: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fp16CollectInput {
    problem: routing::RoutingProblem,
    expected_attempt: u64,
    outputs: Vec<completion::RoutedRawOutput>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleDomainInput {
    scales: Vec<u8>,
    gscales: Vec<f32>,
    k: u64,
}
const MAX_V41_POSITIONS: usize = 4096;
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
enum DeepseekV41Input {
    Mxfp4Values {
        rows: u64,
        cols: u64,
        codes: Vec<u8>,
        scales: Vec<u8>,
        positions: Vec<[u64; 2]>,
    },
    Mxfp4Matvec {
        rows: u64,
        cols: u64,
        codes: Vec<u8>,
        scales: Vec<u8>,
        input_bits: Vec<u64>,
    },
    PreDownActivation {
        gate_bits: Vec<u32>,
        up_bits: Vec<u32>,
        #[serde(default)]
        routing_weight_bits: Option<u32>,
    },
    Fp8ActQuantReference {
        input_bits: Vec<u32>,
    },
}
fn deepseek_v41_ops(input: DeepseekV41Input) -> Result<Value> {
    let scope = json!({"scope":"CPU_FORMAT_OPERATOR_ONLY","model":deepseek_v41::MODEL,
        "revision":deepseek_v41::REVISION,"gpu_executed":false,"cuda_executed":false,
        "gpu_kernel_claim":false,"full_expert_forward":false,"full_moe":false,"full_model":false});
    match input {
        DeepseekV41Input::Mxfp4Values {
            rows,
            cols,
            codes,
            scales,
            positions,
        } => {
            if positions.is_empty() || positions.len() > MAX_V41_POSITIONS {
                return Err(Error::new("INVALID_INPUT", "positions must be 1..=4096"));
            }
            let matrix = deepseek_v41::Mxfp4Matrix::new(rows, cols, codes, scales)?;
            if positions
                .iter()
                .any(|&[r, c]| r >= matrix.rows() || c >= matrix.cols())
            {
                return Err(Error::new(
                    "INVALID_INPUT",
                    "MXFP4 coordinate outside the logical matrix",
                ));
            }
            let values = positions
                .iter()
                .map(|&[r, c]| {
                    let (nibble, scale) = matrix.codes_at(r, c)?;
                    let v = matrix.value(r, c)?;
                    Ok(json!({"row":r,"col":c,"nibble":nibble,"scale_code":scale,
                        "value":v,"f64_bits":format!("{:016x}", v.to_bits())}))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(
                json!({"capability":scope,"op":"mxfp4-values","evidence_level":"CPU_F64_EXACT_DYADIC",
                "summary":matrix.summary(),"values":values}),
            )
        }
        DeepseekV41Input::Mxfp4Matvec {
            rows,
            cols,
            codes,
            scales,
            input_bits,
        } => {
            let matrix = deepseek_v41::Mxfp4Matrix::new(rows, cols, codes, scales)?;
            let input: Vec<f64> = input_bits.into_iter().map(f64::from_bits).collect();
            let output = matrix.matvec_reference(&input)?;
            Ok(json!({"capability":scope,"op":"mxfp4-matvec",
                "evidence_level":"CPU_FP64_REFERENCE",
                "reference_work_limit_macs":deepseek_v41::MAX_MXFP4_REFERENCE_MACS,
                "dense_weights_materialized":false,
                "output_f64_bits":output.iter().map(|x| format!("{:016x}",x.to_bits())).collect::<Vec<_>>()}))
        }
        DeepseekV41Input::PreDownActivation {
            gate_bits,
            up_bits,
            routing_weight_bits,
        } => {
            let gate: Vec<f32> = gate_bits.into_iter().map(f32::from_bits).collect();
            let up: Vec<f32> = up_bits.into_iter().map(f32::from_bits).collect();
            let weight = routing_weight_bits.map(f32::from_bits);
            let out = deepseek_v41::pre_down_activation(&gate, &up, weight)?;
            Ok(
                json!({"capability":scope,"op":"pre-down-activation","evidence_level":"CPU_FP32",
                "swiglu_limit":deepseek_v41::SWIGLU_LIMIT,"precast_dtype":"F32",
                "bf16_cast":false,"activation_quant":false,"down_gemm":false,
                "activation_bits":out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()}),
            )
        }
        DeepseekV41Input::Fp8ActQuantReference { input_bits } => {
            let input: Vec<f32> = input_bits.into_iter().map(f32::from_bits).collect();
            let quantized = deepseek_v41::fp8_activation_quantize_reference(&input)?;
            Ok(json!({"capability":scope,"op":"fp8-act-quant-reference",
                "evidence_level":"CPU_FP32_E4M3FN_UE8M0_REFERENCE",
                "block_size":deepseek_v41::FP8_ACTIVATION_BLOCK,
                "input_domain":"finite FP32 bits; exact BF16 values are included",
                "gemm_executed":false,"e4m3_bytes":quantized.values(),
                "scale_codes":quantized.scale_codes()}))
        }
    }
}
fn decode<T: serde::de::DeserializeOwned>(b: &[u8]) -> Result<T> {
    serde_json::from_slice(b).map_err(|e| Error::new("JSON_INPUT", e.to_string()))
}
fn resident_completion_value(
    launch_input: LaunchInput,
    output: completion::RawOutput,
) -> Result<Value> {
    let stack = resident::pack(&launch_input.weights)?;
    let prepared = launch::prepare(
        &stack,
        &launch_input.dispatch,
        &launch_input.expected_stamp,
        &launch_input.options,
    )?;
    let receipts = completion::from_output(&prepared, &output)?;
    Ok(json!({
        "completions":receipts,
        "evidence_level":"CPU_OUTPUT_REPACK_ONLY",
        "cuda_executed":false
    }))
}

fn resident_completion_raw_fp16_parts(
    bytes: &[u8],
    header_end: usize,
    input: RawFp16OutputInput,
) -> Result<Value> {
    let value_count = input
        .output
        .rows
        .checked_mul(input.output.columns)
        .filter(|&count| count <= completion::MAX_OUTPUT_VALUES)
        .ok_or_else(|| {
            Error::new(
                "CAPACITY_EXCEEDED",
                "raw FP16 completion value count overflows or exceeds its configured bound",
            )
        })?;
    if input.output.rows > 4096 || input.output.columns == 0 || input.output.columns > 65_536 {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "raw FP16 completion rows or columns exceed the admitted bound",
        ));
    }
    let body_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::new("OVERFLOW", "raw FP16 completion body length overflow"))?;
    let body_start = header_end + 1;
    let actual_body_bytes = bytes.len() - body_start;
    if actual_body_bytes < body_bytes {
        return Err(Error::new(
            "RAW_OUTPUT_TRUNCATED",
            "binary FP16 body ended before its declared rows and columns",
        ));
    }
    if actual_body_bytes > body_bytes {
        return Err(Error::new(
            "RAW_OUTPUT_TRAILING_DATA",
            "binary FP16 body has bytes after its declared rows and columns",
        ));
    }
    let fp16_bits = bytes[body_start..]
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    resident_completion_value(
        input.launch,
        completion::RawOutput {
            stamp: input.output.stamp,
            rank: input.output.rank,
            rows: input.output.rows,
            columns: input.output.columns,
            fp16_bits,
        },
    )
}

fn compressed_series_raw_fp16_parts(
    bytes: &[u8],
    header_end: usize,
    input: RawFp16CompressedSeriesInput,
) -> Result<session::CompressedSeriesReport> {
    if !input.series.inputs.is_empty() || input.series.inputs_fp16_bits.is_some() {
        return Err(Error::new(
            "INVALID_INPUT",
            "raw FP16 series metadata must not also contain activation values",
        ));
    }
    if input.rows > 4096
        || input.columns == 0
        || input.columns > 65_536
        || input.rows != input.series.routing.routes.len()
        || input.columns != input.series.routing.hidden
    {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "raw FP16 series rows or columns differ from the admitted routing dimensions",
        ));
    }
    let value_count = input
        .rows
        .checked_mul(input.columns)
        .ok_or_else(|| Error::new("OVERFLOW", "raw FP16 series value count overflow"))?;
    let body_bytes = value_count
        .checked_mul(std::mem::size_of::<u16>())
        .filter(|&size| size <= session::MAX_RAW_ACTIVATION_BYTES)
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "raw FP16 series body exceeds its bounded activation limit",
            )
        })?;
    let body_start = header_end
        .checked_add(1)
        .filter(|&start| start <= bytes.len())
        .ok_or_else(|| Error::new("RAW_INPUT_TRUNCATED", "raw FP16 header has no body"))?;
    let actual_body_bytes = bytes.len() - body_start;
    if actual_body_bytes < body_bytes {
        return Err(Error::new(
            "RAW_INPUT_TRUNCATED",
            "binary FP16 series body ended before its declared rows and columns",
        ));
    }
    if actual_body_bytes > body_bytes {
        return Err(Error::new(
            "RAW_INPUT_TRAILING_DATA",
            "binary FP16 series body has bytes after its declared rows and columns",
        ));
    }

    let mut rows = Vec::new();
    rows.try_reserve_exact(input.rows)
        .map_err(|_| Error::new("CAPACITY_EXCEEDED", "cannot allocate raw FP16 rows"))?;
    let mut values = bytes[body_start..].chunks_exact(2);
    for _ in 0..input.rows {
        let mut row = Vec::new();
        row.try_reserve_exact(input.columns)
            .map_err(|_| Error::new("CAPACITY_EXCEEDED", "cannot allocate raw FP16 row"))?;
        for _ in 0..input.columns {
            let pair = values
                .next()
                .ok_or_else(|| Error::new("RAW_INPUT_TRUNCATED", "raw FP16 value missing"))?;
            row.push(u16::from_le_bytes([pair[0], pair[1]]));
        }
        rows.push(row);
    }
    if values.next().is_some() {
        return Err(Error::new(
            "RAW_INPUT_TRAILING_DATA",
            "binary FP16 series body contains an undeclared value",
        ));
    }
    let mut series = input.series;
    series.inputs_fp16_bits = Some(rows);
    session::run_compressed_series(&series)
}

fn read_bounded_artifact(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)
        .map_err(|error| Error::new("ARTIFACT_IO", format!("could not open artifact: {error}")))?;
    let metadata = file.metadata().map_err(|error| {
        Error::new(
            "ARTIFACT_IO",
            format!("could not inspect artifact: {error}"),
        )
    })?;
    if !metadata.is_file() || metadata.len() > maximum as u64 {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "DeepSeek expert artifact is not a regular file or exceeds the 64 MiB bound",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .map_err(|error| {
            Error::new(
                "ARTIFACT_IO",
                format!("could not reserve bounded artifact buffer: {error}"),
            )
        })?;
    let mut limited = file.take(maximum as u64 + 1);
    limited
        .read_to_end(&mut bytes)
        .map_err(|error| Error::new("ARTIFACT_IO", format!("could not read artifact: {error}")))?;
    if bytes.len() > maximum {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "DeepSeek expert artifact grew beyond the 64 MiB bound while reading",
        ));
    }
    Ok(bytes)
}

fn read_resident_merge_artifact(path: &Path, expected_sha256: &str) -> Result<Vec<u8>> {
    const MAX_ARTIFACT_BYTES: usize = 128 * 1024 * 1024 + 8 * 1024 * 1024 + 8;
    if !voltafabric::sha256::is_hex_digest(expected_sha256) {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "expected resident artifact SHA-256 must be 64 lowercase hexadecimal characters",
        ));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|error| Error::new("ARTIFACT_IO", format!("cannot open merge input: {error}")))?;
    let metadata = file
        .metadata()
        .map_err(|error| Error::new("ARTIFACT_IO", format!("cannot stat merge input: {error}")))?;
    if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES as u64 {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "resident merge input must be a regular artifact within the 136 MiB bound",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .map_err(|error| {
            Error::new(
                "ARTIFACT_IO",
                format!("cannot reserve merge input: {error}"),
            )
        })?;
    file.take(MAX_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| Error::new("ARTIFACT_IO", format!("cannot read merge input: {error}")))?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "resident merge input grew beyond its bounded read limit",
        ));
    }
    if voltafabric::sha256::hex(&bytes) != expected_sha256 {
        return Err(Error::new(
            "ARTIFACT_DIGEST_MISMATCH",
            "resident merge input differs from its declared SHA-256",
        ));
    }
    Ok(bytes)
}
fn value<T: Serialize>(v: T) -> Result<CliPayload> {
    serde_json::to_value(v)
        .map(CliPayload::Json)
        .map_err(|e| Error::new("JSON_OUTPUT", e.to_string()))
}
// Large owned snapshots bypass serde_json::Value and a giant output String.
#[derive(Serialize)]
#[serde(untagged)]
enum CliPayload {
    Json(Value),
    Imported(checkpoint::Imported),
    Resident(resident::ResidentStack),
    Artifact(resident_artifact::ArtifactReceipt),
}
impl CliPayload {
    fn reported_success(&self) -> bool {
        match self {
            Self::Json(v) => v.get("ok").and_then(Value::as_bool).unwrap_or(true),
            _ => true,
        }
    }
}
#[derive(Serialize)]
struct Success {
    ok: bool,
    result: CliPayload,
    schema_version: u8,
}
fn run() -> Result<CliPayload> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err(Error::new("USAGE","vf <capabilities|layout|memory|moe|dispatch|collect|collect-routing|collect-fp16|preweighted-fp32-contract|collect-preweighted-fp32|dequant|sm70-scale-domain|deepseek-v41-ops|routing|rank-run|session-run|session-run-mxfp4|session-run-compressed|qpn2-pack|qpn2-unpack|qpn2-dequant|resident-import|resident-inspect|resident-pack|resident-bind|resident-launch|resident-completions|resident-export|resident-artifact-merge|traffic|plan|pipeline|schedule> [input.json|-]"));
    }
    let cmd = args[0]
        .to_str()
        .ok_or_else(|| Error::new("USAGE", "non-UTF8 command"))?;
    if cmd == "capabilities" {
        if args.len() != 1 {
            return Err(Error::new("USAGE", "capabilities takes no input"));
        }
        return value(
            json!({"cpu_reference":true,"cpu_rank_workers":true,"cpu_process_workers":true,"quantized_local_slot_adapter":true,"qpn2_cpu_codec":true,"metadata_only_collection":true,"cpu_preweighted_fp32_collection":true,"cpu_mxfp4_r9_session":true,"native_checkpoint_import":true,"owned_quantized_buffers":true,"host_launch_metadata":true,"compact_output_receipts":true,"fp16_routing_collection":true,"compressed_resident_artifact":true,"analytical_planner":true,"raw_fp16_completion_cli":true,"raw_fp16_compressed_series_cli":true,"cuda_compiled":false,"cuda_executed":false,"pcie_tested":false,"rdma_tested":false,"live_cloud_workers_used":false,"performance_claim":"NONE"}),
        );
    }
    let reader: Box<dyn Read> = if args.len() == 2 && args[1] != "-" {
        Box::new(
            std::fs::File::open(Path::new(&args[1]))
                .map_err(|e| Error::new("INPUT_IO", e.to_string()))?,
        )
    } else {
        Box::new(io::stdin())
    };
    let mut bytes = Vec::new();
    reader
        .take(MAX_INPUT as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::new("INPUT_IO", e.to_string()))?;
    if bytes.len() > MAX_INPUT {
        return Err(Error::new("INPUT_TOO_LARGE", "maximum JSON input 16 MiB"));
    }
    match cmd {
        "sm70-scale-domain" => {
            let p: ScaleDomainInput = decode(&bytes)?;
            quant::validate_sm70_scale_domain(&p.scales, &p.gscales, p.k)?;
            value(json!({"scope":"CPU_SCALE_ADMISSION_ONLY","cuda_executed":false}))
        }
        "deepseek-v41-ops" => value(deepseek_v41_ops(decode(&bytes)?)?),
        "layout" => {
            let p: LayoutInput = decode(&bytes)?;
            value(memory::nvfp4_layout(
                p.rows,
                p.columns,
                p.alignment,
                p.global_scale_bytes,
            )?)
        }
        "memory" => value(decode::<memory::Budget>(&bytes)?.check()?),
        "moe" => {
            let p: moe::Problem = decode(&bytes)?;
            let reference = moe::reference(&p)?;
            let partitioned = moe::parallel(&p)?;
            let max_abs = reference
                .iter()
                .flatten()
                .zip(partitioned.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            value(
                json!({"evidence_level":"cpu_f64","reference":reference,"partitioned":partitioned,"max_abs_error":max_abs,"stamp":p.stamp}),
            )
        }
        "dispatch" => value(moe::dispatch(&decode::<moe::Problem>(&bytes)?)?),
        "collect" => {
            let p: CollectInput = decode(&bytes)?;
            let mut c = moe::Collector::new(&p.problem)?;
            if p.expired {
                c.expire();
            }
            for r in p.completions {
                c.accept(r)?;
            }
            value(c.finish()?)
        }
        "collect-routing" => {
            let p: CollectRoutingInput = decode(&bytes)?;
            let mut collector = moe::Collector::from_routing(&p.problem)?;
            if p.expired {
                collector.expire();
            }
            for completion in p.completions {
                collector.accept(completion)?;
            }
            value(collector.finish()?)
        }
        "preweighted-fp32-contract" => {
            let p: routing::RoutingProblem = decode(&bytes)?;
            value(preweighted::Collector::new(&p)?.contract())
        }
        "collect-preweighted-fp32" => {
            let p: CollectPreweightedFp32Input = decode(&bytes)?;
            let mut collector = preweighted::Collector::new(&p.problem)?;
            let contract = collector.contract().clone();
            if p.expired {
                collector.expire();
            }
            for completion in p.completions {
                collector.accept(completion)?;
            }
            let bits: Vec<Vec<u32>> = collector
                .finish()?
                .into_iter()
                .map(|row| row.into_iter().map(f32::to_bits).collect())
                .collect();
            value(
                json!({"contract":contract,"fp32_bits":bits,"scope":"CPU_RECEIPT_COLLECTION_ONLY"}),
            )
        }
        "collect-fp16" => {
            let p: Fp16CollectInput = decode(&bytes)?;
            if p.expected_attempt == 0 {
                return Err(Error::new(
                    "INVALID_INPUT",
                    "expected FP16 collection attempt must be positive",
                ));
            }
            let mut collector = moe::Collector::from_routing(&p.problem)?;
            let mut accepted_rows = 0usize;
            let mut fp16_payload_bytes = 0usize;
            for output in p.outputs {
                fp16_payload_bytes = fp16_payload_bytes
                    .checked_add(
                        output
                            .fp16_bits
                            .len()
                            .checked_mul(std::mem::size_of::<u16>())
                            .ok_or_else(|| {
                                Error::new("OVERFLOW", "FP16 completion payload size overflow")
                            })?,
                    )
                    .ok_or_else(|| {
                        Error::new("OVERFLOW", "FP16 completion payload sum overflow")
                    })?;
                accepted_rows = accepted_rows
                    .checked_add(completion::accept_routed_output(
                        p.expected_attempt,
                        output,
                        &mut collector,
                    )?)
                    .ok_or_else(|| Error::new("OVERFLOW", "accepted row count overflow"))?;
            }
            let output = collector.finish()?;
            value(json!({
                "output": output,
                "accepted_rows": accepted_rows,
                "fp16_payload_bytes": fp16_payload_bytes,
                "evidence_level": "CPU_FP16_ROUTING_COLLECTION",
                "cuda_executed": false
            }))
        }
        "dequant" => value(quant::dequantize(&decode(&bytes)?)?),
        "rank-run" => {
            let p: process::ProcessProblem = decode(&bytes)?;
            let executable =
                std::env::current_exe().map_err(|e| Error::new("WORKER_IO", e.to_string()))?;
            let worker = executable.with_file_name(if cfg!(windows) {
                "vf-rank-worker.exe"
            } else {
                "vf-rank-worker"
            });
            value(process::run(&p, &worker)?)
        }
        "session-run" => value(session::run_series(&decode(&bytes)?)?),
        "session-run-mxfp4" => value(session::mxfp4::run_series(decode(&bytes)?)?),
        "session-run-compressed" => {
            let raw_header = bytes
                .iter()
                .position(|byte| *byte == b'\n')
                .and_then(|header_end| {
                    decode::<RawFp16CompressedSeriesInput>(&bytes[..header_end])
                        .ok()
                        .map(|input| (header_end, input))
                });
            if let Some((header_end, input)) = raw_header {
                value(compressed_series_raw_fp16_parts(&bytes, header_end, input)?)
            } else {
                value(session::run_compressed_series(&decode(&bytes)?)?)
            }
        }
        "routing" => value(routing::dispatch(&decode(&bytes)?)?),
        "qpn2-pack" => value(qpn2::pack(&decode(&bytes)?)?),
        "qpn2-unpack" => value(qpn2::unpack(&decode(&bytes)?)?),
        "qpn2-dequant" => value(qpn2::dequantize(&decode(&bytes)?)?),
        "resident-completions" => {
            let raw_header = bytes
                .iter()
                .position(|byte| *byte == b'\n')
                .and_then(|header_end| {
                    decode::<RawFp16OutputInput>(&bytes[..header_end])
                        .ok()
                        .map(|input| (header_end, input))
                });
            if let Some((header_end, input)) = raw_header {
                value(resident_completion_raw_fp16_parts(
                    &bytes, header_end, input,
                )?)
            } else {
                let p: OutputInput = decode(&bytes)?;
                value(resident_completion_value(p.launch, p.output)?)
            }
        }
        "resident-launch" => {
            let p: LaunchInput = decode(&bytes)?;
            let stack = resident::pack(&p.weights)?;
            value(launch::prepare(
                &stack,
                &p.dispatch,
                &p.expected_stamp,
                &p.options,
            )?)
        }
        "resident-import" => Ok(CliPayload::Imported(checkpoint::load(&decode(&bytes)?)?)),
        "resident-pack" => Ok(CliPayload::Resident(resident::pack(&decode(&bytes)?)?)),
        "resident-inspect" => {
            let imported = checkpoint::load(&decode(&bytes)?)?;
            let stack = &imported.stack;
            value(json!({
                "scope":"CPU_COMPRESSED_INSPECTION_ONLY", "cuda_executed":false,
                "model_revision":stack.model_revision(), "layer":stack.layer(),
                "rank":stack.rank(), "epoch":stack.epoch(),
                "physical_experts":stack.physical_experts(),
                "resident_order":stack.resident_order(), "shape":stack.shape(),
                "payload_bytes":stack.payload_bytes(),
                "io_bytes":imported.io_bytes, "header_bytes":imported.header_bytes,
                "file_bytes":imported.file_bytes,
                "plane_elements":{
                    "codes13":stack.codes13().len(), "codes2":stack.codes2().len(),
                    "scales13":stack.scales13().len(), "scales2":stack.scales2().len(),
                    "gscales13":stack.gscales13().len(), "gscales2":stack.gscales2().len()
                }
            }))
        }
        "resident-bind" => {
            let p: ResidentBindInput = decode(&bytes)?;
            let stack = resident::pack(&p.weights)?;
            stack.bind_dispatch(&p.dispatch)?;
            value(
                json!({"bound":true,"rank":stack.rank(),"epoch":stack.epoch(),
                "payload_bytes":stack.payload_bytes(),"empty":stack.is_empty(),
                "evidence_level":"CPU_COMPRESSED_BUFFERS","cuda_executed":false}),
            )
        }
        "resident-export" => {
            let input: ResidentExportInput = decode(&bytes)?;
            let stack = match input.source.kind.as_str() {
                "checkpoint" => {
                    let request: checkpoint::Request = serde_json::from_value(input.source.request)
                        .map_err(|err| Error::new("JSON_INPUT", err.to_string()))?;
                    checkpoint::load(&request)?.stack
                }
                "resident" => {
                    let request: resident::ResidentInput =
                        serde_json::from_value(input.source.request)
                            .map_err(|err| Error::new("JSON_INPUT", err.to_string()))?;
                    resident::pack(&request)?
                }
                "deepseek-mxfp4-expert" => {
                    let request: DeepseekMxfp4ExpertExportRequest =
                        serde_json::from_value(input.source.request)
                            .map_err(|err| Error::new("JSON_INPUT", err.to_string()))?;
                    let bytes = read_bounded_artifact(
                        &request.artifact_path,
                        deepseek_v41::MAX_MXFP4_EXPERT_ARTIFACT_BYTES as usize,
                    )?;
                    deepseek_v41::Mxfp4Expert::from_safetensors_expert_artifact(
                        bytes,
                        request.layer,
                        request.expert,
                        &request.expected_sha256,
                    )?
                    .to_qpn2_resident_stack(
                        request.rank,
                        request.local,
                        request.epoch,
                    )?
                }
                _ => {
                    return Err(Error::new(
                        "INVALID_INPUT",
                        "source.kind must be checkpoint, resident or deepseek-mxfp4-expert",
                    ))
                }
            };
            Ok(CliPayload::Artifact(resident_artifact::export(
                &stack,
                &input.output,
            )?))
        }
        "resident-artifact-merge" => {
            const MAX_INPUTS: usize = 64;
            const MAX_TOTAL_SOURCE_BYTES: u64 = 160 * 1024 * 1024;
            let input: ResidentArtifactMergeInput = decode(&bytes)?;
            if input.artifacts.is_empty() || input.artifacts.len() > MAX_INPUTS {
                return Err(Error::new(
                    "ARTIFACT_INVALID",
                    "resident artifact merge requires between 1 and 64 inputs",
                ));
            }
            let mut total_source_bytes = 0u64;
            for source in &input.artifacts {
                let metadata = std::fs::symlink_metadata(&source.path).map_err(|error| {
                    Error::new("ARTIFACT_IO", format!("cannot stat merge input: {error}"))
                })?;
                if !metadata.file_type().is_file() {
                    return Err(Error::new(
                        "ARTIFACT_INVALID",
                        "resident merge input must be a regular non-symlink file",
                    ));
                }
                total_source_bytes = total_source_bytes
                    .checked_add(metadata.len())
                    .ok_or_else(|| Error::new("ARTIFACT_INVALID", "merge input size overflow"))?;
                if total_source_bytes > MAX_TOTAL_SOURCE_BYTES {
                    return Err(Error::new(
                        "ARTIFACT_INVALID",
                        "resident merge inputs exceed the aggregate 160 MiB read bound",
                    ));
                }
            }
            let mut owned = Vec::with_capacity(input.artifacts.len());
            for source in &input.artifacts {
                owned.push(read_resident_merge_artifact(
                    &source.path,
                    &source.expected_sha256,
                )?);
            }
            let borrowed: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            let stack = resident_artifact::merge(&borrowed)?;
            Ok(CliPayload::Artifact(resident_artifact::export(
                &stack,
                &input.output,
            )?))
        }
        "traffic" => {
            let p: TrafficInput = decode(&bytes)?;
            value(traffic::wire_bytes(p.hidden, &p.counts, p.wire)?)
        }
        "plan" => value(planner::plan(&decode(&bytes)?)?),
        "pipeline" => value(pipeline::partition(&decode(&bytes)?)?),
        "schedule" => value(schedule::schedule(&decode::<Vec<schedule::Job>>(&bytes)?)?),
        _ => Err(Error::new("USAGE", format!("unsupported command: {cmd}"))),
    }
}
fn main() {
    let stdout = io::stdout();
    let mut writer = io::BufWriter::with_capacity(64 * 1024, stdout.lock());
    let (written, code) = match run() {
        Ok(v) => {
            let success = v.reported_success();
            let out = Success {
                ok: success,
                result: v,
                schema_version: 1,
            };
            (
                serde_json::to_writer_pretty(&mut writer, &out),
                if success { 0 } else { 2 },
            )
        }
        Err(e) => (
            serde_json::to_writer_pretty(
                &mut writer,
                &json!({"schema_version":1,"ok":false,"error":e}),
            ),
            2,
        ),
    };
    if let Err(error) = written
        .map_err(io::Error::other)
        .and_then(|_| {
            writer.write_all(
                b"
",
            )
        })
        .and_then(|_| writer.flush())
    {
        eprintln!("JSON output failed: {error}");
        std::process::exit(1);
    }
    std::process::exit(code);
}
