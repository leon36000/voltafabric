//! Persistent local CPU coordinator for Strata-style unweighted expert slots.
//! This is not a live donor callback, GPU synchronization or a distributed lease.
use crate::{
    invalid,
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    process, routing,
    session::{Endpoint, Session},
    strata_ingress,
    strata_native_layout::NativeLayout,
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, Write},
    path::Path,
    time::Duration,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Init {
    pub template: routing::RoutingProblem,
    pub model_experts: Vec<u32>,
    pub index: String,
    pub pack_path: String,
    pub ledger_path: String,
    pub endpoints: Vec<Endpoint>,
    pub bindings: Vec<Binding>,
    /// Limit on total resident payload bytes; excludes protocol buffers and RSS.
    #[serde(default)]
    pub resident_payload_limit: Option<u64>,
    pub timeout_ms: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    EnterBinary,
    Batch {
        batch: u64,
        tokens: usize,
        top_k: usize,
        inputs: Vec<f32>,
        ids: Vec<i32>,
        kinds: Vec<i32>,
    },
    Shutdown,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotRow {
    pub position: usize,
    pub value: Vec<f32>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reply {
    Ready { ranks: Vec<u32> },
    BinaryReady,
    Completed { batch: u64, cpu_rows: Vec<SlotRow> },
    Closed,
    Failed { error: Error },
}

struct Coordinator {
    template: routing::RoutingProblem,
    model_experts: Vec<u32>,
    sessions: Vec<Session>,
    ledger: NativeBatchLedger,
    timeout: Duration,
}
impl Coordinator {
    fn open(init: Init) -> Result<Self> {
        routing::validate(&init.template)?;
        if init.template.hidden != 2560
            || !(1..=4).contains(&init.endpoints.len())
            || init.bindings.is_empty()
            || init.bindings.len() > crate::native_ggml_cpu::MAX_NATIVE_EXPERTS
            || !(1..=60_000).contains(&init.timeout_ms)
            || !Path::new(&init.pack_path).is_absolute()
            || !Path::new(&init.ledger_path).is_absolute()
        {
            return Err(invalid("native bridge init bounds"));
        }
        let model: BTreeSet<_> = init.model_experts.iter().copied().collect();
        if model.is_empty()
            || model.len() != init.model_experts.len()
            || model.len() > 1024
            || init.template.experts.iter().any(|e| !model.contains(e))
        {
            return Err(invalid("native bridge model experts"));
        }
        let mut endpoints = BTreeMap::new();
        for endpoint in &init.endpoints {
            if endpoint.argv.len() != 1
                || !Path::new(&endpoint.argv[0]).is_absolute()
                || endpoints.insert(endpoint.rank, endpoint).is_some()
            {
                return Err(invalid("native bridge requires unique local executables"));
            }
        }
        if endpoints.len() != init.template.resident_order.len() {
            return Err(invalid("native bridge rank coverage"));
        }
        let ledger = NativeBatchLedger::open(Path::new(&init.ledger_path))?;
        // A FIFO must not hang admission before its type can be checked.
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        let mut file = options
            .open(&init.pack_path)
            .map_err(|e| Error::new("NATIVE_PACK_IO", e.to_string()))?;
        let metadata = file
            .metadata()
            .map_err(|e| Error::new("NATIVE_PACK_IO", e.to_string()))?;
        if !metadata.is_file() {
            return Err(invalid("native pack must be a regular file"));
        }
        let layout = NativeLayout::parse_sc117_v3(&init.index, metadata.len())?;
        let limit = init
            .resident_payload_limit
            .unwrap_or(crate::native_ggml_cpu::DEFAULT_NATIVE_PAYLOAD_BYTES);
        if limit == 0 || limit > crate::native_ggml_cpu::MAX_NATIVE_PAYLOAD_BYTES {
            return Err(invalid("native bridge payload budget"));
        }
        let total = init.bindings.iter().try_fold(0_u64, |sum, binding| {
            let span = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
            sum.checked_add(span.bytes)
                .ok_or_else(|| invalid("native bridge payload overflow"))
        })?;
        if total > limit {
            return Err(invalid("native bridge payload budget exceeded"));
        }
        let mut loads = BTreeMap::<u32, Vec<&Binding>>::new();
        for (&rank, order) in &init.template.resident_order {
            if !endpoints.contains_key(&rank)
                || order.is_empty()
                || order.len() > crate::native_ggml_cpu::MAX_NATIVE_EXPERTS
            {
                return Err(invalid("native bridge resident order"));
            }
            let mut owners = Vec::new();
            for &local in order {
                let found: Vec<_> = init
                    .bindings
                    .iter()
                    .filter(|b| b.rank == rank && b.local == local)
                    .collect();
                if found.len() != 1 {
                    return Err(invalid("native bridge binding coverage"));
                }
                let b = found[0];
                if b.model_revision != init.template.stamp.model_revision
                    || b.layer != init.template.stamp.layer
                    || b.epoch != init.template.stamp.epoch
                    || !init
                        .template
                        .registry
                        .copies
                        .iter()
                        .any(|r| r.rank == rank && r.local == local && r.expert == b.expert)
                {
                    return Err(invalid("native bridge binding identity"));
                }
                owners.push(b);
            }
            loads.insert(rank, owners);
        }
        if loads.values().map(Vec::len).sum::<usize>() != init.bindings.len() {
            return Err(invalid("native bridge extra binding"));
        }
        // Validate the complete binding map first, then keep only one verified
        // payload in this process while resident workers receive their weights.
        // Sessions are owned until Ready; errors drop, kill and reap them.
        let timeout = Duration::from_millis(init.timeout_ms);
        let mut sessions = Vec::new();
        for (rank, owners) in loads {
            let mut session = Session::connect(endpoints[&rank], timeout)?;
            for binding in &owners {
                let verified = layout.read_verified_expert(
                    &mut file,
                    binding.layer as usize,
                    u64::from(binding.expert),
                    &binding.index_sha256,
                    &binding.blob_sha256,
                )?;
                session.load_native_ggml(
                    binding,
                    &init.index,
                    metadata.len(),
                    &verified.bytes,
                    timeout,
                )?;
            }
            session.seal_native_ggml(owners.len(), timeout)?;
            sessions.push(session);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let after = file
                .metadata()
                .map_err(|e| Error::new("NATIVE_PACK_IO", e.to_string()))?;
            if (
                metadata.dev(),
                metadata.ino(),
                metadata.len(),
                metadata.mtime(),
                metadata.mtime_nsec(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ) != (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            ) {
                return Err(invalid("native pack changed during admission"));
            }
        }
        Ok(Self {
            template: init.template,
            model_experts: init.model_experts,
            sessions,
            ledger,
            timeout,
        })
    }
    fn batch(
        &mut self,
        batch: u64,
        tokens: usize,
        top_k: usize,
        inputs: &[f32],
        ids: &[i32],
        kinds: &[i32],
    ) -> Result<Vec<SlotRow>> {
        let count = tokens
            .checked_mul(top_k)
            .and_then(|n| n.checked_mul(2560))
            .filter(|n| *n <= 1_048_576)
            .ok_or_else(|| invalid("native bridge output bound"))?;
        let mut template = self.template.clone();
        template.stamp.batch = batch;
        let plan = strata_ingress::admit_strata_unweighted_f64(
            &template,
            &template.stamp,
            template.stamp.layer,
            tokens,
            top_k,
            inputs,
            ids,
            kinds,
            &self.model_experts,
        )?;
        let mut staged = vec![0.0; count];
        plan.execute_native_fp32(&mut self.sessions, &self.ledger, self.timeout, &mut staged)?;
        let mut rows = Vec::new();
        for (position, &kind) in kinds.iter().enumerate() {
            if kind == -1 {
                rows.push(SlotRow {
                    position,
                    value: staged[position * 2560..(position + 1) * 2560].to_vec(),
                });
            }
        }
        Ok(rows)
    }
    fn close(self) -> Result<()> {
        for session in self.sessions {
            let receipt = session.close(self.timeout);
            if receipt.error.is_some() || receipt.exit_code != Some(0) || receipt.killed {
                return Err(invalid("native bridge worker close failed"));
            }
        }
        Ok(())
    }
}
fn send(output: &mut impl Write, reply: &Reply) -> Result<()> {
    process::write_frame(output, &process::encode(reply)?, "native bridge reply")
}

/// VFB1/VFR1 is a bounded little-endian private-pipe protocol. A receiver must
/// stage the entire response and validate coverage before publishing any rows.
fn binary_loop(
    mut coordinator: Coordinator,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    fn u32_at(b: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
    }
    fn u64_at(b: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap())
    }
    fn reply(batch: u64, status: u32, rows: &[SlotRow], layer: Option<u32>) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + rows.len() * (4 + 2560 * 4));
        out.extend_from_slice(if layer.is_some() { b"VFR2" } else { b"VFR1" });
        out.extend_from_slice(&status.to_le_bytes());
        out.extend_from_slice(&batch.to_le_bytes());
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        out.extend_from_slice(&(2560_u32).to_le_bytes());
        if let Some(layer) = layer {
            out.extend_from_slice(&layer.to_le_bytes());
        }
        for row in rows {
            out.extend_from_slice(&(row.position as u32).to_le_bytes());
            for value in &row.value {
                out.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        out
    }
    loop {
        let mut header = [0_u8; 24];
        input
            .read_exact(&mut header)
            .map_err(|e| Error::new("BRIDGE_IO", e.to_string()))?;
        let v2 = &header[..4] == b"VFB2";
        if !v2 && &header[..4] != b"VFB1" {
            return Err(invalid("native bridge binary magic"));
        }
        let layer = v2.then_some(coordinator.template.stamp.layer);
        if v2 {
            let mut requested = [0_u8; 4];
            input
                .read_exact(&mut requested)
                .map_err(|e| Error::new("BRIDGE_IO", e.to_string()))?;
            if u32::from_le_bytes(requested) != coordinator.template.stamp.layer {
                process::write_frame(
                    output,
                    &reply(u64_at(&header, 8), 1, &[], layer),
                    "binary bridge layer rejection",
                )?;
                return Err(invalid("native bridge layer mismatch"));
            }
        }
        let op = u32_at(&header, 4);
        let batch = u64_at(&header, 8);
        let tokens = u32_at(&header, 16) as usize;
        let top_k = u32_at(&header, 20) as usize;
        if op == 2 && batch == 0 && tokens == 0 && top_k == 0 {
            coordinator.close()?;
            process::write_frame(output, &reply(0, 2, &[], layer), "binary bridge shutdown")?;
            return Ok(());
        }
        if op != 1
            || !(1..=8).contains(&tokens)
            || !(1..=64).contains(&top_k)
            || tokens * top_k > 128
        {
            process::write_frame(
                output,
                &reply(batch, 1, &[], layer),
                "binary bridge rejection",
            )?;
            return Err(invalid("native bridge binary dimensions or operation"));
        }
        let slots = tokens * top_k;
        let input_count = tokens * 2560;
        let mut body = vec![0_u8; (input_count + 2 * slots) * 4];
        input
            .read_exact(&mut body)
            .map_err(|e| Error::new("BRIDGE_IO", e.to_string()))?;
        let floats = (0..input_count)
            .map(|i| f32::from_bits(u32_at(&body, i * 4)))
            .collect::<Vec<_>>();
        let ids = (0..slots)
            .map(|i| u32_at(&body, (input_count + i) * 4) as i32)
            .collect::<Vec<_>>();
        let kinds = (0..slots)
            .map(|i| u32_at(&body, (input_count + slots + i) * 4) as i32)
            .collect::<Vec<_>>();
        match coordinator.batch(batch, tokens, top_k, &floats, &ids, &kinds) {
            Ok(rows) => process::write_frame(
                output,
                &reply(batch, 0, &rows, layer),
                "binary bridge reply",
            )?,
            Err(error) => {
                process::write_frame(
                    output,
                    &reply(batch, 1, &[], layer),
                    "binary bridge failure",
                )?;
                return Err(error);
            }
        }
    }
}
/// Serve one bounded line-framed conversation. Any error closes the coordinator;
/// the durable claim remains consumed and no partial batch reply is emitted.
pub fn serve(input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    let init = process::read_message::<Init>(input)?
        .ok_or_else(|| invalid("native bridge must begin with init"))?;
    let mut coordinator = match Coordinator::open(init) {
        Ok(value) => value,
        Err(error) => {
            send(
                output,
                &Reply::Failed {
                    error: error.clone(),
                },
            )?;
            return Err(error);
        }
    };
    send(
        output,
        &Reply::Ready {
            ranks: coordinator.sessions.iter().map(|s| s.rank).collect(),
        },
    )?;
    loop {
        match process::read_message::<Request>(input)? {
            Some(Request::Batch {
                batch,
                tokens,
                top_k,
                inputs,
                ids,
                kinds,
            }) => match coordinator.batch(batch, tokens, top_k, &inputs, &ids, &kinds) {
                Ok(cpu_rows) => send(output, &Reply::Completed { batch, cpu_rows })?,
                Err(error) => {
                    send(
                        output,
                        &Reply::Failed {
                            error: error.clone(),
                        },
                    )?;
                    return Err(error);
                }
            },
            Some(Request::Shutdown) => {
                coordinator.close()?;
                return send(output, &Reply::Closed);
            }
            Some(Request::EnterBinary) => {
                send(output, &Reply::BinaryReady)?;
                return binary_loop(coordinator, input, output);
            }
            _ => return Err(invalid("native bridge unexpected request or EOF")),
        }
    }
}
pub fn main() -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(&mut stdin.lock(), &mut stdout.lock())
}
