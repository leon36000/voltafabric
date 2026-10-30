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
    pub timeout_ms: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
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
            || init.bindings.len() > 4
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
        let mut loads = BTreeMap::<u32, Vec<(Binding, Vec<u8>)>>::new();
        for (&rank, order) in &init.template.resident_order {
            if !endpoints.contains_key(&rank) || order.is_empty() || order.len() > 4 {
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
                let verified = layout.read_verified_expert(
                    &mut file,
                    b.layer as usize,
                    u64::from(b.expert),
                    &b.index_sha256,
                    &b.blob_sha256,
                )?;
                owners.push((b.clone(), verified.bytes));
            }
            loads.insert(rank, owners);
        }
        if loads.values().map(Vec::len).sum::<usize>() != init.bindings.len() {
            return Err(invalid("native bridge extra binding"));
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
        let timeout = Duration::from_millis(init.timeout_ms);
        let mut sessions = Vec::new();
        for (rank, owners) in loads {
            let mut session = Session::connect(endpoints[&rank], timeout)?;
            for (binding, bytes) in &owners {
                session.load_native_ggml(binding, &init.index, metadata.len(), bytes, timeout)?;
            }
            session.seal_native_ggml(owners.len(), timeout)?;
            sessions.push(session);
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
            _ => return Err(invalid("native bridge unexpected request or EOF")),
        }
    }
}
pub fn main() -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(&mut stdin.lock(), &mut stdout.lock())
}
