//! Actual SC117 expert eviction/rebinding check on one local CPU session.
use std::{
    collections::BTreeMap,
    fs::File,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    sc117_expert_origin::Origin,
    session::{Endpoint, NativeEviction, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
fn hash(values: &[Vec<f64>]) -> String {
    sha256::hex(
        &values
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn run() -> Result<(), E> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: PACK INDEX SHARD WORKER LEDGER".into());
    }
    let mut pack = File::open(&args[1])?;
    let index = std::fs::read_to_string(&args[2])?;
    if sha256::hex(index.as_bytes()) != INDEX_SHA {
        return Err("index digest".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let mut origin = Origin::open(File::open(&args[3])?, &layout)?;
    let ids = [434u32, 308, 37, 309];
    let mut blobs = Vec::new();
    let mut bindings = Vec::new();
    for (local, &expert) in ids.iter().enumerate() {
        let digest = origin.expert_digest(&layout, 0, expert.into())?;
        let blob = layout.read_verified_expert(&mut pack, 0, expert.into(), INDEX_SHA, &digest)?;
        bindings.push(Binding {
            model_revision: "sc117-local-observation".into(),
            rank: 2,
            layer: 0,
            expert,
            local: local as u32,
            epoch: 1,
            index_sha256: INDEX_SHA.into(),
            blob_sha256: digest,
        });
        blobs.push(blob.bytes);
    }
    let ledger = NativeBatchLedger::open(Path::new(&args[5]))?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r463-eviction-{}-{attempt}", std::process::id());
    let timeout = Duration::from_secs(30);
    let mut session = Session::connect(
        &Endpoint {
            rank: 2,
            argv: vec![args[4].clone()],
        },
        timeout,
    )?;
    for i in 0..3 {
        session.load_native_ggml(
            &bindings[i],
            &index,
            pack.metadata()?.len(),
            &blobs[i],
            timeout,
        )?;
    }
    session.seal_native_ggml(3, timeout)?;
    let stamp = |batch| Stamp {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        plan_id: "r463-local-cpu".into(),
        layer: 0,
        batch,
        epoch: 1,
    };
    let problem = |batch, local: usize, order: Vec<u32>| RoutingProblem {
        hidden: 2560,
        experts: order.iter().map(|&i| ids[i as usize]).collect(),
        routes: vec![vec![Choice {
            expert: ids[local],
            weight: 1.0,
        }]],
        registry: Registry {
            epoch: 1,
            copies: order
                .iter()
                .map(|&i| Replica {
                    expert: ids[i as usize],
                    rank: 2,
                    local: i,
                })
                .collect(),
        },
        stamp: stamp(batch),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, order)]),
    };
    let input = vec![(0..2560)
        .map(|i| ((i % 31) - 15) as f32 * 0.0002)
        .collect::<Vec<_>>()];
    let first = problem(0, 2, vec![0, 1, 2]);
    let baseline = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &first,
        &input,
        &ledger,
        ledger.reserve_ticket(&first.stamp, attempt)?,
        timeout,
    )?;
    let operation = sha256::hex(format!("{run_id}:evict:308:{attempt}").as_bytes());
    session.evict_native_ggml(
        &NativeEviction {
            stamp: &first.stamp,
            attempt,
            operation_id: &operation,
            binding: &bindings[1],
            expected_bytes: blobs[1].len() as u64,
            expected_remaining: 2,
        },
        timeout,
    )?;
    let after = problem(1, 2, vec![0, 2]);
    let result = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &after,
        &input,
        &ledger,
        ledger.reserve_ticket(&after.stamp, attempt.checked_add(1).ok_or("attempt")?)?,
        timeout,
    )?;
    if baseline.rows() != result.rows() {
        return Err("retained expert changed after eviction".into());
    }
    session.extend_native_ggml(
        &bindings[3],
        &index,
        pack.metadata()?.len(),
        &blobs[3],
        timeout,
    )?;
    let last = problem(2, 3, vec![0, 2, 3]);
    let result3 = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &last,
        &input,
        &ledger,
        ledger.reserve_ticket(&last.stamp, attempt.checked_add(2).ok_or("attempt")?)?,
        timeout,
    )?;
    let hashes = [
        hash(baseline.rows()),
        hash(result.rows()),
        hash(result3.rows()),
    ];
    if hashes[0] == hashes[2] {
        return Err("distinct experts produced identical output".into());
    }
    let receipt = session.close(timeout);
    if receipt.killed
        || receipt.exit_code != Some(0)
        || receipt.error.is_some()
        || receipt.bye_execs != Some(3)
    {
        return Err("worker close".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"one local CPU worker, real experts, synthetic input","run_id":run_id,"attempt_base":attempt,"experts":ids,"expert_sha256":bindings.iter().map(|b|&b.blob_sha256).collect::<Vec<_>>(),"eviction_operation_id":operation,"evicted_expert":308,"freed_payload_bytes":blobs[1].len(),"output_sha256":hashes,"workers_closed":1,"execs":3})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("eviction qualification refused: {e}");
        std::process::exit(1)
    }
}
