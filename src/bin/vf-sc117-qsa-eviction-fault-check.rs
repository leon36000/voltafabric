//! Local CPU QSA batch: idle-rank refusal and suppressed eviction acknowledgement.
use std::{
    collections::BTreeMap,
    fs,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Replica},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    sc117_expert_origin::Origin,
    sc117_expert_source::{self, Config, Seed, Source},
    sc117_ffn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_qsa_stage, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const WORKER: &str = "ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const TRACE: &str = "a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2";
const SIZE: u64 = 50_292_326_400;
fn read(p: &str, h: &str) -> Result<Vec<u8>, E> {
    let b = fs::read(p)?;
    if sha256::hex(&b) != h {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn hash(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
#[allow(clippy::too_many_arguments)]
fn case(
    a: &[String],
    index: &str,
    layout: &NativeLayout,
    manifest: &[u8],
    inputs: &[Vec<f32>],
    attempt: u64,
    idle: bool,
) -> Result<serde_json::Value, E> {
    let run_id = format!(
        "r560-{}-{attempt}-{}",
        std::process::id(),
        if idle { "idle" } else { "lost" }
    );
    let plan_id = format!("{}-{}", a[9], if idle { "idle" } else { "lost" });
    let id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 3,
        epoch: 1,
    };
    let mut dense = fs::File::open(&a[3])?;
    let mut shard = fs::File::open(&a[4])?;
    let mut pack = fs::File::open(&a[5])?;
    let qsa = sc117_qsa_stage::Stage::load(
        serde_json::from_slice(manifest)?,
        id.clone(),
        &mut dense,
        &mut shard,
        4,
    )?;
    let ffn = sc117_ffn_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer3-ffn.json"))?,
        id.clone(),
        &mut dense,
    )?;
    let shared = sc117_shared_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer3-shared.json"))?,
        id.clone(),
        &mut shard,
    )?;
    let mut origin = Origin::open(shard.try_clone()?, layout)?;
    let mut sessions = Vec::new();
    let mut replicas = Vec::new();
    let mut seeds = Vec::new();
    let mut placement = if idle {
        vec![(0u32, 2u32, 0u32), (1, 2, 1)]
    } else {
        vec![(0u32, 2u32, 0u32), (1, 3, 0)]
    };
    let spec: serde_json::Value = serde_json::from_slice(&read(
        &a[11],
        "3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20",
    )?)?;
    if idle {
        for (local, key) in spec["experts"]
            .as_object()
            .ok_or("expert spec")?
            .keys()
            .enumerate()
        {
            placement.push((key.parse()?, 3, local as u32));
        }
        if placement.len() != 19 {
            return Err("seed count".into());
        }
    }
    for (expert, rank, local) in placement {
        let digest = origin.expert_digest(layout, 3, expert.into())?;
        let blob = layout.read_verified_expert(&mut pack, 3, expert.into(), INDEX, &digest)?;
        let b = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 3,
            expert,
            local,
            epoch: 1,
            index_sha256: INDEX.into(),
            blob_sha256: digest,
        };
        if !sessions.iter().any(|s: &Session| s.rank() == rank) {
            let argv = if !idle && rank == 2 {
                vec![
                    "/usr/bin/python3".into(),
                    a[12].clone(),
                    a[8].clone(),
                    a[13].clone(),
                    WORKER.into(),
                ]
            } else {
                vec![
                    a[8].clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    WORKER.into(),
                ]
            };
            sessions.push(Session::connect(
                &Endpoint { rank, argv },
                Duration::from_secs(60),
            )?);
        }
        let s = sessions
            .iter_mut()
            .find(|s| s.rank() == rank)
            .ok_or("rank session")?;
        s.load_native_ggml(&b, index, SIZE, &blob.bytes, Duration::from_secs(60))?;
        replicas.push(Replica {
            expert,
            rank,
            local,
        });
        seeds.push(Seed {
            layer: 3,
            expert,
            rank,
            local,
        });
    }
    for s in &mut sessions {
        let count = replicas.iter().filter(|r| r.rank == s.rank()).count();
        s.seal_native_ggml(count, Duration::from_secs(60))?;
    }
    let order = BTreeMap::from([
        (
            2,
            replicas
                .iter()
                .filter(|r| r.rank == 2)
                .map(|r| r.local)
                .collect(),
        ),
        (
            3,
            replicas
                .iter()
                .filter(|r| r.rank == 3)
                .map(|r| r.local)
                .collect(),
        ),
    ]);
    let owners = Owners {
        sessions,
        registry: Registry {
            epoch: 1,
            copies: replicas,
        },
        preferred: BTreeMap::new(),
        order,
    };
    let layer = Layer::qsa(id.clone(), qsa, ffn, shared, owners)?;
    let ledger_path = if idle { &a[7] } else { &a[6] };
    let ledger = NativeBatchLedger::open(std::path::Path::new(ledger_path))?;
    let mut segment = Segment::new(
        vec![layer],
        plan_id.clone(),
        ledger,
        Duration::from_secs(60),
    )?;
    let limit = 64 * 1024 * 1024;
    let config = Config {
        identity: sc117_expert_source::Identity {
            run_id: run_id.clone(),
            model_revision: id.model_revision.clone(),
            plan_id: plan_id.clone(),
            epoch: 1,
        },
        ranks: vec![2, 3],
        seeds: &seeds,
        limit,
    };
    let mut source = Source::new(&mut pack, layout, index, origin, config)?;
    let result = segment.advance_qsa_batch_with_source(attempt, inputs, &mut source)?;
    let hashes = result.iter().map(|v| hash(v)).collect::<Vec<_>>();
    if hashes
        != [
            "896ee2013680ac93e47da33b1b250f2931d549143855590ecad3a51c14241349",
            "c65230176ea43c513bdcd8bbdcc3c5ae7f5b2e0a6469966d394907d55e50b862",
        ]
        || segment.position() != 2
        || segment.is_poisoned()
    {
        return Err("batch mismatch".into());
    }
    let binding = source.resident_binding(3, 0)?;
    if binding.rank != 2 {
        return Err("unexpected binding".into());
    }
    let before = source.used_bytes();
    let claims_before = fs::read_dir(ledger_path)?.count();
    let error = match segment.evict_owner_after_commit(&mut source, &binding) {
        Ok(_) => return Err("unexpected eviction success".into()),
        Err(e) => e,
    };
    let claims_after = fs::read_dir(ledger_path)?.count();
    if source.used_bytes() != before
        || source.resident_binding(3, 0)? != binding
        || segment.position() != 2
    {
        return Err("eviction error changed accounting".into());
    }
    if idle {
        if !error.message.contains("did not participate")
            || segment.is_poisoned()
            || claims_after != claims_before
        {
            return Err("idle rank refusal failed".into());
        }
    } else {
        if !segment.is_poisoned()
            || claims_after != claims_before + 1
            || segment
                .advance_qsa_batch_with_source(attempt + 1, inputs, &mut source)
                .is_ok()
            || fs::read_dir(ledger_path)?.count() != claims_after
        {
            return Err("lost acknowledgement did not poison".into());
        }
    }
    let admissions = source.admissions().len();
    let used = source.used_bytes();
    let rank_used = source.rank_used_bytes().clone();
    drop(source);
    let receipts = segment.close_with_counts();
    if receipts.len() != 2
        || receipts.iter().any(|(r, n, raw)| {
            let expected = if idle && r.rank == 2 { 0 } else { 1 };
            *n != expected
                || *raw != *n
                || ((idle || r.rank != 2)
                    && (r.killed
                        || r.error.is_some()
                        || r.exit_code != Some(0)
                        || r.bye_execs != Some(*n)))
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"idle":idle,"run_id":run_id,"plan_id":plan_id,"admissions":admissions,"used_bytes":used,"rank_used_bytes":rank_used,"residual_sha256":hashes,"claims_before":claims_before,"claims_after":claims_after,"error_code":error.code,"error_message":error.message,"workers":receipts.iter().map(|(r,n,_)|serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n,"exit_code":r.exit_code})).collect::<Vec<_>>()}),
    )
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 14 {
        return Err("usage: TRACE QSA_MANIFEST DENSE SHARD PACK LEDGER_LOST LEDGER_IDLE WORKER PLAN_SHA INDEX SPEC PROXY MARKER".into());
    }
    if !sha256::is_hex_digest(&a[9]) {
        return Err("plan".into());
    }
    let trace = read(&a[1], TRACE)?;
    if trace.len() != 92 * 47620 * 4 {
        return Err("trace size".into());
    }
    let manifest = read(
        &a[2],
        "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21",
    )?;
    read(&a[8], WORKER)?;
    let index = fs::read_to_string(&a[10])?;
    if sha256::hex(index.as_bytes()) != INDEX {
        return Err("index".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let inputs: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            trace[i * 47620 * 4..i * 47620 * 4 + 40960]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let idle = case(&a, &index, &layout, &manifest, &inputs, attempt, true)?;
    let lost = case(&a, &index, &layout, &manifest, &inputs, attempt + 1, false)?;
    Ok(serde_json::json!({"status":"pass","idle":idle,"lost":lost}))
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("dynamic batch refused: {e}");
            std::process::exit(1)
        }
    }
}
