//! Isolated first-layer transaction for one observed token; no partial output.
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
    sc117_embedding,
    sc117_expert_origin::Origin,
    sc117_expert_source::{Config, Seed, Source},
    sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_post_attention, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const SIZE: u64 = 50_292_326_400;
fn hash(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: layer0 DENSE SHARD PACK INDEX LEDGER".into());
    }
    let plan = std::env::var("VF_R611_PLAN")?;
    let run = std::env::var("VF_R611_RUN")?;
    let worker = std::env::var("VF_R611_WORKER")?;
    let wh = std::env::var("VF_R611_WORKER_SHA256")?;
    let mode = std::env::var("VF_R611_BACKEND")?;
    if !sha256::is_hex_digest(&plan)
        || !sha256::is_hex_digest(&wh)
        || !run.starts_with("r611-")
        || !matches!(mode.as_str(), "scalar" | "vector")
        || sha256::hex(&fs::read(&worker)?) != wh
    {
        return Err("invalid identity or executable".into());
    }
    let index = fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX {
        return Err("index".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let mut dense = fs::File::open(&a[1])?;
    let mut shard = fs::File::open(&a[2])?;
    let mut pack = fs::File::open(&a[3])?;
    let table = sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let embedding = table.gather(248045)?;
    if hash(&embedding) != "ac914238a95d7950bcab749cf5d9358b3467ca3bafc520b1040a113568608423" {
        return Err("embedding differs from R609".into());
    }
    let input: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
    let id = Identity {
        run_id: run.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        epoch: 1,
    };
    let gdn = sc117_gdn_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-gdn.json"))?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let post = sc117_post_attention::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-post.json"))?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let shared = sc117_shared_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-shared.json"))?,
        id.clone(),
        &mut shard,
    )?;
    let mut origin = Origin::open(shard.try_clone()?, &layout)?;
    let timeout = Duration::from_secs(60);
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut seeds = Vec::new();
    for (expert, rank) in [(0u32, 2u32), (1, 3)] {
        let digest = origin.expert_digest(&layout, 0, expert.into())?;
        let data = layout.read_verified_expert(&mut pack, 0, expert.into(), INDEX, &digest)?;
        let binding = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 0,
            expert,
            local: 0,
            epoch: 1,
            index_sha256: INDEX.into(),
            blob_sha256: digest,
        };
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![
                    worker.clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    wh.clone(),
                ],
            },
            timeout,
        )?;
        if mode == "scalar" {
            session.load_native_scalar_ggml(&binding, &index, SIZE, &data.bytes, timeout)?
        } else {
            session.load_native_ggml(&binding, &index, SIZE, &data.bytes, timeout)?
        }
        session.seal_native_ggml(1, timeout)?;
        sessions.push(session);
        copies.push(Replica {
            expert,
            rank,
            local: 0,
        });
        seeds.push(Seed {
            layer: 0,
            expert,
            rank,
            local: 0,
        });
    }
    let owners = Owners {
        sessions,
        registry: Registry { epoch: 1, copies },
        preferred: BTreeMap::new(),
        order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let layer = Layer::gdn(id, gdn, post, shared, owners)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[5]))?;
    let plan_id = format!("r611-{mode}-{plan}");
    let mut segment = Segment::new(vec![layer], plan_id.clone(), ledger, timeout)?;
    let mut source = Source::new(
        &mut pack,
        &layout,
        &index,
        origin,
        Config {
            identity: voltafabric::sc117_expert_source::Identity {
                run_id: run.clone(),
                model_revision: "sc117-local-observation".into(),
                plan_id: plan_id.clone(),
                epoch: 1,
            },
            ranks: vec![2, 3],
            seeds: &seeds,
            limit: 96 * 1024 * 1024,
        },
    )?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let output = segment.advance_token_with_source(attempt, 248045, &input, &mut source)?;
    if segment.is_poisoned() || segment.position() != 1 {
        return Err("segment state".into());
    }
    let admissions = source.admissions().to_vec();
    drop(source);
    let closes = segment.close_with_counts();
    if closes.len() != 2
        || closes.iter().any(|(r, n, raw)| {
            r.killed
                || r.error.is_some()
                || r.exit_code != Some(0)
                || r.bye_execs != Some(*n)
                || *n != 1
                || *raw != 1
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"run_id":run,"plan_id":plan_id,"attempt":attempt,"backend":mode,"token":248045,"embedding_sha256":hash(&embedding),"residual_sha256":hash(&output),"residual":output,"admissions":admissions,"workers_closed":closes.len()}),
    )
}
fn main() {
    match run() {
        Ok(x) => println!("{x}"),
        Err(e) => {
            eprintln!("layer0 refused: {e}");
            std::process::exit(1)
        }
    }
}
