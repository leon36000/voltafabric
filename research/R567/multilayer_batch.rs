//! Local CPU-only two-layer batch qualification. Never replay a submitted plan.
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
    sc117_expert_source::{self, Config, Seed, Source},
    sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_post_attention, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const WORKER: &str = "ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const SIZE: u64 = 50_292_326_400;
fn hash(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
#[allow(clippy::too_many_arguments)]
fn case(
    a: &[String],
    index: &str,
    layout: &NativeLayout,
    inputs: &[Vec<f32>],
    attempt: u64,
    fault: bool,
) -> Result<serde_json::Value, E> {
    let run_id = format!(
        "r567-{}-{attempt}-{}",
        std::process::id(),
        if fault { "fault" } else { "ok" }
    );
    let plan_id = format!("{}-{}", a[8], if fault { "fault" } else { "ok" });
    let mut dense = fs::File::open(&a[1])?;
    let mut shard = fs::File::open(&a[2])?;
    let mut pack = fs::File::open(&a[3])?;
    let mut origin = Origin::open(shard.try_clone()?, layout)?;
    let timeout = Duration::from_secs(60);
    let mut layers = Vec::new();
    let mut seeds = Vec::new();
    for layer in 0..2 {
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let (gm, pm, sm) = if layer == 0 {
            (
                include_str!("../../fixtures/r438/layer0-gdn.json"),
                include_str!("../../fixtures/r438/layer0-post.json"),
                include_str!("../../fixtures/r438/layer0-shared.json"),
            )
        } else {
            (
                include_str!("../../fixtures/r438/layer1-gdn.json"),
                include_str!("../../fixtures/r438/layer1-post.json"),
                include_str!("../../fixtures/r438/layer1-shared.json"),
            )
        };
        let gdn = sc117_gdn_stage::Stage::load(
            serde_json::from_str(gm)?,
            id.clone(),
            &mut dense,
            &mut shard,
        )?;
        let post = sc117_post_attention::Stage::load(
            serde_json::from_str(pm)?,
            id.clone(),
            &mut dense,
            &mut shard,
        )?;
        let shared =
            sc117_shared_stage::Stage::load(serde_json::from_str(sm)?, id.clone(), &mut shard)?;
        let mut sessions = Vec::new();
        let mut copies = Vec::new();
        for (expert, rank) in [(0u32, 2u32), (1, 3)] {
            let digest = origin.expert_digest(layout, layer as usize, expert.into())?;
            let data = layout.read_verified_expert(
                &mut pack,
                layer as usize,
                expert.into(),
                INDEX,
                &digest,
            )?;
            let binding = Binding {
                model_revision: id.model_revision.clone(),
                rank,
                layer,
                expert,
                local: 0,
                epoch: 1,
                index_sha256: INDEX.into(),
                blob_sha256: digest,
            };
            let mut s = Session::connect(
                &Endpoint {
                    rank,
                    argv: vec![
                        a[6].clone(),
                        "--guarded-local".into(),
                        "--expected-exe-sha256".into(),
                        WORKER.into(),
                    ],
                },
                timeout,
            )?;
            s.load_native_ggml(&binding, index, SIZE, &data.bytes, timeout)?;
            s.seal_native_ggml(1, timeout)?;
            sessions.push(s);
            copies.push(Replica {
                expert,
                rank,
                local: 0,
            });
            seeds.push(Seed {
                layer,
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
        let mut stage = Layer::gdn(id, gdn, post, shared, owners)?;
        if layer == 1 {
            let mb = fs::read(&a[10])?;
            if sha256::hex(&mb)
                != "436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824"
            {
                return Err("PLE manifest hash".into());
            }
            let m = serde_json::from_slice(&mb)?;
            let w = voltafabric::sc117_ple_weights::Weights::load(
                &mut shard,
                &m,
                "sc117-local-observation",
            )?;
            let t =
                voltafabric::sc117_ple::Table::open(fs::File::open(&a[9])?, 28_800_138_432, 192)?;
            stage = stage.attach_ple(t, w)?;
        }
        layers.push(stage);
    }
    let ledger = NativeBatchLedger::open(std::path::Path::new(if fault { &a[7] } else { &a[5] }))?;
    let mut segment = Segment::new_ple_cpu_qualification(layers, plan_id.clone(), ledger, timeout)?;
    let limit = if fault {
        2 * 2_176_000 + 2 * 1_510_400 + 17 * 2_176_000 + 1_510_400
    } else {
        96 * 1024 * 1024
    };
    let mut source = Source::new(
        &mut pack,
        layout,
        index,
        origin,
        Config {
            identity: sc117_expert_source::Identity {
                run_id: run_id.clone(),
                model_revision: "sc117-local-observation".into(),
                plan_id: plan_id.clone(),
                epoch: 1,
            },
            ranks: vec![2, 3],
            seeds: &seeds,
            limit,
        },
    )?;
    let before = segment.ple_qualification_snapshot()?;
    let result = segment.advance_token_batch_with_source(attempt, &[42, 73], inputs, &mut source);
    let after = segment.ple_qualification_snapshot()?;
    let hashes = if fault {
        let e = match result {
            Err(e) => e,
            Ok(_) => return Err("budget unexpectedly accepted".into()),
        };
        if !e.message.contains("budget")
            || !segment.is_poisoned()
            || segment.position() != 0
            || after.position != 0
            || after.attention_position != 0
            || after.history_sha256 != before.history_sha256
            || source.admissions().len() != 18
            || segment
                .advance_token_batch_with_source(attempt + 1, &[42, 73], inputs, &mut source)
                .is_ok()
        {
            return Err("later failure violated terminal state".into());
        }
        Vec::new()
    } else {
        let rows = result?;
        let h = rows.iter().map(|r| hash(r)).collect::<Vec<_>>();
        if h != [
            "656e0f77c4945a76460bc1e23009de17d956f68d0ac38ddce6b34f9c98e26945",
            "6d810f022869fb34e7568bad1c486ed6a54b9e3321281400e319044092a18421",
        ] || segment.is_poisoned()
            || segment.position() != 2
            || after.position != 2
            || after.attention_position != 2
            || after.previous != [42, 73]
            || source.admissions().len() != 32
        {
            return Err("batch output or state differs".into());
        }
        h
    };
    let admissions = source.admissions().len();
    drop(source);
    let receipts = segment.close_with_counts();
    if receipts.len() != 4
        || receipts.iter().enumerate().any(|(i, (r, n, raw))| {
            r.killed
                || r.error.is_some()
                || r.exit_code != Some(0)
                || r.bye_execs != Some(*n)
                || *raw != *n
                || *n != if fault && i >= 2 { 0 } else { 1 }
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"fault":fault,"run_id":run_id,"plan_id":plan_id,"admissions":admissions,"residual_sha256":hashes,"workers":receipts.iter().map(|(r,n,_)|serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n,"exit_code":r.exit_code})).collect::<Vec<_>>()}),
    )
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 11 {
        return Err(
            "usage DENSE SHARD PACK INDEX LEDGER_OK WORKER LEDGER_FAIL PLAN PLE_SHARD PLE_MANIFEST"
                .into(),
        );
    }
    if !sha256::is_hex_digest(&a[8]) || sha256::hex(&fs::read(&a[6])?) != WORKER {
        return Err("plan/worker".into());
    }
    let index = fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX {
        return Err("index".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let table = sc117_embedding::Table::open(fs::File::open(&a[2])?, 55_063_448_064, 545_644_544)?;
    let mut inputs = Vec::new();
    for (token, expected) in [
        (
            42,
            "ae94a5f1a2e54b8f2e9719409640badfbba583e83ae82ad66987da58ee64ae31",
        ),
        (
            73,
            "5c1ffc574e109b8aa5738f7b0872f9d8a60f864d6dd8ada409e470f2d5649c04",
        ),
    ] {
        let row = table.gather(token)?;
        if hash(&row) != expected {
            return Err("embedding hash".into());
        }
        inputs.push((0..4).flat_map(|_| row.iter().copied()).collect())
    }
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let fault = case(&a, &index, &layout, &inputs, attempt, true)?;
    let success = case(&a, &index, &layout, &inputs, attempt + 1, false)?;
    Ok(serde_json::json!({"status":"pass","fault":fault,"success":success}))
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("multi-layer batch refused: {e}");
            std::process::exit(1)
        }
    }
}
