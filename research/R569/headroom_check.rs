//! Local CPU-only four-layer mixed-attention batch qualification. Never replay a submitted plan.
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
    sc117_ffn_stage, sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_post_attention, sc117_qsa_stage, sc117_shared_stage,
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
        "r569-{}-{attempt}-{}",
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
    for layer in 0..4 {
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let sm = match layer {
            0 => include_str!("../../fixtures/r438/layer0-shared.json"),
            1 => include_str!("../../fixtures/r438/layer1-shared.json"),
            2 => include_str!("../../fixtures/r438/layer2-shared.json"),
            3 => include_str!("../../fixtures/r431/shared3.json"),
            _ => return Err("layer".into()),
        };
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
        let mut stage = if layer == 3 {
            let qsa = sc117_qsa_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r430/layer3.json"))?,
                id.clone(),
                &mut dense,
                &mut shard,
                4,
            )?;
            let ffn = sc117_ffn_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r431/layer3.json"))?,
                id.clone(),
                &mut dense,
            )?;
            Layer::qsa(id, qsa, ffn, shared, owners)?
        } else {
            let (gm, pm) = match layer {
                0 => (
                    include_str!("../../fixtures/r438/layer0-gdn.json"),
                    include_str!("../../fixtures/r438/layer0-post.json"),
                ),
                1 => (
                    include_str!("../../fixtures/r438/layer1-gdn.json"),
                    include_str!("../../fixtures/r438/layer1-post.json"),
                ),
                2 => (
                    include_str!("../../fixtures/r438/layer2-gdn.json"),
                    include_str!("../../fixtures/r438/layer2-post.json"),
                ),
                _ => return Err("layer".into()),
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
            Layer::gdn(id, gdn, post, shared, owners)?
        };
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
    let limit = 384 * 1024 * 1024;
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
    let first = segment.advance_token_batch_with_source(attempt, &[42, 73], inputs, &mut source)?;
    let first_hashes = first.iter().map(|r| hash(r)).collect::<Vec<_>>();
    if first_hashes
        != [
            "11b2d8fab349ab75aafb932a0b841c26fbfa1d4bf268bcc46aed1706fbda6d97",
            "e0a6f600e6d208bf2d7c4c82b748a1bf3333f86c5d17b0fb9f724524d7253227",
        ]
    {
        return Err("first batch mismatch".into());
    }
    let used_before = source.used_bytes();
    let reserve = source
        .limit_bytes()
        .checked_sub(used_before)
        .ok_or("budget")?
        .checked_add(4_000_000)
        .ok_or("headroom")?;
    if segment
        .advance_token_batch_with_headroom(
            attempt + 1,
            &[42, voltafabric::sc117_ple::VOCAB],
            inputs,
            &mut source,
            reserve,
            8,
        )
        .is_ok()
        || source.used_bytes() != used_before
        || segment.position() != 2
        || segment.is_poisoned()
    {
        return Err("preflight mutated state".into());
    }
    let (second, evictions) = if fault {
        let (rows, evictions) = segment.advance_token_batch_with_headroom(
            attempt + 1,
            &[42, 73],
            inputs,
            &mut source,
            reserve,
            8,
        )?;
        if evictions.is_empty() {
            return Err("expected eviction".into());
        }
        (rows, evictions)
    } else {
        (
            segment.advance_token_batch_with_source(attempt + 1, &[42, 73], inputs, &mut source)?,
            Vec::new(),
        )
    };
    let hashes = second.iter().map(|r| hash(r)).collect::<Vec<_>>();
    let after = segment.ple_qualification_snapshot()?;
    if segment.position() != 4
        || segment.is_poisoned()
        || after.position != 4
        || after.attention_position != 4
        || after.previous != [42, 73]
    {
        return Err("second batch state".into());
    }
    let evicted=evictions.iter().map(|e|serde_json::json!({"layer":e.binding.layer,"expert":e.binding.expert,"rank":e.binding.rank,"operation_id":e.operation_id,"bytes":e.released_payload_bytes})).collect::<Vec<_>>();
    let admissions = source.admissions().len();
    drop(source);
    let receipts = segment.close_with_counts();
    if receipts.len() != 8
        || receipts.iter().any(|(r, n, raw)| {
            r.killed
                || r.error.is_some()
                || r.exit_code != Some(0)
                || r.bye_execs != Some(*n)
                || *raw != *n
                || *n != 2
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"fault":fault,"run_id":run_id,"plan_id":plan_id,"admissions":admissions,"residual_sha256":hashes,"evictions":evicted,"workers":receipts.iter().map(|(r,n,_)|serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n,"exit_code":r.exit_code})).collect::<Vec<_>>()}),
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
    if fault["residual_sha256"] != success["residual_sha256"] {
        return Err("headroom changed outputs".into());
    }
    Ok(serde_json::json!({"status":"pass","headroom":fault,"baseline":success}))
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
