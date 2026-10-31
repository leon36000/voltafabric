//! Bounded local CPU check of dynamic SC117 owner admission for a PLE/GDN batch.
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
const TRACE: &str = "d5b979462ad00496a7c06406e8634f5bc58c69b0c92b4f18049fea25476eee25";
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
    fault: bool,
) -> Result<serde_json::Value, E> {
    let run_id = format!(
        "r566-{}-{attempt}-{}",
        std::process::id(),
        if fault { "fault" } else { "ok" }
    );
    let plan_id = format!("{}-{}", a[9], if fault { "fault" } else { "ok" });
    let id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        epoch: 1,
    };
    let mut dense = fs::File::open(&a[3])?;
    let mut shard = fs::File::open(&a[4])?;
    let mut pack = fs::File::open(&a[5])?;
    let gdn = sc117_gdn_stage::Stage::load(
        serde_json::from_slice(manifest)?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let post = sc117_post_attention::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer1-post.json"))?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let shared = sc117_shared_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer1-shared.json"))?,
        id.clone(),
        &mut shard,
    )?;
    let mut origin = Origin::open(shard.try_clone()?, layout)?;
    let mut sessions = Vec::new();
    let mut replicas = Vec::new();
    let mut seeds = Vec::new();
    for (expert, rank) in [(0u32, 2u32), (1, 3)] {
        let digest = origin.expert_digest(layout, 1, expert.into())?;
        let blob = layout.read_verified_expert(&mut pack, 1, expert.into(), INDEX, &digest)?;
        let b = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 1,
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
                    a[8].clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    WORKER.into(),
                ],
            },
            Duration::from_secs(60),
        )?;
        s.load_native_ggml(&b, index, SIZE, &blob.bytes, Duration::from_secs(60))?;
        s.seal_native_ggml(1, Duration::from_secs(60))?;
        sessions.push(s);
        replicas.push(Replica {
            expert,
            rank,
            local: 0,
        });
        seeds.push(Seed {
            layer: 1,
            expert,
            rank,
            local: 0,
        });
    }
    let owners = Owners {
        sessions,
        registry: Registry {
            epoch: 1,
            copies: replicas,
        },
        preferred: BTreeMap::new(),
        order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let manifest_bytes = read(
        &a[12],
        "436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824",
    )?;
    let ple_manifest: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_slice(&manifest_bytes)?;
    let weights = voltafabric::sc117_ple_weights::Weights::load(
        &mut shard,
        &ple_manifest,
        &id.model_revision,
    )?;
    let table = voltafabric::sc117_ple::Table::open(fs::File::open(&a[11])?, 28_800_138_432, 192)?;
    let layer = Layer::gdn(id.clone(), gdn, post, shared, owners)?.attach_ple(table, weights)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(if fault { &a[7] } else { &a[6] }))?;
    let mut segment = Segment::new_ple_cpu_qualification(
        vec![layer],
        plan_id.clone(),
        ledger,
        Duration::from_secs(60),
    )?;
    let limit = if fault {
        3 * 1_510_400
    } else {
        48 * 1024 * 1024
    };
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
    let mut source = if fault {
        Source::new(&mut pack, layout, index, origin, config)?
    } else {
        Source::new_with_rank_limits(
            &mut pack,
            layout,
            index,
            origin,
            config,
            BTreeMap::from([(2, 24 * 1024 * 1024), (3, 24 * 1024 * 1024)]),
        )?
    };
    let before = segment.ple_qualification_snapshot()?;
    let result = segment.advance_gdn_ple_batch_with_source(attempt, &[42, 73], inputs, &mut source);
    let snapshot = segment.ple_qualification_snapshot()?;
    let hashes = if fault {
        let e = match result {
            Err(e) => e,
            Ok(_) => return Err("budget unexpectedly accepted".into()),
        };
        if !e.message.contains("budget")
            || !segment.is_poisoned()
            || snapshot.position != 0
            || snapshot.attention_position != 0
            || snapshot.previous != [-1, -1]
            || snapshot.history_sha256 != before.history_sha256
            || source.admissions().len() != 1
            || segment
                .advance_gdn_ple_batch_with_source(attempt + 1, &[42, 73], inputs, &mut source)
                .is_ok()
        {
            return Err("budget failure did not poison".into());
        }
        Vec::new()
    } else {
        let result = result?;
        let h = result.iter().map(|v| hash(v)).collect::<Vec<_>>();
        if h != [
            "656e0f77c4945a76460bc1e23009de17d956f68d0ac38ddce6b34f9c98e26945",
            "6d810f022869fb34e7568bad1c486ed6a54b9e3321281400e319044092a18421",
        ] || segment.position() != 2
            || segment.is_poisoned()
            || snapshot.position != 2
            || snapshot.attention_position != 2
            || snapshot.previous != [42, 73]
            || source.admissions().len() != 15
        {
            return Err("dynamic output or admissions mismatch".into());
        }
        h
    };
    let admissions = source.admissions().len();
    let used = source.used_bytes();
    let rank_used = source.rank_used_bytes().clone();
    drop(source);
    let receipts = segment.close_with_counts();
    if receipts.len() != 2
        || receipts.iter().any(|(r, n, raw)| {
            r.killed
                || r.error.is_some()
                || r.exit_code != Some(0)
                || r.bye_execs != Some(*n)
                || *raw != *n
                || *n != if fault { 0 } else { 1 }
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"fault":fault,"run_id":run_id,"plan_id":plan_id,"admissions":admissions,"used_bytes":used,"rank_used_bytes":rank_used,"residual_sha256":hashes,"workers":receipts.iter().map(|(r,n,_)|serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n,"exit_code":r.exit_code})).collect::<Vec<_>>()}),
    )
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 13 {
        return Err("usage: RESIDUAL GDN_MANIFEST DENSE SHARD PACK LEDGER_OK LEDGER_FAIL WORKER PLAN_SHA INDEX PLE_SHARD PLE_MANIFEST".into());
    }
    if !sha256::is_hex_digest(&a[9]) {
        return Err("plan".into());
    }
    let trace = read(&a[1], TRACE)?;

    let manifest = read(
        &a[2],
        "e38a25d4581b0343b2e6e0ae375744c4b2409f4730cc0d2d91fc9b4127058511",
    )?;
    read(&a[8], WORKER)?;
    let index = fs::read_to_string(&a[10])?;
    if sha256::hex(index.as_bytes()) != INDEX {
        return Err("index".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let observed: serde_json::Value = serde_json::from_slice(&trace)?;
    let mut inputs = Vec::new();
    for (i, expected) in [
        "f929a224afaf2832954242730520d71845cd16be3184b7ec0e7c192bed4bb0e8",
        "22821acfb7a7bd0264080eb57331dd0cf6d056e914c2bc6a54b8eddaafdd3257",
    ]
    .iter()
    .enumerate()
    {
        let row: Vec<f32> = observed["layer0_residuals"][i]
            .as_array()
            .ok_or("missing row")?
            .iter()
            .map(|v| v.as_f64().map(|v| v as f32).ok_or("invalid row"))
            .collect::<Result<_, _>>()?;
        if row.len() != 10240 || hash(&row) != *expected {
            return Err("layer0 input hash".into());
        }
        inputs.push(row);
    }
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let fault = case(&a, &index, &layout, &manifest, &inputs, attempt, true)?;
    let ok = case(&a, &index, &layout, &manifest, &inputs, attempt + 1, false)?;
    Ok(serde_json::json!({"status":"pass","fault":fault,"success":ok}))
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
