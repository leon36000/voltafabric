//! Local CPU check of QSA batch headroom and acknowledged eviction.
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
    serial: bool,
) -> Result<serde_json::Value, E> {
    let run_id = format!(
        "r559-{}-{attempt}-{}",
        std::process::id(),
        if serial { "serial" } else { "batch" }
    );
    let plan_id = format!("{}-{}", a[9], if serial { "serial" } else { "batch" });
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
        8,
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
    for (expert, rank) in [(0u32, 2u32), (1, 3)] {
        let digest = origin.expert_digest(layout, 3, expert.into())?;
        let blob = layout.read_verified_expert(&mut pack, 3, expert.into(), INDEX, &digest)?;
        let b = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 3,
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
            layer: 3,
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
    let layer = Layer::qsa(id.clone(), qsa, ffn, shared, owners)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(if serial { &a[7] } else { &a[6] }))?;
    let mut segment = Segment::new(
        vec![layer],
        plan_id.clone(),
        ledger,
        Duration::from_secs(60),
    )?;
    let limit = if serial {
        80 * 1_971_200
    } else {
        40 * 1_971_200
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
    let mut source = Source::new(&mut pack, layout, index, origin, config)?;
    let mut hashes = Vec::new();
    let mut evictions = Vec::new();
    if serial {
        for (i, input) in inputs.iter().enumerate() {
            let output = segment.advance_with_source(attempt + i as u64, input, &mut source)?;
            hashes.push(hash(&output));
        }
    } else {
        for (i, chunk) in inputs.chunks(2).enumerate() {
            let (rows, receipts) = segment.advance_qsa_batch_with_headroom(
                attempt + i as u64,
                chunk,
                &mut source,
                20 * 1_971_200,
                64,
            )?;
            hashes.extend(rows.iter().map(|v| hash(v)));
            for r in receipts {
                if i == 0
                    || r.stamp.batch != (i as u64 - 1) * 2
                    || r.attempt != attempt + i as u64 - 1
                {
                    return Err("eviction identity mismatch".into());
                }
                evictions.push(serde_json::json!({"batch":r.stamp.batch,"rank":r.binding.rank,"expert":r.binding.expert,"operation_id":r.operation_id}));
            }
        }
        if evictions.is_empty() {
            return Err("expected eviction did not occur".into());
        }
        let used_before = source.used_bytes();
        let admissions_before = source.admissions().len();
        if segment
            .advance_qsa_batch_with_headroom(
                attempt + 3,
                &inputs[..3],
                &mut source,
                20 * 1_971_200,
                64,
            )
            .is_ok()
            || segment.is_poisoned()
            || segment.position() != 6
            || source.used_bytes() != used_before
            || source.admissions().len() != admissions_before
        {
            return Err("capacity preflight changed live state".into());
        }
    }
    if segment.position() != 6
        || segment.is_poisoned()
        || hashes.len() != 6
        || hashes[..2]
            != [
                "896ee2013680ac93e47da33b1b250f2931d549143855590ecad3a51c14241349",
                "c65230176ea43c513bdcd8bbdcc3c5ae7f5b2e0a6469966d394907d55e50b862",
            ]
    {
        return Err("position or baseline mismatch".into());
    }
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
                || *n == 0
        })
    {
        return Err("worker close".into());
    }
    Ok(
        serde_json::json!({"serial":serial,"run_id":run_id,"plan_id":plan_id,"admissions":admissions,"used_bytes":used,"rank_used_bytes":rank_used,"residual_sha256":hashes,"evictions":evictions,"workers":receipts.iter().map(|(r,n,_)|serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n,"exit_code":r.exit_code})).collect::<Vec<_>>()}),
    )
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 11 {
        return Err("usage: TRACE QSA_MANIFEST DENSE SHARD PACK LEDGER_OK LEDGER_FAIL WORKER PLAN_SHA INDEX".into());
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
    let inputs: Vec<Vec<f32>> = (0..6)
        .map(|i| {
            trace[i * 47620 * 4..i * 47620 * 4 + 40960]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let serial = case(&a, &index, &layout, &manifest, &inputs, attempt, true)?;
    let batch = case(
        &a,
        &index,
        &layout,
        &manifest,
        &inputs,
        attempt + 100,
        false,
    )?;
    if serial["residual_sha256"] != batch["residual_sha256"] {
        return Err("batch and sequential outputs differ".into());
    }
    Ok(serde_json::json!({"status":"pass","serial":serial,"batch":batch}))
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
