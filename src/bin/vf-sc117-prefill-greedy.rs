//! Bounded SC117 CPU generation qualification with pre-position expert headroom.
use std::{
    collections::BTreeMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Replica},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    sc117_ffn_stage, sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_post_attention, sc117_qsa_stage, sc117_router_cpu, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX_HASH: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";

struct Manifests {
    g: &'static str,
    p: &'static str,
    s: &'static str,
}
fn manifests(layer: u32) -> Manifests {
    match layer {
        0 => Manifests {
            g: include_str!("../../fixtures/r438/layer0-gdn.json"),
            p: include_str!("../../fixtures/r438/layer0-post.json"),
            s: include_str!("../../fixtures/r438/layer0-shared.json"),
        },
        1 => Manifests {
            g: include_str!("../../fixtures/r438/layer1-gdn.json"),
            p: include_str!("../../fixtures/r438/layer1-post.json"),
            s: include_str!("../../fixtures/r438/layer1-shared.json"),
        },
        2 => Manifests {
            g: include_str!("../../fixtures/r438/layer2-gdn.json"),
            p: include_str!("../../fixtures/r438/layer2-post.json"),
            s: include_str!("../../fixtures/r438/layer2-shared.json"),
        },
        4 => Manifests {
            g: include_str!("../../fixtures/r438/layer4-gdn.json"),
            p: include_str!("../../fixtures/r438/layer4-post.json"),
            s: include_str!("../../fixtures/r438/layer4-shared.json"),
        },
        5 => Manifests {
            g: include_str!("../../fixtures/r438/layer5-gdn.json"),
            p: include_str!("../../fixtures/r438/layer5-post.json"),
            s: include_str!("../../fixtures/r438/layer5-shared.json"),
        },
        6 => Manifests {
            g: include_str!("../../fixtures/r438/layer6-gdn.json"),
            p: include_str!("../../fixtures/r438/layer6-post.json"),
            s: include_str!("../../fixtures/r438/layer6-shared.json"),
        },
        8 => Manifests {
            g: include_str!("../../fixtures/r438/layer8-gdn.json"),
            p: include_str!("../../fixtures/r438/layer8-post.json"),
            s: include_str!("../../fixtures/r438/layer8-shared.json"),
        },
        9 => Manifests {
            g: include_str!("../../fixtures/r438/layer9-gdn.json"),
            p: include_str!("../../fixtures/r438/layer9-post.json"),
            s: include_str!("../../fixtures/r438/layer9-shared.json"),
        },
        10 => Manifests {
            g: include_str!("../../fixtures/r438/layer10-gdn.json"),
            p: include_str!("../../fixtures/r438/layer10-post.json"),
            s: include_str!("../../fixtures/r438/layer10-shared.json"),
        },
        12 => Manifests {
            g: include_str!("../../fixtures/r438/layer12-gdn.json"),
            p: include_str!("../../fixtures/r438/layer12-post.json"),
            s: include_str!("../../fixtures/r438/layer12-shared.json"),
        },
        13 => Manifests {
            g: include_str!("../../fixtures/r438/layer13-gdn.json"),
            p: include_str!("../../fixtures/r438/layer13-post.json"),
            s: include_str!("../../fixtures/r438/layer13-shared.json"),
        },
        14 => Manifests {
            g: include_str!("../../fixtures/r438/layer14-gdn.json"),
            p: include_str!("../../fixtures/r438/layer14-post.json"),
            s: include_str!("../../fixtures/r438/layer14-shared.json"),
        },
        16 => Manifests {
            g: include_str!("../../fixtures/r438/layer16-gdn.json"),
            p: include_str!("../../fixtures/r438/layer16-post.json"),
            s: include_str!("../../fixtures/r438/layer16-shared.json"),
        },
        17 => Manifests {
            g: include_str!("../../fixtures/r438/layer17-gdn.json"),
            p: include_str!("../../fixtures/r438/layer17-post.json"),
            s: include_str!("../../fixtures/r438/layer17-shared.json"),
        },
        18 => Manifests {
            g: include_str!("../../fixtures/r438/layer18-gdn.json"),
            p: include_str!("../../fixtures/r438/layer18-post.json"),
            s: include_str!("../../fixtures/r438/layer18-shared.json"),
        },
        20 => Manifests {
            g: include_str!("../../fixtures/r438/layer20-gdn.json"),
            p: include_str!("../../fixtures/r438/layer20-post.json"),
            s: include_str!("../../fixtures/r438/layer20-shared.json"),
        },
        21 => Manifests {
            g: include_str!("../../fixtures/r438/layer21-gdn.json"),
            p: include_str!("../../fixtures/r438/layer21-post.json"),
            s: include_str!("../../fixtures/r438/layer21-shared.json"),
        },
        22 => Manifests {
            g: include_str!("../../fixtures/r438/layer22-gdn.json"),
            p: include_str!("../../fixtures/r438/layer22-post.json"),
            s: include_str!("../../fixtures/r438/layer22-shared.json"),
        },
        24 => Manifests {
            g: include_str!("../../fixtures/r438/layer24-gdn.json"),
            p: include_str!("../../fixtures/r438/layer24-post.json"),
            s: include_str!("../../fixtures/r438/layer24-shared.json"),
        },
        25 => Manifests {
            g: include_str!("../../fixtures/r438/layer25-gdn.json"),
            p: include_str!("../../fixtures/r438/layer25-post.json"),
            s: include_str!("../../fixtures/r438/layer25-shared.json"),
        },
        26 => Manifests {
            g: include_str!("../../fixtures/r438/layer26-gdn.json"),
            p: include_str!("../../fixtures/r438/layer26-post.json"),
            s: include_str!("../../fixtures/r438/layer26-shared.json"),
        },
        28 => Manifests {
            g: include_str!("../../fixtures/r438/layer28-gdn.json"),
            p: include_str!("../../fixtures/r438/layer28-post.json"),
            s: include_str!("../../fixtures/r438/layer28-shared.json"),
        },
        29 => Manifests {
            g: include_str!("../../fixtures/r438/layer29-gdn.json"),
            p: include_str!("../../fixtures/r438/layer29-post.json"),
            s: include_str!("../../fixtures/r438/layer29-shared.json"),
        },
        30 => Manifests {
            g: include_str!("../../fixtures/r438/layer30-gdn.json"),
            p: include_str!("../../fixtures/r438/layer30-post.json"),
            s: include_str!("../../fixtures/r438/layer30-shared.json"),
        },
        32 => Manifests {
            g: include_str!("../../fixtures/r438/layer32-gdn.json"),
            p: include_str!("../../fixtures/r438/layer32-post.json"),
            s: include_str!("../../fixtures/r438/layer32-shared.json"),
        },
        33 => Manifests {
            g: include_str!("../../fixtures/r438/layer33-gdn.json"),
            p: include_str!("../../fixtures/r438/layer33-post.json"),
            s: include_str!("../../fixtures/r438/layer33-shared.json"),
        },
        34 => Manifests {
            g: include_str!("../../fixtures/r438/layer34-gdn.json"),
            p: include_str!("../../fixtures/r438/layer34-post.json"),
            s: include_str!("../../fixtures/r438/layer34-shared.json"),
        },
        36 => Manifests {
            g: include_str!("../../fixtures/r438/layer36-gdn.json"),
            p: include_str!("../../fixtures/r438/layer36-post.json"),
            s: include_str!("../../fixtures/r438/layer36-shared.json"),
        },
        37 => Manifests {
            g: include_str!("../../fixtures/r438/layer37-gdn.json"),
            p: include_str!("../../fixtures/r438/layer37-post.json"),
            s: include_str!("../../fixtures/r438/layer37-shared.json"),
        },
        38 => Manifests {
            g: include_str!("../../fixtures/r438/layer38-gdn.json"),
            p: include_str!("../../fixtures/r438/layer38-post.json"),
            s: include_str!("../../fixtures/r438/layer38-shared.json"),
        },
        40 => Manifests {
            g: include_str!("../../fixtures/r438/layer40-gdn.json"),
            p: include_str!("../../fixtures/r438/layer40-post.json"),
            s: include_str!("../../fixtures/r438/layer40-shared.json"),
        },
        41 => Manifests {
            g: include_str!("../../fixtures/r438/layer41-gdn.json"),
            p: include_str!("../../fixtures/r438/layer41-post.json"),
            s: include_str!("../../fixtures/r438/layer41-shared.json"),
        },
        42 => Manifests {
            g: include_str!("../../fixtures/r438/layer42-gdn.json"),
            p: include_str!("../../fixtures/r438/layer42-post.json"),
            s: include_str!("../../fixtures/r438/layer42-shared.json"),
        },
        44 => Manifests {
            g: include_str!("../../fixtures/r438/layer44-gdn.json"),
            p: include_str!("../../fixtures/r438/layer44-post.json"),
            s: include_str!("../../fixtures/r438/layer44-shared.json"),
        },
        45 => Manifests {
            g: include_str!("../../fixtures/r438/layer45-gdn.json"),
            p: include_str!("../../fixtures/r438/layer45-post.json"),
            s: include_str!("../../fixtures/r438/layer45-shared.json"),
        },
        46 => Manifests {
            g: include_str!("../../fixtures/r438/layer46-gdn.json"),
            p: include_str!("../../fixtures/r438/layer46-post.json"),
            s: include_str!("../../fixtures/r438/layer46-shared.json"),
        },
        _ => unreachable!(),
    }
}

fn qsa_manifests(layer: u32) -> (&'static str, &'static str, &'static str) {
    match layer {
        3 => (
            include_str!("../../fixtures/r430/layer3.json"),
            include_str!("../../fixtures/r431/layer3.json"),
            include_str!("../../fixtures/r431/shared3.json"),
        ),
        7 => (
            include_str!("../../fixtures/r437/layer7.json"),
            include_str!("../../fixtures/r438/layer7-ffn.json"),
            include_str!("../../fixtures/r438/layer7-shared.json"),
        ),
        11 => (
            include_str!("../../fixtures/r437/layer11.json"),
            include_str!("../../fixtures/r438/layer11-ffn.json"),
            include_str!("../../fixtures/r438/layer11-shared.json"),
        ),
        15 => (
            include_str!("../../fixtures/r437/layer15.json"),
            include_str!("../../fixtures/r438/layer15-ffn.json"),
            include_str!("../../fixtures/r438/layer15-shared.json"),
        ),
        19 => (
            include_str!("../../fixtures/r437/layer19.json"),
            include_str!("../../fixtures/r438/layer19-ffn.json"),
            include_str!("../../fixtures/r438/layer19-shared.json"),
        ),
        23 => (
            include_str!("../../fixtures/r437/layer23.json"),
            include_str!("../../fixtures/r438/layer23-ffn.json"),
            include_str!("../../fixtures/r438/layer23-shared.json"),
        ),
        27 => (
            include_str!("../../fixtures/r437/layer27.json"),
            include_str!("../../fixtures/r438/layer27-ffn.json"),
            include_str!("../../fixtures/r438/layer27-shared.json"),
        ),
        31 => (
            include_str!("../../fixtures/r437/layer31.json"),
            include_str!("../../fixtures/r438/layer31-ffn.json"),
            include_str!("../../fixtures/r438/layer31-shared.json"),
        ),
        35 => (
            include_str!("../../fixtures/r437/layer35.json"),
            include_str!("../../fixtures/r438/layer35-ffn.json"),
            include_str!("../../fixtures/r438/layer35-shared.json"),
        ),
        39 => (
            include_str!("../../fixtures/r437/layer39.json"),
            include_str!("../../fixtures/r438/layer39-ffn.json"),
            include_str!("../../fixtures/r438/layer39-shared.json"),
        ),
        43 => (
            include_str!("../../fixtures/r437/layer43.json"),
            include_str!("../../fixtures/r438/layer43-ffn.json"),
            include_str!("../../fixtures/r438/layer43-shared.json"),
        ),
        47 => (
            include_str!("../../fixtures/r437/layer47.json"),
            include_str!("../../fixtures/r438/layer47-ffn.json"),
            include_str!("../../fixtures/r438/layer47-shared.json"),
        ),
        _ => unreachable!(),
    }
}

fn load(
    layer: u32,
    id: &Identity,
    dense: &mut std::fs::File,
    shard: &mut std::fs::File,
) -> Result<
    (
        sc117_gdn_stage::Stage,
        sc117_post_attention::Stage,
        sc117_shared_stage::Stage,
    ),
    E,
> {
    let m = manifests(layer);
    Ok((
        sc117_gdn_stage::Stage::load(serde_json::from_str(m.g)?, id.clone(), dense, shard)?,
        sc117_post_attention::Stage::load(serde_json::from_str(m.p)?, id.clone(), dense, shard)?,
        sc117_shared_stage::Stage::load(serde_json::from_str(m.s)?, id.clone(), shard)?,
    ))
}
fn bytes_hash(values: &[f32]) -> String {
    let b: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    sha256::hex(&b)
}
struct Seeds {
    ids: Vec<u32>,
    hashes: Vec<String>,
}

// Remote execution remains subject to external resource admission. Construct
// a fixed SSH invocation; never interpolate arbitrary text into a shell command.
fn rank_endpoint(
    rank: u32,
    local: &str,
    host: Option<&str>,
    remote: Option<&str>,
) -> Result<Endpoint, E> {
    if !(2..=3).contains(&rank) || local.is_empty() {
        return Err("invalid worker endpoint".into());
    }
    let argv = match (host, remote) {
        (None, None) => vec![local.into()],
        (Some(host), Some(remote)) => {
            let valid_host = !host.is_empty()
                && host.len() <= 253
                && host.as_bytes()[0].is_ascii_alphanumeric()
                && host
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c));
            let valid_path = remote.starts_with('/')
                && remote.len() <= 4096
                && remote
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c));
            if !valid_host || !valid_path || remote == "/" {
                return Err("invalid SSH worker host or path".into());
            }
            vec![
                "/usr/bin/ssh".into(),
                "-T".into(),
                "-o".into(),
                "BatchMode=yes".into(),
                "-o".into(),
                "StrictHostKeyChecking=yes".into(),
                "-o".into(),
                "ConnectTimeout=10".into(),
                host.into(),
                remote.into(),
            ]
        }
        _ => return Err("SSH worker host and path must be supplied together".into()),
    };
    Ok(Endpoint { rank, argv })
}

fn optional_env(name: &str) -> Result<Option<String>, E> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(serde::Serialize)]
struct WorkerPlan<'a> {
    version: &'static str,
    model_revision: &'static str,
    index_sha256: &'static str,
    epoch: u64,
    layers: usize,
    payload_budget: u64,
    rank_limits: &'a Option<BTreeMap<u32, u64>>,
    endpoints: &'a BTreeMap<u32, Endpoint>,
}

fn worker_plan_id(
    endpoints: &BTreeMap<u32, Endpoint>,
    payload_budget: u64,
    rank_limits: &Option<BTreeMap<u32, u64>>,
) -> Result<String, E> {
    if payload_budget == 0
        || endpoints.len() != 2
        || [2, 3].iter().any(|rank| {
            endpoints
                .get(rank)
                .is_none_or(|v| v.rank != *rank || v.argv.is_empty())
        })
        || rank_limits.as_ref().is_some_and(|limits| {
            limits.len() != 2
                || [2, 3].iter().any(|rank| {
                    limits
                        .get(rank)
                        .is_none_or(|&n| n == 0 || n > payload_budget)
                })
        })
    {
        return Err("invalid worker plan".into());
    }
    let plan = WorkerPlan {
        version: "vf-sc117-worker-plan-v1",
        model_revision: "sc117-local-observation",
        index_sha256: INDEX_HASH,
        epoch: 1,
        layers: 48,
        payload_budget,
        rank_limits,
        endpoints,
    };
    Ok(format!("r484-{}", sha256::hex(&serde_json::to_vec(&plan)?)))
}

fn owners(
    layer: u32,
    preview: &Seeds,
    endpoints: &BTreeMap<u32, Endpoint>,
    index: &str,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
) -> Result<Owners, E> {
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut order = BTreeMap::new();
    for rank in 2..=3 {
        let endpoint = endpoints.get(&rank).ok_or("worker endpoint missing")?;
        let mut session = Session::connect(endpoint, Duration::from_secs(30))?;
        let mut locals = Vec::new();
        for (slot, &expert) in preview
            .ids
            .iter()
            .enumerate()
            .filter(|(i, _)| 2 + (*i as u32 % 2) == rank)
        {
            let local = (slot / 2) as u32;
            let blob = layout.read_verified_expert(
                pack,
                layer as usize,
                expert.into(),
                INDEX_HASH,
                &preview.hashes[slot],
            )?;
            let bind = Binding {
                model_revision: "sc117-local-observation".into(),
                rank,
                layer,
                expert,
                local,
                epoch: 1,
                index_sha256: INDEX_HASH.into(),
                blob_sha256: preview.hashes[slot].clone(),
            };
            session.load_native_ggml(
                &bind,
                index,
                pack.metadata()?.len(),
                &blob.bytes,
                Duration::from_secs(30),
            )?;
            copies.push(Replica {
                expert,
                rank,
                local,
            });
            locals.push(local);
        }
        session.seal_native_ggml(locals.len(), Duration::from_secs(30))?;
        order.insert(rank, locals);
        sessions.push(session);
    }
    Ok(Owners {
        sessions,
        registry: Registry { epoch: 1, copies },
        preferred: BTreeMap::new(),
        order,
    })
}

const BUDGET: u64 = 4 * 1024 * 1024 * 1024;
fn valid_close(
    receipt: &voltafabric::session::CloseReceipt,
    expected: u64,
    positions: u64,
) -> bool {
    !receipt.killed
        && receipt.exit_code == Some(0)
        && receipt.error.is_none()
        && receipt.bye_execs == Some(expected)
        && expected <= positions
}
fn run() -> Result<(), E> {
    let a: Vec<_> = std::env::args().collect();
    if !(11..=12).contains(&a.len()) {
        return Err(
            "usage: check DENSE SHARD PACK INDEX SESSION LEDGER PLE_SHARD PLE_MANIFEST LAYERS PROMPT_TOKEN_IDS [MAX_NEW_TOKENS]"
                .into(),
        );
    }
    let layer_count: usize = a[9].parse()?;
    if layer_count != 48 {
        return Err("full greedy qualification requires 48 layers".into());
    }
    let prompt_tokens: Vec<u32> = a[10].split(',').map(str::parse).collect::<Result<_, _>>()?;
    let max_new_tokens: usize = if a.len() == 12 { a[11].parse()? } else { 2 };
    if !(1..=8).contains(&max_new_tokens) {
        return Err("invalid generation limit".into());
    }
    if !(1..=8).contains(&prompt_tokens.len())
        || prompt_tokens.iter().any(|&t| t >= 248_320 || t == 248_046)
    {
        return Err("invalid prompt tokens".into());
    }
    let mut endpoints = BTreeMap::new();
    for rank in 2..=3 {
        let host = optional_env(&format!("VF_RANK{rank}_SSH_HOST"))?;
        let remote = optional_env(&format!("VF_RANK{rank}_SSH_WORKER"))?;
        endpoints.insert(
            rank,
            rank_endpoint(rank, &a[5], host.as_deref(), remote.as_deref())?,
        );
    }
    let mut dense = std::fs::File::open(&a[1])?;
    let mut shard = std::fs::File::open(&a[2])?;
    let mut pack = std::fs::File::open(&a[3])?;
    let index = std::fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX_HASH {
        return Err("index digest".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let worst_case_headroom =
        layout.worst_case_admission_bytes(sc117_router_cpu::ROUTED_EXPERTS)?;
    let budget = match std::env::var("VF_CPU_EXPERT_BUDGET") {
        Ok(value) => value.parse::<u64>()?,
        Err(std::env::VarError::NotPresent) => BUDGET,
        Err(error) => return Err(error.into()),
    };
    let seed_bytes = layout.worst_case_admission_bytes(2)?;
    if budget > BUDGET
        || seed_bytes
            .checked_add(worst_case_headroom)
            .is_none_or(|minimum| budget < minimum)
    {
        return Err("expert budget cannot cover seeds and worst-case headroom".into());
    }
    let rank_limits = match (
        std::env::var("VF_RANK2_BUDGET"),
        std::env::var("VF_RANK3_BUDGET"),
    ) {
        (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => None,
        (Ok(a), Ok(b)) => {
            let limits = BTreeMap::from([(2, a.parse::<u64>()?), (3, b.parse::<u64>()?)]);
            let seed_per_rank = layout.worst_case_admission_bytes(1)?;
            if limits.values().any(|&v| v < seed_per_rank || v > budget) {
                return Err("rank budget cannot cover seeds or exceeds global budget".into());
            }
            Some(limits)
        }
        _ => return Err("both rank budgets must be supplied".into()),
    };
    let plan_id = worker_plan_id(&endpoints, budget, &rank_limits)?;
    let m: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_reader(std::fs::File::open(&a[8])?)?;
    let weights =
        voltafabric::sc117_ple_weights::Weights::load(&mut shard, &m, "sc117-local-observation")?;
    let ple =
        voltafabric::sc117_ple::Table::open(std::fs::File::open(&a[7])?, 28_800_138_432, 192)?;
    let token_table =
        voltafabric::sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r484-prefill-greedy-{}-{attempt}", std::process::id());
    let head_id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 48,
        epoch: 1,
    };
    let head_manifest: voltafabric::sc117_head_cpu::Manifest =
        serde_json::from_str(include_str!("../../fixtures/r429/head.json"))?;
    let mut head = voltafabric::sc117_head_cpu::Head::load(
        head_manifest,
        head_id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut layers = Vec::new();
    let mut used = 0u64;
    let mut origin = voltafabric::sc117_expert_origin::Origin::open(shard.try_clone()?, &layout)?;
    let mut seeds = Vec::new();
    let mut seed_records = Vec::new();
    for layer in 0..layer_count {
        let ids = vec![0, 1];
        let hashes = ids
            .iter()
            .map(|&expert| origin.expert_digest(&layout, layer, expert.into()))
            .collect::<voltafabric::Result<Vec<_>>>()?;
        let seed = Seeds { ids, hashes };
        seeds.push(seed.hashes.clone());
        for (slot, &expert) in seed.ids.iter().enumerate() {
            seed_records.push(voltafabric::sc117_expert_source::Seed {
                layer: layer as u32,
                expert,
                rank: 2 + slot as u32,
                local: 0,
            });
        }
        for &expert in &seed.ids {
            used = used
                .checked_add(layout.expert(layer, expert.into())?.bytes)
                .ok_or("seed budget overflow")?;
        }
        if used > budget {
            return Err("seed budget exceeded".into());
        }
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer: layer as u32,
            epoch: 1,
        };
        let owners = owners(layer as u32, &seed, &endpoints, &index, &mut pack, &layout)?;
        if layer % 4 == 3 {
            let qsa = sc117_qsa_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).0)?,
                id.clone(),
                &mut dense,
                &mut shard,
                prompt_tokens.len() + max_new_tokens - 1,
            )?;
            let ffn = sc117_ffn_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).1)?,
                id.clone(),
                &mut dense,
            )?;
            let shared = sc117_shared_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).2)?,
                id.clone(),
                &mut shard,
            )?;
            layers.push(Layer::qsa(id, qsa, ffn, shared, owners)?);
        } else {
            let (g, post, shared) = load(layer as u32, &id, &mut dense, &mut shard)?;
            layers.push(Layer::gdn(id, g, post, shared, owners)?);
        }
    }
    // Replace layer 1 while retaining the layer order and owners.
    let mut iter = layers.into_iter();
    let first = iter.next().ok_or("missing layer 0")?;
    let second = iter
        .next()
        .ok_or("missing layer 1")?
        .attach_ple(ple, weights)?;
    let mut layers = vec![first, second];
    layers.extend(iter);
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let mut segment = Segment::new_ple_cpu_qualification(
        layers,
        plan_id.clone(),
        ledger,
        Duration::from_secs(30),
    )?;
    let source_config = voltafabric::sc117_expert_source::Config {
        identity: voltafabric::sc117_expert_source::Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            plan_id: plan_id.clone(),
            epoch: 1,
        },
        ranks: vec![2, 3],
        seeds: &seed_records,
        limit: budget,
    };
    let mut source = if let Some(limits) = rank_limits.clone() {
        voltafabric::sc117_expert_source::Source::new_with_rank_limits(
            &mut pack,
            &layout,
            &index,
            origin,
            source_config,
            limits,
        )?
    } else {
        voltafabric::sc117_expert_source::Source::new(
            &mut pack,
            &layout,
            &index,
            origin,
            source_config,
        )?
    };
    if source.used_bytes() != used {
        return Err("seed budget mismatch".into());
    }
    let mut input_tokens = Vec::new();
    let mut embedding_hashes = Vec::new();
    let mut output_hashes = Vec::new();
    let mut logits_hashes = Vec::new();
    let mut generated = Vec::new();
    let mut evictions = Vec::new();
    let mut rank_history = Vec::new();
    let mut current_token = prompt_tokens[0];
    for pos in 0..(prompt_tokens.len() + max_new_tokens - 1) {
        if pos < prompt_tokens.len() {
            current_token = prompt_tokens[pos];
        }
        input_tokens.push(current_token);
        let embedding = token_table.gather(current_token)?;
        embedding_hashes.push(bytes_hash(&embedding));
        let input: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
        let (output, removed) = segment.advance_token_with_headroom(
            attempt.checked_add(pos as u64).ok_or("attempt overflow")?,
            current_token as i32,
            &input,
            &mut source,
            worst_case_headroom,
            768,
        )?;
        evictions.extend(removed);
        if segment.position() != pos as u64 + 1 || segment.is_poisoned() {
            return Err("segment did not commit complete position".into());
        }
        let rank_used = source.rank_used_bytes();
        let accounted = rank_used
            .values()
            .try_fold(0u64, |total, value| total.checked_add(*value));
        if accounted != Some(source.used_bytes())
            || rank_limits.as_ref().is_some_and(|limits| {
                rank_used
                    .iter()
                    .any(|(rank, used)| limits.get(rank).is_none_or(|limit| used > limit))
            })
        {
            return Err("rank payload accounting mismatch".into());
        }
        rank_history.push(rank_used.clone());
        output_hashes.push(bytes_hash(&output));
        if pos + 1 < prompt_tokens.len() {
            continue;
        }
        let prediction = head.evaluate(&head_id, &output)?;
        logits_hashes.push(bytes_hash(&prediction.logits));
        generated.push(prediction.greedy);
        current_token = prediction.greedy;
        if current_token == 248_046 {
            break;
        }
    }
    let receipts = segment.close_with_counts();
    if receipts.len() != layer_count * 2
        || receipts
            .iter()
            .any(|(r, expected)| !valid_close(r, *expected, input_tokens.len() as u64))
    {
        return Err("worker shutdown receipt".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"up to eight supplied prompt tokens and eight greedy predictions through 48 CPU layers", "worker_transports":endpoints.iter().map(|(rank,endpoint)| (*rank, if endpoint.argv.len()==1 { "local" } else { "ssh" })).collect::<BTreeMap<_,_>>(), "expert_headroom_bytes":worst_case_headroom, "evictions":evictions, "requested_max_new_tokens":max_new_tokens, "layer_count":layer_count,"run_id":run_id,"plan_id":plan_id,"attempt_base":attempt,"prompt_tokens":prompt_tokens,"input_tokens":input_tokens,"generated_token_ids":generated,"logits_sha256":logits_hashes,"embedding_sha256":embedding_hashes,"output_sha256":output_hashes,"seed_experts":[0,1],"seed_hashes":seeds,"workers_closed":receipts.len(),"worker_exec_counts":receipts.iter().map(|(r, expected)| (r.rank, *expected, r.bye_execs)).collect::<Vec<_>>(),"payload_budget_bytes":budget,"resident_payload_bytes":source.used_bytes(),"rank_used_bytes":source.rank_used_bytes(),"rank_limits":rank_limits,"rank_history":rank_history,"dynamic_admissions":source.admissions()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("prefill-greedy qualification refused: {e}");
        std::process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_plan_binds_rank_endpoints_and_capacities() {
        let rank2 = rank_endpoint(2, "/worker", None, None).unwrap();
        let rank3 = rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/worker")).unwrap();
        let endpoints = BTreeMap::from([(2, rank2.clone()), (3, rank3.clone())]);
        let limits = Some(BTreeMap::from([(2, 100), (3, 200)]));
        let id = worker_plan_id(&endpoints, 300, &limits).unwrap();
        assert_eq!(id.len(), 69);
        assert_eq!(
            id,
            "r484-077e885b41da741d94ad4d21c8325f06d462670eb0a299572392434a39ec7054"
        );
        assert_eq!(id, worker_plan_id(&endpoints, 300, &limits).unwrap());
        let changed_host = BTreeMap::from([
            (2, rank2.clone()),
            (
                3,
                rank_endpoint(3, "/worker", Some("pc2-lan"), Some("/worker")).unwrap(),
            ),
        ]);
        assert_ne!(id, worker_plan_id(&changed_host, 300, &limits).unwrap());
        let changed_path = BTreeMap::from([
            (2, rank2.clone()),
            (
                3,
                rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/other")).unwrap(),
            ),
        ]);
        assert_ne!(id, worker_plan_id(&changed_path, 300, &limits).unwrap());
        assert_ne!(id, worker_plan_id(&endpoints, 301, &limits).unwrap());
        assert_ne!(id, worker_plan_id(&endpoints, 300, &None).unwrap());
        assert!(worker_plan_id(
            &BTreeMap::from([(2, rank2), (3, rank3.clone()), (4, rank3)]),
            300,
            &limits
        )
        .is_err());
        assert!(worker_plan_id(&endpoints, 300, &Some(BTreeMap::from([(2, 100)]))).is_err());
    }
    #[test]
    fn rank_endpoints_are_explicit_and_reject_shell_metacharacters() {
        let local = rank_endpoint(2, "/opt/vf-rank-session", None, None).unwrap();
        assert_eq!(local.rank, 2);
        assert_eq!(local.argv, vec!["/opt/vf-rank-session"]);
        let remote = rank_endpoint(
            3,
            "/opt/vf-rank-session",
            Some("pc1-lan"),
            Some("/opt/vf-rank-session"),
        )
        .unwrap();
        assert_eq!(remote.rank, 3);
        assert_eq!(
            remote.argv,
            vec![
                "/usr/bin/ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "pc1-lan",
                "/opt/vf-rank-session"
            ]
        );
        assert!(rank_endpoint(3, "/worker", Some("pc1-lan"), None).is_err());
        assert!(rank_endpoint(3, "/worker", None, Some("/worker")).is_err());
        for host in ["", "-oProxyCommand", "pc1;id", "user@host", "pc1\n"] {
            assert!(rank_endpoint(3, "/worker", Some(host), Some("/worker")).is_err());
        }
        for path in ["worker", "/", "/worker;id", "/worker$(id)", "/worker name"] {
            assert!(rank_endpoint(3, "/worker", Some("pc1-lan"), Some(path)).is_err());
        }
        assert!(rank_endpoint(4, "/worker", None, None).is_err());
    }
    #[test]
    fn close_requires_actual_dispatched_count_not_every_position() {
        let mut r = voltafabric::session::CloseReceipt {
            rank: 2,
            pid: 1,
            bye_execs: Some(3),
            peak_rss_kib: None,
            exit_code: Some(0),
            killed: false,
            stderr: String::new(),
            stderr_bytes: 0,
            error: None,
        };
        assert!(valid_close(&r, 3, 9));
        assert!(!valid_close(&r, 4, 9));
        assert!(!valid_close(&r, 3, 2));
        r.killed = true;
        assert!(!valid_close(&r, 3, 9));
        r.killed = false;
        r.bye_execs = None;
        assert!(!valid_close(&r, 3, 9));
        r.bye_execs = Some(3);
        r.exit_code = Some(1);
        assert!(!valid_close(&r, 3, 9));
    }
}
