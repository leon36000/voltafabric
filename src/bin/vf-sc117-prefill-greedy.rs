//! Bounded two-token prompt and two-token CPU generation qualification.
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
    sc117_post_attention, sc117_qsa_stage, sc117_shared_stage,
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

fn owners(
    layer: u32,
    preview: &Seeds,
    worker: &str,
    index: &str,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
) -> Result<Owners, E> {
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut order = BTreeMap::new();
    for rank in 2..=3 {
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![worker.into()],
            },
            Duration::from_secs(30),
        )?;
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
    if !(1..=4).contains(&max_new_tokens) {
        return Err("invalid generation limit".into());
    }
    if !(1..=2).contains(&prompt_tokens.len())
        || prompt_tokens.iter().any(|&t| t >= 248_320 || t == 248_046)
    {
        return Err("invalid prompt tokens".into());
    }
    let mut dense = std::fs::File::open(&a[1])?;
    let mut shard = std::fs::File::open(&a[2])?;
    let mut pack = std::fs::File::open(&a[3])?;
    let index = std::fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX_HASH {
        return Err("index digest".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let m: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_reader(std::fs::File::open(&a[8])?)?;
    let weights =
        voltafabric::sc117_ple_weights::Weights::load(&mut shard, &m, "sc117-local-observation")?;
    let ple =
        voltafabric::sc117_ple::Table::open(std::fs::File::open(&a[7])?, 28_800_138_432, 192)?;
    let token_table =
        voltafabric::sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r470-prefill-greedy-{}-{attempt}", std::process::id());
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
        if used > BUDGET {
            return Err("seed budget exceeded".into());
        }
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer: layer as u32,
            epoch: 1,
        };
        let owners = owners(layer as u32, &seed, &a[5], &index, &mut pack, &layout)?;
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
        "r470-cpu-only".into(),
        ledger,
        Duration::from_secs(30),
    )?;
    let mut source = voltafabric::sc117_expert_source::Source::new(
        &mut pack,
        &layout,
        &index,
        origin,
        voltafabric::sc117_expert_source::Config {
            identity: voltafabric::sc117_expert_source::Identity {
                run_id: run_id.clone(),
                model_revision: "sc117-local-observation".into(),
                plan_id: "r470-cpu-only".into(),
                epoch: 1,
            },
            ranks: vec![2, 3],
            seeds: &seed_records,
            limit: BUDGET,
        },
    )?;
    if source.used_bytes() != used {
        return Err("seed budget mismatch".into());
    }
    let mut input_tokens = Vec::new();
    let mut embedding_hashes = Vec::new();
    let mut output_hashes = Vec::new();
    let mut logits_hashes = Vec::new();
    let mut generated = Vec::new();
    let mut current_token = prompt_tokens[0];
    for pos in 0..(prompt_tokens.len() + max_new_tokens - 1) {
        if pos < prompt_tokens.len() {
            current_token = prompt_tokens[pos];
        }
        input_tokens.push(current_token);
        let embedding = token_table.gather(current_token)?;
        embedding_hashes.push(bytes_hash(&embedding));
        let input: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
        let output = segment.advance_token_with_source(
            attempt.checked_add(pos as u64).ok_or("attempt overflow")?,
            current_token as i32,
            &input,
            &mut source,
        )?;
        if segment.position() != pos as u64 + 1 || segment.is_poisoned() {
            return Err("segment did not commit complete position".into());
        }
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
    let receipts = segment.close();
    if receipts.len() != layer_count * 2
        || receipts.iter().any(|r| {
            r.killed
                || r.exit_code != Some(0)
                || r.error.is_some()
                || r.bye_execs != Some(input_tokens.len() as u64)
        })
    {
        return Err("worker shutdown receipt".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"up to two supplied prompt tokens and four greedy predictions through 48 local CPU layers", "requested_max_new_tokens":max_new_tokens, "layer_count":layer_count,"run_id":run_id,"attempt_base":attempt,"prompt_tokens":prompt_tokens,"input_tokens":input_tokens,"generated_token_ids":generated,"logits_sha256":logits_hashes,"embedding_sha256":embedding_hashes,"output_sha256":output_hashes,"seed_experts":[0,1],"seed_hashes":seeds,"workers_closed":receipts.len(),"payload_budget_bytes":BUDGET,"resident_payload_bytes":source.used_bytes(),"dynamic_admissions":source.admissions()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("prefill-greedy qualification refused: {e}");
        std::process::exit(1)
    }
}
