//! Four-layer SC117 CPU qualification; not whole-model generation.
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    completion::{self, OutputSlot, RoutedRawOutput},
    gdn_cpu::Identity,
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::{Binding, Resident},
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    sc117_ffn_stage, sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_layer_completion::{self, LayerInputs, StampedValues},
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
fn binding(
    layout: &NativeLayout,
    pack: &mut std::fs::File,
    layer: u32,
    expert: u32,
    rank: u32,
    local: u32,
) -> Result<
    (
        Binding,
        voltafabric::strata_native_layout::VerifiedNativeExpert,
    ),
    E,
> {
    let span = layout.expert(layer as usize, expert.into())?;
    let mut bytes = vec![0; span.bytes as usize];
    pack.seek(SeekFrom::Start(span.offset))?;
    pack.read_exact(&mut bytes)?;
    let digest = sha256::hex(&bytes);
    let verified =
        layout.read_verified_expert(pack, layer as usize, expert.into(), INDEX_HASH, &digest)?;
    let binding = Binding {
        model_revision: "sc117-local-observation".into(),
        rank,
        layer,
        expert,
        local,
        epoch: 1,
        index_sha256: INDEX_HASH.into(),
        blob_sha256: digest,
    };
    Ok((binding, verified))
}

struct Preview {
    ids: Vec<u32>,
    hashes: Vec<String>,
    residual: Vec<f32>,
}
enum PreviewAttention {
    Qsa(sc117_qsa_stage::Stage, sc117_ffn_stage::Stage),
    Gdn(sc117_gdn_stage::Stage, sc117_post_attention::Stage),
}
struct PreviewStage {
    id: Identity,
    attention: PreviewAttention,
    shared: sc117_shared_stage::Stage,
}
impl PreviewStage {
    fn load(layer: u32, dense: &mut std::fs::File, shard: &mut std::fs::File) -> Result<Self, E> {
        let id = Identity {
            run_id: format!("r436-preview-{layer}"),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let (attention, shared) = if layer == 3 {
            let qsa = sc117_qsa_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r430/layer3.json"))?,
                id.clone(),
                dense,
                shard,
                2,
            )?;
            let ffn = sc117_ffn_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r431/layer3.json"))?,
                id.clone(),
                dense,
            )?;
            let shared = sc117_shared_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r431/shared3.json"))?,
                id.clone(),
                shard,
            )?;
            (PreviewAttention::Qsa(qsa, ffn), shared)
        } else {
            let (g, post, shared) = load(layer, &id, dense, shard)?;
            (PreviewAttention::Gdn(g, post), shared)
        };
        Ok(Self {
            id,
            attention,
            shared,
        })
    }
}
enum Candidate {
    Qsa(sc117_qsa_stage::Proposal),
    Gdn(sc117_gdn_stage::Proposal),
}
fn preview(
    stage: &mut PreviewStage,
    input: &[f32],
    position: u64,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
) -> Result<Preview, E> {
    let id = &stage.id;
    let layer = id.layer;
    let stamp = Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "preview-only-no-worker".into(),
        layer,
        batch: position,
        epoch: 1,
    };
    let (route, candidate) = match &mut stage.attention {
        PreviewAttention::Qsa(qsa, ffn) => {
            let p = qsa.prepare(id, position as usize, input)?;
            let r = ffn.compute(id, input, &p.output().attention, &p.output().inject)?;
            (r, Candidate::Qsa(p))
        }
        PreviewAttention::Gdn(g, post) => {
            let p = g.prepare(&stamp, position + 1, position, input)?;
            let r = post.compute(id, p.output(), input)?;
            (
                sc117_ffn_stage::Routed {
                    residual: r.residual,
                    mixed: r.mixed,
                    inject: r.inject,
                    choices: r.choices,
                },
                Candidate::Gdn(p),
            )
        }
    };
    let sv = stage.shared.compute(id, &route.mixed)?;
    let mut copies = Vec::new();
    let mut packets = Vec::new();
    let mut ids = Vec::new();
    let mut hashes = Vec::new();
    for (slot, choice) in route.choices.iter().enumerate() {
        let rank = 2 + (slot as u32 % 2);
        let local = (slot / 2) as u32;
        let (bind, blob) = binding(layout, pack, layer, choice.expert, rank, local)?;
        let mut resident = Resident::load(layout, blob, bind.clone())?;
        let row = resident.forward(&route.mixed)?;
        let bits = row
            .iter()
            .map(|x| completion::encode_fp16(f64::from(*x)))
            .collect::<voltafabric::Result<Vec<_>>>()?;
        copies.push(Replica {
            expert: choice.expert,
            rank,
            local,
        });
        packets.push(RoutedRawOutput {
            stamp: stamp.clone(),
            rank,
            attempt: position + 1,
            rows: 1,
            columns: 2560,
            slots: vec![OutputSlot {
                token: 0,
                slot,
                expert: choice.expert,
                local,
            }],
            fp16_bits: bits,
        });
        ids.push(choice.expert);
        hashes.push(bind.blob_sha256);
    }
    let problem = RoutingProblem {
        hidden: 2560,
        experts: ids.clone(),
        routes: vec![route
            .choices
            .iter()
            .map(|c| Choice {
                expert: c.expert,
                weight: f64::from(c.weight),
            })
            .collect()],
        registry: Registry { epoch: 1, copies },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0, 1, 2, 3, 4]), (3, vec![0, 1, 2, 3, 4])]),
    };
    let stamped = |values| StampedValues {
        stamp: stamp.clone(),
        attempt: position + 1,
        values,
    };
    let inputs = LayerInputs {
        problem,
        attempt: position + 1,
        outputs: packets,
        shared: stamped(sv),
        attention_residual: stamped(route.residual),
        injection: stamped(route.inject),
    };
    let residual = match (&mut stage.attention, candidate) {
        (PreviewAttention::Qsa(qsa, _), Candidate::Qsa(p)) => {
            let r = sc117_layer_completion::finish(inputs)?;
            qsa.commit(p)?;
            r
        }
        (PreviewAttention::Gdn(g, _), Candidate::Gdn(p)) => g.commit_collected(p, inputs)?.residual,
        _ => return Err("preview candidate mismatch".into()),
    };
    Ok(Preview {
        ids,
        hashes,
        residual,
    })
}

fn owners(
    layer: u32,
    preview: &Preview,
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

fn run() -> Result<(), E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 9 {
        return Err(
            "usage: check DENSE SHARD PACK INDEX SESSION LEDGER PLE_SHARD PLE_MANIFEST".into(),
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
    let m: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_reader(std::fs::File::open(&a[8])?)?;
    let weights =
        voltafabric::sc117_ple_weights::Weights::load(&mut shard, &m, "sc117-local-observation")?;
    let ple =
        voltafabric::sc117_ple::Table::open(std::fs::File::open(&a[7])?, 28_800_138_432, 192)?;
    let token_table =
        voltafabric::sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let tokens = [42u32, 73u32];
    let mut inputs = Vec::new();
    let mut embedding_hashes = Vec::new();
    for &token in &tokens {
        let v = token_table.gather(token)?;
        embedding_hashes.push(bytes_hash(&v));
        inputs.push((0..4).flat_map(|_| v.iter().copied()).collect::<Vec<f32>>());
    }
    let mut stages = Vec::new();
    for layer in 0..4 {
        stages.push(PreviewStage::load(layer, &mut dense, &mut shard)?);
    }
    let mut previous = [-1, -1];
    let mut history = vec![0.0; 10240 * 9];
    let mut per_layer: Vec<Vec<Preview>> = (0..4).map(|_| Vec::new()).collect();
    for (pos, input) in inputs.iter().enumerate() {
        let mut residual = input.clone();
        for layer in 0..4 {
            if layer == 1 {
                let (_, emb) = ple.gather(tokens[pos] as i32, previous)?;
                let c = weights.evaluate(&emb, &residual, &history)?;
                residual = c.residual;
                history = c.next_history;
                previous = [previous[1], tokens[pos] as i32];
            }
            let p = preview(
                &mut stages[layer],
                &residual,
                pos as u64,
                &mut pack,
                &layout,
            )?;
            residual = p.residual.clone();
            per_layer[layer].push(p);
        }
    }
    drop(stages);
    let union = |items: &[Preview]| -> Result<Preview, E> {
        let mut m = BTreeMap::new();
        for p in items {
            for (&id, hash) in p.ids.iter().zip(&p.hashes) {
                if let Some(old) = m.insert(id, hash.clone()) {
                    if old != *hash {
                        return Err("expert hash changed".into());
                    }
                }
            }
        }
        let (ids, hashes) = m.into_iter().unzip();
        Ok(Preview {
            ids,
            hashes,
            residual: Vec::new(),
        })
    };
    let unions = per_layer
        .iter()
        .map(|p| union(p))
        .collect::<Result<Vec<_>, _>>()?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r454-four-layers-{}-{attempt}", std::process::id());
    let mut layers = Vec::new();
    for (layer, u) in unions.iter().enumerate() {
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer: layer as u32,
            epoch: 1,
        };
        let owners = owners(layer as u32, u, &a[5], &index, &mut pack, &layout)?;
        if layer == 3 {
            let qsa = sc117_qsa_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r430/layer3.json"))?,
                id.clone(),
                &mut dense,
                &mut shard,
                2,
            )?;
            let ffn = sc117_ffn_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r431/layer3.json"))?,
                id.clone(),
                &mut dense,
            )?;
            let shared = sc117_shared_stage::Stage::load(
                serde_json::from_str(include_str!("../../fixtures/r431/shared3.json"))?,
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
        "r454-cpu-only".into(),
        ledger,
        Duration::from_secs(30),
    )?;
    let mut output_hashes = Vec::new();
    for pos in 0..2 {
        let output = segment.advance_token(
            attempt.checked_add(pos as u64).ok_or("attempt overflow")?,
            tokens[pos] as i32,
            &inputs[pos],
        )?;
        if segment.position() != pos as u64 + 1
            || segment.is_poisoned()
            || output != per_layer[3][pos].residual
        {
            return Err("four-layer segment differs from shared-component preview".into());
        }
        output_hashes.push(bytes_hash(&output));
    }
    let receipts = segment.close();
    if receipts.len() != 8
        || receipts.iter().any(|r| {
            r.killed || r.exit_code != Some(0) || r.error.is_some() || r.bye_execs != Some(2)
        })
    {
        return Err("worker shutdown receipt".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"two observed tokens through GDN layers 0-2 and QSA layer 3 with PLE; local CPU", "run_id":run_id,"attempt_base":attempt,"tokens":tokens,"embedding_sha256":embedding_hashes,"layer_sha256":per_layer.iter().map(|p|p.iter().map(|x|bytes_hash(&x.residual)).collect::<Vec<_>>()).collect::<Vec<_>>(),"output_sha256":output_hashes,"routes":per_layer.iter().map(|p|p.iter().map(|x|&x.ids).collect::<Vec<_>>()).collect::<Vec<_>>(),"unions":unions.iter().map(|u|&u.ids).collect::<Vec<_>>(),"union_hashes":unions.iter().map(|u|&u.hashes).collect::<Vec<_>>(),"workers_closed":receipts.len()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("four-layer qualification refused: {e}");
        std::process::exit(1)
    }
}
