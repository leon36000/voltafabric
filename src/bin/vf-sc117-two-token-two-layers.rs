//! Two actual embeddings through layers 0 and 1, qualification only.
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
    sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, Segment},
    sc117_layer_completion::{LayerInputs, StampedValues},
    sc117_post_attention, sc117_shared_stage,
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
struct PreviewStage {
    id: Identity,
    g: sc117_gdn_stage::Stage,
    post: sc117_post_attention::Stage,
    shared: sc117_shared_stage::Stage,
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
    let candidate = stage.g.prepare(&stamp, position + 1, position, input)?;
    let route = stage.post.compute(id, candidate.output(), input)?;
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
    let residual = stage.g.commit_collected(candidate, inputs)?.residual;
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
    let manifest: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_reader(std::fs::File::open(&a[8])?)?;
    let weights = voltafabric::sc117_ple_weights::Weights::load(
        &mut shard,
        &manifest,
        "sc117-local-observation",
    )?;
    let table =
        voltafabric::sc117_ple::Table::open(std::fs::File::open(&a[7])?, 28_800_138_432, 192)?;
    let tokens = [42, 73];
    let token_table =
        voltafabric::sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let mut inputs = Vec::new();
    let mut embedding_hashes = Vec::new();
    for &token in &tokens {
        let v = token_table.gather(token)?;
        embedding_hashes.push(bytes_hash(&v));
        inputs.push((0..4).flat_map(|_| v.iter().copied()).collect::<Vec<f32>>());
    }
    let mut stages = Vec::new();
    for layer in 0..=1 {
        let id = Identity {
            run_id: format!("r453-preview-{layer}"),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let (g, post, shared) = load(layer, &id, &mut dense, &mut shard)?;
        stages.push(PreviewStage {
            id,
            g,
            post,
            shared,
        });
    }
    let mut previous = [-1, -1];
    let mut history = vec![0.0; 10240 * 9];
    let mut p0 = Vec::new();
    let mut p1 = Vec::new();
    let mut ple_hashes = Vec::new();
    let mut ple_residuals = Vec::new();
    for (pos, input) in inputs.iter().enumerate() {
        p0.push(preview(
            &mut stages[0],
            input,
            pos as u64,
            &mut pack,
            &layout,
        )?);
        let (_, emb) = table.gather(tokens[pos] as i32, previous)?;
        let candidate = weights.evaluate(&emb, &p0[pos].residual, &history)?;
        ple_hashes.push(bytes_hash(&candidate.residual));
        ple_residuals.push(candidate.residual.clone());
        p1.push(preview(
            &mut stages[1],
            &candidate.residual,
            pos as u64,
            &mut pack,
            &layout,
        )?);
        history = candidate.next_history;
        previous = [previous[1], tokens[pos] as i32];
    }
    drop(stages);
    let union = |previews: &[Preview]| -> Result<Preview, E> {
        let mut m = BTreeMap::new();
        for p in previews {
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
    let u0 = union(&p0)?;
    let u1 = union(&p1)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r453-two-tokens-{}-{attempt}", std::process::id());
    let mut layers = Vec::new();
    for (layer, preview) in [(0, &u0), (1, &u1)] {
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let (g, post, shared) = load(layer, &id, &mut dense, &mut shard)?;
        let owners = owners(layer, preview, &a[5], &index, &mut pack, &layout)?;
        layers.push(Layer::gdn(id, g, post, shared, owners)?);
    }
    let l = layers
        .pop()
        .ok_or("missing layer 1")?
        .attach_ple(table, weights)?;
    layers.push(l);
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let mut segment = Segment::new_ple_cpu_qualification(
        layers,
        "r453-cpu-only".into(),
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
            || output != p1[pos].residual
        {
            return Err("segment differs from shared-component preview".into());
        }
        output_hashes.push(bytes_hash(&output));
    }
    let receipts = segment.close();
    if receipts.len() != 4
        || receipts.iter().any(|r| {
            r.killed || r.exit_code != Some(0) || r.error.is_some() || r.bye_execs != Some(2)
        })
    {
        return Err("worker shutdown receipt".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"two actual token embeddings, two real layers and PLE; local CPU", "run_id":run_id,"attempt_base":attempt,"tokens":tokens,"embedding_sha256":embedding_hashes,"layer0_sha256":p0.iter().map(|p|bytes_hash(&p.residual)).collect::<Vec<_>>(),"ple_residual_sha256":ple_hashes,"ple_residuals":ple_residuals,"layer0_residuals":p0.iter().map(|p|&p.residual).collect::<Vec<_>>(),"output_sha256":output_hashes,"layer0_routes":p0.iter().map(|p|&p.ids).collect::<Vec<_>>(),"layer1_routes":p1.iter().map(|p|&p.ids).collect::<Vec<_>>(),"layer0_union":u0.ids,"layer0_hashes":u0.hashes,"layer1_union":u1.ids,"layer1_hashes":u1.hashes,"workers_closed":receipts.len()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("PLE two-position qualification refused: {e}");
        std::process::exit(1)
    }
}
