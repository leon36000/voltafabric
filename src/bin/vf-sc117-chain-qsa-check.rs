//! One fresh two-layer local CPU integration check. No GPU or distributed lease.
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
        4 => Manifests {
            g: include_str!("../../fixtures/r432/gdn.json"),
            p: include_str!("../../fixtures/r432/post.json"),
            s: include_str!("../../fixtures/r432/shared.json"),
        },
        5 => Manifests {
            g: include_str!("../../fixtures/r434/gdn5.json"),
            p: include_str!("../../fixtures/r434/post5.json"),
            s: include_str!("../../fixtures/r434/shared5.json"),
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
fn preview(
    layer: u32,
    input: &[f32],
    dense: &mut std::fs::File,
    shard: &mut std::fs::File,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
) -> Result<Preview, E> {
    let id = Identity {
        run_id: format!("r435-preview-{layer}"),
        model_revision: "sc117-local-observation".into(),
        layer,
        epoch: 1,
    };
    let stamp = Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "preview-only-no-worker".into(),
        layer,
        batch: 0,
        epoch: 1,
    };
    let (route, mut shared) = if layer == 3 {
        let mut qsa = sc117_qsa_stage::Stage::load(
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
        let proposal = qsa.prepare(&id, 0, input)?;
        let r = ffn.compute(
            &id,
            input,
            &proposal.output().attention,
            &proposal.output().inject,
        )?;
        let shared = sc117_shared_stage::Stage::load(
            serde_json::from_str(include_str!("../../fixtures/r431/shared3.json"))?,
            id.clone(),
            shard,
        )?;
        (r, shared)
    } else {
        let (mut g, mut post, shared) = load(layer, &id, dense, shard)?;
        let candidate = g.prepare(&stamp, 1, 0, input)?;
        let r = post.compute(&id, candidate.output(), input)?;
        (
            sc117_ffn_stage::Routed {
                residual: r.residual,
                mixed: r.mixed,
                inject: r.inject,
                choices: r.choices,
            },
            shared,
        )
    };
    let sv = shared.compute(&id, &route.mixed)?;
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
            attempt: 1,
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
        attempt: 1,
        values,
    };
    let residual = sc117_layer_completion::finish(LayerInputs {
        problem,
        attempt: 1,
        outputs: packets,
        shared: stamped(sv),
        attention_residual: stamped(route.residual),
        injection: stamped(route.inject),
    })?;
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
    if a.len() != 7 {
        return Err("usage: chain-check DENSE SHARD PACK INDEX WORKER LEDGER".into());
    }
    let mut dense = std::fs::File::open(&a[1])?;
    let mut shard = std::fs::File::open(&a[2])?;
    let mut pack = std::fs::File::open(&a[3])?;
    let index = std::fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX_HASH {
        return Err("index digest".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let input: Vec<f32> = (0..10240)
        .map(|i| ((i % 149) as f32 - 74.0) * 0.00011)
        .collect();
    let p3 = preview(3, &input, &mut dense, &mut shard, &mut pack, &layout)?;
    let p4 = preview(4, &p3.residual, &mut dense, &mut shard, &mut pack, &layout)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r435-chain-{}-{attempt}", std::process::id());
    let mut layers = Vec::new();
    for (layer, preview) in [(3, &p3), (4, &p4)] {
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let owners = owners(layer, preview, &a[5], &index, &mut pack, &layout)?;
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
            let (g, post, shared) = load(layer, &id, &mut dense, &mut shard)?;
            layers.push(Layer::gdn(id, g, post, shared, owners)?);
        }
    }
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let mut segment = Segment::new(
        layers,
        "r435-cpu-only".into(),
        ledger,
        Duration::from_secs(30),
    )?;
    let output = segment.advance(attempt, &input)?;
    if segment.position() != 1 || segment.is_poisoned() || output != p4.residual {
        return Err("chain differs from in-process preview".into());
    }
    let receipts = segment.close();
    if receipts.len() != 4
        || receipts.iter().any(|r| {
            r.killed || r.exit_code != Some(0) || r.error.is_some() || r.bye_execs != Some(1)
        })
    {
        return Err("worker shutdown receipt".into());
    }
    println!(
        "{}",
        serde_json::json!({"scope":"QSA layer 3 then GDN layer 4, local CPU only", "run_id":run_id, "attempt":attempt,
        "input_sha256":bytes_hash(&input), "layer3_sha256":bytes_hash(&p3.residual), "output_sha256":bytes_hash(&output),
        "layer3_experts":p3.ids, "layer4_experts":p4.ids, "workers_closed":receipts.len()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("chain check refused: {e}");
        std::process::exit(1);
    }
}
