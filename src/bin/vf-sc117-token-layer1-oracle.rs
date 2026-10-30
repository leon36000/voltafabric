//! One fresh layer-1 local CPU diagnostic; not independent parity or generation.
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
    sc117_bound_layer, sc117_gdn_stage,
    sc117_layer_chain::Owners,
    sc117_layer_completion::{self, LayerInputs, StampedValues},
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
    assert_eq!(layer, 1);
    Manifests {
        g: include_str!("../../fixtures/r438/layer1-gdn.json"),
        p: include_str!("../../fixtures/r438/layer1-post.json"),
        s: include_str!("../../fixtures/r438/layer1-shared.json"),
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
        run_id: format!("r434-preview-{layer}"),
        model_revision: "sc117-local-observation".into(),
        layer,
        epoch: 1,
    };
    let (mut g, mut post, mut shared) = load(layer, &id, dense, shard)?;
    let stamp = Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "preview-only-no-worker".into(),
        layer,
        batch: 0,
        epoch: 1,
    };
    let candidate = g.prepare(&stamp, 1, 0, input)?;
    let route = post.compute(&id, candidate.output(), input)?;
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
    if a.len() != 10 {
        return Err(
            "usage: check DENSE SHARD PACK INDEX WORKER LEDGER PLE_SHARD PLE_MANIFEST RESIDUAL"
                .into(),
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
    let input_bytes = std::fs::read(&a[9])?;
    if input_bytes.len() != 40960
        || sha256::hex(&input_bytes)
            != "f929a224afaf2832954242730520d71845cd16be3184b7ec0e7c192bed4bb0e8"
    {
        return Err("layer 0 residual identity".into());
    }
    let input: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let (_, embedding) = table.gather(42, [-1, -1])?;
    let pre = weights.evaluate(&embedding, &input, &vec![0.0; 10240 * 9])?;
    let preview = preview(1, &pre.residual, &mut dense, &mut shard, &mut pack, &layout)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r451-layer-oracle-{}-{attempt}", std::process::id());
    let id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        epoch: 1,
    };
    let (mut g, mut post, mut shared_stage) = load(1, &id, &mut dense, &mut shard)?;
    let mut owners = owners(1, &preview, &a[5], &index, &mut pack, &layout)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let stamp = Stamp {
        run_id: run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "r451-oracle-cpu".into(),
        layer: 1,
        batch: 0,
        epoch: 1,
    };
    let ticket = ledger.reserve_ticket(&stamp, attempt)?;
    let proposal = g.prepare_reserved(&ledger, &ticket, 0, &pre.residual)?;
    let o = proposal.output();
    let gdn: Vec<f32> = o
        .mixed
        .iter()
        .chain(&o.inject)
        .chain(&o.qkv)
        .chain(&o.recurrent)
        .chain(&o.gate)
        .chain(&o.beta)
        .copied()
        .collect();
    let mut extra = sc117_post_attention::Stage::load(
        serde_json::from_str(manifests(1).p)?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let traced = extra.compute(&id, proposal.output(), &pre.residual)?;
    let routed =
        sc117_bound_layer::Routed::prepare(&mut post, &ledger, &ticket, proposal, &pre.residual)?;
    let activation = routed.activation().to_vec();
    let post_attention = routed.attention_residual().to_vec();
    let inject = routed.injection().to_vec();
    if traced.residual != post_attention || traced.mixed != activation || traced.inject != inject {
        return Err("trace differs from bound route".into());
    }
    let choices = routed.choices().to_vec();
    let shared = routed.shared(&mut shared_stage)?;
    let shared_values = shared.values().to_vec();
    let collected = routed.dispatch(
        &mut owners.sessions,
        owners.registry.clone(),
        owners.preferred.clone(),
        owners.order.clone(),
        &ledger,
        ticket,
        Duration::from_secs(30),
    )?;
    let collector = collected.rows()[0].clone();
    let committed = routed.commit(&mut g, shared, collected)?;
    if g.next_token() != 1 || committed.residual != preview.residual {
        return Err("bound output differs from preview".into());
    }
    let receipts: Vec<_> = owners
        .sessions
        .into_iter()
        .map(|s| s.close(Duration::from_secs(30)))
        .collect();
    if receipts.len() != 2
        || receipts.iter().any(|r| {
            r.killed || r.exit_code != Some(0) || r.error.is_some() || r.bye_execs != Some(1)
        })
    {
        return Err("unclean worker shutdown".into());
    }
    println!(
        "{}",
        serde_json::json!({"run_id":run_id,"attempt":attempt,"ple_residual":pre.residual,"gdn":gdn,"z":traced.z,"normalized":traced.normalized,"attention":traced.attention,"activation":activation,"post_attention":post_attention,"injection":inject,"experts":choices.iter().map(|c|c.expert).collect::<Vec<_>>(),"weights":choices.iter().map(|c|c.weight).collect::<Vec<_>>(),"logits":choices.iter().map(|c|c.logit).collect::<Vec<_>>(),"collector":collector,"shared":shared_values,"residual_sha256":bytes_hash(&committed.residual),"workers_closed":receipts.len()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("PLE qualification refused: {e}");
        std::process::exit(1);
    }
}
