//! Fault injection against a child session owned by this fresh CPU diagnostic.
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

fn children() -> Result<std::collections::BTreeSet<u32>, E> {
    let p = format!("/proc/self/task/{}/children", std::process::id());
    std::fs::read_to_string(p)?
        .split_whitespace()
        .map(|x| x.parse::<u32>().map_err(Into::into))
        .collect()
}
fn owners(
    layer: u32,
    preview: &Preview,
    worker: &str,
    index: &str,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
) -> Result<(Owners, u32), E> {
    let mut victim = None;
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut order = BTreeMap::new();
    for rank in 2..=3 {
        let before = children()?;
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![worker.into()],
            },
            Duration::from_secs(30),
        )?;
        if rank == 3 {
            let after = children()?;
            let new: Vec<_> = after.difference(&before).copied().collect();
            if new.len() != 1 {
                return Err("cannot identify owned rank 3 child".into());
            }
            victim = Some(new[0]);
        }
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
    Ok((
        Owners {
            sessions,
            registry: Registry { epoch: 1, copies },
            preferred: BTreeMap::new(),
            order,
        },
        victim.ok_or("missing victim")?,
    ))
}

fn run() -> Result<(), E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 9 {
        return Err(
            "usage: check DENSE SHARD PACK INDEX WORKER LEDGER PLE_SHARD PLE_MANIFEST".into(),
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
    let input: Vec<f32> = (0..10240)
        .map(|i| ((i % 137) as f32 - 68.0) * 0.0001)
        .collect();
    let (_, embedding) = table.gather(42, [-1, -1])?;
    let pre = weights.evaluate(&embedding, &input, &vec![0.0; 10240 * 9])?;
    let preview = preview(1, &pre.residual, &mut dense, &mut shard, &mut pack, &layout)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r448-ple-fault-{}-{attempt}", std::process::id());
    let id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        epoch: 1,
    };
    let (g, post, shared) = load(1, &id, &mut dense, &mut shard)?;
    let (owners, victim) = owners(1, &preview, &a[5], &index, &mut pack, &layout)?;
    let layer = Layer::gdn(id, g, post, shared, owners)?.attach_ple(table, weights)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let mut segment = Segment::new_ple_cpu_qualification(
        vec![layer],
        "r448-fault-cpu".into(),
        ledger,
        Duration::from_secs(30),
    )?;
    let before = segment.ple_qualification_snapshot()?;
    if !children()?.contains(&victim) {
        return Err("victim is not our current child".into());
    }
    let status = std::process::Command::new("/bin/kill")
        .args(["-KILL", &victim.to_string()])
        .status()?;
    if !status.success() {
        return Err("failed to kill owned rank-3 child".into());
    }
    let error = match segment.advance_token(attempt, 42, &input) {
        Ok(_) => return Err("partial output accepted".into()),
        Err(e) => e,
    };
    let after = segment.ple_qualification_snapshot()?;
    let before_rank2 = before
        .sessions
        .iter()
        .find(|s| s.0 == 2)
        .ok_or("rank 2 missing")?;
    let after_rank2 = after
        .sessions
        .iter()
        .find(|s| s.0 == 2)
        .ok_or("rank 2 missing")?;
    let received = after_rank2.1.saturating_sub(before_rank2.1);
    if received < 1
        || !segment.is_poisoned()
        || segment.position() != 0
        || after.position != 0
        || after.attention_position != 0
        || before.history_sha256 != after.history_sha256
        || before.previous != after.previous
        || after.sessions.iter().any(|s| s.2)
    {
        return Err("failure atomicity or rank-2 progress not established".into());
    }
    if segment
        .advance_token(attempt.checked_add(1).ok_or("overflow")?, 73, &input)
        .is_ok()
    {
        return Err("poisoned segment accepted retry".into());
    }
    let ledger2 = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let stamp = Stamp {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        plan_id: "r448-fault-cpu".into(),
        layer: 1,
        batch: 0,
        epoch: 1,
    };
    if ledger2
        .reserve_ticket(&stamp, attempt.checked_add(1).ok_or("overflow")?)
        .is_ok()
    {
        return Err("consumed claim accepted".into());
    }
    let receipts = segment.close();
    println!(
        "{}",
        serde_json::json!({"scope":"CPU rank-3 child killed before dispatch; rank 2 delivered one response; no partial output", "run_id":run_id,"attempt":attempt,"error":format!("{error}"),"rank2_new_frames":received,"expected_preview_sha256":bytes_hash(&preview.residual),"before":before,"after":after,"segment_poisoned":true,"duplicate_refused":true,"receipts":receipts})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("PLE qualification refused: {e}");
        std::process::exit(1);
    }
}
