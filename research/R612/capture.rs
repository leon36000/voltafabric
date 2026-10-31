//! Commit one observed layer-0 transaction before releasing its diagnostic values.
use std::{
    collections::BTreeMap,
    fs,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Replica, Stamp},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    sc117_bound_layer, sc117_embedding, sc117_gdn_stage, sc117_post_attention, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const SIZE: u64 = 50_292_326_400;
const IDS: [u32; 10] = [136, 317, 77, 257, 175, 414, 373, 142, 2, 226];
const DIGESTS: [&str; 10] = [
    "4afdad83a3af091e94d0b4265dda2bd8f905aa818b831940cd34e9009d205511",
    "f5714f3b62a17ed4e9f812e905eb978ae8f91c1ef7a8cebc8fa9fe42b7c42d35",
    "4c2fee7f5f2128f8acc3c644169d2d47e106bf53f7b8f76d9f4def7ba9c346fb",
    "72331450edfab4d94b6e6e0fdca6f40e811fbfb6da3d1f371754527c6b80ca1d",
    "f277de1ae0ecee1b82e108ad8a573c92e24b9db74d3dd91169d1b0e66b5c8d7f",
    "d3ac7f1e77dd59092aac4db08b5d3dd7c7db5e00d93e48480e69dbf5a07a4276",
    "f665692c081f9b3adcd3dc5e75401035fe930ab36c746fa530e146931fe594d5",
    "682f89850023e924ba9924309c9cc9559ae744d17f5cd6e0c10758a410c5ccd1",
    "b428f2d34ca933aa7c89b4f6fe87db30a575000d44fcbd4eeac1185cbe3eac60",
    "ab47298249651d73228d2661919d087826d107f7989f0fd104f26e8d9efd8b4c",
];
fn hash(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage DENSE SHARD PACK INDEX LEDGER".into());
    }
    let plan = std::env::var("VF_R612_PLAN")?;
    let run = std::env::var("VF_R612_RUN")?;
    let worker = std::env::var("VF_R612_WORKER")?;
    let wh = std::env::var("VF_R612_WORKER_SHA256")?;
    let mode = std::env::var("VF_R612_BACKEND")?;
    if !sha256::is_hex_digest(&plan)
        || !sha256::is_hex_digest(&wh)
        || !run.starts_with("r612-")
        || !matches!(mode.as_str(), "scalar" | "vector")
        || sha256::hex(&fs::read(&worker)?) != wh
    {
        return Err("identity or executable".into());
    }
    let index = fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX {
        return Err("index".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let mut dense = fs::File::open(&a[1])?;
    let mut shard = fs::File::open(&a[2])?;
    let mut pack = fs::File::open(&a[3])?;
    let table = sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let embedding = table.gather(248045)?;
    if hash(&embedding) != "ac914238a95d7950bcab749cf5d9358b3467ca3bafc520b1040a113568608423" {
        return Err("embedding".into());
    }
    let input: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
    let id = Identity {
        run_id: run.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        epoch: 1,
    };
    let mut gdn = sc117_gdn_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-gdn.json"))?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut post = sc117_post_attention::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-post.json"))?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut shared = sc117_shared_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer0-shared.json"))?,
        id.clone(),
        &mut shard,
    )?;
    let timeout = Duration::from_secs(60);
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut orders = BTreeMap::new();
    for rank in 2..=3 {
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![
                    worker.clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    wh.clone(),
                ],
            },
            timeout,
        )?;
        let mut locals = Vec::new();
        for (slot, &expert) in IDS
            .iter()
            .enumerate()
            .filter(|(i, _)| 2 + (*i as u32) % 2 == rank)
        {
            let local = (slot / 2) as u32;
            locals.push(local);
            let blob =
                layout.read_verified_expert(&mut pack, 0, expert.into(), INDEX, DIGESTS[slot])?;
            let binding = Binding {
                model_revision: id.model_revision.clone(),
                rank,
                layer: 0,
                expert,
                local,
                epoch: 1,
                index_sha256: INDEX.into(),
                blob_sha256: DIGESTS[slot].into(),
            };
            if mode == "scalar" {
                session.load_native_scalar_ggml(&binding, &index, SIZE, &blob.bytes, timeout)?
            } else {
                session.load_native_ggml(&binding, &index, SIZE, &blob.bytes, timeout)?
            }
            copies.push(Replica {
                expert,
                rank,
                local,
            });
        }
        session.seal_native_ggml(5, timeout)?;
        orders.insert(rank, locals);
        sessions.push(session);
    }
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[5]))?;
    let stamp = Stamp {
        run_id: run.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: format!("r612-{mode}-{plan}"),
        layer: 0,
        batch: 0,
        epoch: 1,
    };
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let ticket = ledger.reserve_ticket(&stamp, attempt)?;
    let proposal = gdn.prepare_reserved(&ledger, &ticket, 0, &input)?;
    let routed = sc117_bound_layer::Routed::prepare(&mut post, &ledger, &ticket, proposal, &input)?;
    let ids: Vec<_> = routed.choices().iter().map(|c| c.expert).collect();
    if ids != IDS {
        return Err("route differs from R611".into());
    }
    let weights: Vec<_> = routed.choices().iter().map(|c| c.weight).collect();
    let activation = routed.activation().to_vec();
    let shared_values = routed.shared(&mut shared)?;
    let collected = routed.dispatch(
        &mut sessions,
        Registry { epoch: 1, copies },
        BTreeMap::new(),
        orders,
        &ledger,
        ticket,
        timeout,
    )?;
    let row = collected.rows()[0].clone();
    let output = routed.commit(&mut gdn, shared_values, collected)?;
    if gdn.next_token() != 1 {
        return Err("state did not advance".into());
    }
    let expected = if mode == "scalar" {
        "36e796998107c83d5aacdcd436325bd6c463f8e8085e56364956b739cee4a5b8"
    } else {
        "dfd0dcd481d28eebfbdc0376299ba272eaae1896ec0c042bb74c5cb2a0108ae3"
    };
    if hash(&output.residual) != expected {
        return Err("residual differs from R611".into());
    }
    for s in sessions {
        let c = s.close(timeout);
        if c.killed || c.error.is_some() || c.exit_code != Some(0) || c.bye_execs != Some(1) {
            return Err("worker close".into());
        }
    }
    Ok(
        serde_json::json!({"run_id":run,"plan_id":stamp.plan_id,"attempt":attempt,"backend":mode,"ids":ids,"weights":weights,"activation_sha256":hash(&activation),"activation":activation,"collector":row,"residual_sha256":expected,"workers_closed":2}),
    )
}
fn main() {
    match run() {
        Ok(x) => println!("{x}"),
        Err(e) => {
            eprintln!("capture refused: {e}");
            std::process::exit(1)
        }
    }
}
