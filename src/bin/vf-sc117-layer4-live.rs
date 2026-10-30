//! Bounded CPU integration diagnostic: actual layer-4 prefix and ten actual owners and shared expert, actual route.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Replica, Stamp},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    sc117_bound_layer,
    sc117_gdn_stage::{Manifest, Stage},
    sc117_post_attention, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
fn open(p: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let f = o.open(p)?;
    if !f.metadata()?.is_file() {
        return Err("not regular".into());
    }
    Ok(f)
}
fn read(p: &str, len: usize, sha: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut f = open(p)?;
    if f.metadata()?.len() != len as u64 {
        return Err("size mismatch".into());
    }
    let mut b = vec![0; len];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != sha {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 11 {
        return Err("usage: vf-sc117-reserved-integration MANIFEST DENSE SHARD RESIDUAL PACK INDEX WORKER LEDGER_DIR POST_MANIFEST SHARED_MANIFEST".into());
    }
    let mut manifest = open(&a[1])?;
    let len = manifest.metadata()?.len();
    if len == 0 || len > 16384 {
        return Err("manifest size".into());
    }
    let mut mb = Vec::new();
    manifest.read_to_end(&mut mb)?;
    if sha256::hex(&mb) != "e828fe4e01fe8c9a86201b79fa40f6e29ceca325b2d0a4ca9be954a939d8afca" {
        return Err("manifest digest".into());
    }
    let m: Manifest = serde_json::from_slice(&mb)?;
    let r = read(
        &a[4],
        40960,
        "653f8385328f68ebebfe966082717b7c566053d540081b0539d5e0370ea15b65",
    )?;
    let residual: Vec<f32> = r
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let index_len = open(&a[6])?.metadata()?.len();
    if index_len == 0 || index_len > 65536 {
        return Err("index bound".into());
    }
    let index_bytes = read(
        &a[6],
        index_len.try_into()?,
        "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
    )?;
    let index = String::from_utf8(index_bytes)?;
    let mut pack = open(&a[5])?;
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let stamp = Stamp {
        run_id: format!("r432-local-{}-{attempt}", std::process::id()),
        model_revision: m.model_revision.clone(),
        plan_id: "r432-layer4-gdn".into(),
        layer: 4,
        batch: 1,
        epoch: 1,
    };
    let id = Identity {
        run_id: stamp.run_id.clone(),
        model_revision: stamp.model_revision.clone(),
        layer: 4,
        epoch: 1,
    };
    let mut dense = open(&a[2])?;
    let mut shard = open(&a[3])?;
    let mut stage = Stage::load(m, id.clone(), &mut dense, &mut shard)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[8]))?;
    let ticket = ledger.reserve_ticket(&stamp, attempt)?;
    let proposal = stage.prepare_reserved(&ledger, &ticket, 0, &residual)?;
    let post_bytes = read(
        &a[9],
        1300,
        "4edc0334cb6bc661f10c74efbfb9989c9fca973cda15baa1b5c90450a5bb6e94",
    )?;
    let post_manifest: sc117_post_attention::Manifest = serde_json::from_slice(&post_bytes)?;
    let mut post =
        sc117_post_attention::Stage::load(post_manifest, id.clone(), &mut dense, &mut shard)?;
    let routed =
        sc117_bound_layer::Routed::prepare(&mut post, &ledger, &ticket, proposal, &residual)?;
    let shared_bytes = read(
        &a[10],
        711,
        "e07e3f8e57da20c1d6a806b6263807d76bd74bbf12d81f1d6d65c0e7be4b195b",
    )?;
    let shared_manifest: sc117_shared_stage::Manifest = serde_json::from_slice(&shared_bytes)?;
    let mut shared_stage =
        sc117_shared_stage::Stage::load(shared_manifest, id.clone(), &mut shard)?;
    let shared = routed.shared(&mut shared_stage)?;
    let ids = [433, 309, 447, 475, 261, 252, 123, 107, 448, 469];
    if routed
        .choices()
        .iter()
        .map(|c| c.expert)
        .collect::<Vec<_>>()
        != ids
    {
        return Err("unexpected route".into());
    }
    let mut activation_bytes = Vec::new();
    for x in routed.activation() {
        activation_bytes.extend_from_slice(&x.to_le_bytes());
    }
    if sha256::hex(&activation_bytes)
        != "92ad15e04506003bb34e72e5d8b50c67f1c14b0a2a29aa05abf4e86869701d4d"
    {
        return Err("unexpected activation".into());
    }
    let hashes = [
        "fb1b61d67ac89b66381e8e5b59e96c63df287edd2747d262b7264e68c68177e0",
        "ad3e98c5e6dad251481cf64c4e4f92c9084be49236a3f4c6e599e62a2794fb7b",
        "ce28378868cfaa7910b3dbfca7320054ad94aea07798f0f510519b7763788c17",
        "4fad7b12ce680a6835f13d4d4ecb67dd2c6b38f64e1b4c305c98e91a99ffb6eb",
        "aa7844bbd260279efcb02e9bc61a42c8054372d31d3eb8630a25552ad9442bab",
        "754e860228131e225c6cbe8812d67d1e2ccc86087923faf3fead6e2427266334",
        "ab79a3fd62af597183f03fd826a819db4da8991d69d6ffa9fc5b19b9f12c53dc",
        "4a57dcc039156d50a74de4c7a3ceff2b480ac9a6f1ac3f1f04b4427436c8e16b",
        "819674af46bf1a1ac908bdb5ceaa24b60b3f8228e47947f186915aba2e528a81",
        "a98997cd1e906bfa8f256f6e1cb7d3e21379fcbb93eb573fe3e5d78ed36b9165",
    ];
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut orders = BTreeMap::new();
    for rank in 2..=3 {
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![a[7].clone()],
            },
            Duration::from_secs(30),
        )?;
        let mut locals = Vec::new();
        for (slot, &expert) in ids
            .iter()
            .enumerate()
            .filter(|(slot, _)| 2 + (*slot as u32) % 2 == rank)
        {
            let local = (slot / 2) as u32;
            locals.push(local);
            let blob = layout.read_verified_expert(
                &mut pack,
                4,
                expert.into(),
                "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
                hashes[slot],
            )?;
            let binding = Binding {
                model_revision: stamp.model_revision.clone(),
                rank,
                layer: 4,
                expert,
                local,
                epoch: 1,
                index_sha256: layout.index_sha256().into(),
                blob_sha256: hashes[slot].into(),
            };
            session.load_native_ggml(
                &binding,
                &index,
                pack.metadata()?.len(),
                &blob.bytes,
                Duration::from_secs(30),
            )?;
            copies.push(Replica {
                expert,
                rank,
                local,
            });
        }
        session.seal_native_ggml(5, Duration::from_secs(30))?;
        orders.insert(rank, locals);
        sessions.push(session);
    }
    let weights: Vec<_> = routed.choices().iter().map(|c| c.weight).collect();
    let post_values = routed.attention_residual().to_vec();
    let inject = routed.injection().to_vec();
    let shared_values = shared.values().to_vec();
    let collected = routed.dispatch(
        &mut sessions,
        Registry { epoch: 1, copies },
        BTreeMap::new(),
        orders,
        &ledger,
        ticket,
        Duration::from_secs(30),
    )?;
    let collected_rows = collected.rows()[0].clone();
    let committed = routed.commit(&mut stage, shared, collected)?;
    if stage.next_token() != 1 {
        return Err("state not committed".into());
    }
    if ledger.reserve_ticket(&stamp, attempt + 1).is_ok() {
        return Err("claim replay accepted".into());
    }
    for s in sessions {
        let c = s.close(Duration::from_secs(30));
        if c.killed || c.exit_code != Some(0) || c.error.is_some() || c.bye_execs.is_none() {
            return Err("worker did not close cleanly".into());
        }
    }
    let mut bytes = Vec::with_capacity(40960);
    for v in &committed.residual {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let info = serde_json::json!({"scope":"CPU layer with bound route, shared and collector, two local ranks, layer-3 input", "run_id":stamp.run_id,"attempt":attempt,"layer":4,"residual_sha256":sha256::hex(&bytes),"next_token":stage.next_token(),"collector":collected_rows,"shared":shared_values,"post_attention":post_values,"injection":inject,"weights":weights});
    writeln!(std::io::stdout().lock(), "{info}")?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("reserved integration refused: {e}");
        std::process::exit(1);
    }
}
