//! Bounded CPU integration diagnostic: actual layer-2 prefix and ten actual owners and shared expert, actual route.
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
fn children() -> Result<std::collections::BTreeSet<u32>, Box<dyn std::error::Error>> {
    let p = format!("/proc/self/task/{}/children", std::process::id());
    std::fs::read_to_string(p)?
        .split_whitespace()
        .map(|x| x.parse::<u32>().map_err(Into::into))
        .collect()
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
    if sha256::hex(&mb) != "d822695e4fceee7b285b510e9ec7f97cc0a1507188b6ca698de5b6b216073348" {
        return Err("manifest digest".into());
    }
    let m: Manifest = serde_json::from_slice(&mb)?;
    let r = read(
        &a[4],
        40960,
        "d7f95ab0388246f01d0345783acbff0ba64e74087cdbd1cb22d624b093cdf579",
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
        run_id: format!("r427-local-{}-{attempt}", std::process::id()),
        model_revision: m.model_revision.clone(),
        plan_id: "r427-fault-after-rank2".into(),
        layer: 2,
        batch: 1,
        epoch: 1,
    };
    let id = Identity {
        run_id: stamp.run_id.clone(),
        model_revision: stamp.model_revision.clone(),
        layer: 2,
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
        1299,
        "8959d5ff4bbac4c6be08e6a978c5d27ffa1f09f8e2f3541add9395368121b0f1",
    )?;
    let post_manifest: sc117_post_attention::Manifest = serde_json::from_slice(&post_bytes)?;
    let mut post =
        sc117_post_attention::Stage::load(post_manifest, id.clone(), &mut dense, &mut shard)?;
    let routed =
        sc117_bound_layer::Routed::prepare(&mut post, &ledger, &ticket, proposal, &residual)?;
    let shared_bytes = read(
        &a[10],
        710,
        "2a9a91e11381fb9ef59636437c7f4649682cc1a89877e44818cd950c097e5d0c",
    )?;
    let shared_manifest: sc117_shared_stage::Manifest = serde_json::from_slice(&shared_bytes)?;
    let mut shared_stage =
        sc117_shared_stage::Stage::load(shared_manifest, id.clone(), &mut shard)?;
    let _shared = routed.shared(&mut shared_stage)?;
    let ids = [75, 12, 209, 181, 257, 272, 480, 492, 159, 255];
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
        != "903a85d59568dc86043d2a16cb1751499f59ee930ccdb61c80aa750556483ced"
    {
        return Err("unexpected activation".into());
    }
    let hashes = [
        "a312f304226bc4cfb7590ea90e9d91babbba67ef0fe4686a0eae85e26a9ed74d",
        "658b2b7b9c7b5055667577babea78964084e1a1dad958c546c46d513249f578e",
        "6271454ee230559ad5f701f94f8af7edc4b3ba7f10ff98b4763d09837c51db61",
        "98a127792dc0ae3709e7c714225f994913404e880e61e55794c0bf4fcb4d77ea",
        "6f626990dabe150f19c95ae1576c1563a98909e2333bb01aea0d0984891491ca",
        "d3db2278072ae0ac7eb8c42b1597c6fea5a10469a81e4bf6b01274eb49fd26c5",
        "a72edc2b4ad6258cb63946914960d24bbbcf763809be93824d0c51e3ce3a727a",
        "2e9bd3f3382db3c813301e257e8d0a603545d6a9e6d939c4715d6d7b4f5457c0",
        "694842ad3656f0f76eb8307520b83c314bb82fa123e8a7762256c4d9a2b06161",
        "8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5",
    ];
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut orders = BTreeMap::new();
    let mut victim = None;
    for rank in 2..=3 {
        let before = children()?;
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![a[7].clone()],
            },
            Duration::from_secs(30),
        )?;
        if rank == 3 {
            let after = children()?;
            let new: Vec<_> = after.difference(&before).copied().collect();
            if new.len() != 1 {
                return Err("cannot identify owned worker".into());
            }
            victim = Some(new[0]);
        }
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
                2,
                expert.into(),
                "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
                hashes[slot],
            )?;
            let binding = Binding {
                model_revision: stamp.model_revision.clone(),
                rank,
                layer: 2,
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
    let victim = victim.ok_or("missing victim")?;
    if !children()?.contains(&victim) {
        return Err("victim is not a current child".into());
    }
    let status = std::process::Command::new("/bin/kill")
        .args(["-KILL", &victim.to_string()])
        .status()?;
    if !status.success() {
        return Err("failed to stop owned rank-3 worker".into());
    }
    let before = sessions[0].traffic.frames_received;
    let result = routed.dispatch(
        &mut sessions,
        Registry { epoch: 1, copies },
        BTreeMap::new(),
        orders,
        &ledger,
        ticket,
        Duration::from_secs(30),
    );
    let error = match result {
        Ok(_) => return Err("partial batch accepted".into()),
        Err(e) => e,
    };
    let received = sessions[0].traffic.frames_received.saturating_sub(before);
    if received < 2 || sessions.iter().any(|s| s.is_usable()) || stage.next_token() != 0 {
        return Err("failure atomicity or first-rank progress not demonstrated".into());
    }
    if ledger.reserve_ticket(&stamp, attempt + 1).is_ok() {
        return Err("claim replay accepted".into());
    }
    let closes: Vec<_> = sessions
        .into_iter()
        .map(|s| s.close(Duration::from_secs(30)))
        .collect();
    let info = serde_json::json!({"scope":"fault-injected local CPU rank-3 process, rank-2 result admitted internally before failure; no output","run_id":stamp.run_id,"attempt":attempt,"rank2_new_frames":received,"next_token":stage.next_token(),"error":format!("{error}"),"closed":closes});
    writeln!(std::io::stdout().lock(), "{info}")?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("reserved integration refused: {e}");
        std::process::exit(1);
    }
}
