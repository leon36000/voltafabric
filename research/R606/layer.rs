//! Single fresh CPU layer transaction using scalar native expert owners.
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
    if a.len() != 12 {
        return Err(
            "usage: two GDN DENSE SHARD RES0 RES1 PACK INDEX WORKER LEDGER POST SHARED".into(),
        );
    }
    let plan = std::env::var("VF_R606_PLAN_SHA256")?;
    let worker = std::env::var("VF_R606_WORKER_SHA256")?;
    let run_id = std::env::var("VF_R606_RUN_ID")?;
    let valid_digest = |s: &str| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    };
    if !valid_digest(&plan)
        || !valid_digest(&worker)
        || !run_id.starts_with("r606-")
        || run_id.len() > 128
    {
        return Err("invalid external identity".into());
    }
    let m: Manifest = serde_json::from_slice(&read(
        &a[1],
        open(&a[1])?.metadata()?.len() as usize,
        "d822695e4fceee7b285b510e9ec7f97cc0a1507188b6ca698de5b6b216073348",
    )?)?;
    let postm: sc117_post_attention::Manifest = serde_json::from_slice(&read(
        &a[10],
        1299,
        "8959d5ff4bbac4c6be08e6a978c5d27ffa1f09f8e2f3541add9395368121b0f1",
    )?)?;
    let sharem: sc117_shared_stage::Manifest = serde_json::from_slice(&read(
        &a[11],
        710,
        "2a9a91e11381fb9ef59636437c7f4649682cc1a89877e44818cd950c097e5d0c",
    )?)?;
    let hashes = [
        "ed1d74a7f93128f6b6a3b30855687a660cba93eca8e1c76c72086dc942fa7061",
        "9a69b12aec635edd0a042269e830779b49be0db018f6812479f617867371d96d",
    ];
    let mut inputs = Vec::new();
    for i in 0..1 {
        inputs.push(
            read(&a[4 + i], 40960, hashes[i])?
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
    }
    let ilen = open(&a[7])?.metadata()?.len();
    if ilen == 0 || ilen > 65536 {
        return Err("index size".into());
    }
    let index = String::from_utf8(read(
        &a[7],
        ilen as usize,
        "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
    )?)?;
    let mut pack = open(&a[6])?;
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let stamp = Stamp {
        run_id,
        model_revision: m.model_revision.clone(),
        plan_id: format!("r606-{plan}"),
        layer: 2,
        batch: 0,
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
    let mut post = sc117_post_attention::Stage::load(postm, id.clone(), &mut dense, &mut shard)?;
    let mut shared_stage = sc117_shared_stage::Stage::load(sharem, id, &mut shard)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[9]))?;
    let experts = [75, 12, 209, 181, 257, 272, 480, 159, 492, 255, 302, 64];
    let digests = [
        "a312f304226bc4cfb7590ea90e9d91babbba67ef0fe4686a0eae85e26a9ed74d",
        "658b2b7b9c7b5055667577babea78964084e1a1dad958c546c46d513249f578e",
        "6271454ee230559ad5f701f94f8af7edc4b3ba7f10ff98b4763d09837c51db61",
        "98a127792dc0ae3709e7c714225f994913404e880e61e55794c0bf4fcb4d77ea",
        "6f626990dabe150f19c95ae1576c1563a98909e2333bb01aea0d0984891491ca",
        "d3db2278072ae0ac7eb8c42b1597c6fea5a10469a81e4bf6b01274eb49fd26c5",
        "a72edc2b4ad6258cb63946914960d24bbbcf763809be93824d0c51e3ce3a727a",
        "694842ad3656f0f76eb8307520b83c314bb82fa123e8a7762256c4d9a2b06161",
        "2e9bd3f3382db3c813301e257e8d0a603545d6a9e6d939c4715d6d7b4f5457c0",
        "8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5",
        "1b00f6b217f13db94e35ae6204921755860556ce547d0ffb70efdb29ff6a6824",
        "ad7cd66d98eb7faa3411047725e79af82714ab94c30f8a43d323587ba68ee4a2",
    ];
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut orders = BTreeMap::new();
    for rank in 2..=3 {
        let mut session = Session::connect(
            &Endpoint {
                rank,
                argv: vec![
                    a[8].clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    worker.clone(),
                ],
            },
            Duration::from_secs(30),
        )?;
        let mut locals = Vec::new();
        for (slot, &expert) in experts
            .iter()
            .take(10)
            .enumerate()
            .filter(|(i, _)| 2 + (*i as u32) % 2 == rank)
        {
            let local = (slot / 2) as u32;
            locals.push(local);
            let blob = layout.read_verified_expert(
                &mut pack,
                2,
                expert.into(),
                "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
                digests[slot],
            )?;
            let binding = Binding {
                model_revision: stamp.model_revision.clone(),
                rank,
                layer: 2,
                expert,
                local,
                epoch: 1,
                index_sha256: layout.index_sha256().into(),
                blob_sha256: digests[slot].into(),
            };
            session.load_native_scalar_ggml(
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
    let expected_ids = [
        vec![75, 12, 209, 181, 257, 272, 480, 159, 492, 255],
        vec![75, 181, 480, 302, 159, 492, 64, 12, 209, 272],
    ];
    let expected_sha = [
        "c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a",
        "9cdd44775e6d5afd50f61662745765e1b1a5701c3f31a85c766f82efed8a3403",
    ];
    let mut reports = Vec::new();
    for i in 0..1 {
        let mut st = stamp.clone();
        st.batch = i as u64;
        let at = attempt + i as u64;
        let ticket = ledger.reserve_ticket(&st, at)?;
        let proposal = stage.prepare_reserved(&ledger, &ticket, i as u64, &inputs[i])?;
        let route =
            sc117_bound_layer::Routed::prepare(&mut post, &ledger, &ticket, proposal, &inputs[i])?;
        let ids: Vec<_> = route.choices().iter().map(|x| x.expert).collect();
        if ids != expected_ids[i] {
            return Err("route changed from preview".into());
        }
        let mut bytes = Vec::new();
        for v in route.activation() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if sha256::hex(&bytes) != expected_sha[i] {
            return Err("activation changed from preview".into());
        }
        let weights: Vec<_> = route.choices().iter().map(|c| c.weight).collect();
        let post_values = route.attention_residual().to_vec();
        let injection = route.injection().to_vec();
        let shared = route.shared(&mut shared_stage)?;
        let shared_values = shared.values().to_vec();
        let collected = route.dispatch(
            &mut sessions,
            Registry {
                epoch: 1,
                copies: copies.clone(),
            },
            BTreeMap::new(),
            orders.clone(),
            &ledger,
            ticket,
            Duration::from_secs(30),
        )?;
        let row = collected.rows()[0].clone();
        let committed = route.commit(&mut stage, shared, collected)?;
        if stage.next_token() != i as u64 + 1 {
            return Err("state progression".into());
        }
        let mut result = Vec::new();
        for x in &committed.residual {
            result.extend_from_slice(&x.to_le_bytes());
        }
        let mut gdn = Vec::new();
        for x in committed
            .output
            .mixed
            .iter()
            .chain(&committed.output.inject)
            .chain(&committed.output.qkv)
            .chain(&committed.output.recurrent)
            .chain(&committed.output.gate)
            .chain(&committed.output.beta)
        {
            gdn.extend_from_slice(&x.to_le_bytes());
        }
        let collector_bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        reports.push(serde_json::json!({"token":i,"ids":ids,"weights":weights,"activation_sha":sha256::hex(&bytes),"gdn_sha":sha256::hex(&gdn),"residual_sha":sha256::hex(&result),"collector_sha":sha256::hex(&collector_bytes),"shared":shared_values,"post":post_values,"injection":injection}));
    }
    let mut closes = Vec::new();
    for s in sessions {
        let c = s.close(Duration::from_secs(30));
        if c.killed || c.exit_code != Some(0) || c.bye_execs != Some(1) {
            return Err("worker close or reuse count".into());
        }
        closes.push(c);
    }
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::json!({"run_id":stamp.run_id,"attempt":attempt,"next_token":stage.next_token(),"tokens":reports,"closed":closes})
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("scalar layer refused: {e}");
        std::process::exit(1);
    }
}
