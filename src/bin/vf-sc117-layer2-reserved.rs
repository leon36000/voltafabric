//! Bounded CPU integration diagnostic: actual layer-2 prefix and ten actual owners and shared expert, actual route.
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom, Write},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::{Binding, MatrixBinding, QuantizedMatrix},
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    sc117_gdn_stage::{Manifest, Stage},
    sc117_layer_completion::StampedValues,
    sc117_post_attention,
    session::{Endpoint, Session},
    sha256, shared_expert_cpu,
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
fn read_at(
    f: &mut std::fs::File,
    off: u64,
    len: usize,
    hash: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(off))?;
    let mut b = vec![0; len];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != hash {
        return Err("shared digest mismatch".into());
    }
    Ok(b)
}
#[allow(clippy::too_many_arguments)]
fn matrix(
    f: &mut std::fs::File,
    off: u64,
    len: usize,
    hash: &str,
    name: &str,
    kind: i32,
    rows: usize,
    columns: usize,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let b = read_at(f, off, len, hash)?;
    let binding = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 2,
        tensor: format!("blk.2.{name}.weight"),
        ggml_type: kind,
        rows,
        columns,
        sha256: hash.into(),
    };
    Ok(QuantizedMatrix::load(binding, &b)?.forward(x)?)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 10 {
        return Err("usage: vf-sc117-reserved-integration MANIFEST DENSE SHARD RESIDUAL PACK INDEX WORKER LEDGER_DIR POST_MANIFEST".into());
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
        "e8d894b8909687b479cd53b5608c4f5aae886f5f04d415f1b14ac64fc2a8f980",
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
        run_id: format!("r423-local-{}-{attempt}", std::process::id()),
        model_revision: m.model_revision.clone(),
        plan_id: "r423-actual-route-shared".into(),
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
    let routed = post.compute(&id, proposal.output(), &residual)?;
    // Match the independently checked R422 observation before any native dispatch.
    let mut observation = Vec::new();
    for x in routed
        .z
        .iter()
        .chain(&routed.normalized)
        .chain(&routed.attention)
        .chain(&routed.residual)
        .chain(&routed.mixed)
        .chain(&routed.inject)
    {
        observation.extend_from_slice(&x.to_le_bytes());
    }
    for c in &routed.choices {
        observation.extend_from_slice(&c.expert.to_le_bytes());
        observation.extend_from_slice(&c.weight.to_le_bytes());
        observation.extend_from_slice(&c.logit.to_le_bytes());
    }
    if sha256::hex(&observation)
        != "1d422bb0c33f854b725b61a924a0f0f0d8d7def1df51a8f3723c9fadbe9e72ed"
    {
        return Err("post-attention observation mismatch".into());
    }
    let ids = [75, 12, 209, 181, 480, 257, 272, 492, 64, 302];
    if routed.choices.iter().map(|c| c.expert).collect::<Vec<_>>() != ids {
        return Err("unexpected route".into());
    }
    let hashes = [
        "a312f304226bc4cfb7590ea90e9d91babbba67ef0fe4686a0eae85e26a9ed74d",
        "658b2b7b9c7b5055667577babea78964084e1a1dad958c546c46d513249f578e",
        "6271454ee230559ad5f701f94f8af7edc4b3ba7f10ff98b4763d09837c51db61",
        "98a127792dc0ae3709e7c714225f994913404e880e61e55794c0bf4fcb4d77ea",
        "a72edc2b4ad6258cb63946914960d24bbbcf763809be93824d0c51e3ce3a727a",
        "6f626990dabe150f19c95ae1576c1563a98909e2333bb01aea0d0984891491ca",
        "d3db2278072ae0ac7eb8c42b1597c6fea5a10469a81e4bf6b01274eb49fd26c5",
        "2e9bd3f3382db3c813301e257e8d0a603545d6a9e6d939c4715d6d7b4f5457c0",
        "ad7cd66d98eb7faa3411047725e79af82714ab94c30f8a43d323587ba68ee4a2",
        "1b00f6b217f13db94e35ae6204921755860556ce547d0ffb70efdb29ff6a6824",
    ];
    let scalar = shared_expert_cpu::scalar_gate(
        &routed.mixed,
        &read_at(
            &mut shard,
            3772567296,
            5120,
            "8fdc1f7226267f8ac55175fd6c58cec66b36988f1eab46524a863253826f6a20",
        )?,
    )?;
    let gate = matrix(
        &mut shard,
        3772572416,
        921600,
        "558cb007c6c7e3c4d7ea99468b94044d31fa13d63d143de651f054adbaa29b07",
        "ffn_gate_shexp",
        12,
        640,
        2560,
        &routed.mixed,
    )?;
    let up = matrix(
        &mut shard,
        4042191616,
        1344000,
        "77a6723149d0fbcc5cbe2c1d4d60b8ac48bef5ba1ef1fc24a63bf39938f72e0a",
        "ffn_up_shexp",
        14,
        640,
        2560,
        &routed.mixed,
    )?;
    let mid = shared_expert_cpu::activation(&gate, &up)?;
    let down = matrix(
        &mut shard,
        3499507456,
        1740800,
        "68c834fd250a4167f190d0cb8e14bf6ef155a9fa2ab39f03f3ca131c93a4d5ad",
        "ffn_down_shexp",
        8,
        2560,
        640,
        &mid,
    )?;
    let shared: Vec<f32> = down.iter().map(|x| x * scalar).collect();
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
    let problem = RoutingProblem {
        hidden: 2560,
        experts: ids.to_vec(),
        routes: vec![routed
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
        resident_order: orders,
    };
    let input = routed.mixed.clone();
    let collected = Session::exec_native_ggml_reserved(
        &mut sessions,
        &problem,
        &[input],
        &ledger,
        ticket,
        Duration::from_secs(30),
    )?;
    let collected_rows = collected.rows()[0].clone();
    let expected_combined = shared_expert_cpu::combine(&collected_rows, &shared)?;
    let expected = voltafabric::hyperconnection_cpu::write(
        &routed.residual,
        &expected_combined,
        &routed.inject,
    )?;
    let val = |v| StampedValues {
        stamp: stamp.clone(),
        attempt,
        values: v,
    };
    let committed = stage.commit_native(
        proposal,
        collected,
        val(shared.clone()),
        val(routed.residual.clone()),
        val(routed.inject.clone()),
    )?;
    if stage.next_token() != 1 || committed.residual != expected {
        return Err("transaction result mismatch".into());
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
    let info = serde_json::json!({"scope":"CPU single layer, actual gate route, two local ranks, actual shared; not full model or remote transport", "run_id":stamp.run_id,"attempt":attempt,"layer":2,"residual_sha256":sha256::hex(&bytes),"next_token":stage.next_token(),"collector":collected_rows,"shared":shared,"post_attention":routed.residual,"injection":routed.inject,"weights":routed.choices.iter().map(|c|c.weight).collect::<Vec<_>>()});
    writeln!(std::io::stdout().lock(), "{info}")?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("reserved integration refused: {e}");
        std::process::exit(1);
    }
}
