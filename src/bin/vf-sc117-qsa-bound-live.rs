//! Local actual layer-3 reserved QSA + FFN integration, diagnostic only.
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
    sc117_ffn_stage, sc117_qsa_bound_layer, sc117_qsa_stage, sc117_shared_stage,
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
        return Err("usage: qsa-bound QSA_MAN FFN_MAN SHARED_MAN DENSE SHARD RESIDUAL PACK INDEX WORKER LEDGER".into());
    }
    let size = |p: &str| -> Result<usize, Box<dyn std::error::Error>> {
        let n = open(p)?.metadata()?.len();
        if n == 0 || n > 16384 {
            return Err("manifest length".into());
        }
        Ok(n as usize)
    };
    let qm: sc117_qsa_stage::Manifest = serde_json::from_slice(&read(
        &a[1],
        size(&a[1])?,
        "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21",
    )?)?;
    let fm: sc117_ffn_stage::Manifest = serde_json::from_slice(&read(
        &a[2],
        size(&a[2])?,
        "79d44d7663231fea2a6c04c309d776cd9552af7d0ecb859c6428b8124351afd1",
    )?)?;
    let sm: sc117_shared_stage::Manifest = serde_json::from_slice(&read(
        &a[3],
        709,
        "66e2f83f66d902aeebea7a1c667cba3960e8dd62235dc56004a99d7b238adebb",
    )?)?;
    let residual: Vec<f32> = read(
        &a[6],
        40960,
        "ffb003ec3d3126d160c7994f4414d02b90ca64e68a4a488dc54ff1856a80fe6d",
    )?
    .chunks_exact(4)
    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
    .collect();
    let ni = open(&a[8])?.metadata()?.len();
    if ni == 0 || ni > 65536 {
        return Err("index size".into());
    }
    let index = String::from_utf8(read(
        &a[8],
        ni as usize,
        "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
    )?)?;
    let mut pack = open(&a[7])?;
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let stamp = Stamp {
        run_id: format!("r431-local-{}-{attempt}", std::process::id()),
        model_revision: qm.model_revision.clone(),
        plan_id: "r431-layer3-actual-qsa".into(),
        layer: 3,
        batch: 1,
        epoch: 1,
    };
    let id = Identity {
        run_id: stamp.run_id.clone(),
        model_revision: stamp.model_revision.clone(),
        layer: 3,
        epoch: 1,
    };
    let mut d = open(&a[4])?;
    let mut shard = open(&a[5])?;
    let mut qsa = sc117_qsa_stage::Stage::load(qm, id.clone(), &mut d, &mut shard, 2)?;
    let ffn = sc117_ffn_stage::Stage::load(fm, id.clone(), &mut d)?;
    let mut shared_stage = sc117_shared_stage::Stage::load(sm, id, &mut shard)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[10]))?;
    let ticket = ledger.reserve_ticket(&stamp, attempt)?;
    let route =
        sc117_qsa_bound_layer::Routed::prepare(&mut qsa, &ffn, &ledger, &ticket, 0, &residual)?;
    if !qsa.is_empty() {
        return Err("cache advanced before collection".into());
    }
    let x = route.attention_prefix();
    let mut bytes = Vec::new();
    for v in x
        .mixed
        .iter()
        .chain(&x.inject)
        .chain(&x.raw_query_gate)
        .chain(&x.query)
        .chain(&x.gate)
        .chain(&x.raw_key)
        .chain(&x.key)
        .chain(&x.value)
        .chain(&x.gated_attention)
        .chain(&x.attention)
    {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if sha256::hex(&bytes) != "82be7f3d202456ffe27837641f2e24be96abac1432a8be318e21a1ba602ec575" {
        return Err("QSA prefix differs".into());
    }
    let ids = [405, 143, 190, 42, 232, 231, 315, 223, 294, 470];
    if route.choices().iter().map(|x| x.expert).collect::<Vec<_>>() != ids {
        return Err("route differs".into());
    }
    let mut data = Vec::new();
    for x in route.activation() {
        data.extend_from_slice(&x.to_le_bytes());
    }
    if sha256::hex(&data) != "0dbad6d54af71538591e93935e24d0acd510134a7b3a671c91f5f8b7486f10d7" {
        return Err("activation differs".into());
    }
    let hashes = [
        "9b6b82fc96643b92e3db30cbccc776a9ddba2c708e1d411c314e3a64622e24e8",
        "40f10027b91e90236398c3007d8df88288242eb46bffdaff9f81af05f65892ab",
        "f689be1ea27dfaed33f8860bdb09681127550e90cd8e1c47cd5bcc51a2042ada",
        "cabc10964137415260e7199497eba4cbf5cc893a5ac70cc8c927a7733bdc8ed4",
        "234e22191fc67a738887c5d6e0bdac52b9e1639dbbca5d9cdb1be35ee9d451b1",
        "f33a25914d5c124b643863947ae1e4b2938aa6ce4c2de6667a60420b829f98d9",
        "33732d673ccb23a0ce37849f056bcf1419b1e5c4d407224cd6cb075dc7f54996",
        "d16123aa0070946a2b290122aa24292893c938f4f0ad7e993f9d469439a3b7a7",
        "10c4f855927a1c68d068687e0dea6176a4548989a5f7e54970de08fafe3a4b9c",
        "3126d776d738c48e3e68927736ddd1f8c93d07ffd84df453bf9255159ed81b84",
    ];
    let shared = route.shared(&mut shared_stage)?;
    let shared_values = shared.values().to_vec();
    let weights: Vec<_> = route.choices().iter().map(|x| x.weight).collect();
    let post = route.attention_residual().to_vec();
    let injection = route.injection().to_vec();
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut orders = BTreeMap::new();
    for rank in 2..=3 {
        let mut sess = Session::connect(
            &Endpoint {
                rank,
                argv: vec![a[9].clone()],
            },
            Duration::from_secs(30),
        )?;
        let mut locals = Vec::new();
        for (slot, &expert) in ids
            .iter()
            .enumerate()
            .filter(|(i, _)| 2 + (*i as u32) % 2 == rank)
        {
            let local = (slot / 2) as u32;
            locals.push(local);
            let blob = layout.read_verified_expert(
                &mut pack,
                3,
                expert.into(),
                "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
                hashes[slot],
            )?;
            let binding = Binding {
                model_revision: stamp.model_revision.clone(),
                rank,
                layer: 3,
                expert,
                local,
                epoch: 1,
                index_sha256: layout.index_sha256().into(),
                blob_sha256: hashes[slot].into(),
            };
            sess.load_native_ggml(
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
        sess.seal_native_ggml(5, Duration::from_secs(30))?;
        orders.insert(rank, locals);
        sessions.push(sess);
    }
    let collected = route.dispatch(
        &mut sessions,
        Registry { epoch: 1, copies },
        BTreeMap::new(),
        orders,
        &ledger,
        ticket,
        Duration::from_secs(30),
    )?;
    let collector = collected.rows()[0].clone();
    let committed = route.commit(&mut qsa, shared, collected)?;
    if qsa.len() != 1 {
        return Err("QSA state not committed".into());
    }
    if ledger.reserve_ticket(&stamp, attempt + 1).is_ok() {
        return Err("claim replay accepted".into());
    }
    for s in sessions {
        let c = s.close(Duration::from_secs(30));
        if c.killed || c.exit_code != Some(0) || c.bye_execs != Some(1) {
            return Err("worker close".into());
        }
    }
    let mut finalbytes = Vec::new();
    for x in &committed.residual {
        finalbytes.extend_from_slice(&x.to_le_bytes());
    }
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::json!({"scope":"layer 3 actual QSA and FFN on local CPU, not full model","run_id":stamp.run_id,"attempt":attempt,"qsa_len":qsa.len(),"residual_sha":sha256::hex(&finalbytes),"weights":weights,"collector":collector,"shared":shared_values,"post":post,"injection":injection})
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("bound QSA refused: {e}");
        std::process::exit(1);
    }
}
