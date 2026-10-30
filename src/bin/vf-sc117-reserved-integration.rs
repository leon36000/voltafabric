//! Bounded CPU integration diagnostic: actual layer-2 prefix and two actual owners, synthetic route/shared.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    sc117_gdn_stage::{Manifest, Stage},
    sc117_layer_completion::StampedValues,
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
    if a.len() != 9 {
        return Err("usage: vf-sc117-reserved-integration MANIFEST DENSE SHARD RESIDUAL PACK INDEX WORKER LEDGER_DIR".into());
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
        run_id: format!("r421-local-{}-{attempt}", std::process::id()),
        model_revision: m.model_revision.clone(),
        plan_id: "r421-synthetic-route-c".into(),
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
    let mut stage = Stage::load(m, id, &mut dense, &mut shard)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[8]))?;
    let ticket = ledger.reserve_ticket(&stamp, attempt)?;
    let proposal = stage.prepare_reserved(&ledger, &ticket, 0, &residual)?;
    let mut sessions = Vec::new();
    let hashes = [
        "35878c2a8c029ebb9eed384ab8c0e9a00bb7a4039963ee943641de12bb8c8deb",
        "7f9f1e13b9d7b059b827bf2cb02debddb0dae32d20a4f540cecb667d410d9b59",
    ];
    for (e, hash) in hashes.iter().enumerate() {
        let expert = e as u32;
        let rank = 2 + expert;
        let blob = layout.read_verified_expert(
            &mut pack,
            2,
            expert.into(),
            "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24",
            hash,
        )?;
        let binding = Binding {
            model_revision: stamp.model_revision.clone(),
            rank,
            layer: 2,
            expert,
            local: 0,
            epoch: 1,
            index_sha256: layout.index_sha256().into(),
            blob_sha256: (*hash).into(),
        };
        let endpoint = Endpoint {
            rank,
            argv: vec![a[7].clone()],
        };
        let mut session = Session::connect(&endpoint, Duration::from_secs(30))?;
        session.load_native_ggml(
            &binding,
            &index,
            pack.metadata()?.len(),
            &blob.bytes,
            Duration::from_secs(30),
        )?;
        session.seal_native_ggml(1, Duration::from_secs(30))?;
        sessions.push(session);
    }
    let problem = RoutingProblem {
        hidden: 2560,
        experts: vec![0, 1],
        routes: vec![vec![
            Choice {
                expert: 0,
                weight: 0.3,
            },
            Choice {
                expert: 1,
                weight: 0.7,
            },
        ]],
        registry: Registry {
            epoch: 1,
            copies: vec![
                Replica {
                    expert: 0,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 1,
                    rank: 3,
                    local: 0,
                },
            ],
        },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    // Distinct synthetic workload from the earlier uncertain attempt.
    let input: Vec<f32> = proposal.output().mixed.iter().map(|v| v * 0.25).collect();
    let collected = Session::exec_native_ggml_reserved(
        &mut sessions,
        &problem,
        &[input],
        &ledger,
        ticket,
        Duration::from_secs(30),
    )?;
    let expected: Vec<f32> = residual
        .iter()
        .enumerate()
        .map(|(i, &r)| r + collected.rows()[0][i % 2560] as f32)
        .collect();
    let val = |v| StampedValues {
        stamp: stamp.clone(),
        attempt,
        values: v,
    };
    let committed = stage.commit_native(
        proposal,
        collected,
        val(vec![0.0; 2560]),
        val(residual.clone()),
        val(vec![0.0; 4]),
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
    let info = serde_json::json!({"scope":"CPU plumbing, two real layer-2 owners with synthetic route/shared and prefix activation; not full layer", "run_id":stamp.run_id,"attempt":attempt,"layer":2,"residual_sha256":sha256::hex(&bytes),"next_token":stage.next_token()});
    writeln!(std::io::stdout().lock(), "{info}")?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("reserved integration refused: {e}");
        std::process::exit(1);
    }
}
