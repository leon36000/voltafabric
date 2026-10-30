//! Fail-closed QSA bound preflight with invalid placement; no worker is launched.
use std::io::Read;
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Stamp},
    native_ledger::NativeBatchLedger,
    sc117_ffn_stage, sc117_qsa_bound_layer, sc117_qsa_stage, sha256,
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
    if a.len() != 6 {
        return Err("usage: reject QSA_MAN FFN_MAN DENSE SHARD LEDGER".into());
    }
    let n = |p: &str| -> Result<usize, Box<dyn std::error::Error>> {
        let v = open(p)?.metadata()?.len();
        if v == 0 || v > 16384 {
            return Err("manifest size".into());
        }
        Ok(v as usize)
    };
    let qm: sc117_qsa_stage::Manifest = serde_json::from_slice(&read(
        &a[1],
        n(&a[1])?,
        "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21",
    )?)?;
    let fm: sc117_ffn_stage::Manifest = serde_json::from_slice(&read(
        &a[2],
        n(&a[2])?,
        "79d44d7663231fea2a6c04c309d776cd9552af7d0ecb859c6428b8124351afd1",
    )?)?;
    let stamp = Stamp {
        run_id: format!("r431-reject-{}", std::process::id()),
        model_revision: qm.model_revision.clone(),
        plan_id: "r431-invalid-registry".into(),
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
    let mut d = open(&a[3])?;
    let mut shard = open(&a[4])?;
    let mut qsa = sc117_qsa_stage::Stage::load(qm, id.clone(), &mut d, &mut shard, 2)?;
    let ffn = sc117_ffn_stage::Stage::load(fm, id, &mut d)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[5]))?;
    let ticket = ledger.reserve_ticket(&stamp, 1)?;
    let mut residual = vec![0.0; 10240];
    residual[0] = 0.031;
    let route =
        sc117_qsa_bound_layer::Routed::prepare(&mut qsa, &ffn, &ledger, &ticket, 0, &residual)?;
    if !qsa.is_empty() {
        return Err("cache advanced during prepare".into());
    }
    let registry = Registry {
        epoch: 1,
        copies: vec![],
    };
    if route
        .dispatch(
            &mut [],
            registry,
            std::collections::BTreeMap::new(),
            std::collections::BTreeMap::new(),
            &ledger,
            ticket,
            std::time::Duration::from_secs(1),
        )
        .is_ok()
    {
        return Err("missing owners accepted".into());
    }
    if !qsa.is_empty() || ledger.reserve_ticket(&stamp, 2).is_ok() {
        return Err("failed transaction advanced or replayed".into());
    }
    println!("missing owners rejected, QSA cache unchanged, claim consumed");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("QSA rejection refused: {e}");
        std::process::exit(1);
    }
}
