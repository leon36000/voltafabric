//! Local preflight: no native worker dispatch or GDN commit.
use std::io::Read;
use voltafabric::{
    gdn_cpu::Identity, moe::Stamp, native_ledger::NativeBatchLedger, sc117_bound_layer,
    sc117_gdn_stage, sc117_post_attention, sc117_shared_stage, sha256,
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
fn exact(p: &str, n: usize, h: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut f = open(p)?;
    if f.metadata()?.len() != n as u64 || !sha256::is_hex_digest(h) {
        return Err("input size or digest syntax".into());
    }
    let mut b = vec![0; n];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != h {
        return Err("input digest mismatch".into());
    }
    Ok(b)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err(
            "usage: preflight GDN_MANIFEST POST_MANIFEST SHARED_MANIFEST DENSE SHARD LEDGER".into(),
        );
    }
    let m: sc117_gdn_stage::Manifest = serde_json::from_slice(&exact(
        &a[1],
        open(&a[1])?.metadata()?.len() as usize,
        "d822695e4fceee7b285b510e9ec7f97cc0a1507188b6ca698de5b6b216073348",
    )?)?;
    let p: sc117_post_attention::Manifest = serde_json::from_slice(&exact(
        &a[2],
        1299,
        "8959d5ff4bbac4c6be08e6a978c5d27ffa1f09f8e2f3541add9395368121b0f1",
    )?)?;
    let h: sc117_shared_stage::Manifest = serde_json::from_slice(&exact(
        &a[3],
        open(&a[3])?.metadata()?.len() as usize,
        "2a9a91e11381fb9ef59636437c7f4649682cc1a89877e44818cd950c097e5d0c",
    )?)?;
    let id = Identity {
        run_id: format!("r426-rejection-{}", std::process::id()),
        model_revision: m.model_revision.clone(),
        layer: 2,
        epoch: 1,
    };
    let mut d = open(&a[4])?;
    let mut f = open(&a[5])?;
    let mut g = sc117_gdn_stage::Stage::load(m, id.clone(), &mut d, &mut f)?;
    let mut p = sc117_post_attention::Stage::load(p, id.clone(), &mut d, &mut f)?;
    let mut h = sc117_shared_stage::Stage::load(h, id.clone(), &mut f)?;
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let stamp = |batch| Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "r426-invalid-registry".into(),
        layer: 2,
        batch,
        epoch: 1,
    };
    let mut x = vec![0.0f32; 10240];
    x[0] = 0.021;
    let ta = ledger.reserve_ticket(&stamp(1), 1)?;
    let proposal = g.prepare_reserved(&ledger, &ta, 0, &x)?;
    let mut wrong = x.clone();
    wrong[0] = 0.022;
    if sc117_bound_layer::Routed::prepare(&mut p, &ledger, &ta, proposal, &wrong).is_ok() {
        return Err("wrong residual accepted".into());
    }
    x[0] = 0.023;
    let tb = ledger.reserve_ticket(&stamp(2), 2)?;
    let proposal = g.prepare_reserved(&ledger, &tb, 0, &x)?;
    let tc = ledger.reserve_ticket(&stamp(3), 3)?;
    if sc117_bound_layer::Routed::prepare(&mut p, &ledger, &tc, proposal, &x).is_ok() {
        return Err("wrong ticket accepted".into());
    }
    x[0] = 0.024;
    let td = ledger.reserve_ticket(&stamp(4), 4)?;
    let proposal = g.prepare_reserved(&ledger, &td, 0, &x)?;
    let route = sc117_bound_layer::Routed::prepare(&mut p, &ledger, &td, proposal, &x)?;
    if route.choices().len() != 10 {
        return Err("wrong route length".into());
    }
    let _shared = route.shared(&mut h)?;
    let invalid = voltafabric::moe::Registry {
        epoch: 1,
        copies: vec![],
    };
    if route
        .dispatch(
            &mut [],
            invalid,
            std::collections::BTreeMap::new(),
            std::collections::BTreeMap::new(),
            &ledger,
            td,
            std::time::Duration::from_secs(1),
        )
        .is_ok()
    {
        return Err("empty registry accepted".into());
    }
    if g.next_token() != 0 {
        return Err("unexpected state commit".into());
    }
    println!("preflight ok: residual/ticket rejected, route and shared produced, empty registry rejected, state unchanged, no worker dispatch");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("preflight refused: {e}");
        std::process::exit(1);
    }
}
