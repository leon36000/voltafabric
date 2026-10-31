//! Local CPU integration check: observed QSA + shared + real expert sessions.
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
    sc117_ffn_stage,
    sc117_qsa_bound_layer::{Routed, RoutedBatch},
    sc117_qsa_stage, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const WORKER: &str = "ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const SPEC: &str = "3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20";
const TRACE: &str = "a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2";
const SIZE: u64 = 50_292_326_400;
fn checked(p: &str, h: &str) -> Result<Vec<u8>, E> {
    let b = fs::read(p)?;
    if sha256::hex(&b) != h {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn hash(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 11 {
        return Err(
            "usage: SPEC TRACE QSA_MANIFEST DENSE SHARD PACK INDEX LEDGER WORKER PLAN_SHA".into(),
        );
    }
    if !sha256::is_hex_digest(&a[10]) {
        return Err("plan hash".into());
    }
    let spec: serde_json::Value = serde_json::from_slice(&checked(&a[1], SPEC)?)?;
    let trace = checked(&a[2], TRACE)?;
    if trace.len() != 92 * 47620 * 4 {
        return Err("trace size".into());
    }
    let manifest = checked(
        &a[3],
        "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21",
    )?;
    let index = String::from_utf8(checked(&a[7], INDEX)?)?;
    checked(&a[9], WORKER)?;
    let records: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            trace[i * 47620 * 4..i * 47620 * 4 + 40960]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let stamp = Stamp {
        run_id: format!("r557-{}-{attempt}", std::process::id()),
        model_revision: "sc117-local-observation".into(),
        plan_id: a[10].clone(),
        layer: 3,
        batch: 0,
        epoch: 1,
    };
    let id = Identity {
        run_id: stamp.run_id.clone(),
        model_revision: stamp.model_revision.clone(),
        layer: 3,
        epoch: 1,
    };
    let mut dense = fs::File::open(&a[4])?;
    let mut shard = fs::File::open(&a[5])?;
    let mut qsa = sc117_qsa_stage::Stage::load(
        serde_json::from_slice(&manifest)?,
        id.clone(),
        &mut dense,
        &mut shard,
        4,
    )?;
    let mut serial = sc117_qsa_stage::Stage::load(
        serde_json::from_slice(&manifest)?,
        id.clone(),
        &mut dense,
        &mut shard,
        4,
    )?;
    let ffn = sc117_ffn_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer3-ffn.json"))?,
        id.clone(),
        &mut dense,
    )?;
    let mut shared = sc117_shared_stage::Stage::load(
        serde_json::from_str(include_str!("../../fixtures/r438/layer3-shared.json"))?,
        id.clone(),
        &mut shard,
    )?;
    let layout = NativeLayout::parse_sc117_v3(&index, SIZE)?;
    let mut pack = fs::File::open(&a[6])?;
    let meta = spec["experts"].as_object().ok_or("experts")?;
    if meta.len() != 17 {
        return Err("expert count".into());
    }
    let mut counters = BTreeMap::from([(2, 0u32), (3, 0u32)]);
    let mut copies = Vec::new();
    let mut buffers = Vec::new();
    for (key, val) in meta {
        let expert: u32 = key.parse()?;
        let rank = u32::try_from(val["rank"].as_u64().ok_or("rank")?)?;
        let local = counters.get_mut(&rank).ok_or("invalid rank")?;
        let binding = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 3,
            expert,
            local: *local,
            epoch: 1,
            index_sha256: INDEX.into(),
            blob_sha256: val["sha256"].as_str().ok_or("digest")?.into(),
        };
        *local += 1;
        let data = layout.read_verified_expert(
            &mut pack,
            3,
            expert.into(),
            INDEX,
            &binding.blob_sha256,
        )?;
        copies.push(Replica {
            expert,
            rank,
            local: binding.local,
        });
        buffers.push((binding, data));
    }
    let order = BTreeMap::from([
        (2, (0..counters[&2]).collect::<Vec<_>>()),
        (3, (0..counters[&3]).collect::<Vec<_>>()),
    ]);
    let registry = Registry { epoch: 1, copies };
    let timeout = Duration::from_secs(60);
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[8]))?;
    let mut sessions = Vec::new();
    for rank in [2, 3] {
        let mut s = Session::connect(
            &Endpoint {
                rank,
                argv: vec![
                    a[9].clone(),
                    "--guarded-local".into(),
                    "--expected-exe-sha256".into(),
                    WORKER.into(),
                ],
            },
            timeout,
        )?;
        for (b, d) in &buffers {
            if b.rank == rank {
                s.load_native_ggml(b, &index, SIZE, &d.bytes, timeout)?;
            }
        }
        s.seal_native_ggml(counters[&rank] as usize, timeout)?;
        sessions.push(s);
    }
    let mut bad_stamp = stamp.clone();
    bad_stamp.batch = 100;
    let bad_ticket = ledger.reserve_ticket(&bad_stamp, attempt)?;
    let mut bad = records.clone();
    bad[1][0] = f32::NAN;
    if RoutedBatch::prepare(&mut qsa, &ffn, &ledger, &bad_ticket, &bad).is_ok() || !qsa.is_empty() {
        return Err("failure changed active cache".into());
    }
    let mut foreign_stamp = stamp.clone();
    foreign_stamp.batch = 98;
    let foreign_ticket = ledger.reserve_ticket(&foreign_stamp, attempt + 1)?;
    let foreign = RoutedBatch::prepare(&mut serial, &ffn, &ledger, &foreign_ticket, &records)?;
    let foreign_shared = foreign.shared(&mut shared)?;
    drop(foreign);
    let mut fault_stamp = stamp.clone();
    fault_stamp.batch = 99;
    let fault_ticket = ledger.reserve_ticket(&fault_stamp, attempt + 2)?;
    let fault = RoutedBatch::prepare(&mut qsa, &ffn, &ledger, &fault_ticket, &records)?;
    let fault_collected = fault.dispatch(
        &mut sessions,
        registry.clone(),
        BTreeMap::new(),
        order.clone(),
        &ledger,
        fault_ticket,
        timeout,
    )?;
    if fault.commit(foreign_shared, fault_collected).is_ok()
        || !qsa.is_empty()
        || !serial.is_empty()
    {
        return Err("foreign transaction committed".into());
    }
    let ticket = ledger.reserve_ticket(&stamp, attempt + 3)?;
    let batch = RoutedBatch::prepare(&mut qsa, &ffn, &ledger, &ticket, &records)?;
    for (i, choices) in batch.choices().enumerate() {
        let expected = spec["inputs"][i]["choices"].as_array().ok_or("choices")?;
        if choices.len() != expected.len()
            || choices.iter().zip(expected).any(|(x, y)| {
                u64::from(x.expert) != y["expert"].as_u64().unwrap_or(u64::MAX)
                    || x.weight.to_bits()
                        != (y["weight"].as_f64().unwrap_or(f64::NAN) as f32).to_bits()
            })
        {
            return Err("route mismatch".into());
        }
    }
    for (i, x) in batch.activations().enumerate() {
        if hash(x)
            != spec["inputs"][i]["activation_sha256"]
                .as_str()
                .ok_or("activation")?
        {
            return Err("activation mismatch".into());
        }
    }
    let shared_batch = batch.shared(&mut shared)?;
    let collected = batch.dispatch(
        &mut sessions,
        registry.clone(),
        BTreeMap::new(),
        order.clone(),
        &ledger,
        ticket,
        timeout,
    )?;
    let output = batch.commit(shared_batch, collected)?;
    let mut expected = Vec::new();
    for (i, input) in records.iter().enumerate() {
        let mut s = stamp.clone();
        s.batch = (i + 1) as u64;
        let t = ledger.reserve_ticket(&s, attempt + i as u64 + 4)?;
        let r = Routed::prepare(&mut serial, &ffn, &ledger, &t, i, input)?;
        let sh = r.shared(&mut shared)?;
        let c = r.dispatch(
            &mut sessions,
            registry.clone(),
            BTreeMap::new(),
            order.clone(),
            &ledger,
            t,
            timeout,
        )?;
        expected.push(r.commit(&mut serial, sh, c)?.residual);
    }
    if output.residuals.len() != 2
        || output.residuals.iter().zip(&expected).any(|(a, b)| {
            a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
        })
        || qsa.len() != 2
        || serial.len() != 2
    {
        return Err("layer output mismatch".into());
    }
    let mut receipts = Vec::new();
    for s in sessions {
        let count = s.traffic.native_ggml_raw_execs;
        let r = s.close(timeout);
        let expected_count = if r.rank == 2 {
            3
        } else if r.rank == 3 {
            4
        } else {
            return Err("unexpected worker rank".into());
        };
        if count != expected_count
            || r.bye_execs != Some(expected_count)
            || r.exit_code != Some(0)
            || r.killed
            || r.error.is_some()
        {
            return Err("worker close".into());
        }
        receipts.push(serde_json::json!({"rank":r.rank,"pid":r.pid,"raw_execs":count,"exit_code":r.exit_code}));
    }
    Ok(
        serde_json::json!({"status":"pass","run_id":stamp.run_id,"plan_id":stamp.plan_id,"residual_sha256":output.residuals.iter().map(|x|hash(x)).collect::<Vec<_>>(),"positions":2,"workers":receipts}),
    )
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("bound batch refused: {e}");
            std::process::exit(1)
        }
    }
}
