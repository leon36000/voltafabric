//! Local CPU qualification of reserved GDN whole-layer batching.
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
    sc117_bound_layer::{Routed, RoutedBatch},
    sc117_gdn_stage, sc117_post_attention, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const INDEX: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const WORKER: &str = "ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const EXPERTS: [(u32, &str); 11] = [
    (
        433,
        "fb1b61d67ac89b66381e8e5b59e96c63df287edd2747d262b7264e68c68177e0",
    ),
    (
        309,
        "ad3e98c5e6dad251481cf64c4e4f92c9084be49236a3f4c6e599e62a2794fb7b",
    ),
    (
        447,
        "ce28378868cfaa7910b3dbfca7320054ad94aea07798f0f510519b7763788c17",
    ),
    (
        475,
        "4fad7b12ce680a6835f13d4d4ecb67dd2c6b38f64e1b4c305c98e91a99ffb6eb",
    ),
    (
        261,
        "aa7844bbd260279efcb02e9bc61a42c8054372d31d3eb8630a25552ad9442bab",
    ),
    (
        252,
        "754e860228131e225c6cbe8812d67d1e2ccc86087923faf3fead6e2427266334",
    ),
    (
        123,
        "ab79a3fd62af597183f03fd826a819db4da8991d69d6ffa9fc5b19b9f12c53dc",
    ),
    (
        107,
        "4a57dcc039156d50a74de4c7a3ceff2b480ac9a6f1ac3f1f04b4427436c8e16b",
    ),
    (
        448,
        "819674af46bf1a1ac908bdb5ceaa24b60b3f8228e47947f186915aba2e528a81",
    ),
    (
        469,
        "a98997cd1e906bfa8f256f6e1cb7d3e21379fcbb93eb573fe3e5d78ed36b9165",
    ),
    (
        232,
        "df715982ea24ff6d4738063487963c12fc4b07d762fb6158dfe50dc3bdad7461",
    ),
];
fn checked(path: &str, hash: &str) -> Result<Vec<u8>, E> {
    let b = fs::read(path)?;
    if sha256::hex(&b) != hash {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn digest(v: &[f32]) -> String {
    sha256::hex(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 11 {
        return Err("usage: GDN POST SHARED DENSE SHARD RESIDUAL PACK INDEX WORKER LEDGER".into());
    }
    let gm = checked(
        &a[1],
        "e828fe4e01fe8c9a86201b79fa40f6e29ceca325b2d0a4ca9be954a939d8afca",
    )?;
    let pm = checked(
        &a[2],
        "4edc0334cb6bc661f10c74efbfb9989c9fca973cda15baa1b5c90450a5bb6e94",
    )?;
    let sm = checked(
        &a[3],
        "e07e3f8e57da20c1d6a806b6263807d76bd74bbf12d81f1d6d65c0e7be4b195b",
    )?;
    let raw = checked(
        &a[6],
        "653f8385328f68ebebfe966082717b7c566053d540081b0539d5e0370ea15b65",
    )?;
    if raw.len() != 40960 {
        return Err("residual size".into());
    }
    let r: Vec<f32> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let inputs = vec![r.clone(), r.iter().map(|x| x * 1.001).collect()];
    let index = String::from_utf8(checked(&a[8], INDEX)?)?;
    checked(&a[9], WORKER)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let plan = std::env::var("VF_R563_PLAN_SHA")?;
    if !sha256::is_hex_digest(&plan) {
        return Err("plan sha".into());
    }
    let stamp = Stamp {
        run_id: format!("r563-{}-{attempt}", std::process::id()),
        model_revision: "sc117-local-observation".into(),
        plan_id: plan,
        layer: 4,
        batch: 0,
        epoch: 1,
    };
    let id = Identity {
        run_id: stamp.run_id.clone(),
        model_revision: stamp.model_revision.clone(),
        layer: 4,
        epoch: 1,
    };
    let mut dense = fs::File::open(&a[4])?;
    let mut shard = fs::File::open(&a[5])?;
    let mut gdn = sc117_gdn_stage::Stage::load(
        serde_json::from_slice(&gm)?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut serial = sc117_gdn_stage::Stage::load(
        serde_json::from_slice(&gm)?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut post = sc117_post_attention::Stage::load(
        serde_json::from_slice(&pm)?,
        id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut shared =
        sc117_shared_stage::Stage::load(serde_json::from_slice(&sm)?, id.clone(), &mut shard)?;
    let mut pack = fs::File::open(&a[7])?;
    let size = pack.metadata()?.len();
    let layout = NativeLayout::parse_sc117_v3(&index, size)?;
    let mut counters = BTreeMap::from([(2, 0u32), (3, 0u32)]);
    let mut copies = Vec::new();
    let mut blobs = Vec::new();
    for (i, (expert, digest)) in EXPERTS.iter().enumerate() {
        let rank = 2 + (i as u32) % 2;
        let local = counters.get_mut(&rank).ok_or("rank")?;
        let binding = Binding {
            model_revision: id.model_revision.clone(),
            rank,
            layer: 4,
            expert: *expert,
            local: *local,
            epoch: 1,
            index_sha256: INDEX.into(),
            blob_sha256: (*digest).into(),
        };
        *local += 1;
        let blob = layout.read_verified_expert(&mut pack, 4, (*expert).into(), INDEX, digest)?;
        copies.push(Replica {
            expert: *expert,
            rank,
            local: binding.local,
        });
        blobs.push((binding, blob));
    }
    let orders = BTreeMap::from([
        (2, (0..counters[&2]).collect::<Vec<_>>()),
        (3, (0..counters[&3]).collect::<Vec<_>>()),
    ]);
    let registry = Registry { epoch: 1, copies };
    let timeout = Duration::from_secs(60);
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[10]))?;
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
        for (b, blob) in &blobs {
            if b.rank == rank {
                s.load_native_ggml(b, &index, size, &blob.bytes, timeout)?
            }
        }
        s.seal_native_ggml(counters[&rank] as usize, timeout)?;
        sessions.push(s);
    }
    let mut bad_stamp = stamp.clone();
    bad_stamp.batch = 10;
    let t = ledger.reserve_ticket(&bad_stamp, attempt)?;
    let mut bad = inputs.clone();
    bad[1][0] = f32::NAN;
    if RoutedBatch::prepare(&mut gdn, &mut post, &ledger, &t, &bad).is_ok() || gdn.next_token() != 0
    {
        return Err("invalid batch advanced".into());
    }
    let mut foreign_stamp = stamp.clone();
    foreign_stamp.batch = 11;
    let t = ledger.reserve_ticket(&foreign_stamp, attempt + 1)?;
    let foreign = RoutedBatch::prepare(&mut serial, &mut post, &ledger, &t, &inputs)?;
    let foreign_shared = foreign.shared(&mut shared)?;
    drop(foreign);
    let mut fault_stamp = stamp.clone();
    fault_stamp.batch = 12;
    let t = ledger.reserve_ticket(&fault_stamp, attempt + 2)?;
    let fault = RoutedBatch::prepare(&mut gdn, &mut post, &ledger, &t, &inputs)?;
    let collected = fault.dispatch(
        &mut sessions,
        registry.clone(),
        BTreeMap::new(),
        orders.clone(),
        &ledger,
        t,
        timeout,
    )?;
    if fault.commit(foreign_shared, collected).is_ok()
        || gdn.next_token() != 0
        || serial.next_token() != 0
    {
        return Err("foreign batch committed".into());
    }
    let t = ledger.reserve_ticket(&stamp, attempt + 3)?;
    let batch = RoutedBatch::prepare(&mut gdn, &mut post, &ledger, &t, &inputs)?;
    let expected_ids = [
        vec![433, 309, 447, 475, 261, 252, 123, 107, 448, 469],
        vec![309, 433, 447, 475, 261, 252, 123, 107, 232, 448],
    ];
    for (actual, expected) in batch.choices().zip(&expected_ids) {
        if actual.iter().map(|c| c.expert).collect::<Vec<_>>() != *expected {
            return Err("route mismatch".into());
        }
    }
    let sh = batch.shared(&mut shared)?;
    let col = batch.dispatch(
        &mut sessions,
        registry.clone(),
        BTreeMap::new(),
        orders.clone(),
        &ledger,
        t,
        timeout,
    )?;
    let result = batch.commit(sh, col)?;
    if digest(&result.residuals[0])
        != "73f76dfe1165e8297193da3222b835d820aacb69e7a094d7d69de60fed9fe08d"
    {
        return Err("first residual differs from R432 independent reconstruction".into());
    }
    let mut expected = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let mut st = stamp.clone();
        st.batch = (i + 1) as u64;
        let t = ledger.reserve_ticket(&st, attempt + i as u64 + 4)?;
        let proposal = serial.prepare_reserved(&ledger, &t, i as u64, input)?;
        let route = Routed::prepare(&mut post, &ledger, &t, proposal, input)?;
        let sh = route.shared(&mut shared)?;
        let col = route.dispatch(
            &mut sessions,
            registry.clone(),
            BTreeMap::new(),
            orders.clone(),
            &ledger,
            t,
            timeout,
        )?;
        expected.push(route.commit(&mut serial, sh, col)?.residual)
    }
    if result.residuals.len() != 2
        || result.residuals.iter().zip(&expected).any(|(x, y)| {
            x.iter()
                .map(|v| v.to_bits())
                .ne(y.iter().map(|v| v.to_bits()))
        })
        || gdn.next_token() != 2
        || serial.next_token() != 2
    {
        return Err("serial mismatch".into());
    }
    let mut receipts = Vec::new();
    for s in sessions {
        let n = s.traffic.native_ggml_raw_execs;
        let r = s.close(timeout);
        if n != 4
            || r.bye_execs != Some(4)
            || r.exit_code != Some(0)
            || r.killed
            || r.error.is_some()
        {
            return Err("worker receipt".into());
        }
        receipts.push(serde_json::json!({"rank":r.rank,"pid":r.pid,"execs":n}));
    }
    Ok(
        serde_json::json!({"status":"pass","run_id":stamp.run_id,"plan":stamp.plan_id,"residual_sha256":result.residuals.iter().map(|x|digest(x)).collect::<Vec<_>>(),"workers":receipts,"positions":2}),
    )
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("GDN batch refused: {e}");
            std::process::exit(1)
        }
    }
}
