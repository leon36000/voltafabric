//! Explicit one-step local diagnostic for the reusable SC117 resident stage.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};
use voltafabric::{
    completion::{OutputSlot, RoutedRawOutput},
    gdn_cpu::Identity,
    moe::{Choice, Registry, Replica, Stamp},
    routing::RoutingProblem,
    sc117_gdn_stage::{Manifest, Stage},
    sc117_layer_completion::{LayerInputs, StampedValues},
    sha256,
};
fn synthetic(stamp: &Stamp, attempt: u64, residual: &[f32]) -> LayerInputs {
    let problem = RoutingProblem {
        hidden: 2560,
        experts: vec![0],
        routes: vec![vec![Choice {
            expert: 0,
            weight: 1.0,
        }]],
        registry: Registry {
            epoch: stamp.epoch,
            copies: vec![Replica {
                expert: 0,
                rank: 0,
                local: 0,
            }],
        },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(0, vec![0])]),
    };
    let values = |v| StampedValues {
        stamp: stamp.clone(),
        attempt,
        values: v,
    };
    let output = RoutedRawOutput {
        stamp: stamp.clone(),
        rank: 0,
        attempt,
        rows: 1,
        columns: 2560,
        slots: vec![OutputSlot {
            token: 0,
            slot: 0,
            expert: 0,
            local: 0,
        }],
        fp16_bits: vec![0; 2560],
    };
    LayerInputs {
        problem,
        attempt,
        outputs: vec![output],
        shared: values(vec![0.0; 2560]),
        attention_residual: values(residual.to_vec()),
        injection: values(vec![0.0; 4]),
    }
}
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
fn bytes(o: &voltafabric::sc117_gdn_stage::Output) -> Vec<u8> {
    o.mixed
        .iter()
        .chain(&o.inject)
        .chain(&o.qkv)
        .chain(&o.recurrent)
        .chain(&o.gate)
        .chain(&o.beta)
        .flat_map(|v| v.to_le_bytes())
        .collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err(
            "expected manifest dense shard residual reference and two new output paths".into(),
        );
    }
    let mb = std::fs::read(&a[1])?;
    if sha256::hex(&mb) != "e828fe4e01fe8c9a86201b79fa40f6e29ceca325b2d0a4ca9be954a939d8afca" {
        return Err("manifest digest".into());
    }
    let m: Manifest = serde_json::from_slice(&mb)?;
    let raw = exact(
        &a[4],
        40960,
        "653f8385328f68ebebfe966082717b7c566053d540081b0539d5e0370ea15b65",
    )?;
    let r: Vec<f32> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let r2: Vec<f32> = r.iter().map(|x| x * 1.001).collect();
    let id = Identity {
        run_id: "r572-gdn-oracle-cpu".into(),
        model_revision: m.model_revision.clone(),
        layer: m.layer,
        epoch: 1,
    };
    let stamp = Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "r572-diagnostic".into(),
        layer: id.layer,
        batch: 1,
        epoch: id.epoch,
    };
    let mut d = open(&a[2])?;
    let mut s = open(&a[3])?;
    let mut batch = Stage::load(m, id.clone(), &mut d, &mut s)?;
    let m: Manifest = serde_json::from_slice(&mb)?;
    let mut seq = Stage::load(m, id.clone(), &mut d, &mut s)?;
    let p0 = seq.prepare(&stamp, 1, 0, &r)?;
    let v0 = bytes(p0.output());
    if v0 != std::fs::read(&a[5])? {
        return Err("first position differs from R432 checked output".into());
    }
    seq.commit_collected(p0, synthetic(&stamp, 1, &r))?;
    let p1 = seq.prepare(&stamp, 1, 1, &r2)?;
    let v1 = bytes(p1.output());
    seq.commit_collected(p1, synthetic(&stamp, 1, &r2))?;
    {
        let mut b = batch.begin_batch(2)?;
        let got0 = bytes(b.prepare_next(&id, &r)?);
        let got1 = bytes(b.prepare_next(&id, &r2)?);
        if got0 != v0 || got1 != v1 {
            return Err("batch/serial mismatch".into());
        }
        for (path, data) in [(&a[6], &got0), (&a[7], &got1)] {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        b.commit()?;
    }
    if batch.next_token() != 2 {
        return Err("batch position".into());
    }
    {
        let mut b = batch.begin_batch(2)?;
        b.prepare_next(&id, &r)?;
        let mut bad = r2.clone();
        bad[0] = f32::NAN;
        if b.prepare_next(&id, &bad).is_ok() || b.commit().is_ok() {
            return Err("failed batch accepted".into());
        }
    }
    if batch.next_token() != 2 {
        return Err("failure advanced state".into());
    }
    {
        let mut b = batch.begin_batch(2)?;
        b.prepare_next(&id, &r)?;
        if b.commit().is_ok() {
            return Err("incomplete batch accepted".into());
        }
    }
    {
        let mut b = batch.begin_batch(1)?;
        let mut wrong = id.clone();
        wrong.epoch += 1;
        if b.prepare_next(&wrong, &r).is_ok() || b.commit().is_ok() {
            return Err("foreign identity accepted".into());
        }
    }
    let p2 = seq.prepare(&stamp, 1, 2, &r)?;
    let v2 = bytes(p2.output());
    {
        let mut b = batch.begin_batch(1)?;
        if bytes(b.prepare_next(&id, &r)?) != v2 {
            return Err("rollback altered recurrence".into());
        }
        b.commit()?;
    }
    if batch.next_token() != 3 || batch.begin_batch(0).is_ok() || batch.begin_batch(129).is_ok() {
        return Err("bounds".into());
    }
    println!("{{\"status\":\"pass\",\"positions\":2,\"first_matches_r432\":true,\"serial_bitwise\":true,\"rollback_verified\":true}}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("GDN batch refused: {e}");
        std::process::exit(1)
    }
}
