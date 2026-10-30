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
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err(
            "usage: vf-sc117-gdn-stage MANIFEST MANIFEST_SHA DENSE SHARD RESIDUAL RESIDUAL_SHA"
                .into(),
        );
    }
    let mf = open(&a[1])?;
    let n = mf.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest size".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(&a[1], n as usize, &a[2])?)?;
    let input = exact(&a[5], 40960, &a[6])?;
    let residual: Vec<f32> = input
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "r418-local-zero-state".into(),
        model_revision: m.model_revision.clone(),
        layer: m.layer,
        epoch: 0,
    };
    let mut d = open(&a[3])?;
    let mut s = open(&a[4])?;
    let mut stage = Stage::load(m, id.clone(), &mut d, &mut s)?;
    let stamp = Stamp {
        run_id: id.run_id.clone(),
        model_revision: id.model_revision.clone(),
        plan_id: "diagnostic-only".into(),
        layer: id.layer,
        batch: 0,
        epoch: id.epoch,
    };
    let rejected = stage.prepare(&stamp, 1, 0, &residual)?;
    if stage
        .commit_collected(rejected, synthetic(&stamp, 99, &residual))
        .is_ok()
        || stage.next_token() != 0
    {
        return Err("wrong attempt advanced token".into());
    }
    let wrong_stamp = stage.prepare(&stamp, 1, 0, &residual)?;
    let mut changed = stamp.clone();
    changed.batch += 1;
    if stage
        .commit_collected(wrong_stamp, synthetic(&changed, 1, &residual))
        .is_ok()
        || stage.next_token() != 0
    {
        return Err("wrong batch advanced token".into());
    }
    let incomplete = stage.prepare(&stamp, 1, 0, &residual)?;
    let mut missing = synthetic(&stamp, 1, &residual);
    missing.outputs.clear();
    if stage.commit_collected(incomplete, missing).is_ok() || stage.next_token() != 0 {
        return Err("incomplete residual advanced token".into());
    }
    let candidate = stage.prepare(&stamp, 2, 0, &residual)?;
    let stale = stage.prepare(&stamp, 2, 0, &residual)?;
    // This synthetic completion exercises state mechanics, not a completed model layer.
    let result = stage
        .commit_collected(candidate, synthetic(&stamp, 2, &residual))?
        .output;
    if stage
        .commit_collected(stale, synthetic(&stamp, 2, &residual))
        .is_ok()
        || stage.next_token() != 1
    {
        return Err("stale candidate advanced token".into());
    }
    if stage.prepare(&stamp, 3, 0, &residual).is_ok() || stage.next_token() != 1 {
        return Err("duplicate token was not refused".into());
    }
    let mut invalid = residual.clone();
    invalid[0] = f32::NAN;
    if stage.prepare(&stamp, 3, 1, &invalid).is_ok() || stage.next_token() != 1 {
        return Err("invalid activation advanced token".into());
    }
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for v in result
        .mixed
        .iter()
        .chain(&result.inject)
        .chain(&result.qkv)
        .chain(&result.recurrent)
        .chain(&result.gate)
        .chain(&result.beta)
    {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("SC117 stage refused: {e}");
        std::process::exit(1);
    }
}
