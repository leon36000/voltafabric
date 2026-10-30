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
    sc117_post_attention, sha256,
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
    if a.len() != 9 {
        return Err("usage: preview GDN_MAN POST_MAN DENSE SHARD RES0 RES1 OUT0 OUT1".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(
        &a[1],
        open(&a[1])?.metadata()?.len() as usize,
        "d822695e4fceee7b285b510e9ec7f97cc0a1507188b6ca698de5b6b216073348",
    )?)?;
    let pm: sc117_post_attention::Manifest = serde_json::from_slice(&exact(
        &a[2],
        1299,
        "8959d5ff4bbac4c6be08e6a978c5d27ffa1f09f8e2f3541add9395368121b0f1",
    )?)?;
    let hashes = [
        "ed1d74a7f93128f6b6a3b30855687a660cba93eca8e1c76c72086dc942fa7061",
        "9a69b12aec635edd0a042269e830779b49be0db018f6812479f617867371d96d",
    ];
    let id = Identity {
        run_id: "r428-preview".into(),
        model_revision: m.model_revision.clone(),
        layer: 2,
        epoch: 1,
    };
    let mut d = open(&a[3])?;
    let mut sh = open(&a[4])?;
    let mut g = Stage::load(m, id.clone(), &mut d, &mut sh)?;
    let mut p = sc117_post_attention::Stage::load(pm, id.clone(), &mut d, &mut sh)?;
    for i in 0..2 {
        let res: Vec<f32> = exact(&a[5 + i], 40960, hashes[i])?
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let stamp = Stamp {
            run_id: id.run_id.clone(),
            model_revision: id.model_revision.clone(),
            plan_id: "r428-preview-synthetic".into(),
            layer: 2,
            batch: i as u64,
            epoch: 1,
        };
        let candidate = g.prepare(&stamp, (i + 1) as u64, i as u64, &res)?;
        let r = p.compute(&id, candidate.output(), &res)?;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&a[7 + i])?;
        for x in candidate
            .output()
            .mixed
            .iter()
            .chain(&candidate.output().inject)
            .chain(&candidate.output().qkv)
            .chain(&candidate.output().recurrent)
            .chain(&candidate.output().gate)
            .chain(&candidate.output().beta)
        {
            out.write_all(&x.to_le_bytes())?;
        }
        let mut activation = Vec::new();
        for x in &r.mixed {
            activation.extend_from_slice(&x.to_le_bytes());
        }
        println!(
            "{}",
            serde_json::json!({"token":i,"ids":r.choices.iter().map(|c|c.expert).collect::<Vec<_>>(),"weights":r.choices.iter().map(|c|c.weight).collect::<Vec<_>>(),"activation_sha":sha256::hex(&activation)})
        );
        g.commit_collected(candidate, synthetic(&stamp, (i + 1) as u64, &res))?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("preview refused: {e}");
        std::process::exit(1);
    }
}
