//! Sequential, bounded-memory CPU checkpoint stage admission check; no owners or generation.
use std::{io::Read, path::Path};
use voltafabric::{
    gdn_cpu::Identity, moe::Stamp, sc117_ffn_stage, sc117_gdn_stage, sc117_post_attention,
    sc117_qsa_stage, sc117_shared_stage, sha256,
};
type E = Box<dyn std::error::Error>;
fn manifest<T: serde::de::DeserializeOwned>(path: &Path) -> Result<(T, String), E> {
    let f = std::fs::File::open(path)?;
    if !f.metadata()?.is_file() || f.metadata()?.len() == 0 || f.metadata()?.len() > 16_384 {
        return Err("manifest length".into());
    }
    let mut data = Vec::new();
    f.take(16_385).read_to_end(&mut data)?;
    if data.len() > 16_384 {
        return Err("manifest grew".into());
    }
    Ok((serde_json::from_slice(&data)?, sha256::hex(&data)))
}
fn hash_f32(values: &[f32]) -> String {
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    sha256::hex(&bytes)
}
fn run() -> Result<(), E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: all-layer-check DENSE SHARD MANIFEST_DIR QSA_MANIFEST_DIR".into());
    }
    let mut dense = std::fs::File::open(&a[1])?;
    let mut shard = std::fs::File::open(&a[2])?;
    let root = Path::new(&a[3]);
    let qroot = Path::new(&a[4]);
    let input: Vec<f32> = (0..10240)
        .map(|i| ((i % 131) as f32 - 65.0) * 0.00012)
        .collect();
    let mut results = Vec::new();
    for layer in 0..48 {
        let id = Identity {
            run_id: "r438-local-prefix".into(),
            model_revision: "sc117-local-observation".into(),
            layer,
            epoch: 1,
        };
        let (shared_m, shared_hash) = manifest(&root.join(format!("layer{layer}-shared.json")))?;
        let mut shared = sc117_shared_stage::Stage::load(shared_m, id.clone(), &mut shard)?;
        let (route, attention_hash, prefix_hash, route_manifest_hash) = if layer % 4 == 3 {
            let (q, qhash) = manifest(&qroot.join(format!("layer{layer}.json")))?;
            let (f, fhash) = manifest(&root.join(format!("layer{layer}-ffn.json")))?;
            let mut qsa = sc117_qsa_stage::Stage::load(q, id.clone(), &mut dense, &mut shard, 1)?;
            let ffn = sc117_ffn_stage::Stage::load(f, id.clone(), &mut dense)?;
            let candidate = qsa.prepare(&id, 0, &input)?;
            let attention_hash = hash_f32(&candidate.output().attention);
            let route = ffn.compute(
                &id,
                &input,
                &candidate.output().attention,
                &candidate.output().inject,
            )?;
            (route, attention_hash, qhash, fhash)
        } else {
            let (g, ghash) = manifest(&root.join(format!("layer{layer}-gdn.json")))?;
            let (p, phash) = manifest(&root.join(format!("layer{layer}-post.json")))?;
            let mut gdn = sc117_gdn_stage::Stage::load(g, id.clone(), &mut dense, &mut shard)?;
            let mut post =
                sc117_post_attention::Stage::load(p, id.clone(), &mut dense, &mut shard)?;
            let stamp = Stamp {
                run_id: id.run_id.clone(),
                model_revision: id.model_revision.clone(),
                plan_id: "r438-no-worker".into(),
                layer,
                batch: 0,
                epoch: 1,
            };
            let candidate = gdn.prepare(&stamp, 1, 0, &input)?;
            let r = post.compute(&id, candidate.output(), &input)?;
            let attention_hash = hash_f32(&r.attention);
            (
                sc117_ffn_stage::Routed {
                    residual: r.residual,
                    mixed: r.mixed,
                    inject: r.inject,
                    choices: r.choices,
                },
                attention_hash,
                ghash,
                phash,
            )
        };
        let shared_values = shared.compute(&id, &route.mixed)?;
        if shared_values.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite shared output".into());
        }
        results.push(
            serde_json::json!({"layer":layer,"prefix_manifest_sha256":prefix_hash,
            "route_manifest_sha256":route_manifest_hash,"shared_manifest_sha256":shared_hash,
            "attention_sha256":attention_hash,"shared_sha256":hash_f32(&shared_values),
            "expert_ids":route.choices.iter().map(|c| c.expert).collect::<Vec<_>>() }),
        );
        eprintln!("checked layer {layer}");
    }
    println!(
        "{}",
        serde_json::json!({"scope":"48 independent local CPU prefix/router/shared checks; not a model chain", "input_sha256":hash_f32(&input), "layers":results})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("all-layer check refused: {e}");
        std::process::exit(1);
    }
}
