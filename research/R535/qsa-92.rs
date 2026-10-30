//! Local no-worker prefix observation for 92 positions; not a completed layer.
use std::io::Write;
use voltafabric::{gdn_cpu::Identity, sc117_qsa_stage};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: qsa-92-position DENSE SHARD".into());
    }
    let id = Identity {
        run_id: "r535-independent-prefix".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 3,
        epoch: 1,
    };
    let mut dense = std::fs::File::open(&args[1])?;
    let mut shard = std::fs::File::open(&args[2])?;
    let manifest = serde_json::from_str(include_str!("../../fixtures/r430/layer3.json"))?;
    let mut stage = sc117_qsa_stage::Stage::load(manifest, id.clone(), &mut dense, &mut shard, 92)?;
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for token in 0..92 {
        let input: Vec<f32> = (0..10240)
            .map(|i| (((i * (token * 2 + 1) + token * 17) % 157) as f32 - 78.0) * 0.00009)
            .collect();
        let candidate = stage.prepare(&id, token, &input)?;
        let output = stage.commit(candidate)?;
        for value in output
            .mixed
            .iter()
            .chain(&output.inject)
            .chain(&output.raw_query_gate)
            .chain(&output.query)
            .chain(&output.gate)
            .chain(&output.raw_key)
            .chain(&output.key)
            .chain(&output.value)
            .chain(&output.gated_attention)
            .chain(&output.attention)
        {
            w.write_all(&value.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("92-position QSA observation refused: {e}");
        std::process::exit(1);
    }
}
