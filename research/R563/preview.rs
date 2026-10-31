use std::fs;
use voltafabric::{gdn_cpu::Identity, sc117_gdn_stage, sc117_post_attention, sha256};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("args: gdn post dense shard residual".into());
    }
    let g = fs::read(&a[1])?;
    let p = fs::read(&a[2])?;
    let r = fs::read(&a[5])?;
    if sha256::hex(&g) != "e828fe4e01fe8c9a86201b79fa40f6e29ceca325b2d0a4ca9be954a939d8afca"
        || sha256::hex(&p) != "4edc0334cb6bc661f10c74efbfb9989c9fca973cda15baa1b5c90450a5bb6e94"
        || sha256::hex(&r) != "653f8385328f68ebebfe966082717b7c566053d540081b0539d5e0370ea15b65"
    {
        return Err("input hash".into());
    }
    let residual: Vec<f32> = r
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "r563-preview".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 4,
        epoch: 1,
    };
    let mut d = fs::File::open(&a[3])?;
    let mut s = fs::File::open(&a[4])?;
    let mut stage =
        sc117_gdn_stage::Stage::load(serde_json::from_slice(&g)?, id.clone(), &mut d, &mut s)?;
    let mut post =
        sc117_post_attention::Stage::load(serde_json::from_slice(&p)?, id.clone(), &mut d, &mut s)?;
    let mut batch = stage.begin_batch(2)?;
    for scale in [1.0, 1.001] {
        let input: Vec<_> = residual.iter().map(|v| v * scale).collect();
        let o = batch.prepare_next(&id, &input)?;
        let r = post.compute(&id, o, &input)?;
        println!(
            "{:?}",
            r.choices.iter().map(|x| x.expert).collect::<Vec<_>>()
        );
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
