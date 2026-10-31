//! Local CPU check of the two-position QSA transaction against R552 trace.
use std::fs;
use voltafabric::{
    gdn_cpu::Identity,
    sc117_qsa_stage::{Manifest, Output, Stage},
    sha256,
};
type E = Box<dyn std::error::Error>;
fn words(o: &Output) -> Vec<f32> {
    [
        &o.mixed,
        &o.inject,
        &o.raw_query_gate,
        &o.query,
        &o.gate,
        &o.raw_key,
        &o.key,
        &o.value,
        &o.gated_attention,
        &o.attention,
    ]
    .into_iter()
    .flat_map(|v| v.iter().copied())
    .collect()
}
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: TRACE MANIFEST DENSE SHARD".into());
    }
    let trace = fs::read(&a[1])?;
    if trace.len() != 92 * 47620 * 4
        || sha256::hex(&trace) != "a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2"
    {
        return Err("trace mismatch".into());
    }
    let manifest = fs::read(&a[2])?;
    if sha256::hex(&manifest) != "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21"
    {
        return Err("manifest mismatch".into());
    }
    let m: Manifest = serde_json::from_slice(&manifest)?;
    let id = Identity {
        run_id: "r556-qsa-batch-trace".into(),
        model_revision: m.model_revision.clone(),
        layer: 3,
        epoch: 1,
    };
    let mut stage = Stage::load(
        m,
        id.clone(),
        &mut fs::File::open(&a[3])?,
        &mut fs::File::open(&a[4])?,
        4,
    )?;
    let records: Vec<Vec<f32>> = (0..2)
        .map(|pos| {
            trace[pos * 47620 * 4..(pos + 1) * 47620 * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    {
        let mut tx = stage.begin_batch(2)?;
        tx.prepare_next(&id, &records[0][..10240])?;
        let other = Identity {
            epoch: 2,
            ..id.clone()
        };
        if tx.prepare_next(&other, &records[1][..10240]).is_ok() || tx.commit().is_ok() {
            return Err("failed transaction accepted".into());
        }
    }
    if !stage.is_empty() {
        return Err("failed batch committed".into());
    }
    {
        let mut tx = stage.begin_batch(2)?;
        tx.prepare_next(&id, &records[0][..10240])?;
        if tx.commit().is_ok() {
            return Err("incomplete transaction accepted".into());
        }
    }
    if !stage.is_empty() {
        return Err("incomplete batch committed".into());
    }
    let mut tx = stage.begin_batch(2)?;
    for record in &records {
        tx.prepare_next(&id, &record[..10240])?;
    }
    let outputs = tx.commit()?;
    if outputs.len() != 2 || stage.len() != 2 {
        return Err("commit size".into());
    }
    for (record, out) in records.iter().zip(&outputs) {
        let actual = words(out);
        if actual.len() != 37380
            || actual
                .iter()
                .zip(&record[10240..])
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("trace output mismatch".into());
        }
    }
    Ok(
        serde_json::json!({"status":"pass","positions":2,"output_f32_elements":74760,"failed_transaction_preserved_cache":true,"trace_sha256":sha256::hex(&trace)}),
    )
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("QSA batch check refused: {e}");
            std::process::exit(1)
        }
    }
}
