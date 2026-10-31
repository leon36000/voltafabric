//! Recover real layer-3 FFN inputs and routes from an authenticated local trace.
use voltafabric::{gdn_cpu::Identity, sc117_ffn_stage::{Manifest, Stage}, sha256};
use std::fs;
type E = Box<dyn std::error::Error>;
fn run() -> Result<serde_json::Value, E> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 { return Err("usage: program TRACE MANIFEST DENSE".into()); }
    let trace = fs::read(&a[1])?;
    if trace.len() != 92 * 47620 * 4 || sha256::hex(&trace) != "a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2" { return Err("trace identity mismatch".into()); }
    let manifest = fs::read(&a[2])?;
    if sha256::hex(&manifest) != "79d44d7663231fea2a6c04c309d776cd9552af7d0ecb859c6428b8124351afd1" { return Err("manifest mismatch".into()); }
    let m: Manifest = serde_json::from_slice(&manifest)?;
    let id = Identity {run_id:"r554-recover-observed-prefill".into(), model_revision:m.model_revision.clone(), layer:3, epoch:1};
    let stage = Stage::load(m, id.clone(), &mut fs::File::open(&a[3])?)?;
    let mut records = Vec::new();
    for pos in 0..2 {
        let b = &trace[pos * 47620 * 4..(pos + 1) * 47620 * 4];
        let values: Vec<f32> = b.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
        if values.iter().any(|v|!v.is_finite()) {return Err("nonfinite trace".into());}
        let routed=stage.compute(&id, &values[..10240], &values[45060..47620], &values[12800..12804])?;
        let mixed_bytes:Vec<u8>=routed.mixed.iter().flat_map(|v|v.to_le_bytes()).collect();
        records.push(serde_json::json!({"position":pos,"choices":routed.choices,"activation_sha256":sha256::hex(&mixed_bytes),"activation_bits":routed.mixed.iter().map(|x|x.to_bits()).collect::<Vec<_>>() }));
    }
    Ok(serde_json::json!({"trace_sha256":"a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2","source_plan":"r544-4dc0e95e92153de53fc65afae1c344deb887644645b3ba5e5c859cea183531b6","layer":3,"records":records}))
}
fn main() {match run(){Ok(v)=>println!("{v}"),Err(e)=>{eprintln!("route recovery refused: {e}");std::process::exit(1)}}}
