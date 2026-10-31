//! Offline assembly of retained layer components; no session or state commit.
use std::{error::Error, io::Write};
use voltafabric::{hyperconnection_cpu, sha256, shared_expert_cpu};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("expected input JSON and output directory".into());
    }
    let raw = std::fs::read(&a[1])?;
    if sha256::hex(&raw) != "31800dc58ccac050049ce417dd0bdb61143e3d022c8ab9cae210f02ccfec8cd4" {
        return Err("historical input hash".into());
    }
    let doc: serde_json::Value = serde_json::from_slice(&raw)?;
    let v = &doc["tokens"][0];
    if v["activation_sha"].as_str()
        != Some("c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a")
    {
        return Err("activation identity".into());
    }
    let vector = |key: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        let values = v[key]
            .as_array()
            .ok_or("field not array")?
            .iter()
            .map(|x| x.as_f64().map(|v| v as f32).ok_or("invalid value"))
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite".into());
        }
        Ok(values)
    };
    let shared = vector("shared")?;
    let post = vector("post")?;
    let injection = vector("injection")?;
    if shared.len() != 2560 || post.len() != 10240 || injection.len() != 4 {
        return Err("shape".into());
    }
    let orig = v["collector"]
        .as_array()
        .ok_or("collector missing")?
        .iter()
        .map(|x| x.as_f64().ok_or("bad collector"))
        .collect::<Result<Vec<_>, _>>()?;
    let raw_new = std::fs::read(".runs/vf-r604-work/collector.f64")?;
    if raw_new.len() != 20480
        || sha256::hex(&raw_new)
            != "ef0a699feb2e5d21cea882c1f7c8cbd517883a2de419e0c511aa91e784514ac5"
    {
        return Err("scalar collector identity".into());
    }
    let new: Vec<f64> = raw_new
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    for (name, row) in [("historical.f32", orig), ("scalar.f32", new)] {
        let block = shared_expert_cpu::combine(&row, &shared)?;
        let result = hyperconnection_cpu::write(&post, &block, &injection)?;
        let bytes: Vec<u8> = result.iter().flat_map(|x| x.to_le_bytes()).collect();
        if name == "historical.f32"
            && sha256::hex(&bytes) != v["residual_sha"].as_str().ok_or("missing hash")?
        {
            return Err("historical residual mismatch".into());
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(std::path::Path::new(&a[2]).join(name))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        println!("{} sha256={}", name, sha256::hex(&bytes));
    }
    Ok(())
}
