//! Bounded CPU research sender for the R257 multipart owner. Not Rust Session integration.
use serde_json::{json, Value};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};
const CHUNK: usize = 4 * 1024 * 1024;
fn reply(reader: &mut impl BufRead) -> Result<Value, String> {
    let mut line = Vec::new();
    reader.take(65537).read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
    if line.is_empty() || line.len() > 65536 || line.last() != Some(&b'\n') { return Err("invalid reply frame; outcome UNKNOWN".into()); }
    serde_json::from_slice(&line).map_err(|e| e.to_string())
}
fn send(writer: &mut impl Write, header: Value, body: &[u8]) -> Result<(), String> {
    let mut b = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    b.push(b'\n');
    if b.len() > 65536 || body.len() > CHUNK { return Err("frame bounds exceeded".into()); }
    writer.write_all(&b).and_then(|_| writer.write_all(body)).and_then(|_| writer.flush()).map_err(|e| format!("write failed, outcome UNKNOWN: {e}"))
}
fn run() -> Result<(), String> {
    let mut raw = String::new(); io::stdin().read_to_string(&mut raw).map_err(|e| e.to_string())?;
    let cfg: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let get = |k: &str| cfg[k].as_str().ok_or_else(|| format!("missing {k}"));
    let path = get("artifact")?; let digest = get("sha256")?;
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if !(1..=256*1024*1024).contains(&size) { return Err("invalid artifact size".into()); }
    let mut child = Command::new(get("python")?).arg("-B").arg(get("worker")?).arg(get("root")?).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e| e.to_string())?;
    let mut tx = child.stdin.take().ok_or("missing stdin")?;
    let mut rx = BufReader::new(child.stdout.take().ok_or("missing stdout")?);
    send(&mut tx,json!({"op":"hello","protocol":1,"rank":2}), &[])?;
    let h=reply(&mut rx)?;
    if h["reply"]!="hello" || h["rank"]!=2 || !h["capabilities"].as_array().is_some_and(|v|v.iter().any(|x|x=="cpu_qpn2_multipart_v1")) || h["max_chunk_bytes"]!=CHUNK {return Err("capability not negotiated; no transfer".into());}
    let identity = cfg.get("identity").ok_or("missing identity")?.clone();
    send(&mut tx,json!({"op":"begin_multipart","identity":identity,"artifact_bytes":size,"artifact_sha256":digest}), &[])?;
    if reply(&mut rx)? != json!({"reply":"multipart_ready","offset":0}) {return Err("begin refused; outcome UNKNOWN".into());}
    let mut buf = vec![0u8; CHUNK]; let mut offset=0u64;
    while offset<size {
        let amount=usize::try_from((size-offset).min(CHUNK as u64)).map_err(|e|e.to_string())?;
        file.read_exact(&mut buf[..amount]).map_err(|e|format!("artifact read failed; outcome UNKNOWN: {e}"))?;
        send(&mut tx,json!({"op":"multipart_chunk","offset":offset,"bytes":amount}),&buf[..amount])?;
        offset+=amount as u64;
        if reply(&mut rx)? != json!({"reply":"chunk_accepted","offset":offset}) {return Err("chunk acknowledgement mismatch; outcome UNKNOWN".into());}
    }
    let mut extra=[0u8;1]; if file.read(&mut extra).map_err(|e|e.to_string())?!=0 {return Err("artifact grew; outcome UNKNOWN".into());}
    send(&mut tx,json!({"op":"multipart_finish"}),&[])?;
    let result=reply(&mut rx)?;
    if result != json!({"reply":"loaded","artifact_sha256":digest,"rank":2,"experts":10}) {return Err("load reply mismatch; outcome UNKNOWN".into());}
    send(&mut tx,json!({"op":"shutdown"}),&[])?;
    if reply(&mut rx)? != json!({"reply":"bye","execs":0}) {return Err("shutdown mismatch; outcome UNKNOWN".into());}
    drop(tx);let status=child.wait().map_err(|e|e.to_string())?; if !status.success() {return Err("worker failed after shutdown".into());}
    println!("{}",json!({"status":"PASS","bytes":size,"sha256":digest,"owner":"READY and CLOSED","protocol":"cpu_qpn2_multipart_v1","gpu":false})); Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("{e}");std::process::exit(1)}}
