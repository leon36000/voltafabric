//! Explicit single observed SC117 token shared-expert CPU diagnostic.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    hyperconnection_cpu,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256, shared_expert_cpu,
};
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
fn read(
    f: &mut std::fs::File,
    off: u64,
    len: usize,
    hash: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(off))?;
    let mut b = vec![0; len];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != hash {
        return Err(format!(
            "digest mismatch at offset {off}, length {len}, observed {}",
            sha256::hex(&b)
        )
        .into());
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
#[allow(clippy::too_many_arguments)]
fn matrix(
    shard: &mut std::fs::File,
    off: u64,
    len: usize,
    hash: &str,
    tensor: &str,
    kind: i32,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = read(shard, off, len, hash)?;
    let b = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        tensor: tensor.into(),
        ggml_type: kind,
        rows,
        columns: cols,
        sha256: hash.into(),
    };
    Ok(QuantizedMatrix::load(b, &bytes)?.forward(x)?)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: vf-sc117-layer1-shared SHARD ROUTE_RESULT COLLECTOR_JSON".into());
    }
    let mut shard = open(&a[1])?;
    let before = shard.metadata()?;
    if before.len() != 55_063_448_064 {
        return Err("shard size".into());
    }
    let mut input = open(&a[2])?;
    if input.metadata()?.len() != 110728 {
        return Err("input size".into());
    }
    let input = floats(&read(
        &mut input,
        0,
        110728,
        "f252ac7c7cf54d959b195db3e704c0de62e17a74417342e81f505a4887492bcf",
    )?);
    let mut cf = open(&a[3])?;
    if cf.metadata()?.len() != 58332 {
        return Err("collector size".into());
    }
    let raw = read(
        &mut cf,
        0,
        58332,
        "06904c1defa825df39d0e167ff0021d08610a79e0d8596523612c2656d54d154",
    )?;
    let v: serde_json::Value = serde_json::from_slice(&raw)?;
    if v["ok"].as_bool() != Some(true) || v["result"]["accepted_rows"].as_u64() != Some(10) {
        return Err("collector evidence".into());
    }
    let row = v["result"]["output"][0]
        .as_array()
        .ok_or("missing collector output")?;
    if row.len() != 2560 {
        return Err("collector width".into());
    }
    let routed = row
        .iter()
        .map(|x| x.as_f64().ok_or("collector value"))
        .collect::<Result<Vec<_>, _>>()?;
    let ginp = read(
        &mut shard,
        2_617_724_800,
        5120,
        "c76074d1062bbb19655287f3839ac6ebaafd2411ff6c563d6736336b68c10e9e",
    )?;
    let scalar = shared_expert_cpu::scalar_gate(&input[25088..27648], &ginp)?;
    let gate = matrix(
        &mut shard,
        2_617_729_920,
        870400,
        "37ba126d3d2f52d8aba60fbe3f609a43df8164b761810feaa33df212d861d5a5",
        "blk.1.ffn_gate_shexp.weight",
        23,
        640,
        2560,
        &input[25088..27648],
    )?;
    let up = matrix(
        &mut shard,
        2_887_297_920,
        921600,
        "87862686c7f62c33c625cfb7e4683f6e49a0a5afc9f1b64e0d819beba9fac5da",
        "blk.1.ffn_up_shexp.weight",
        12,
        640,
        2560,
        &input[25088..27648],
    )?;
    let mid = shared_expert_cpu::activation(&gate, &up)?;
    let down = matrix(
        &mut shard,
        2_344_664_960,
        1740800,
        "6d73cd2e8af36682abeafb6d9125ad030baa7bf33dd4cd49b4e962e77f8febd2",
        "blk.1.ffn_down_shexp.weight",
        8,
        2560,
        640,
        &mid,
    )?;
    let shared: Vec<f32> = down.iter().map(|&x| x * scalar).collect();
    let combined = shared_expert_cpu::combine(&routed, &shared)?;
    let residual =
        hyperconnection_cpu::write(&input[14848..25088], &combined, &input[27648..27652])?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = shard.metadata()?;
        if (
            before.dev(),
            before.ino(),
            before.len(),
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec(),
        ) != (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec(),
        ) {
            return Err("shard changed".into());
        }
    }
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for x in gate
        .iter()
        .chain(&up)
        .chain(&mid)
        .chain(&down)
        .chain(&shared)
        .chain(&combined)
        .chain(&residual)
        .chain(std::iter::once(&scalar))
    {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("shared output refused: {e}");
        std::process::exit(1);
    }
}
