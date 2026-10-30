//! Zero-history, one-token SC117 GDN preprocess CPU diagnostic.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    gdn_cpu::{ConvHistory, Identity},
    sha256,
};
fn open_regular(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let f = o.open(path)?;
    if !f.metadata()?.is_file() {
        return Err("not a regular file".into());
    }
    Ok(f)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: vf-sc117-gdn-preprocess DENSE QKV CONV_SHA QKV_SHA".into());
    }
    let mut dense = open_regular(&args[1])?;
    let before = dense.metadata()?;
    if before.len() != 1_538_035_200 {
        return Err("unexpected dense pack".into());
    }
    dense.seek(SeekFrom::Start(42_726_592))?;
    let mut conv = vec![0_u8; 163_840];
    dense.read_exact(&mut conv)?;
    if !sha256::is_hex_digest(&args[3]) || sha256::hex(&conv) != args[3] {
        return Err("convolution digest mismatch".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = dense.metadata()?;
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
            return Err("dense pack changed during read".into());
        }
    }
    let mut input = open_regular(&args[2])?;
    if input.metadata()?.len() != 40_960 {
        return Err("unexpected QKV length".into());
    }
    let mut bytes = vec![0_u8; 40_960];
    input.read_exact(&mut bytes)?;
    if !sha256::is_hex_digest(&args[4]) || sha256::hex(&bytes) != args[4] {
        return Err("QKV digest mismatch".into());
    }
    let qkv: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let kernel: Vec<f32> = conv
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "local-zero-history".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        epoch: 0,
    };
    let mut state = ConvHistory::new(id.clone(), 10240)?;
    let out = state.step(&id, &qkv, &kernel, 16, 48, 128, 1e-6)?;
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for value in out.q.iter().chain(&out.k).chain(&out.v) {
        stdout.write_all(&value.to_le_bytes())?;
    }
    stdout.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("GDN preprocess refused: {e}");
        std::process::exit(1);
    }
}
