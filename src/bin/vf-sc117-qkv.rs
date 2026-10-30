//! Bounded local SC117 layer-0 Q6_K QKV projection; no GPU execution.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256,
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: vf-sc117-qkv SHARD INPUT_F32 EXPECTED_TENSOR_SHA256".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(&args[1])?;
    let before = file.metadata()?;
    if !before.is_file() || before.len() != 55_063_448_064 {
        return Err("unexpected GGUF shard".into());
    }
    file.seek(SeekFrom::Start(892_207_104))?;
    let mut weights = vec![0_u8; 21_504_000];
    file.read_exact(&mut weights)?;
    if !sha256::is_hex_digest(&args[3]) || sha256::hex(&weights) != args[3] {
        return Err("QKV tensor digest mismatch".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = file.metadata()?;
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
            return Err("GGUF shard changed during read".into());
        }
    }
    let mut input = options.open(&args[2])?;
    if !input.metadata()?.is_file() || input.metadata()?.len() != 2560 * 4 {
        return Err("unexpected activation file".into());
    }
    let mut raw = vec![0; 2560 * 4];
    input.read_exact(&mut raw)?;
    let activation: Vec<f32> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let binding = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        tensor: "blk.0.attn_qkv.weight".into(),
        ggml_type: 14,
        rows: 10240,
        columns: 2560,
        sha256: args[3].clone(),
    };
    let mut matrix = QuantizedMatrix::load(binding, &weights)?;
    let out = matrix.forward(&activation)?;
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for v in out {
        stdout.write_all(&v.to_le_bytes())?;
    }
    stdout.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("QKV refused: {e}");
        std::process::exit(1);
    }
}
