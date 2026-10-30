//! Bounded SC117 layer-0 first-token CPU diagnostic; pinned local observations only.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    gdn_cpu::{project_gates, GdnState, Identity},
    sha256,
};

fn file(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let f = options.open(path)?;
    if !f.metadata()?.is_file() {
        return Err("not a regular file".into());
    }
    Ok(f)
}
fn span(
    f: &mut std::fs::File,
    offset: u64,
    len: usize,
    digest: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0; len];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != digest {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: vf-sc117-gdn-token DENSE MIXED QKV".into());
    }
    let mut dense = file(&args[1])?;
    let before = dense.metadata()?;
    if before.len() != 1_538_035_200 {
        return Err("unexpected dense pack size".into());
    }
    let a = floats(&span(
        &mut dense,
        42_234_880,
        192,
        "6e38a1018f33d34678019e53974ad84738381bd8ac092f65596d4cf800ded59d",
    )?);
    let alpha = span(
        &mut dense,
        42_235_072,
        245_760,
        "c02197d7f8c87944947b2a386a0e6d6e941537c6f699f24c6c29c3c1973a2ae2",
    )?;
    let beta = span(
        &mut dense,
        42_480_832,
        245_760,
        "855a7f5dc31e5c919644406adfd97f5c81261dd64525bad9e96110451b1a661b",
    )?;
    let conv = floats(&span(
        &mut dense,
        42_726_592,
        163_840,
        "86971b2b293c0f8d662169eac001156cb7d5d81bbb28fe75d78e358f1c39b476",
    )?);
    let dt = floats(&span(
        &mut dense,
        42_890_432,
        192,
        "61f670e1bfb6524a4ea2aa4bb8f9dbaeaba94c752f7485380ca626b336b8f0c6",
    )?);
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
            return Err("dense changed during read".into());
        }
    }
    let mut mixed_file = file(&args[2])?;
    if mixed_file.metadata()?.len() != 10240 {
        return Err("mixed length".into());
    }
    let mixed = floats(&span(
        &mut mixed_file,
        0,
        10240,
        "9df9ccdc3759a810606a4bc9594177cb4c9eff411c123cd5b196200128f5ce4f",
    )?);
    let mut qkv_file = file(&args[3])?;
    if qkv_file.metadata()?.len() != 40960 {
        return Err("QKV length".into());
    }
    let qkv = floats(&span(
        &mut qkv_file,
        0,
        40960,
        "2967a329365c17bb10ec76211c650d6dc1502bf273abf07c75466a1443c90f64",
    )?);
    let (gate, beta) = project_gates(&mixed, &alpha, &beta, &dt, &a)?;
    let identity = Identity {
        run_id: "r409-token42-local".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        epoch: 0,
    };
    let mut state = GdnState::new(identity.clone(), 16, 48, 128)?;
    let output = state.step(&identity, &qkv, &conv, 1e-6, &gate, &beta)?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    // Diagnostic layout: 6144 recurrence values, 48 gates, then 48 betas.
    for value in output.iter().chain(&gate).chain(&beta) {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("SC117 GDN diagnostic refused: {e}");
        std::process::exit(1);
    }
}
