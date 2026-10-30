//! Local token-42 attention write and FFN read, with pinned observed SC117 spans.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{hyperconnection_cpu, sha256};
fn open(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let f = o.open(path)?;
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
    if args.len() != 5 {
        return Err("usage: vf-sc117-router-input DENSE EMBEDDING MIXED GDN_OUTPUT".into());
    }
    let mut d = open(&args[1])?;
    let before = d.metadata()?;
    if before.len() != 1_538_035_200 {
        return Err("dense size".into());
    }
    let mut e = open(&args[2])?;
    if e.metadata()?.len() != 10240 {
        return Err("embedding size".into());
    }
    let embedding = floats(&read(
        &mut e,
        0,
        10240,
        "ae94a5f1a2e54b8f2e9719409640badfbba583e83ae82ad66987da58ee64ae31",
    )?);
    let mut mixed_file = open(&args[3])?;
    if mixed_file.metadata()?.len() != 10240 {
        return Err("mixed size".into());
    }
    let mixed = floats(&read(
        &mut mixed_file,
        0,
        10240,
        "9df9ccdc3759a810606a4bc9594177cb4c9eff411c123cd5b196200128f5ce4f",
    )?);
    let mut block_file = open(&args[4])?;
    if block_file.metadata()?.len() != 59392 {
        return Err("GDN output size".into());
    }
    let block = floats(&read(
        &mut block_file,
        0,
        59392,
        "beb3875a163c8cd56708a5c99d9de2f336373b62cbb4f2cad6eb3486c098f375",
    )?);
    let mut residual = Vec::with_capacity(10240);
    for _ in 0..4 {
        residual.extend_from_slice(&embedding);
    }
    let spans = [
        (
            22410240,
            40960,
            "a2be4ca661ebebcffddfa452dab947242d92a1f71f14a5392110be41e55398a3",
        ),
        (
            15774720,
            6553600,
            "0cf47e5bf283e8974c8d24bb3dfc1aef536b7d0daff9ac69f9ae8f214e6aed6a",
        ),
        (
            22451200,
            6553600,
            "6e87eb16e95e4e24cb83f3852d4200bdff7da87d8e8989022fe8012b88c2f978",
        ),
        (
            22328320,
            81920,
            "97548dcae39b501c2958bb607d3954b484fce8abe4ae758e8bb09e988d37abc9",
        ),
        (
            35640320,
            40960,
            "5287f71db818f0a54dea7e2509986b40de62d3837f0be7e57fd438e8cfbc64b2",
        ),
        (
            29004800,
            6553600,
            "9e1587dc09770060822aea6d122ad6be1653e5e8bbddc08f7f87f32e2b3e3785",
        ),
        (
            35681280,
            6553600,
            "83b4b781daafd109692f877649268ab9a3a4b503c343f3d3b981f099cfa5b992",
        ),
        (
            35558400,
            81920,
            "d6314c879c97c0dabd28fe94f7c96f4615c0c6d3acad9de62e60798e04776b4e",
        ),
    ];
    let mut p = Vec::new();
    for (off, len, hash) in spans {
        p.push(read(&mut d, off, len, hash)?);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = d.metadata()?;
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
            return Err("dense changed".into());
        }
    }
    let first = hyperconnection_cpu::read(
        &residual,
        &floats(&p[0]),
        &p[1],
        &p[2],
        &p[3],
        2560,
        4,
        320,
        1e-6,
    )?;
    if first
        .mixed
        .iter()
        .zip(&mixed)
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("mixed activation mismatch".into());
    }
    let post = hyperconnection_cpu::write(&residual, &block[12288..], &first.inject)?;
    let ffn = hyperconnection_cpu::read(
        &post,
        &floats(&p[4]),
        &p[5],
        &p[6],
        &p[7],
        2560,
        4,
        320,
        1e-6,
    )?;
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    // 2560 router-input floats, four FFN inject floats, then 10240 post-attention residual floats.
    for v in ffn.mixed.iter().chain(&ffn.inject).chain(&post) {
        output.write_all(&v.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("SC117 router input refused: {e}");
        std::process::exit(1);
    }
}
