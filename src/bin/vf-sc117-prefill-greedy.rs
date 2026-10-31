//! Bounded SC117 CPU generation qualification with pre-position expert headroom.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    gdn_cpu::Identity,
    moe::{Registry, Replica},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    sc117_ffn_stage, sc117_gdn_stage,
    sc117_layer_chain::{Layer, Owners, QsaQualificationTrace, Segment},
    sc117_post_attention, sc117_qsa_stage, sc117_router_cpu, sc117_shared_stage,
    session::{Endpoint, Session},
    sha256,
    strata_native_layout::NativeLayout,
};
type E = Box<dyn std::error::Error>;
const MAX_POSITIONS: usize = 128;
const TRACE_MAX_POSITIONS: usize = 96;
const TRACE_WORDS: usize = 37_380;
const TRACE_DOMAIN: &[u8] = b"vf-qsa-trace-v1-layer3-96-f32le\0";
const INPUT_TRACE_WORDS: usize = TRACE_WORDS + 10_240;
const INPUT_TRACE_DOMAIN: &[u8] = b"vf-qsa-trace-v2-layer3-input-96-f32le\0";
#[derive(Clone, Copy, PartialEq, Eq)]
enum TraceMode {
    Off,
    Attention,
    WithInput,
}
impl TraceMode {
    fn words(self) -> usize {
        if self == Self::WithInput {
            INPUT_TRACE_WORDS
        } else {
            TRACE_WORDS
        }
    }
}
const INDEX_HASH: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";

struct Manifests {
    g: &'static str,
    p: &'static str,
    s: &'static str,
}
fn manifests(layer: u32) -> Manifests {
    match layer {
        0 => Manifests {
            g: include_str!("../../fixtures/r438/layer0-gdn.json"),
            p: include_str!("../../fixtures/r438/layer0-post.json"),
            s: include_str!("../../fixtures/r438/layer0-shared.json"),
        },
        1 => Manifests {
            g: include_str!("../../fixtures/r438/layer1-gdn.json"),
            p: include_str!("../../fixtures/r438/layer1-post.json"),
            s: include_str!("../../fixtures/r438/layer1-shared.json"),
        },
        2 => Manifests {
            g: include_str!("../../fixtures/r438/layer2-gdn.json"),
            p: include_str!("../../fixtures/r438/layer2-post.json"),
            s: include_str!("../../fixtures/r438/layer2-shared.json"),
        },
        4 => Manifests {
            g: include_str!("../../fixtures/r438/layer4-gdn.json"),
            p: include_str!("../../fixtures/r438/layer4-post.json"),
            s: include_str!("../../fixtures/r438/layer4-shared.json"),
        },
        5 => Manifests {
            g: include_str!("../../fixtures/r438/layer5-gdn.json"),
            p: include_str!("../../fixtures/r438/layer5-post.json"),
            s: include_str!("../../fixtures/r438/layer5-shared.json"),
        },
        6 => Manifests {
            g: include_str!("../../fixtures/r438/layer6-gdn.json"),
            p: include_str!("../../fixtures/r438/layer6-post.json"),
            s: include_str!("../../fixtures/r438/layer6-shared.json"),
        },
        8 => Manifests {
            g: include_str!("../../fixtures/r438/layer8-gdn.json"),
            p: include_str!("../../fixtures/r438/layer8-post.json"),
            s: include_str!("../../fixtures/r438/layer8-shared.json"),
        },
        9 => Manifests {
            g: include_str!("../../fixtures/r438/layer9-gdn.json"),
            p: include_str!("../../fixtures/r438/layer9-post.json"),
            s: include_str!("../../fixtures/r438/layer9-shared.json"),
        },
        10 => Manifests {
            g: include_str!("../../fixtures/r438/layer10-gdn.json"),
            p: include_str!("../../fixtures/r438/layer10-post.json"),
            s: include_str!("../../fixtures/r438/layer10-shared.json"),
        },
        12 => Manifests {
            g: include_str!("../../fixtures/r438/layer12-gdn.json"),
            p: include_str!("../../fixtures/r438/layer12-post.json"),
            s: include_str!("../../fixtures/r438/layer12-shared.json"),
        },
        13 => Manifests {
            g: include_str!("../../fixtures/r438/layer13-gdn.json"),
            p: include_str!("../../fixtures/r438/layer13-post.json"),
            s: include_str!("../../fixtures/r438/layer13-shared.json"),
        },
        14 => Manifests {
            g: include_str!("../../fixtures/r438/layer14-gdn.json"),
            p: include_str!("../../fixtures/r438/layer14-post.json"),
            s: include_str!("../../fixtures/r438/layer14-shared.json"),
        },
        16 => Manifests {
            g: include_str!("../../fixtures/r438/layer16-gdn.json"),
            p: include_str!("../../fixtures/r438/layer16-post.json"),
            s: include_str!("../../fixtures/r438/layer16-shared.json"),
        },
        17 => Manifests {
            g: include_str!("../../fixtures/r438/layer17-gdn.json"),
            p: include_str!("../../fixtures/r438/layer17-post.json"),
            s: include_str!("../../fixtures/r438/layer17-shared.json"),
        },
        18 => Manifests {
            g: include_str!("../../fixtures/r438/layer18-gdn.json"),
            p: include_str!("../../fixtures/r438/layer18-post.json"),
            s: include_str!("../../fixtures/r438/layer18-shared.json"),
        },
        20 => Manifests {
            g: include_str!("../../fixtures/r438/layer20-gdn.json"),
            p: include_str!("../../fixtures/r438/layer20-post.json"),
            s: include_str!("../../fixtures/r438/layer20-shared.json"),
        },
        21 => Manifests {
            g: include_str!("../../fixtures/r438/layer21-gdn.json"),
            p: include_str!("../../fixtures/r438/layer21-post.json"),
            s: include_str!("../../fixtures/r438/layer21-shared.json"),
        },
        22 => Manifests {
            g: include_str!("../../fixtures/r438/layer22-gdn.json"),
            p: include_str!("../../fixtures/r438/layer22-post.json"),
            s: include_str!("../../fixtures/r438/layer22-shared.json"),
        },
        24 => Manifests {
            g: include_str!("../../fixtures/r438/layer24-gdn.json"),
            p: include_str!("../../fixtures/r438/layer24-post.json"),
            s: include_str!("../../fixtures/r438/layer24-shared.json"),
        },
        25 => Manifests {
            g: include_str!("../../fixtures/r438/layer25-gdn.json"),
            p: include_str!("../../fixtures/r438/layer25-post.json"),
            s: include_str!("../../fixtures/r438/layer25-shared.json"),
        },
        26 => Manifests {
            g: include_str!("../../fixtures/r438/layer26-gdn.json"),
            p: include_str!("../../fixtures/r438/layer26-post.json"),
            s: include_str!("../../fixtures/r438/layer26-shared.json"),
        },
        28 => Manifests {
            g: include_str!("../../fixtures/r438/layer28-gdn.json"),
            p: include_str!("../../fixtures/r438/layer28-post.json"),
            s: include_str!("../../fixtures/r438/layer28-shared.json"),
        },
        29 => Manifests {
            g: include_str!("../../fixtures/r438/layer29-gdn.json"),
            p: include_str!("../../fixtures/r438/layer29-post.json"),
            s: include_str!("../../fixtures/r438/layer29-shared.json"),
        },
        30 => Manifests {
            g: include_str!("../../fixtures/r438/layer30-gdn.json"),
            p: include_str!("../../fixtures/r438/layer30-post.json"),
            s: include_str!("../../fixtures/r438/layer30-shared.json"),
        },
        32 => Manifests {
            g: include_str!("../../fixtures/r438/layer32-gdn.json"),
            p: include_str!("../../fixtures/r438/layer32-post.json"),
            s: include_str!("../../fixtures/r438/layer32-shared.json"),
        },
        33 => Manifests {
            g: include_str!("../../fixtures/r438/layer33-gdn.json"),
            p: include_str!("../../fixtures/r438/layer33-post.json"),
            s: include_str!("../../fixtures/r438/layer33-shared.json"),
        },
        34 => Manifests {
            g: include_str!("../../fixtures/r438/layer34-gdn.json"),
            p: include_str!("../../fixtures/r438/layer34-post.json"),
            s: include_str!("../../fixtures/r438/layer34-shared.json"),
        },
        36 => Manifests {
            g: include_str!("../../fixtures/r438/layer36-gdn.json"),
            p: include_str!("../../fixtures/r438/layer36-post.json"),
            s: include_str!("../../fixtures/r438/layer36-shared.json"),
        },
        37 => Manifests {
            g: include_str!("../../fixtures/r438/layer37-gdn.json"),
            p: include_str!("../../fixtures/r438/layer37-post.json"),
            s: include_str!("../../fixtures/r438/layer37-shared.json"),
        },
        38 => Manifests {
            g: include_str!("../../fixtures/r438/layer38-gdn.json"),
            p: include_str!("../../fixtures/r438/layer38-post.json"),
            s: include_str!("../../fixtures/r438/layer38-shared.json"),
        },
        40 => Manifests {
            g: include_str!("../../fixtures/r438/layer40-gdn.json"),
            p: include_str!("../../fixtures/r438/layer40-post.json"),
            s: include_str!("../../fixtures/r438/layer40-shared.json"),
        },
        41 => Manifests {
            g: include_str!("../../fixtures/r438/layer41-gdn.json"),
            p: include_str!("../../fixtures/r438/layer41-post.json"),
            s: include_str!("../../fixtures/r438/layer41-shared.json"),
        },
        42 => Manifests {
            g: include_str!("../../fixtures/r438/layer42-gdn.json"),
            p: include_str!("../../fixtures/r438/layer42-post.json"),
            s: include_str!("../../fixtures/r438/layer42-shared.json"),
        },
        44 => Manifests {
            g: include_str!("../../fixtures/r438/layer44-gdn.json"),
            p: include_str!("../../fixtures/r438/layer44-post.json"),
            s: include_str!("../../fixtures/r438/layer44-shared.json"),
        },
        45 => Manifests {
            g: include_str!("../../fixtures/r438/layer45-gdn.json"),
            p: include_str!("../../fixtures/r438/layer45-post.json"),
            s: include_str!("../../fixtures/r438/layer45-shared.json"),
        },
        46 => Manifests {
            g: include_str!("../../fixtures/r438/layer46-gdn.json"),
            p: include_str!("../../fixtures/r438/layer46-post.json"),
            s: include_str!("../../fixtures/r438/layer46-shared.json"),
        },
        _ => unreachable!(),
    }
}

fn qsa_manifests(layer: u32) -> (&'static str, &'static str, &'static str) {
    match layer {
        3 => (
            include_str!("../../fixtures/r430/layer3.json"),
            include_str!("../../fixtures/r431/layer3.json"),
            include_str!("../../fixtures/r431/shared3.json"),
        ),
        7 => (
            include_str!("../../fixtures/r437/layer7.json"),
            include_str!("../../fixtures/r438/layer7-ffn.json"),
            include_str!("../../fixtures/r438/layer7-shared.json"),
        ),
        11 => (
            include_str!("../../fixtures/r437/layer11.json"),
            include_str!("../../fixtures/r438/layer11-ffn.json"),
            include_str!("../../fixtures/r438/layer11-shared.json"),
        ),
        15 => (
            include_str!("../../fixtures/r437/layer15.json"),
            include_str!("../../fixtures/r438/layer15-ffn.json"),
            include_str!("../../fixtures/r438/layer15-shared.json"),
        ),
        19 => (
            include_str!("../../fixtures/r437/layer19.json"),
            include_str!("../../fixtures/r438/layer19-ffn.json"),
            include_str!("../../fixtures/r438/layer19-shared.json"),
        ),
        23 => (
            include_str!("../../fixtures/r437/layer23.json"),
            include_str!("../../fixtures/r438/layer23-ffn.json"),
            include_str!("../../fixtures/r438/layer23-shared.json"),
        ),
        27 => (
            include_str!("../../fixtures/r437/layer27.json"),
            include_str!("../../fixtures/r438/layer27-ffn.json"),
            include_str!("../../fixtures/r438/layer27-shared.json"),
        ),
        31 => (
            include_str!("../../fixtures/r437/layer31.json"),
            include_str!("../../fixtures/r438/layer31-ffn.json"),
            include_str!("../../fixtures/r438/layer31-shared.json"),
        ),
        35 => (
            include_str!("../../fixtures/r437/layer35.json"),
            include_str!("../../fixtures/r438/layer35-ffn.json"),
            include_str!("../../fixtures/r438/layer35-shared.json"),
        ),
        39 => (
            include_str!("../../fixtures/r437/layer39.json"),
            include_str!("../../fixtures/r438/layer39-ffn.json"),
            include_str!("../../fixtures/r438/layer39-shared.json"),
        ),
        43 => (
            include_str!("../../fixtures/r437/layer43.json"),
            include_str!("../../fixtures/r438/layer43-ffn.json"),
            include_str!("../../fixtures/r438/layer43-shared.json"),
        ),
        47 => (
            include_str!("../../fixtures/r437/layer47.json"),
            include_str!("../../fixtures/r438/layer47-ffn.json"),
            include_str!("../../fixtures/r438/layer47-shared.json"),
        ),
        _ => unreachable!(),
    }
}

fn load(
    layer: u32,
    id: &Identity,
    dense: &mut std::fs::File,
    shard: &mut std::fs::File,
) -> Result<
    (
        sc117_gdn_stage::Stage,
        sc117_post_attention::Stage,
        sc117_shared_stage::Stage,
    ),
    E,
> {
    let m = manifests(layer);
    Ok((
        sc117_gdn_stage::Stage::load(serde_json::from_str(m.g)?, id.clone(), dense, shard)?,
        sc117_post_attention::Stage::load(serde_json::from_str(m.p)?, id.clone(), dense, shard)?,
        sc117_shared_stage::Stage::load(serde_json::from_str(m.s)?, id.clone(), shard)?,
    ))
}
fn bytes_hash(values: &[f32]) -> String {
    let b: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    sha256::hex(&b)
}
struct Seeds {
    ids: Vec<u32>,
    hashes: Vec<String>,
}

// Remote execution remains subject to external resource admission. Construct
// a fixed SSH invocation; never interpolate arbitrary text into a shell command.
fn rank_endpoint(
    rank: u32,
    local: &str,
    host: Option<&str>,
    remote: Option<&str>,
    digest: Option<&str>,
) -> Result<Endpoint, E> {
    if !(2..=3).contains(&rank) || !std::path::Path::new(local).is_absolute() {
        return Err("invalid worker endpoint".into());
    }
    let argv = match (host, remote, digest) {
        (None, None, Some(digest)) if sha256::is_hex_digest(digest) => vec![
            local.into(),
            "--guarded-local".into(),
            "--expected-exe-sha256".into(),
            digest.into(),
        ],
        (Some(host), Some(remote), Some(digest)) => {
            let valid_host = !host.is_empty()
                && host.len() <= 253
                && host.as_bytes()[0].is_ascii_alphanumeric()
                && host
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c));
            let valid_path = remote.starts_with('/')
                && remote.len() <= 4096
                && remote
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c));
            if !valid_host || !valid_path || remote == "/" || !sha256::is_hex_digest(digest) {
                return Err("invalid SSH worker host or path".into());
            }
            vec![
                "/usr/bin/ssh".into(),
                "-T".into(),
                "-o".into(),
                "BatchMode=yes".into(),
                "-o".into(),
                "StrictHostKeyChecking=yes".into(),
                "-o".into(),
                "ConnectTimeout=10".into(),
                host.into(),
                remote.into(),
                "--guarded-remote".into(),
                "--expected-exe-sha256".into(),
                digest.into(),
            ]
        }
        _ => return Err("SSH worker host, path and digest must be supplied together".into()),
    };
    Ok(Endpoint { rank, argv })
}

// Make the key file an immutable snapshot. The plan names the caller's
// canonical path and digest; SSH reads only the sealed descriptor at runtime.
#[cfg(target_os = "linux")]
fn pin_known_hosts(
    endpoint: &mut Endpoint,
    host: &str,
    path: &str,
    expected: &str,
) -> Result<std::fs::File, E> {
    use std::os::unix::fs::MetadataExt;
    if !path.starts_with('/')
        || path.len() > 4096
        || path == "/"
        || !path
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c))
        || !sha256::is_hex_digest(expected)
        || endpoint.argv.first().map(String::as_str) != Some("/usr/bin/ssh")
    {
        return Err("invalid SSH known-hosts input".into());
    }
    let input = std::fs::File::open(path)?;
    let before = input.metadata()?;
    if !before.is_file() || before.len() == 0 || before.len() > 4096 {
        return Err("invalid SSH known-hosts size".into());
    }
    let mut bytes = Vec::new();
    input.take(4097).read_to_end(&mut bytes)?;
    let after = std::fs::metadata(path)?;
    let identity = |m: &std::fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    if identity(&before) != identity(&after)
        || bytes.len() as u64 != before.len()
        || sha256::hex(&bytes) != expected
    {
        return Err("SSH known-hosts changed or digest mismatch".into());
    }
    let line = std::str::from_utf8(&bytes)?
        .strip_suffix('\n')
        .unwrap_or(std::str::from_utf8(&bytes)?);
    let parts: Vec<_> = line.split(' ').collect();
    if parts.len() != 3
        || parts[0] != host
        || parts[1] != "ssh-ed25519"
        || parts[2].is_empty()
        || !parts[2]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+/=".contains(&b))
    {
        return Err("SSH known-hosts must contain exactly one matching ed25519 key".into());
    }
    let fd = rustix::fs::memfd_create(
        "vf-known-hosts",
        rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC,
    )?;
    let mut sealed = std::fs::File::from(fd);
    sealed.write_all(&bytes)?;
    let seals = rustix::fs::SealFlags::WRITE
        | rustix::fs::SealFlags::GROW
        | rustix::fs::SealFlags::SHRINK
        | rustix::fs::SealFlags::SEAL;
    rustix::fs::fcntl_add_seals(&sealed, seals)?;
    if !rustix::fs::fcntl_get_seals(&sealed)?.contains(seals) {
        return Err("cannot seal SSH known-hosts".into());
    }
    let options = [
        format!("UserKnownHostsFile={path}"),
        "GlobalKnownHostsFile=/dev/null".into(),
        format!("HostKeyAlias={host}"),
        "VerifyHostKeyDNS=no".into(),
        "UpdateHostKeys=no".into(),
        "KnownHostsCommand=none".into(),
        "CheckHostIP=no".into(),
        "ProxyCommand=none".into(),
        "ProxyJump=none".into(),
        "PermitLocalCommand=no".into(),
        "ClearAllForwardings=yes".into(),
    ];
    let mut at = 1;
    for option in options {
        endpoint.argv.insert(at, "-o".into());
        endpoint.argv.insert(at + 1, option);
        at += 2;
    }
    Ok(sealed)
}
#[cfg(not(target_os = "linux"))]
fn pin_known_hosts(_: &mut Endpoint, _: &str, _: &str, _: &str) -> Result<std::fs::File, E> {
    Err("sealed SSH host key snapshot requires Linux".into())
}

/// Cooperative lease identity. The operator must pin the actual shared lease;
/// this check cannot establish that other services cooperate with that lock.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedLease {
    path: String,
    dev: u64,
    ino: u64,
    uid: u32,
}
#[cfg(target_os = "linux")]
fn lease_path_matches(spec: &SharedLease) -> Result<(), E> {
    use std::os::unix::fs::MetadataExt;
    if !std::path::Path::new(&spec.path).is_absolute()
        || spec.path.len() > 4096
        || spec.uid != rustix::process::geteuid().as_raw()
    {
        return Err("invalid shared lease identity".into());
    }
    let m = std::fs::symlink_metadata(&spec.path)?;
    if !m.is_file() || (m.dev(), m.ino(), m.uid()) != (spec.dev, spec.ino, spec.uid) {
        return Err("shared lease path changed".into());
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn verify_shared_lease(spec: &SharedLease, fd: i32) -> Result<(), E> {
    use std::os::unix::fs::MetadataExt;
    if fd < 3 {
        return Err("invalid shared lease descriptor".into());
    }
    lease_path_matches(spec)?;
    let m = std::fs::metadata(format!("/proc/self/fd/{fd}"))?;
    if !m.is_file() || (m.dev(), m.ino(), m.uid()) != (spec.dev, spec.ino, spec.uid) {
        return Err("shared lease descriptor changed".into());
    }
    let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}"))?;
    let locks: Vec<Vec<_>> = info
        .lines()
        .filter(|line| line.starts_with("lock:"))
        .map(|line| line.split_whitespace().collect())
        .collect();
    if locks.len() != 1
        || locks[0].len() != 9
        || locks[0][2..5] != ["FLOCK", "ADVISORY", "WRITE"]
        || locks[0][7..9] != ["0", "EOF"]
    {
        return Err("exclusive shared lease flock required".into());
    }
    let mut parts = locks[0][6].split(':');
    let major = u32::from_str_radix(parts.next().ok_or("lease lock device")?, 16)?;
    let minor = u32::from_str_radix(parts.next().ok_or("lease lock device")?, 16)?;
    let ino = parts.next().ok_or("lease lock inode")?.parse::<u64>()?;
    if parts.next().is_some()
        || major != rustix::fs::major(spec.dev)
        || minor != rustix::fs::minor(spec.dev)
        || ino != spec.ino
    {
        return Err("shared lease lock identity mismatch".into());
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn lease_path_matches(_: &SharedLease) -> Result<(), E> {
    Err("shared lease requires Linux".into())
}
#[cfg(not(target_os = "linux"))]
fn verify_shared_lease(_: &SharedLease, _: i32) -> Result<(), E> {
    Err("shared lease requires Linux".into())
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Namespace {
    root: String,
    dev: u64,
    ino: u64,
    uid: u32,
    base_id: String,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NamespaceContext {
    plan_id: String,
    shared_lease: SharedLease,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NamespaceRecord {
    status: String,
    scope: String,
    base_id: String,
    effective_run_id: String,
    reserved_ns: u64,
    context: NamespaceContext,
}
#[cfg(target_os = "linux")]
fn open_namespace(spec: &Namespace) -> Result<std::fs::File, E> {
    use std::os::unix::fs::MetadataExt;
    let base = spec
        .base_id
        .strip_prefix("r53-")
        .ok_or("invalid R53 namespace")?;
    if !(8..=60).contains(&base.len())
        || !base
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        || !std::path::Path::new(&spec.root).is_absolute()
        || spec.root.len() > 4096
        || spec.uid != rustix::process::geteuid().as_raw()
    {
        return Err("invalid R53 identity".into());
    }
    let fd = rustix::fs::open(
        &spec.root,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    let directory = std::fs::File::from(fd);
    let m = directory.metadata()?;
    if !m.is_dir()
        || (m.dev(), m.ino(), m.uid()) != (spec.dev, spec.ino, spec.uid)
        || m.mode() & 0o077 != 0
    {
        return Err("R53 directory identity or permissions changed".into());
    }
    Ok(directory)
}
#[cfg(target_os = "linux")]
fn consume_namespace(spec: &Namespace, plan: &str, lease: &SharedLease) -> Result<(), E> {
    use std::os::unix::fs::MetadataExt;
    let directory = open_namespace(spec)?;
    for effective in [
        spec.base_id.clone(),
        format!("{}-duplicate-slot", spec.base_id),
    ] {
        let fd = rustix::fs::openat(
            &directory,
            format!("{effective}.json"),
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        let file = std::fs::File::from(fd);
        let before = file.metadata()?;
        if !before.is_file()
            || before.uid() != spec.uid
            || before.mode() & 0o077 != 0
            || before.nlink() != 1
            || before.len() == 0
            || before.len() > 65536
        {
            return Err("invalid R53 reservation file".into());
        }
        let mut bytes = Vec::new();
        (&file).take(65537).read_to_end(&mut bytes)?;
        let after = file.metadata()?;
        let identity = |m: &std::fs::Metadata| {
            (
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        };
        if identity(&before) != identity(&after) || bytes.len() as u64 != before.len() {
            return Err("R53 reservation changed".into());
        }
        let record: NamespaceRecord = serde_json::from_slice(&bytes)?;
        if record.status != "RESERVED_UNKNOWN_NO_REPLAY"
            || record.scope
                != "Namespace consumption only; see linked execution receipt for actual outcome"
            || record.base_id != spec.base_id
            || record.effective_run_id != effective
            || record.reserved_ns == 0
            || record.context.plan_id != plan
            || record.context.shared_lease != *lease
        {
            return Err("R53 reservation identity mismatch".into());
        }
    }
    // This single-use marker is consumed even if the write, sync, or subsequent
    // launch fails. Never remove it or retry an UNKNOWN operation.
    let name = format!("{}-sc117-started.json", spec.base_id);
    let fd = rustix::fs::openat(
        &directory,
        name.as_str(),
        rustix::fs::OFlags::WRONLY
            | rustix::fs::OFlags::CREATE
            | rustix::fs::OFlags::EXCL
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::from_bits_truncate(0o600),
    )?;
    let mut file = std::fs::File::from(fd);
    let value = serde_json::json!({"status":"RESERVED_UNKNOWN_NO_REPLAY","base_id":spec.base_id,"plan_id":plan});
    serde_json::to_writer(&mut file, &value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    directory.sync_all()?;
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn open_namespace(_: &Namespace) -> Result<std::fs::File, E> {
    Err("R53 namespace requires Linux".into())
}
#[cfg(not(target_os = "linux"))]
fn consume_namespace(_: &Namespace, _: &str, _: &SharedLease) -> Result<(), E> {
    Err("R53 namespace requires Linux".into())
}

fn optional_env(name: &str) -> Result<Option<String>, E> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeMath {
    Vector,
    Scalar,
}
impl NativeMath {
    fn requires_expected(self, remote: bool) -> bool {
        remote || self == Self::Scalar
    }
    fn validate_transport(self, remote: bool) -> Result<(), E> {
        if remote && self == Self::Scalar {
            return Err("scalar CPU backend is not qualified over SSH".into());
        }
        Ok(())
    }
    fn label(self) -> &'static str {
        match self {
            Self::Vector => "vector-ggml-v1",
            Self::Scalar => "scalar-reference-ggml-v1",
        }
    }
}
fn parse_native_math(value: Option<&str>) -> Result<NativeMath, E> {
    match value {
        None | Some("vector") => Ok(NativeMath::Vector),
        Some("scalar") if cfg!(feature = "native-scalar-reference") => Ok(NativeMath::Scalar),
        _ => Err("unsupported or unavailable native CPU backend".into()),
    }
}
fn backend_plan(base: &str, backend: NativeMath) -> String {
    match backend {
        NativeMath::Vector => base.to_owned(),
        NativeMath::Scalar => {
            let mut bytes = b"vf-sc117-scalar-reference-backend-v1\0".to_vec();
            bytes.extend_from_slice(base.as_bytes());
            format!("r600-{}", sha256::hex(&bytes))
        }
    }
}

// An external controller supplies the admitted plan. This comparison is not
// itself a lease, attestation, or resource reservation.
fn verify_expected_plan(plan: &str, expected: Option<&str>, remote: bool) -> Result<(), E> {
    match expected {
        Some(value) if value == plan => Ok(()),
        None if !remote => Ok(()),
        _ => Err("expected worker plan missing or mismatched".into()),
    }
}

// Remote producer arithmetic is not qualified as interchangeable. Keep each
// previously admitted (layer, expert) on its original rank after eviction.
// This guard does not establish equivalence between different producers.
fn require_remote_affinity(remote: bool, sticky: bool) -> Result<(), E> {
    if remote && !sticky {
        return Err("SSH workers require sticky expert rank affinity".into());
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct WorkerPlan<'a> {
    version: &'static str,
    wire_policy: &'static str,
    eviction_policy: &'static str,
    origin_policy: &'static str,
    model_revision: &'static str,
    index_sha256: &'static str,
    ple_manifest_sha256: &'a str,
    epoch: u64,
    layers: usize,
    prompt_length: usize,
    prompt_tokens_sha256: &'a str,
    max_new_tokens: usize,
    eos_token_id: u32,
    prompt_mode: &'static str,
    payload_budget: u64,
    rank_limits: &'a Option<BTreeMap<u32, u64>>,
    endpoints: &'a BTreeMap<u32, Endpoint>,
    known_hosts_sha256: &'a BTreeMap<u32, String>,
    shared_lease: &'a Option<SharedLease>,
    r53_namespace: &'a Option<Namespace>,
}

fn prompt_tokens_hash(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    sha256::hex(&bytes)
}

fn admission_plan(base: &str) -> String {
    let mut bytes = b"vf-sc117-layer-admission-preflight-v1\0".to_vec();
    bytes.extend_from_slice(base.as_bytes());
    format!("r579-{}", sha256::hex(&bytes))
}

fn trace_plan(base: &str, mode: TraceMode) -> String {
    let (prefix, domain) = match mode {
        TraceMode::Off => return base.to_owned(),
        TraceMode::Attention => ("r537-", TRACE_DOMAIN),
        TraceMode::WithInput => ("r544-", INPUT_TRACE_DOMAIN),
    };
    let mut bytes = domain.to_vec();
    bytes.extend_from_slice(base.as_bytes());
    format!("{prefix}{}", sha256::hex(&bytes))
}
fn batch_plan(base: &str, batched: bool) -> String {
    if !batched {
        return base.to_owned();
    }
    let mut bytes = b"vf-prompt-batch-two-local-cpu-v1\0".to_vec();
    bytes.extend_from_slice(base.as_bytes());
    format!("r570-{}", sha256::hex(&bytes))
}

// Domain-separated plan: identical identities must not silently change the
// rank chosen on readmission when a mixed-architecture fleet is used.
fn affinity_plan(base: &str, enabled: bool) -> String {
    if !enabled {
        return base.to_owned();
    }
    let mut bytes = b"vf-expert-rank-affinity-v1\0".to_vec();
    bytes.extend_from_slice(base.as_bytes());
    format!("r576-{}", sha256::hex(&bytes))
}

fn append_qsa_trace(
    bytes: &mut Vec<u8>,
    trace: QsaQualificationTrace,
    run_id: &str,
    plan_id: &str,
    position: usize,
    attempt: u64,
    mode: TraceMode,
) -> Result<(), E> {
    let stamp = &trace.stamp;
    if position >= TRACE_MAX_POSITIONS
        || mode == TraceMode::Off
        || bytes.len() != position * mode.words() * 4
        || stamp.run_id != run_id
        || stamp.model_revision != "sc117-local-observation"
        || stamp.plan_id != plan_id
        || stamp.layer != 3
        || stamp.batch != position as u64
        || stamp.epoch != 1
        || trace.attempt != attempt
    {
        return Err("QSA trace identity or position".into());
    }
    let o = trace.output;
    if mode == TraceMode::WithInput
        && (trace.input.len() != 10_240 || trace.input.iter().any(|v| !v.is_finite()))
    {
        return Err("QSA trace input shape or value".into());
    }
    let slices: [(&[f32], usize); 10] = [
        (&o.mixed, 2560),
        (&o.inject, 4),
        (&o.raw_query_gate, 12288),
        (&o.query, 6144),
        (&o.gate, 6144),
        (&o.raw_key, 512),
        (&o.key, 512),
        (&o.value, 512),
        (&o.gated_attention, 6144),
        (&o.attention, 2560),
    ];
    if slices
        .iter()
        .any(|(s, n)| s.len() != *n || s.iter().any(|v| !v.is_finite()))
    {
        return Err("QSA trace shape or value".into());
    }
    if mode == TraceMode::WithInput {
        for value in &trace.input {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for (slice, _) in slices {
        for value in slice {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn worker_plan_id(
    endpoints: &BTreeMap<u32, Endpoint>,
    payload_budget: u64,
    rank_limits: &Option<BTreeMap<u32, u64>>,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    prompt_mode: &'static str,
    ple_manifest_sha256: &str,
    known_hosts_sha256: &BTreeMap<u32, String>,
    shared_lease: &Option<SharedLease>,
    r53_namespace: &Option<Namespace>,
) -> Result<String, E> {
    let prompt_length = prompt_tokens.len();
    if !sha256::is_hex_digest(ple_manifest_sha256)
        || !["plain", "chat"].contains(&prompt_mode)
        || eos_token_id >= 248_320
        || prompt_tokens
            .iter()
            .any(|&t| t >= 248_320 || (prompt_mode == "plain" && t == eos_token_id))
        || payload_budget == 0
        || !(1..=MAX_POSITIONS).contains(&prompt_length)
        || !(1..=MAX_POSITIONS).contains(&max_new_tokens)
        || prompt_length + max_new_tokens - 1 > MAX_POSITIONS
        || endpoints.len() != 2
        || (shared_lease.is_some() == known_hosts_sha256.is_empty())
        || (r53_namespace.is_some() == known_hosts_sha256.is_empty())
        || known_hosts_sha256
            .iter()
            .any(|(rank, hash)| !endpoints.contains_key(rank) || !sha256::is_hex_digest(hash))
        || endpoints.iter().any(|(rank, endpoint)| {
            let remote = endpoint
                .argv
                .get(1)
                .is_none_or(|arg| arg != "--guarded-local");
            remote != known_hosts_sha256.contains_key(rank)
        })
        || [2, 3].iter().any(|rank| {
            endpoints
                .get(rank)
                .is_none_or(|v| v.rank != *rank || v.argv.is_empty())
        })
        || rank_limits.as_ref().is_some_and(|limits| {
            limits.len() != 2
                || [2, 3].iter().any(|rank| {
                    limits
                        .get(rank)
                        .is_none_or(|&n| n == 0 || n > payload_budget)
                })
        })
    {
        return Err("invalid worker plan".into());
    }
    let prompt_tokens_sha256 = prompt_tokens_hash(prompt_tokens);
    let plan = WorkerPlan {
        version: "vf-sc117-worker-plan-v16",
        wire_policy: "native-f32-raw-if-advertised-otherwise-json-v1",
        eviction_policy: "rank-lru-then-size-v2",
        origin_policy: "stable-source-digest-cache-v1",
        model_revision: "sc117-local-observation",
        index_sha256: INDEX_HASH,
        ple_manifest_sha256,
        epoch: 1,
        layers: 48,
        prompt_length,
        prompt_tokens_sha256: &prompt_tokens_sha256,
        max_new_tokens,
        eos_token_id,
        prompt_mode,
        payload_budget,
        rank_limits,
        endpoints,
        known_hosts_sha256,
        shared_lease,
        r53_namespace,
    };
    Ok(format!("r531-{}", sha256::hex(&serde_json::to_vec(&plan)?)))
}

fn owners(
    layer: u32,
    preview: &Seeds,
    endpoints: &BTreeMap<u32, Endpoint>,
    index: &str,
    pack: &mut std::fs::File,
    layout: &NativeLayout,
    backend: NativeMath,
) -> Result<Owners, E> {
    let mut sessions = Vec::new();
    let mut copies = Vec::new();
    let mut order = BTreeMap::new();
    for rank in 2..=3 {
        let endpoint = endpoints.get(&rank).ok_or("worker endpoint missing")?;
        let mut session = Session::connect(endpoint, Duration::from_secs(30))?;
        let mut locals = Vec::new();
        for (slot, &expert) in preview
            .ids
            .iter()
            .enumerate()
            .filter(|(i, _)| 2 + (*i as u32 % 2) == rank)
        {
            let local = (slot / 2) as u32;
            let blob = layout.read_verified_expert(
                pack,
                layer as usize,
                expert.into(),
                INDEX_HASH,
                &preview.hashes[slot],
            )?;
            let bind = Binding {
                model_revision: "sc117-local-observation".into(),
                rank,
                layer,
                expert,
                local,
                epoch: 1,
                index_sha256: INDEX_HASH.into(),
                blob_sha256: preview.hashes[slot].clone(),
            };
            match backend {
                NativeMath::Vector => session.load_native_ggml(
                    &bind,
                    index,
                    pack.metadata()?.len(),
                    &blob.bytes,
                    Duration::from_secs(30),
                )?,
                NativeMath::Scalar => {
                    #[cfg(feature = "native-scalar-reference")]
                    session.load_native_scalar_ggml(
                        &bind,
                        index,
                        pack.metadata()?.len(),
                        &blob.bytes,
                        Duration::from_secs(30),
                    )?;
                    #[cfg(not(feature = "native-scalar-reference"))]
                    return Err("scalar backend not compiled".into());
                }
            }
            copies.push(Replica {
                expert,
                rank,
                local,
            });
            locals.push(local);
        }
        session.seal_native_ggml(locals.len(), Duration::from_secs(30))?;
        order.insert(rank, locals);
        sessions.push(session);
    }
    Ok(Owners {
        sessions,
        registry: Registry { epoch: 1, copies },
        preferred: BTreeMap::new(),
        order,
    })
}

const BUDGET: u64 = 4 * 1024 * 1024 * 1024;
// Before consuming a remote namespace or starting any worker, mirror the
// first position's rank-headroom requirement. There is no committed position
// yet, so the eviction policy cannot make up for insufficient initial space.
fn initial_rank_headroom(
    caps: &BTreeMap<u32, u64>,
    seed_per_rank: u64,
    headroom: u64,
) -> Result<(), E> {
    let required = seed_per_rank
        .checked_add(headroom)
        .ok_or("rank budget overflow")?;
    if caps.is_empty()
        || caps.values().any(|&cap| cap < seed_per_rank)
        || !caps.values().any(|&cap| cap >= required)
    {
        return Err("rank budget cannot cover initial seeds and first headroom".into());
    }
    Ok(())
}
fn valid_close(
    receipt: &voltafabric::session::CloseReceipt,
    expected: u64,
    positions: u64,
) -> bool {
    !receipt.killed
        && receipt.exit_code == Some(0)
        && receipt.error.is_none()
        && receipt.bye_execs == Some(expected)
        && expected <= positions
}
fn run() -> Result<(), E> {
    let mut a: Vec<_> = std::env::args().collect();
    let plan_only = a.get(1).is_some_and(|arg| arg == "--plan-only");
    if plan_only {
        a.remove(1);
    }
    if !(11..=14).contains(&a.len()) {
        return Err(
            "usage: check DENSE SHARD PACK INDEX SESSION LEDGER PLE_SHARD PLE_MANIFEST LAYERS PROMPT_TOKEN_IDS [MAX_NEW_TOKENS [EOS_TOKEN_ID [chat]]]"
                .into(),
        );
    }
    let layer_count: usize = a[9].parse()?;
    if layer_count != 48 {
        return Err("full greedy qualification requires 48 layers".into());
    }
    let prompt_tokens: Vec<u32> = a[10].split(',').map(str::parse).collect::<Result<_, _>>()?;
    let max_new_tokens: usize = if a.len() >= 12 { a[11].parse()? } else { 2 };
    let eos_token_id: u32 = if a.len() >= 13 {
        a[12].parse()?
    } else {
        248_046
    };
    let prompt_mode = if a.len() == 14 {
        if a[13] != "chat" {
            return Err("invalid prompt mode".into());
        }
        "chat"
    } else {
        "plain"
    };
    if eos_token_id >= 248_320 {
        return Err("invalid EOS token".into());
    }
    if !(1..=MAX_POSITIONS).contains(&max_new_tokens) {
        return Err("invalid generation limit".into());
    }
    if !(1..=MAX_POSITIONS).contains(&prompt_tokens.len())
        || prompt_tokens.len() + max_new_tokens - 1 > MAX_POSITIONS
        || prompt_tokens
            .iter()
            .any(|&t| t >= 248_320 || (prompt_mode == "plain" && t == eos_token_id))
    {
        return Err("invalid prompt tokens".into());
    }
    let native_math = parse_native_math(optional_env("VF_SC117_NATIVE_BACKEND")?.as_deref())?;
    let trace_mode = match optional_env("VF_QSA_TRACE_LAYER")?.as_deref() {
        None => TraceMode::Off,
        Some("3") => TraceMode::Attention,
        Some("3:input") => TraceMode::WithInput,
        _ => return Err("unsupported QSA trace mode".into()),
    };
    let trace_qsa = trace_mode != TraceMode::Off;
    let batched_prompt = match optional_env("VF_PREFILL_GROUP_SIZE")?.as_deref() {
        None | Some("1") => false,
        Some("2") => true,
        _ => return Err("unsupported prefill group size".into()),
    };
    let rank_affinity = match optional_env("VF_EXPERT_RANK_AFFINITY")?.as_deref() {
        None | Some("off") => false,
        Some("sticky") => true,
        _ => return Err("unsupported expert rank affinity".into()),
    };
    if rank_affinity && trace_qsa {
        return Err("rank affinity does not support QSA trace".into());
    }
    if batched_prompt && trace_qsa {
        return Err("batched prefill does not support QSA trace".into());
    }
    if trace_qsa && prompt_tokens.len() + max_new_tokens - 1 > TRACE_MAX_POSITIONS {
        return Err("QSA trace position limit".into());
    }
    let mut endpoints = BTreeMap::new();
    let mut host_pins = BTreeMap::new();
    let mut host_digests = BTreeMap::new();
    for rank in 2..=3 {
        let host = optional_env(&format!("VF_RANK{rank}_SSH_HOST"))?;
        let remote = optional_env(&format!("VF_RANK{rank}_SSH_WORKER"))?;
        let remote_digest = optional_env(&format!("VF_RANK{rank}_SSH_WORKER_SHA256"))?;
        let local_digest = optional_env("VF_LOCAL_WORKER_SHA256")?;
        let digest = if host.is_none() && remote.is_none() && remote_digest.is_none() {
            local_digest.as_deref()
        } else {
            remote_digest.as_deref()
        };
        let mut endpoint = rank_endpoint(rank, &a[5], host.as_deref(), remote.as_deref(), digest)?;
        let path = optional_env(&format!("VF_RANK{rank}_SSH_KNOWN_HOSTS"))?;
        let host_hash = optional_env(&format!("VF_RANK{rank}_SSH_KNOWN_HOSTS_SHA256"))?;
        match (host.as_deref(), path.as_deref(), host_hash.as_deref()) {
            (Some(host), Some(path), Some(hash)) => {
                host_pins.insert(rank, pin_known_hosts(&mut endpoint, host, path, hash)?);
                host_digests.insert(rank, hash.to_owned());
            }
            (None, None, None) => (),
            _ => return Err("SSH known-hosts path and digest must be supplied together".into()),
        }
        endpoints.insert(rank, endpoint);
    }
    require_remote_affinity(!host_digests.is_empty(), rank_affinity)?;
    native_math.validate_transport(!host_digests.is_empty())?;
    if trace_qsa && !host_digests.is_empty() {
        return Err("QSA trace requires local CPU workers".into());
    }
    if batched_prompt && !host_digests.is_empty() {
        return Err("batched prefill requires local CPU workers".into());
    }
    let shared_lease: Option<SharedLease> = match (
        host_digests.is_empty(),
        optional_env("VF_SHARED_LEASE_SPEC")?,
    ) {
        (true, None) => None,
        (false, Some(json)) if json.len() <= 8192 => {
            let spec: SharedLease = serde_json::from_str(&json)?;
            lease_path_matches(&spec)?;
            Some(spec)
        }
        _ => return Err("explicit shared lease identity required for SSH".into()),
    };
    let r53_namespace: Option<Namespace> = match (
        host_digests.is_empty(),
        optional_env("VF_R53_NAMESPACE_SPEC")?,
    ) {
        (true, None) => None,
        (false, Some(json)) if json.len() <= 8192 => {
            let spec: Namespace = serde_json::from_str(&json)?;
            open_namespace(&spec)?;
            Some(spec)
        }
        _ => return Err("explicit R53 namespace identity required for SSH".into()),
    };
    let mut dense = std::fs::File::open(&a[1])?;
    let mut shard = std::fs::File::open(&a[2])?;
    let mut pack = std::fs::File::open(&a[3])?;
    let index = std::fs::read_to_string(&a[4])?;
    if sha256::hex(index.as_bytes()) != INDEX_HASH {
        return Err("index digest".into());
    }
    let layout = NativeLayout::parse_sc117_v3(&index, pack.metadata()?.len())?;
    let worst_case_headroom =
        layout.worst_case_admission_bytes(sc117_router_cpu::ROUTED_EXPERTS)?;
    let budget = match std::env::var("VF_CPU_EXPERT_BUDGET") {
        Ok(value) => value.parse::<u64>()?,
        Err(std::env::VarError::NotPresent) => BUDGET,
        Err(error) => return Err(error.into()),
    };
    let batch_headroom = if batched_prompt {
        layout.worst_case_admission_bytes(sc117_router_cpu::ROUTED_EXPERTS * 2)?
    } else {
        worst_case_headroom
    };
    let seed_bytes = layout.worst_case_admission_bytes(2)?;
    if budget > BUDGET
        || seed_bytes
            .checked_add(batch_headroom)
            .is_none_or(|minimum| budget < minimum)
    {
        return Err("expert budget cannot cover seeds and worst-case headroom".into());
    }
    let rank_limits = match (
        std::env::var("VF_RANK2_BUDGET"),
        std::env::var("VF_RANK3_BUDGET"),
    ) {
        (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => None,
        (Ok(a), Ok(b)) => {
            let limits = BTreeMap::from([(2, a.parse::<u64>()?), (3, b.parse::<u64>()?)]);
            let seed_per_rank = layout.worst_case_admission_bytes(1)?;
            if limits.values().any(|&v| v > budget) {
                return Err("rank budget exceeds global budget".into());
            }
            initial_rank_headroom(&limits, seed_per_rank, batch_headroom)?;
            Some(limits)
        }
        _ => return Err("both rank budgets must be supplied".into()),
    };
    // Read once: the plan and weight loader must refer to the same bounded bytes.
    let manifest_file = std::fs::File::open(&a[8])?;
    if !manifest_file.metadata()?.is_file() {
        return Err("PLE manifest not regular".into());
    }
    let mut manifest_bytes = Vec::new();
    manifest_file
        .take(65_537)
        .read_to_end(&mut manifest_bytes)?;
    if manifest_bytes.is_empty() || manifest_bytes.len() > 65_536 {
        return Err("PLE manifest size".into());
    }
    let manifest: voltafabric::sc117_ple_weights::Manifest =
        serde_json::from_slice(&manifest_bytes)?;
    let manifest_hash = sha256::hex(&manifest_bytes);
    let base_plan_id = worker_plan_id(
        &endpoints,
        budget,
        &rank_limits,
        &prompt_tokens,
        max_new_tokens,
        eos_token_id,
        prompt_mode,
        &manifest_hash,
        &host_digests,
        &shared_lease,
        &r53_namespace,
    )?;
    let plan_id = backend_plan(
        &affinity_plan(
            &batch_plan(
                &trace_plan(&admission_plan(&base_plan_id), trace_mode),
                batched_prompt,
            ),
            rank_affinity,
        ),
        native_math,
    );
    if plan_only {
        println!(
            "{}",
            serde_json::json!({"plan_id":plan_id, "native_math_backend":native_math.label(), "prompt_tokens_sha256":prompt_tokens_hash(&prompt_tokens)})
        );
        return Ok(());
    }
    let expected = optional_env("VF_EXPECTED_PLAN_ID")?;
    verify_expected_plan(
        &plan_id,
        expected.as_deref(),
        native_math.requires_expected(
            endpoints
                .values()
                .any(|v| v.argv.get(1).is_none_or(|arg| arg != "--guarded-local")),
        ),
    )?;
    let lease_fd = if let Some(spec) = &shared_lease {
        let fd = optional_env("VF_SHARED_LEASE_FD")?
            .ok_or("shared lease fd required")?
            .parse::<i32>()?;
        verify_shared_lease(spec, fd)?;
        Some(fd)
    } else {
        None
    };
    if let (Some(namespace), Some(lease)) = (&r53_namespace, &shared_lease) {
        consume_namespace(namespace, &plan_id, lease)?;
    }
    // The sealed memfd stays alive for all SSH children. Keep the canonical
    // path in the plan; only replace the runtime option with our immutable fd.
    #[cfg(target_os = "linux")]
    for (rank, file) in &host_pins {
        use std::os::fd::AsRawFd;
        let argv = &mut endpoints.get_mut(rank).ok_or("missing SSH endpoint")?.argv;
        let option = argv
            .iter_mut()
            .find(|arg| arg.starts_with("UserKnownHostsFile="))
            .ok_or("missing SSH known-hosts option")?;
        *option = format!(
            "UserKnownHostsFile=/proc/{}/fd/{}",
            std::process::id(),
            file.as_raw_fd()
        );
    }
    let weights = voltafabric::sc117_ple_weights::Weights::load(
        &mut shard,
        &manifest,
        "sc117-local-observation",
    )?;
    let ple =
        voltafabric::sc117_ple::Table::open(std::fs::File::open(&a[7])?, 28_800_138_432, 192)?;
    let token_table =
        voltafabric::sc117_embedding::Table::open(shard.try_clone()?, 55_063_448_064, 545_644_544)?;
    let attempt = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
    let run_id = format!("r494-prefill-greedy-{}-{attempt}", std::process::id());
    let head_id = Identity {
        run_id: run_id.clone(),
        model_revision: "sc117-local-observation".into(),
        layer: 48,
        epoch: 1,
    };
    let head_manifest: voltafabric::sc117_head_cpu::Manifest =
        serde_json::from_str(include_str!("../../fixtures/r429/head.json"))?;
    let mut head = voltafabric::sc117_head_cpu::Head::load(
        head_manifest,
        head_id.clone(),
        &mut dense,
        &mut shard,
    )?;
    let mut layers = Vec::new();
    let mut used = 0u64;
    let mut origin = voltafabric::sc117_expert_origin::Origin::open(shard.try_clone()?, &layout)?;
    let mut seeds = Vec::new();
    let mut seed_records = Vec::new();
    for layer in 0..layer_count {
        let ids = vec![0, 1];
        let hashes = ids
            .iter()
            .map(|&expert| origin.expert_digest(&layout, layer, expert.into()))
            .collect::<voltafabric::Result<Vec<_>>>()?;
        let seed = Seeds { ids, hashes };
        seeds.push(seed.hashes.clone());
        for (slot, &expert) in seed.ids.iter().enumerate() {
            seed_records.push(voltafabric::sc117_expert_source::Seed {
                layer: layer as u32,
                expert,
                rank: 2 + slot as u32,
                local: 0,
            });
        }
        for &expert in &seed.ids {
            used = used
                .checked_add(layout.expert(layer, expert.into())?.bytes)
                .ok_or("seed budget overflow")?;
        }
        if used > budget {
            return Err("seed budget exceeded".into());
        }
        let id = Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            layer: layer as u32,
            epoch: 1,
        };
        if let (Some(spec), Some(fd)) = (&shared_lease, lease_fd) {
            verify_shared_lease(spec, fd)?;
        }
        let owners = owners(
            layer as u32,
            &seed,
            &endpoints,
            &index,
            &mut pack,
            &layout,
            native_math,
        )?;
        if layer % 4 == 3 {
            let qsa = sc117_qsa_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).0)?,
                id.clone(),
                &mut dense,
                &mut shard,
                prompt_tokens.len() + max_new_tokens - 1,
            )?;
            let ffn = sc117_ffn_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).1)?,
                id.clone(),
                &mut dense,
            )?;
            let shared = sc117_shared_stage::Stage::load(
                serde_json::from_str(qsa_manifests(id.layer).2)?,
                id.clone(),
                &mut shard,
            )?;
            layers.push(Layer::qsa(id, qsa, ffn, shared, owners)?);
        } else {
            let (g, post, shared) = load(layer as u32, &id, &mut dense, &mut shard)?;
            layers.push(Layer::gdn(id, g, post, shared, owners)?);
        }
    }
    // Replace layer 1 while retaining the layer order and owners.
    let mut iter = layers.into_iter();
    let first = iter.next().ok_or("missing layer 0")?;
    let second = iter
        .next()
        .ok_or("missing layer 1")?
        .attach_ple(ple, weights)?;
    let mut layers = vec![first, second];
    layers.extend(iter);
    let ledger = NativeBatchLedger::open(std::path::Path::new(&a[6]))?;
    let mut segment = Segment::new_ple_cpu_qualification(
        layers,
        plan_id.clone(),
        ledger,
        Duration::from_secs(30),
    )?;
    let source_config = voltafabric::sc117_expert_source::Config {
        identity: voltafabric::sc117_expert_source::Identity {
            run_id: run_id.clone(),
            model_revision: "sc117-local-observation".into(),
            plan_id: plan_id.clone(),
            epoch: 1,
        },
        ranks: vec![2, 3],
        seeds: &seed_records,
        limit: budget,
    };
    let mut source = if let Some(limits) = rank_limits.clone() {
        voltafabric::sc117_expert_source::Source::new_with_rank_limits(
            &mut pack,
            &layout,
            &index,
            origin,
            source_config,
            limits,
        )?
    } else {
        voltafabric::sc117_expert_source::Source::new(
            &mut pack,
            &layout,
            &index,
            origin,
            source_config,
        )?
    };
    if rank_affinity {
        source.enable_rank_affinity()?;
    }
    if source.used_bytes() != used {
        return Err("seed budget mismatch".into());
    }
    let mut input_tokens = Vec::new();
    let mut embedding_hashes = Vec::new();
    let mut output_hashes = Vec::new();
    let mut logits_hashes = Vec::new();
    let mut generated = Vec::new();
    let mut evictions = Vec::new();
    let mut rank_history = Vec::new();
    let mut trace_bytes = Vec::new();
    let mut current_token = prompt_tokens[0];
    let mut pos = 0;
    while pos < prompt_tokens.len() + max_new_tokens - 1 {
        if let (Some(spec), Some(fd)) = (&shared_lease, lease_fd) {
            verify_shared_lease(spec, fd)?;
        }
        if batched_prompt && pos + 1 < prompt_tokens.len() {
            let tokens = &prompt_tokens[pos..pos + 2];
            let mut batch_inputs = Vec::with_capacity(2);
            for &token in tokens {
                input_tokens.push(token);
                let embedding = token_table.gather(token)?;
                embedding_hashes.push(bytes_hash(&embedding));
                batch_inputs.push(
                    (0..4)
                        .flat_map(|_| embedding.iter().copied())
                        .collect::<Vec<f32>>(),
                );
            }
            let position_attempt = attempt.checked_add(pos as u64).ok_or("attempt overflow")?;
            let token_ids = [tokens[0] as i32, tokens[1] as i32];
            let (outputs, removed) = segment.advance_token_batch_with_headroom(
                position_attempt,
                &token_ids,
                &batch_inputs,
                &mut source,
                batch_headroom,
                768,
            )?;
            if outputs.len() != 2 || segment.position() != pos as u64 + 2 || segment.is_poisoned() {
                return Err("segment did not commit complete batch".into());
            }
            evictions.extend(removed);
            let rank_used = source.rank_used_bytes();
            let accounted = rank_used
                .values()
                .try_fold(0u64, |total, value| total.checked_add(*value));
            if accounted != Some(source.used_bytes())
                || rank_limits.as_ref().is_some_and(|limits| {
                    rank_used
                        .iter()
                        .any(|(rank, used)| limits.get(rank).is_none_or(|limit| used > limit))
                })
            {
                return Err("rank payload accounting mismatch".into());
            }
            for output in &outputs {
                output_hashes.push(bytes_hash(output));
                rank_history.push(rank_used.clone());
            }
            pos += 2;
            if pos == prompt_tokens.len() {
                let prediction = head.evaluate(&head_id, &outputs[1])?;
                logits_hashes.push(bytes_hash(&prediction.logits));
                generated.push(prediction.greedy);
                current_token = prediction.greedy;
                if current_token == eos_token_id {
                    break;
                }
            }
            continue;
        }
        if pos < prompt_tokens.len() {
            current_token = prompt_tokens[pos];
        }
        input_tokens.push(current_token);
        let embedding = token_table.gather(current_token)?;
        embedding_hashes.push(bytes_hash(&embedding));
        let input: Vec<f32> = (0..4).flat_map(|_| embedding.iter().copied()).collect();
        let position_attempt = attempt.checked_add(pos as u64).ok_or("attempt overflow")?;
        let (output, removed) = if trace_qsa {
            let (output, removed, trace) = segment.advance_token_with_headroom_qsa_trace(
                position_attempt,
                current_token as i32,
                &input,
                &mut source,
                worst_case_headroom,
                768,
                3,
            )?;
            append_qsa_trace(
                &mut trace_bytes,
                trace,
                &run_id,
                &plan_id,
                pos,
                position_attempt,
                trace_mode,
            )?;
            (output, removed)
        } else {
            segment.advance_token_with_headroom(
                position_attempt,
                current_token as i32,
                &input,
                &mut source,
                worst_case_headroom,
                768,
            )?
        };
        if let (Some(spec), Some(fd)) = (&shared_lease, lease_fd) {
            verify_shared_lease(spec, fd)?;
        }
        evictions.extend(removed);
        if segment.position() != pos as u64 + 1 || segment.is_poisoned() {
            return Err("segment did not commit complete position".into());
        }
        let rank_used = source.rank_used_bytes();
        let accounted = rank_used
            .values()
            .try_fold(0u64, |total, value| total.checked_add(*value));
        if accounted != Some(source.used_bytes())
            || rank_limits.as_ref().is_some_and(|limits| {
                rank_used
                    .iter()
                    .any(|(rank, used)| limits.get(rank).is_none_or(|limit| used > limit))
            })
        {
            return Err("rank payload accounting mismatch".into());
        }
        rank_history.push(rank_used.clone());
        output_hashes.push(bytes_hash(&output));
        if pos + 1 < prompt_tokens.len() {
            pos += 1;
            continue;
        }
        let prediction = head.evaluate(&head_id, &output)?;
        logits_hashes.push(bytes_hash(&prediction.logits));
        generated.push(prediction.greedy);
        current_token = prediction.greedy;
        if current_token == eos_token_id {
            break;
        }
        pos += 1;
    }
    let receipts = segment.close_with_counts();
    if receipts.len() != layer_count * 2
        || receipts.iter().any(|(r, expected, raw)| {
            *raw > *expected || !valid_close(r, *expected, input_tokens.len() as u64)
        })
    {
        return Err("worker shutdown receipt".into());
    }
    let (cache_hits, cache_misses, cache_entries) = source.origin_digest_cache_stats();
    let mut receipt = serde_json::json!({"scope":"up to 128 prompt tokens and 128 predictions within 128 input positions through 48 CPU layers", "worker_transports":endpoints.iter().map(|(rank,endpoint)| (*rank, if endpoint.argv.get(1).is_some_and(|arg| arg == "--guarded-local") { "local" } else { "ssh" })).collect::<BTreeMap<_,_>>(), "expert_headroom_bytes":worst_case_headroom, "evictions":evictions, "requested_max_new_tokens":max_new_tokens, "eos_token_id":eos_token_id, "prompt_mode":prompt_mode, "layer_count":layer_count,"run_id":run_id,"plan_id":plan_id,"attempt_base":attempt,"prompt_tokens_sha256":prompt_tokens_hash(&prompt_tokens),"prompt_tokens":prompt_tokens,"input_tokens":input_tokens,"generated_token_ids":generated,"logits_sha256":logits_hashes,"embedding_sha256":embedding_hashes,"output_sha256":output_hashes,"seed_experts":[0,1],"seed_hashes":seeds,"workers_closed":receipts.len(),"worker_exec_counts":receipts.iter().map(|(r, expected, _)| (r.rank, *expected, r.bye_execs)).collect::<Vec<_>>(),"worker_raw_exec_counts":receipts.iter().map(|(r, _, raw)| (r.rank, *raw)).collect::<Vec<_>>(),"payload_budget_bytes":budget,"resident_payload_bytes":source.used_bytes(),"rank_used_bytes":source.rank_used_bytes(),"rank_limits":rank_limits,"rank_affinity":rank_affinity,"rank_history":rank_history,"dynamic_admissions":source.admissions(),"source_digest_cache":{"hits":cache_hits,"misses":cache_misses,"entries":cache_entries}});
    receipt["native_math_backend"] = serde_json::json!(native_math.label());
    if trace_qsa {
        if trace_bytes.len() != input_tokens.len() * trace_mode.words() * 4 {
            return Err("QSA trace incomplete".into());
        }
        let version = if trace_mode == TraceMode::WithInput {
            "vf-qsa-trace-v2-layer3-input-f32le"
        } else {
            "vf-qsa-trace-v1-layer3-f32le"
        };
        receipt["qsa_trace"] = serde_json::json!({"version":version, "layer":3, "positions":input_tokens.len(), "words_per_position":trace_mode.words(), "sha256":sha256::hex(&trace_bytes), "base64":voltafabric::base64::encode(&trace_bytes)});
    }
    println!("{receipt}");
    drop(host_pins);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("prefill-greedy qualification refused: {e}");
        std::process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn remote_workers_require_sticky_expert_affinity() {
        assert!(super::require_remote_affinity(false, false).is_ok());
        assert!(super::require_remote_affinity(false, true).is_ok());
        assert!(super::require_remote_affinity(true, true).is_ok());
        assert!(super::require_remote_affinity(true, false).is_err());
    }
    #[cfg(feature = "native-scalar-reference")]
    #[test]
    fn scalar_seed_loading_uses_the_explicit_session_opcode() {
        let index = include_str!("../../fixtures/strata_sc117_native_experts_v3.txt");
        let layout = NativeLayout::parse_sc117_v3(index, 50_292_326_400).unwrap();
        let span = layout.expert(2, 0).unwrap();
        assert_eq!(span.bytes, 1_971_200);
        let seeds = Seeds {
            ids: vec![0, 1],
            hashes: vec![
                "94ed1ed77bb5bbdd3136799a00c0ed4301802ceb3553185f621d3bd59b223c8e".into();
                2
            ],
        };
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!(
            ".runs/vf-r600-work/sparse-pack-{}-{suffix}",
            std::process::id()
        );
        let mut pack = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        pack.set_len(50_292_326_400).unwrap();
        let python = r#"import sys,json,os
b=sys.stdin.buffer
h=json.loads(b.readline())
assert h['op']=='hello'
print(json.dumps({'reply':'hello','hello':{'protocol':1,'rank':h['rank'],'arch':'test','os':'test','endian':'little','build':'test','capabilities':['cpu_f64_dense_moe_v1','cpu_strata_sc117_ggml_fp16_v1','cpu_strata_sc117_scalar_reference_fp16_v1'],'max_message_bytes':16777216,'pid':os.getpid()}}),flush=True)
h=json.loads(b.readline()); assert h['op']=='load_native_scalar_ggml_raw'
n=h['artifact_bytes']
while n:
 x=b.read(min(n,65536)); assert x; n-=len(x)
print(json.dumps({'reply':'native_scalar_ggml_loaded','binding':h['binding']}),flush=True)
h=json.loads(b.readline()); assert h['op']=='seal_native_ggml'
print(json.dumps({'reply':'native_ggml_ready','experts':1}),flush=True)
h=json.loads(b.readline()); assert h['op']=='shutdown'
print(json.dumps({'reply':'bye','execs':0,'peak_rss_kib':None}),flush=True)
"#;
        let endpoints = [2, 3]
            .into_iter()
            .map(|rank| {
                (
                    rank,
                    Endpoint {
                        rank,
                        argv: vec!["python3".into(), "-c".into(), python.into()],
                    },
                )
            })
            .collect();
        let owners = owners(
            2,
            &seeds,
            &endpoints,
            index,
            &mut pack,
            &layout,
            NativeMath::Scalar,
        )
        .unwrap();
        assert_eq!(owners.registry.copies.len(), 2);
        for session in owners.sessions {
            let receipt = session.close(Duration::from_secs(10));
            assert_eq!(receipt.exit_code, Some(0));
            assert_eq!(receipt.bye_execs, Some(0));
        }
    }

    #[test]
    fn scalar_backend_has_a_distinct_plan_and_requires_local_expected_plan() {
        let base = format!("r576-{}", "a".repeat(64));
        assert_eq!(backend_plan(&base, NativeMath::Vector), base);
        let scalar = backend_plan(&base, NativeMath::Scalar);
        assert_eq!(
            scalar,
            "r600-484f0177d81fc89ce4ed802eccc9abf178bda45cda900667b6a49edbcd2a1295"
        );
        assert_ne!(scalar, base);
        assert_ne!(
            scalar,
            backend_plan(&format!("r576-{}", "b".repeat(64)), NativeMath::Scalar)
        );
        assert!(
            verify_expected_plan(&scalar, None, NativeMath::Scalar.requires_expected(false))
                .is_err()
        );
        assert!(verify_expected_plan(
            &scalar,
            Some(&scalar),
            NativeMath::Scalar.requires_expected(false)
        )
        .is_ok());
        assert!(NativeMath::Scalar.validate_transport(true).is_err());
        assert!(NativeMath::Scalar.validate_transport(false).is_ok());
        assert!(NativeMath::Vector.validate_transport(true).is_ok());
        assert!(parse_native_math(Some("unsupported")).is_err());
        assert_eq!(parse_native_math(None).unwrap(), NativeMath::Vector);
        assert_eq!(
            parse_native_math(Some("scalar")).is_ok(),
            cfg!(feature = "native-scalar-reference")
        );
    }

    #[test]
    fn admission_preflight_changes_plan_identity() {
        let base = format!("r531-{}", "a".repeat(64));
        assert_eq!(
            admission_plan(&base),
            "r579-d7ce1f1ccbdebf7b385d5223e8bafce849a044ef408da644339907582e01bc77"
        );
        assert_ne!(
            admission_plan(&base),
            admission_plan(&format!("r531-{}", "b".repeat(64)))
        );
    }

    #[test]
    fn sticky_plan_binds_policy_and_base() {
        let base = format!("r531-{}", "a".repeat(64));
        assert_eq!(affinity_plan(&base, false), base);
        let sticky = affinity_plan(&base, true);
        assert_eq!(
            sticky,
            "r576-14d8a594dfddda0d9fb3f4ec30d5c53b0c5b1672eb6f88903ebfd05d5354f557"
        );
        assert_ne!(
            sticky,
            affinity_plan(&format!("r531-{}", "b".repeat(64)), true)
        );
    }

    #[test]
    fn initial_rank_headroom_is_rejected_before_worker_start() {
        let caps = BTreeMap::from([(2, 20), (3, 18)]);
        assert!(initial_rank_headroom(&caps, 10, 10).is_ok());
        assert!(initial_rank_headroom(&caps, 10, 11).is_err());
        assert!(initial_rank_headroom(&BTreeMap::new(), 10, 1).is_err());
        assert!(initial_rank_headroom(&caps, u64::MAX, 1).is_err());
        // A rank that cannot even fit its own initial residents must fail,
        // even if another rank could hold the entire headroom.
        assert!(initial_rank_headroom(&BTreeMap::from([(2, 9), (3, 100)]), 10, 5).is_err());
    }

    #[test]
    fn sc117_initial_seed_bound_covers_every_layer_and_rank() {
        let index = include_str!("../../fixtures/strata_sc117_native_experts_v3.txt");
        let layout = NativeLayout::parse_sc117_v3(index, 50_292_326_400).unwrap();
        let bound = layout.worst_case_admission_bytes(1).unwrap();
        for expert in [0, 1] {
            let actual: u64 = (0..48)
                .map(|layer| layout.expert(layer, expert).unwrap().bytes)
                .sum();
            assert_eq!(actual, bound);
        }
        let reserve = layout
            .worst_case_admission_bytes(sc117_router_cpu::ROUTED_EXPERTS * 2)
            .unwrap();
        let caps = BTreeMap::from([(2, bound + reserve - 1), (3, bound + reserve - 1)]);
        assert!(initial_rank_headroom(&caps, bound, reserve).is_err());
        let caps = BTreeMap::from([(2, bound + reserve), (3, bound)]);
        assert!(initial_rank_headroom(&caps, bound, reserve).is_ok());
    }

    #[test]
    fn batched_prompt_plan_is_distinct_and_binds_the_base() {
        let base = format!("r531-{}", "a".repeat(64));
        assert_eq!(super::batch_plan(&base, false), base);
        let batched = super::batch_plan(&base, true);
        assert!(batched.starts_with("r570-") && batched.len() == 69);
        assert_ne!(
            batched,
            super::batch_plan(&format!("r531-{}", "b".repeat(64)), true)
        );
    }
    use super::*;

    #[test]
    fn trace_plan_and_records_are_bound_and_reject_partial_data() {
        let base = format!("r531-{}", "a".repeat(64));
        let plan = trace_plan(&base, TraceMode::Attention);
        let input_plan = trace_plan(&base, TraceMode::WithInput);
        assert_eq!(trace_plan(&base, TraceMode::Off), base);
        assert!(plan.starts_with("r537-"));
        assert!(input_plan.starts_with("r544-"));
        assert_ne!(plan, input_plan);
        assert_ne!(
            plan,
            trace_plan(&format!("r531-{}", "b".repeat(64)), TraceMode::Attention)
        );
        let trace = || QsaQualificationTrace {
            stamp: voltafabric::moe::Stamp {
                run_id: "run".into(),
                model_revision: "sc117-local-observation".into(),
                plan_id: plan.clone(),
                layer: 3,
                batch: 0,
                epoch: 1,
            },
            attempt: 17,
            input: vec![2.0; 10240],
            output: sc117_qsa_stage::Output {
                mixed: vec![1.0; 2560],
                inject: vec![0.0; 4],
                raw_query_gate: vec![0.0; 12288],
                query: vec![0.0; 6144],
                gate: vec![0.0; 6144],
                raw_key: vec![0.0; 512],
                key: vec![0.0; 512],
                value: vec![0.0; 512],
                gated_attention: vec![0.0; 6144],
                attention: vec![0.0; 2560],
            },
        };
        let mut bytes = Vec::new();
        assert!(append_qsa_trace(
            &mut bytes,
            trace(),
            "wrong",
            &plan,
            0,
            17,
            TraceMode::Attention
        )
        .is_err());
        assert!(bytes.is_empty());
        let mut malformed = trace();
        malformed.output.gate.pop();
        assert!(append_qsa_trace(
            &mut bytes,
            malformed,
            "run",
            &plan,
            0,
            17,
            TraceMode::Attention
        )
        .is_err());
        assert!(bytes.is_empty());
        append_qsa_trace(
            &mut bytes,
            trace(),
            "run",
            &plan,
            0,
            17,
            TraceMode::Attention,
        )
        .unwrap();
        assert_eq!(bytes.len(), TRACE_WORDS * 4);
        assert_eq!(&bytes[..4], &1.0f32.to_le_bytes());
        assert!(append_qsa_trace(
            &mut bytes,
            trace(),
            "run",
            &plan,
            1,
            18,
            TraceMode::Attention
        )
        .is_err());
        assert_eq!(bytes.len(), TRACE_WORDS * 4);
        let mut input_trace = trace();
        input_trace.stamp.plan_id = input_plan.clone();
        let mut v2 = Vec::new();
        let mut invalid = trace();
        invalid.stamp.plan_id = input_plan.clone();
        invalid.input.pop();
        assert!(append_qsa_trace(
            &mut v2,
            invalid,
            "run",
            &input_plan,
            0,
            17,
            TraceMode::WithInput
        )
        .is_err());
        assert!(v2.is_empty());
        append_qsa_trace(
            &mut v2,
            input_trace,
            "run",
            &input_plan,
            0,
            17,
            TraceMode::WithInput,
        )
        .unwrap();
        assert_eq!(v2.len(), INPUT_TRACE_WORDS * 4);
        assert_eq!(&v2[..4], &2.0f32.to_le_bytes());
        assert_eq!(&v2[10240 * 4..10240 * 4 + 4], &1.0f32.to_le_bytes());
    }
    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    #[test]
    fn expected_plan_is_checked_before_remote_work() {
        let plan = "r512-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(verify_expected_plan(plan, Some(plan), true).is_ok());
        assert!(verify_expected_plan(plan, None, true).is_err());
        assert!(verify_expected_plan(plan, Some("r512-wrong"), true).is_err());
        assert!(verify_expected_plan(plan, None, false).is_ok());
        assert!(verify_expected_plan(plan, Some("r512-wrong"), false).is_err());
    }
    #[test]
    fn worker_plan_binds_rank_endpoints_and_capacities() {
        let namespace = Some(Namespace {
            root: "/ledger".into(),
            dev: 3,
            ino: 4,
            uid: 3,
            base_id: "r53-test1234".into(),
        });
        let lease = Some(SharedLease {
            path: "/lease".into(),
            dev: 1,
            ino: 2,
            uid: 3,
        });
        let known_hosts = BTreeMap::from([(3u32, DIGEST.to_owned())]);
        let rank2 = rank_endpoint(2, "/worker", None, None, Some(DIGEST)).unwrap();
        let rank3 =
            rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/worker"), Some(DIGEST)).unwrap();
        let endpoints = BTreeMap::from([(2, rank2.clone()), (3, rank3.clone())]);
        let limits = Some(BTreeMap::from([(2, 100), (3, 200)]));
        let id = worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace,
        )
        .unwrap();
        assert_eq!(id.len(), 69);
        assert_eq!(
            id,
            "r531-f353bd74941ccb5dda477a988229569c1b014dd4e9624268d0b964c29e90cc0f"
        );
        let changed_lease = Some(SharedLease {
            path: "/lease".into(),
            dev: 1,
            ino: 9,
            uid: 3,
        });
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &changed_lease,
                &namespace
            )
            .unwrap()
        );
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &None,
            &namespace
        )
        .is_err());
        let changed_host_key = BTreeMap::from([(3u32, "b".repeat(64))]);
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &changed_host_key,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &BTreeMap::new(),
            &lease,
            &namespace
        )
        .is_err());
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                &"b".repeat(64),
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "plain",
            "BAD",
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        let p12 = worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[1, 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace,
        )
        .unwrap();
        let p21 = worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[2, 1],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace,
        )
        .unwrap();
        assert_ne!(id, p12);
        assert_ne!(p12, p21);
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[248_320],
            1,
            248_046,
            "chat",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[248_046],
            1,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[248_046],
            1,
            248_046,
            "chat",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_ok());
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "chat",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "invalid",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert_eq!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        let changed_local = BTreeMap::from([
            (
                2,
                rank_endpoint(2, "/worker", None, None, Some(&"b".repeat(64))).unwrap(),
            ),
            (3, rank3.clone()),
        ]);
        assert_ne!(
            id,
            worker_plan_id(
                &changed_local,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        let changed_host = BTreeMap::from([
            (2, rank2.clone()),
            (
                3,
                rank_endpoint(3, "/worker", Some("pc2-lan"), Some("/worker"), Some(DIGEST))
                    .unwrap(),
            ),
        ]);
        assert_ne!(
            id,
            worker_plan_id(
                &changed_host,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        let changed_path = BTreeMap::from([
            (2, rank2.clone()),
            (
                3,
                rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/other"), Some(DIGEST)).unwrap(),
            ),
        ]);
        assert_ne!(
            id,
            worker_plan_id(
                &changed_path,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        let changed_digest = BTreeMap::from([
            (2, rank2.clone()),
            (
                3,
                rank_endpoint(
                    3,
                    "/worker",
                    Some("pc1-lan"),
                    Some("/worker"),
                    Some(&"b".repeat(64)),
                )
                .unwrap(),
            ),
        ]);
        assert_ne!(
            id,
            worker_plan_id(
                &changed_digest,
                300,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                301,
                &limits,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &None,
                &[7; 2],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 3],
                16,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                32,
                248_046,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert_ne!(
            id,
            worker_plan_id(
                &endpoints,
                300,
                &limits,
                &[7; 2],
                16,
                248_047,
                "plain",
                DIGEST,
                &known_hosts,
                &lease,
                &namespace
            )
            .unwrap()
        );
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            16,
            248_320,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 2],
            129,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 0],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 32],
            8,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_ok());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 128],
            2,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 8],
            32,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_ok());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 129],
            1,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 1],
            128,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_ok());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &limits,
            &[7; 128],
            1,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_ok());
        assert!(worker_plan_id(
            &BTreeMap::from([(2, rank2), (3, rank3.clone()), (4, rank3)]),
            300,
            &limits,
            &[7; 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
        assert!(worker_plan_id(
            &endpoints,
            300,
            &Some(BTreeMap::from([(2, 100)])),
            &[7; 2],
            16,
            248_046,
            "plain",
            DIGEST,
            &known_hosts,
            &lease,
            &namespace
        )
        .is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn r53_requires_both_matching_reservations_and_consumes_once() {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
        let name = format!(
            "vf-test-r53-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(name);
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let meta = std::fs::metadata(&root).unwrap();
        let namespace = Namespace {
            root: root.to_str().unwrap().into(),
            dev: meta.dev(),
            ino: meta.ino(),
            uid: meta.uid(),
            base_id: "r53-test1234".into(),
        };
        let lease = SharedLease {
            path: "/lease".into(),
            dev: 1,
            ino: 2,
            uid: meta.uid(),
        };
        let plan = format!("r529-{}", "a".repeat(64));
        let write_record = |effective: &str| {
            let value = serde_json::json!({"status":"RESERVED_UNKNOWN_NO_REPLAY", "scope":"Namespace consumption only; see linked execution receipt for actual outcome", "base_id":namespace.base_id, "effective_run_id":effective, "reserved_ns":1, "context":{"plan_id":plan,"shared_lease":lease}});
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(root.join(format!("{effective}.json")))
                .unwrap();
            serde_json::to_writer(&mut file, &value).unwrap();
        };
        write_record(&namespace.base_id);
        assert!(consume_namespace(&namespace, &plan, &lease).is_err());
        assert!(!root.join("r53-test1234-sc117-started.json").exists());
        write_record("r53-test1234-duplicate-slot");
        assert!(consume_namespace(&namespace, "wrong-plan", &lease).is_err());
        assert!(!root.join("r53-test1234-sc117-started.json").exists());
        assert!(consume_namespace(&namespace, &plan, &lease).is_ok());
        assert!(consume_namespace(&namespace, &plan, &lease).is_err());
        // Synthetic records stay in the isolated TMPDIR; no actual operation ran.
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn shared_lease_requires_exact_descriptor_and_exclusive_lock() {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::MetadataExt;
        let name = format!(
            "vf-test-lease-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let m = file.metadata().unwrap();
        let spec = SharedLease {
            path: path.to_str().unwrap().into(),
            dev: m.dev(),
            ino: m.ino(),
            uid: m.uid(),
        };
        assert!(verify_shared_lease(&spec, file.as_raw_fd()).is_err());
        rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive).unwrap();
        assert!(verify_shared_lease(&spec, file.as_raw_fd()).is_ok());
        let other = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(verify_shared_lease(&spec, other.as_raw_fd()).is_err());
        let wrong = SharedLease {
            ino: spec.ino + 1,
            ..spec.clone()
        };
        assert!(verify_shared_lease(&wrong, file.as_raw_fd()).is_err());
        let wrong_uid = SharedLease {
            uid: spec.uid.wrapping_add(1),
            ..spec.clone()
        };
        assert!(verify_shared_lease(&wrong_uid, file.as_raw_fd()).is_err());
        assert!(verify_shared_lease(&spec, -1).is_err());
        rustix::fs::flock(&file, rustix::fs::FlockOperation::Unlock).unwrap();
        assert!(verify_shared_lease(&spec, file.as_raw_fd()).is_err());
        // The zero-byte test file remains in the isolated TMPDIR for inspection.
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn known_hosts_are_single_pinned_and_sealed() {
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::os::fd::AsRawFd;
        let fd =
            rustix::fs::memfd_create("vf-test-original", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let mut original = std::fs::File::from(fd);
        let contents = b"pc1-lan ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
        original.write_all(contents).unwrap();
        original.flush().unwrap();
        let hash = sha256::hex(contents);
        let mut endpoint =
            rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/worker"), Some(DIGEST)).unwrap();
        let path = format!("/proc/self/fd/{}", original.as_raw_fd());
        let path = path.as_str();
        assert!(pin_known_hosts(&mut endpoint.clone(), "other", path, &hash).is_err());
        assert!(pin_known_hosts(&mut endpoint.clone(), "pc1-lan", path, &"b".repeat(64)).is_err());
        let mut pinned = pin_known_hosts(&mut endpoint, "pc1-lan", path, &hash).unwrap();
        assert!(endpoint
            .argv
            .contains(&format!("UserKnownHostsFile={path}")));
        assert!(endpoint.argv.contains(&"HostKeyAlias=pc1-lan".to_owned()));
        original.write_all(b"changed").unwrap();
        assert!(pinned.write_all(b"changed").is_err());
        pinned.seek(SeekFrom::Start(0)).unwrap();
        let mut read = Vec::new();
        pinned.read_to_end(&mut read).unwrap();
        assert_eq!(read, contents);
        let path = format!("/proc/{}/fd/{}", std::process::id(), pinned.as_raw_fd());
        let child = std::process::Command::new("/bin/cat")
            .arg(path)
            .output()
            .unwrap();
        assert!(child.status.success());
        assert_eq!(child.stdout, contents);
        let pinned_path = format!("/proc/{}/fd/{}", std::process::id(), pinned.as_raw_fd());
        let lookup = std::process::Command::new("/usr/bin/ssh-keygen")
            .args(["-F", "pc1-lan", "-f", &pinned_path])
            .output()
            .unwrap();
        assert!(lookup.status.success(), "{:?}", lookup.stderr);
        *endpoint
            .argv
            .iter_mut()
            .find(|a| a.starts_with("UserKnownHostsFile="))
            .unwrap() = format!("UserKnownHostsFile={pinned_path}");
        let effective = std::process::Command::new("/usr/bin/ssh")
            .arg("-G")
            .args(&endpoint.argv[1..])
            .output()
            .unwrap();
        assert!(effective.status.success(), "{:?}", effective.stderr);
        let config = String::from_utf8(effective.stdout).unwrap();
        assert!(config
            .lines()
            .any(|l| l == format!("userknownhostsfile {pinned_path}")));
        assert!(config
            .lines()
            .any(|l| l == "globalknownhostsfile /dev/null"));
        assert!(config.lines().any(|l| l == "hostkeyalias pc1-lan"));
        let path = format!("/proc/self/fd/{}", original.as_raw_fd());
        let path = path.as_str();
        assert!(pin_known_hosts(&mut endpoint, "pc1-lan", path, &hash).is_err());
    }
    #[test]
    fn rank_endpoints_are_explicit_and_reject_shell_metacharacters() {
        let local = rank_endpoint(2, "/opt/vf-rank-session", None, None, Some(DIGEST)).unwrap();
        assert_eq!(local.rank, 2);
        assert_eq!(
            local.argv,
            vec![
                "/opt/vf-rank-session",
                "--guarded-local",
                "--expected-exe-sha256",
                DIGEST
            ]
        );
        assert!(rank_endpoint(2, "/worker", None, None, None).is_err());
        assert!(rank_endpoint(2, "/worker", None, None, Some("BAD")).is_err());
        assert!(rank_endpoint(2, "worker", None, None, Some(DIGEST)).is_err());
        let remote = rank_endpoint(
            3,
            "/opt/vf-rank-session",
            Some("pc1-lan"),
            Some("/opt/vf-rank-session"),
            Some(DIGEST),
        )
        .unwrap();
        assert_eq!(remote.rank, 3);
        assert_eq!(
            remote.argv,
            vec![
                "/usr/bin/ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "pc1-lan",
                "/opt/vf-rank-session",
                "--guarded-remote",
                "--expected-exe-sha256",
                DIGEST
            ]
        );
        assert!(rank_endpoint(3, "/worker", Some("pc1-lan"), None, None).is_err());
        assert!(rank_endpoint(3, "/worker", None, Some("/worker"), None).is_err());
        assert!(rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/worker"), None).is_err());
        for bad in [
            "",
            "ABC",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaA",
        ] {
            assert!(
                rank_endpoint(3, "/worker", Some("pc1-lan"), Some("/worker"), Some(bad)).is_err()
            );
        }
        for host in ["", "-oProxyCommand", "pc1;id", "user@host", "pc1\n"] {
            assert!(
                rank_endpoint(3, "/worker", Some(host), Some("/worker"), Some(DIGEST)).is_err()
            );
        }
        for path in ["worker", "/", "/worker;id", "/worker$(id)", "/worker name"] {
            assert!(
                rank_endpoint(3, "/worker", Some("pc1-lan"), Some(path), Some(DIGEST)).is_err()
            );
        }
        assert!(rank_endpoint(4, "/worker", None, None, None).is_err());
    }
    #[test]
    fn close_requires_actual_dispatched_count_not_every_position() {
        let mut r = voltafabric::session::CloseReceipt {
            rank: 2,
            pid: 1,
            bye_execs: Some(3),
            peak_rss_kib: None,
            exit_code: Some(0),
            killed: false,
            stderr: String::new(),
            stderr_bytes: 0,
            error: None,
        };
        assert!(valid_close(&r, 3, 9));
        assert!(!valid_close(&r, 4, 9));
        assert!(!valid_close(&r, 3, 2));
        r.killed = true;
        assert!(!valid_close(&r, 3, 9));
        r.killed = false;
        r.bye_execs = None;
        assert!(!valid_close(&r, 3, 9));
        r.bye_execs = Some(3);
        r.exit_code = Some(1);
        assert!(!valid_close(&r, 3, 9));
    }
}
