//! Selective Qwen NVFP4 safetensors import. All weights remain compressed on CPU.
//! Metadata labels are caller checks, not authentication. Sources must be quiescent.
use crate::{invalid, moe, quant, resident, Error, Result};
use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use serde::{
    de::{IgnoredAny, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};
pub const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub id: u32,
    pub local: u32,
    pub source_expert: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub model_revision: String,
    pub layer: u32,
    pub rank: u32,
    pub registry: moe::Registry,
    pub resident_order: Vec<u32>,
    pub experts: Vec<Selection>,
    /// Exact labels expected by this caller. Not a hash or upstream attestation.
    pub expected_metadata: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub path: PathBuf,
    pub spec: Spec,
}
#[derive(Debug, Serialize)]
pub struct Imported {
    // Keep the CLI snapshot byte order when streamed directly.
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub io_bytes: u64,
    pub stack: resident::ResidentStack,
}

struct UniqueLabels;
impl<'de> Deserialize<'de> for UniqueLabels {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Check;
        impl<'de> Visitor<'de> for Check {
            type Value = UniqueLabels;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("bounded unique metadata labels")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if name.len() > 256 || names.len() >= 64 || !names.insert(name) {
                        return Err(serde::de::Error::custom(
                            "duplicate/oversized metadata label",
                        ));
                    }
                    let value = map.next_value::<String>()?;
                    if value.len() > 4096 {
                        return Err(serde::de::Error::custom("metadata value exceeds bound"));
                    }
                }
                Ok(UniqueLabels)
            }
        }
        d.deserialize_map(Check)
    }
}

struct UniqueHeader;
impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Check;
        impl<'de> Visitor<'de> for Check {
            type Value = UniqueHeader;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("bounded unique safetensors tensor names")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    let labels = name == "__metadata__";
                    if name.len() > 512 || names.len() >= 65536 || !names.insert(name) {
                        return Err(serde::de::Error::custom(
                            "duplicate/oversized safetensors name or tensor count",
                        ));
                    }
                    if labels {
                        map.next_value::<UniqueLabels>()?;
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(UniqueHeader)
            }
        }
        d.deserialize_map(Check)
    }
}
fn io(e: std::io::Error) -> Error {
    Error::new("CHECKPOINT_IO", e.to_string())
}
fn json(e: serde_json::Error) -> Error {
    Error::new("CHECKPOINT_HEADER", e.to_string())
}
fn bytes<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut b = Vec::new();
    b.try_reserve_exact(n)
        .map_err(|e| Error::new("ALLOCATION_FAILED", e.to_string()))?;
    b.resize(n, 0);
    r.read_exact(&mut b).map_err(io)?;
    Ok(b)
}
fn prefix(layer: u32, e: u32, plane: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{e}.{plane}")
}
fn info<'a>(m: &'a Metadata, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&'a TensorInfo> {
    let t = m
        .info(name)
        .ok_or_else(|| invalid(format!("missing selected tensor {name}")))?;
    if t.dtype != dtype || t.shape != shape {
        return Err(invalid(format!("wrong dtype/shape for {name}")));
    }
    Ok(t)
}
fn shape(m: &Metadata, p: &str) -> Result<quant::PackedRows> {
    let name = format!("{p}.weight");
    let t = m
        .info(&name)
        .ok_or_else(|| invalid(format!("missing selected tensor {name}")))?;
    if t.dtype != Dtype::U8 || t.shape.len() != 2 {
        return Err(invalid("weight must be packed U8 rows by columns/2"));
    }
    let rows = t.shape[0];
    let columns = t.shape[1]
        .checked_mul(2)
        .ok_or_else(|| invalid("column overflow"))?;
    info(
        m,
        &format!("{p}.weight_scale"),
        Dtype::F8_E4M3,
        &[rows, columns / 16],
    )?;
    info(m, &format!("{p}.weight_scale_2"), Dtype::F32, &[])?;
    Ok(quant::PackedRows {
        rows: rows as u64,
        columns: columns as u64,
        codes: Vec::new(),
        scales: Vec::new(),
        global_dequant: 1.0,
    })
}
fn tensor<R: Read + Seek>(
    r: &mut R,
    m: &Metadata,
    name: &str,
    start: u64,
    read_bytes: &mut u64,
) -> Result<Vec<u8>> {
    let t = m
        .info(name)
        .ok_or_else(|| invalid("selected tensor missing after preflight"))?;
    r.seek(SeekFrom::Start(
        start
            .checked_add(t.data_offsets.0 as u64)
            .ok_or_else(|| invalid("offset overflow"))?,
    ))
    .map_err(io)?;
    let b = bytes(r, t.data_offsets.1 - t.data_offsets.0)?;
    *read_bytes = read_bytes
        .checked_add(b.len() as u64)
        .ok_or_else(|| invalid("byte count overflow"))?;
    Ok(b)
}
/// Read only the bounded header and nine tensors per selected expert, then reuse
/// resident validation/permutation. No tensor outside the selection is read.
/// Payload limit is final compressed buffers, not RSS; header/source/output coexist.
pub fn read<R: Read + Seek>(r: &mut R, s: &Spec) -> Result<Imported> {
    if s.experts.len() > 1024
        || s.expected_metadata.len() > 64
        || s.expected_metadata
            .iter()
            .any(|(k, v)| k.len() > 256 || v.len() > 4096)
    {
        return Err(invalid(
            "checkpoint selection or expected metadata too large",
        ));
    }
    let file_bytes = r.seek(SeekFrom::End(0)).map_err(io)?;
    r.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut length = [0; 8];
    r.read_exact(&mut length).map_err(io)?;
    let n = u64::from_le_bytes(length);
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err(invalid("checkpoint header exceeds 8 MiB bound or is empty"));
    }
    let start = n.checked_add(8).ok_or_else(|| invalid("header overflow"))?;
    if start > file_bytes {
        return Err(invalid("truncated checkpoint header"));
    }
    let header = bytes(r, n as usize)?;
    serde_json::from_slice::<UniqueHeader>(&header).map_err(json)?;
    let m: Metadata = serde_json::from_slice(&header).map_err(json)?;
    if start.checked_add(m.data_len() as u64) != Some(file_bytes) {
        return Err(invalid(
            "safetensors offsets do not cover exact file length",
        ));
    }
    for (k, v) in &s.expected_metadata {
        if m.metadata().as_ref().and_then(|x| x.get(k)) != Some(v) {
            return Err(invalid(format!("checkpoint label differs: {k}")));
        }
    }
    let mut input = resident::ResidentInput {
        model_revision: s.model_revision.clone(),
        layer: s.layer,
        rank: s.rank,
        registry: s.registry.clone(),
        resident_order: s.resident_order.clone(),
        experts: Vec::new(),
    };
    for e in &s.experts {
        input.experts.push(resident::ResidentExpert {
            id: e.id,
            local: e.local,
            gate: shape(&m, &prefix(s.layer, e.source_expert, "gate_proj"))?,
            up: shape(&m, &prefix(s.layer, e.source_expert, "up_proj"))?,
            down: shape(&m, &prefix(s.layer, e.source_expert, "down_proj"))?,
        });
    }
    // Check uniform shapes, exact registry/order and final payload before reading weights.
    resident::preflight(&input)?;
    let mut io_bytes = start;
    for (e, target) in s.experts.iter().zip(&mut input.experts) {
        for (plane, p) in [
            ("gate_proj", &mut target.gate),
            ("up_proj", &mut target.up),
            ("down_proj", &mut target.down),
        ] {
            let pre = prefix(s.layer, e.source_expert, plane);
            p.codes = tensor(r, &m, &format!("{pre}.weight"), start, &mut io_bytes)?;
            p.scales = tensor(r, &m, &format!("{pre}.weight_scale"), start, &mut io_bytes)?;
            let f = tensor(
                r,
                &m,
                &format!("{pre}.weight_scale_2"),
                start,
                &mut io_bytes,
            )?;
            p.global_dequant = f64::from(f32::from_le_bytes(
                f.try_into().map_err(|_| invalid("F32 scalar length"))?,
            ));
        }
    }
    Ok(Imported {
        stack: resident::pack(&input)?,
        io_bytes,
        header_bytes: n,
        file_bytes,
    })
}
/// Local CLI/file API only. Check regular type before/after open. Unix open uses
/// O_NONBLOCK so a regular-to-FIFO race cannot wait for a writer. No service/RPC.
/// Detect ordinary size/mtime changes; not protection against a malicious writer
/// restoring metadata. The caller must keep the source immutable during import.
pub(crate) fn open_nonblocking(path: &std::path::Path) -> std::io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    options.open(path)
}

pub fn load(p: &Request) -> Result<Imported> {
    let before = std::fs::metadata(&p.path).map_err(io)?;
    if !before.is_file() {
        return Err(invalid("checkpoint path must be a regular file"));
    }
    let mut f = open_nonblocking(&p.path).map_err(io)?;
    let initial = f.metadata().map_err(io)?;
    if !initial.is_file() {
        return Err(invalid("opened checkpoint is not regular"));
    }
    let result = read(&mut f, &p.spec)?;
    let after = f.metadata().map_err(io)?;
    if initial.len() != after.len()
        || initial.modified().map_err(io)? != after.modified().map_err(io)?
    {
        return Err(Error::new(
            "CHECKPOINT_CHANGED",
            "checkpoint size/mtime changed during import",
        ));
    }
    Ok(result)
}

#[cfg(all(test, unix))]
mod file_tests {
    #[test]
    fn fifo_open_is_nonblocking_before_file_type_check() {
        let base = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
        let dir = base.join("checkpoint-fifo-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "fifo-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .unwrap()
            .success());
        let result = super::open_nonblocking(&path);
        std::fs::remove_file(&path).unwrap();
        let file = result.unwrap();
        assert!(!file.metadata().unwrap().is_file());
    }
}
