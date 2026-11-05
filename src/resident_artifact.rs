//! Stable, CPU-only SafeTensors export for validated resident stacks.
use crate::{
    resident::{self, ResidentStack},
    Error, Result,
};
use safetensors::tensor::{serialize, Dtype, Metadata, SafeTensors, TensorInfo, TensorView, View};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const SCHEMA: &str = "QPN2_RESIDENT_V1";
const STAGING_ATTEMPTS: u32 = 64;
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExpertInfo {
    pub id: u32,
    pub local: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactInfo {
    pub model_revision: String,
    pub layer: u32,
    pub rank: u32,
    pub epoch: u64,
    pub payload_bytes: u64,
    pub physical_experts: Vec<PhysicalExpertInfo>,
    pub resident_order: Vec<u32>,
    pub shape: [usize; 5],
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactReceipt {
    pub meta: ArtifactInfo,
    pub file_bytes: u64,
    pub scope: &'static str,
    pub cuda_executed: bool,
}

fn io_error(code: &'static str, context: &str, err: impl std::fmt::Display) -> Error {
    Error::new(code, format!("{context}: {err}"))
}

fn artifact_info(stack: &ResidentStack) -> ArtifactInfo {
    let (e, n13, k13, n2, k2) = stack.shape();
    ArtifactInfo {
        model_revision: stack.model_revision().to_owned(),
        layer: stack.layer(),
        rank: stack.rank(),
        epoch: stack.epoch(),
        payload_bytes: stack.payload_bytes(),
        physical_experts: stack
            .physical_experts()
            .iter()
            .map(|expert| PhysicalExpertInfo {
                id: expert.id(),
                local: expert.local(),
            })
            .collect(),
        resident_order: stack.resident_order().to_vec(),
        shape: [e, n13 as usize, k13 as usize, n2 as usize, k2 as usize],
    }
}

fn f32_le_bytes(values: &[f32]) -> Result<Vec<u8>> {
    let capacity = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::new("ARTIFACT_INVALID", "global scale byte length overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|err| io_error("ARTIFACT_IO", "cannot allocate global scale byte view", err))?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

type TensorViews<'a> = Vec<(String, TensorView<'a>)>;
type ArtifactMetadata = HashMap<String, String>;

fn build_tensors<'a>(
    stack: &'a ResidentStack,
    global13: &'a [u8],
    global2: &'a [u8],
) -> Result<(TensorViews<'a>, ArtifactMetadata)> {
    let (e, n13, _, n2, _) = stack.shape();
    let views = vec![
        (
            "codes13".to_owned(),
            TensorView::new(Dtype::U8, vec![stack.codes13().len()], stack.codes13()),
        ),
        (
            "scales13".to_owned(),
            TensorView::new(Dtype::U8, vec![stack.scales13().len()], stack.scales13()),
        ),
        (
            "codes2".to_owned(),
            TensorView::new(Dtype::U8, vec![stack.codes2().len()], stack.codes2()),
        ),
        (
            "scales2".to_owned(),
            TensorView::new(Dtype::U8, vec![stack.scales2().len()], stack.scales2()),
        ),
        (
            "gscales13".to_owned(),
            TensorView::new(Dtype::F32, vec![e, (n13 / 32) as usize], global13),
        ),
        (
            "gscales2".to_owned(),
            TensorView::new(Dtype::F32, vec![e, (n2 / 32) as usize], global2),
        ),
    ];
    let mut tensors = Vec::with_capacity(6);
    for (name, view) in views {
        tensors.push((
            name,
            view.map_err(|err| io_error("ARTIFACT_INVALID", "tensor view rejected", err))?,
        ));
    }
    let info = artifact_info(stack);
    let serialized_info = serde_json::to_string(&info).map_err(|err| {
        io_error(
            "ARTIFACT_METADATA",
            "resident metadata serialization failed",
            err,
        )
    })?;
    let metadata = HashMap::from([
        ("vf_schema".to_owned(), SCHEMA.to_owned()),
        ("vf_info".to_owned(), serialized_info),
    ]);
    Ok((tensors, metadata))
}

fn staging_path(output: &Path, attempt: u32) -> Result<(PathBuf, File)> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .ok_or_else(|| Error::new("INVALID_INPUT", "output path must name a file"))?;
    let mut stage_name = name.to_os_string();
    let serial = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
    stage_name.push(format!(
        ".vf-stage-{}-{serial}-{attempt}",
        std::process::id()
    ));
    let stage = parent.join(stage_name);
    match OpenOptions::new().write(true).create_new(true).open(&stage) {
        Ok(file) => Ok((stage, file)),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Err(io_error(
            "STAGE_COLLISION",
            "staging name already exists",
            err,
        )),
        Err(err) => Err(io_error("ARTIFACT_IO", "cannot create staging file", err)),
    }
}

fn publish_with(output: &Path, write_stage: impl FnOnce(&Path) -> Result<()>) -> Result<u64> {
    if output.file_name().is_none() {
        return Err(Error::new("INVALID_INPUT", "output path must name a file"));
    }
    let mut stage_and_file = None;
    for attempt in 0..STAGING_ATTEMPTS {
        match staging_path(output, attempt) {
            Ok(pair) => {
                stage_and_file = Some(pair);
                break;
            }
            Err(err) if err.code == "STAGE_COLLISION" => continue,
            Err(err) => return Err(err),
        }
    }
    let (stage, file) = stage_and_file
        .ok_or_else(|| Error::new("ARTIFACT_IO", "staging filename retry limit reached"))?;
    drop(file);

    if let Err(err) = write_stage(&stage) {
        let _ = fs::remove_file(&stage);
        return Err(err);
    }
    let stage_file = match File::open(&stage) {
        Ok(file) => file,
        Err(err) => {
            let _ = fs::remove_file(&stage);
            return Err(io_error(
                "ARTIFACT_IO",
                "cannot reopen staged artifact",
                err,
            ));
        }
    };
    if let Err(err) = stage_file.sync_all() {
        drop(stage_file);
        let _ = fs::remove_file(&stage);
        return Err(io_error("ARTIFACT_IO", "cannot sync staged artifact", err));
    }
    let file_bytes = match stage_file.metadata() {
        Ok(meta) => meta.len(),
        Err(err) => {
            drop(stage_file);
            let _ = fs::remove_file(&stage);
            return Err(io_error("ARTIFACT_IO", "cannot stat staged artifact", err));
        }
    };
    drop(stage_file);

    if let Err(err) = fs::hard_link(&stage, output) {
        let _ = fs::remove_file(&stage);
        let code = if err.kind() == std::io::ErrorKind::AlreadyExists {
            "OUTPUT_EXISTS"
        } else {
            "ARTIFACT_PUBLISH"
        };
        return Err(io_error(code, "cannot atomically publish artifact", err));
    }
    let cleanup = fs::remove_file(&stage);
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let sync_result = File::open(parent).and_then(|dir| dir.sync_all());
    if let Err(err) = cleanup {
        return Err(io_error(
            "PUBLICATION_UNCERTAIN",
            "artifact is linked but staging cleanup failed",
            err,
        ));
    }
    if let Err(err) = sync_result {
        return Err(io_error(
            "PUBLICATION_UNCERTAIN",
            "artifact is linked but directory sync failed",
            err,
        ));
    }
    Ok(file_bytes)
}

/// Export an already validated resident stack as an atomically published SafeTensors artifact.
pub fn export(stack: &ResidentStack, output: &Path) -> Result<ArtifactReceipt> {
    let global13 = f32_le_bytes(stack.gscales13())?;
    let global2 = f32_le_bytes(stack.gscales2())?;
    let (tensors, metadata) = build_tensors(stack, &global13, &global2)?;
    let expected_bytes = stack.payload_bytes();
    let lengths = tensors
        .iter()
        .try_fold(0u64, |sum, (_, tensor)| {
            sum.checked_add(tensor.data().len() as u64)
        })
        .ok_or_else(|| Error::new("ARTIFACT_INVALID", "tensor payload length overflow"))?;
    if lengths != expected_bytes {
        return Err(Error::new(
            "ARTIFACT_INVALID",
            "resident payload byte count does not match tensor views",
        ));
    }
    let bytes = canonical(serialize(tensors, Some(metadata)).map_err(|err| {
        io_error(
            "ARTIFACT_SERIALIZE",
            "SafeTensors serialization failed",
            err,
        )
    })?)?;
    let file_bytes = publish_with(output, |stage| {
        fs::write(stage, &bytes).map_err(|err| {
            io_error(
                "ARTIFACT_SERIALIZE",
                "SafeTensors serialization failed",
                err,
            )
        })
    })?;
    Ok(ArtifactReceipt {
        meta: artifact_info(stack),
        file_bytes,
        scope: "CPU_COMPRESSED_EXPORT_ONLY",
        cuda_executed: false,
    })
}

/// The same SafeTensors file with recursively sorted header JSON keys. `safetensors`
/// writes `__metadata__` in `HashMap` order, which differs from one process to the next:
/// until A53 the same stack exported to one of two byte strings. Header text is padded
/// with spaces to 8 bytes, as `safetensors` pads it; the payload bytes are unchanged.
/// Sorting keys keeps the header's length, so the header is rewritten in place and the
/// file is held once, not twice; a header of another length gets a new buffer.
// serde_json may use either BTreeMap or insertion-ordered Map, depending on
// features unified by optional dependencies. Make canonical bytes independent of that.
fn sort_json_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            let mut entries: Vec<_> = std::mem::take(object).into_iter().collect();
            for (_, child) in &mut entries {
                sort_json_keys(child);
            }
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            object.extend(entries);
        }
        serde_json::Value::Array(values) => {
            for child in values {
                sort_json_keys(child);
            }
        }
        _ => {}
    }
}

fn canonical(mut raw: Vec<u8>) -> Result<Vec<u8>> {
    let malformed = || Error::new("ARTIFACT_SERIALIZE", "SafeTensors header is malformed");
    let n = raw
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(malformed)?;
    let header = raw.get(8..8 + n).ok_or_else(malformed)?;
    let mut header: serde_json::Value = serde_json::from_slice(header).map_err(|_| malformed())?;
    sort_json_keys(&mut header);
    let mut text = serde_json::to_vec(&header).map_err(|_| malformed())?;
    text.resize(text.len().div_ceil(8) * 8, b' ');
    if text.len() == n {
        raw[8..8 + n].copy_from_slice(&text);
        return Ok(raw);
    }
    let mut out = Vec::with_capacity(raw.len());
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    out.extend_from_slice(&raw[8 + n..]);
    Ok(out)
}

/// The exact bytes `export` publishes for this stack, in memory.
pub fn encode(stack: &ResidentStack) -> Result<Vec<u8>> {
    let global13 = f32_le_bytes(stack.gscales13())?;
    let global2 = f32_le_bytes(stack.gscales2())?;
    let (tensors, metadata) = build_tensors(stack, &global13, &global2)?;
    canonical(serialize(tensors, Some(metadata)).map_err(|err| {
        io_error(
            "ARTIFACT_SERIALIZE",
            "SafeTensors serialization failed",
            err,
        )
    })?)
}

fn invalid_artifact(message: &str) -> Error {
    Error::new("ARTIFACT_INVALID", message)
}

fn u8_tensor(file: &SafeTensors<'_>, name: &str) -> Result<Vec<u8>> {
    let view = file
        .tensor(name)
        .map_err(|_| invalid_artifact("a QPN2 resident tensor is missing"))?;
    if view.dtype() != Dtype::U8 || view.shape() != [view.data().len()] {
        return Err(invalid_artifact(
            "QPN2 code and scale tensors are one-dimensional U8",
        ));
    }
    Ok(view.data().to_vec())
}

fn f32_tensor(file: &SafeTensors<'_>, name: &str, experts: usize) -> Result<Vec<f32>> {
    let view = file
        .tensor(name)
        .map_err(|_| invalid_artifact("a QPN2 resident tensor is missing"))?;
    let data = view.data();
    if view.dtype() != Dtype::F32 || view.shape().len() != 2 || view.shape()[0] != experts {
        return Err(invalid_artifact(
            "tile factor tensors are F32 [experts, tiles]",
        ));
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// Compare the canonical serialization in bounded pieces. Re-serializing the whole
// payload here used to allocate another artifact-sized Vec during admission.
fn canonical_encoding_matches(stack: &ResidentStack, bytes: &[u8]) -> Result<bool> {
    let global13 = f32_le_bytes(stack.gscales13())?;
    let global2 = f32_le_bytes(stack.gscales2())?;
    let (mut views, metadata) = build_tensors(stack, &global13, &global2)?;
    // Must match safetensors 0.8's prepare ordering exactly.
    views.sort_by(|(left_name, left), (right_name, right)| {
        right
            .dtype()
            .cmp(&left.dtype())
            .then(left_name.cmp(right_name))
    });
    let mut offset = 0usize;
    let mut infos = Vec::with_capacity(views.len());
    for (name, view) in &views {
        let end = offset
            .checked_add(view.data_len())
            .ok_or_else(|| invalid_artifact("tensor offsets overflow"))?;
        infos.push((
            name.clone(),
            TensorInfo {
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
                data_offsets: (offset, end),
            },
        ));
        offset = end;
    }
    let header = Metadata::new(Some(metadata), infos)
        .map_err(|err| io_error("ARTIFACT_SERIALIZE", "tensor metadata invalid", err))?;
    let mut value = serde_json::to_value(header)
        .map_err(|err| io_error("ARTIFACT_SERIALIZE", "header serialization failed", err))?;
    sort_json_keys(&mut value);
    let mut text = serde_json::to_vec(&value)
        .map_err(|err| io_error("ARTIFACT_SERIALIZE", "header serialization failed", err))?;
    text.resize(text.len().div_ceil(8) * 8, b' ');
    let Some(header_end) = 8usize.checked_add(text.len()) else {
        return Ok(false);
    };
    if bytes.get(..8) != Some(&(text.len() as u64).to_le_bytes())
        || bytes.get(8..header_end) != Some(text.as_slice())
        || bytes.len().checked_sub(header_end) != Some(offset)
    {
        return Ok(false);
    }
    let mut cursor = header_end;
    for (_, view) in &views {
        let data = view.data();
        let end = cursor
            .checked_add(data.len())
            .ok_or_else(|| invalid_artifact("tensor offsets overflow"))?;
        if bytes.get(cursor..end) != Some(data) {
            return Ok(false);
        }
        cursor = end;
    }
    Ok(cursor == bytes.len())
}

/// Rebuild a resident stack from the bytes of a `QPN2_RESIDENT_V1` artifact. The stack
/// passes the same validation `pack` gives its output, and the bytes must be exactly the
/// ones `encode` writes for it, so an artifact's SHA-256 names one stack and one only.
pub fn import(bytes: &[u8]) -> Result<ResidentStack> {
    let (_, header) = SafeTensors::read_metadata(bytes)
        .map_err(|err| io_error("ARTIFACT_INVALID", "not a SafeTensors file", err))?;
    let metadata = header.metadata().as_ref();
    let schema = metadata.and_then(|m| m.get("vf_schema"));
    let info = metadata.and_then(|m| m.get("vf_info"));
    let (Some(schema), Some(info)) = (schema, info) else {
        return Err(invalid_artifact("artifact lacks vf_schema or vf_info"));
    };
    if schema != SCHEMA {
        return Err(invalid_artifact("artifact schema is not QPN2_RESIDENT_V1"));
    }
    let info: ArtifactInfo = serde_json::from_str(info)
        .map_err(|err| io_error("ARTIFACT_INVALID", "malformed vf_info", err))?;
    let file = SafeTensors::deserialize(bytes)
        .map_err(|err| io_error("ARTIFACT_INVALID", "not a SafeTensors file", err))?;
    let experts = info.physical_experts.len();
    let stack = resident::from_parts(resident::Parts {
        codes13: u8_tensor(&file, "codes13")?,
        scales13: u8_tensor(&file, "scales13")?,
        gscales13: f32_tensor(&file, "gscales13", experts)?,
        codes2: u8_tensor(&file, "codes2")?,
        scales2: u8_tensor(&file, "scales2")?,
        gscales2: f32_tensor(&file, "gscales2", experts)?,
        model_revision: info.model_revision,
        layer: info.layer,
        rank: info.rank,
        epoch: info.epoch,
        physical_experts: info
            .physical_experts
            .iter()
            .map(|p| (p.id, p.local))
            .collect(),
        resident_order: info.resident_order,
        shape: info.shape,
        payload_bytes: info.payload_bytes,
    })?;
    if !canonical_encoding_matches(&stack, bytes)? {
        return Err(Error::new(
            "ARTIFACT_NOT_CANONICAL",
            "artifact bytes differ from the canonical encoding of their content",
        ));
    }
    Ok(stack)
}

/// Merge complete, independently validated compressed resident artifacts without
/// dequantizing their QPN2 planes. Input artifact order defines physical order;
/// every artifact must describe the same model revision, layer, rank, epoch, and
/// expert geometry. Duplicate global or local expert identities are rejected.
pub fn merge(artifacts: &[&[u8]]) -> Result<ResidentStack> {
    const MAX_ARTIFACTS: usize = 64;
    if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACTS {
        return Err(invalid_artifact(
            "resident artifact merge requires between 1 and 64 inputs",
        ));
    }
    let stacks: Vec<ResidentStack> = artifacts
        .iter()
        .map(|bytes| import(bytes))
        .collect::<Result<_>>()?;
    let first = stacks
        .first()
        .ok_or_else(|| invalid_artifact("resident artifact merge has no inputs"))?;
    if first.is_empty() {
        return Err(invalid_artifact(
            "empty resident artifacts cannot be merged into a compute stack",
        ));
    }
    let (_, n13, k13, n2, k2) = first.shape();
    let mut expert_count = 0usize;
    let mut payload_bytes = 0u64;
    let mut physical_experts = Vec::new();
    let mut resident_order = Vec::new();
    let mut global_ids = std::collections::BTreeSet::new();
    let mut local_ids = std::collections::BTreeSet::new();
    for stack in &stacks {
        let (count, stack_n13, stack_k13, stack_n2, stack_k2) = stack.shape();
        if stack.is_empty()
            || stack.model_revision() != first.model_revision()
            || stack.layer() != first.layer()
            || stack.rank() != first.rank()
            || stack.epoch() != first.epoch()
            || (stack_n13, stack_k13, stack_n2, stack_k2) != (n13, k13, n2, k2)
        {
            return Err(invalid_artifact(
                "resident artifacts differ in identity or expert geometry",
            ));
        }
        expert_count = expert_count
            .checked_add(count)
            .ok_or_else(|| invalid_artifact("resident expert count overflow"))?;
        if expert_count > 1024 {
            return Err(invalid_artifact(
                "merged resident stack exceeds 1024 experts",
            ));
        }
        payload_bytes = payload_bytes
            .checked_add(stack.payload_bytes())
            .ok_or_else(|| invalid_artifact("merged resident payload overflow"))?;
        if payload_bytes > crate::resident::MAX_PAYLOAD_BYTES {
            return Err(Error::new(
                "CAPACITY_EXCEEDED",
                "merged compressed resident payload exceeds 128 MiB",
            ));
        }
        for expert in stack.physical_experts() {
            if !global_ids.insert(expert.id()) || !local_ids.insert(expert.local()) {
                return Err(invalid_artifact(
                    "merged resident artifacts repeat a global or local expert ID",
                ));
            }
            physical_experts.push((expert.id(), expert.local()));
        }
        resident_order.extend_from_slice(stack.resident_order());
    }
    if expert_count == 0 {
        return Err(invalid_artifact("merged resident stack is empty"));
    }

    macro_rules! concatenate_plane {
        ($method:ident) => {{
            let capacity = stacks
                .iter()
                .try_fold(0usize, |sum, stack| sum.checked_add(stack.$method().len()))
                .ok_or_else(|| invalid_artifact("merged QPN2 plane length overflow"))?;
            let mut joined = Vec::new();
            joined.try_reserve_exact(capacity).map_err(|error| {
                io_error(
                    "ARTIFACT_IO",
                    "cannot reserve merged compressed plane",
                    error,
                )
            })?;
            for stack in &stacks {
                joined.extend_from_slice(stack.$method());
            }
            joined
        }};
    }

    let n13 = usize::try_from(n13)
        .map_err(|_| invalid_artifact("merged resident dimension exceeds usize"))?;
    let k13 = usize::try_from(k13)
        .map_err(|_| invalid_artifact("merged resident dimension exceeds usize"))?;
    let n2 = usize::try_from(n2)
        .map_err(|_| invalid_artifact("merged resident dimension exceeds usize"))?;
    let k2 = usize::try_from(k2)
        .map_err(|_| invalid_artifact("merged resident dimension exceeds usize"))?;
    resident::from_parts(resident::Parts {
        model_revision: first.model_revision().to_owned(),
        layer: first.layer(),
        rank: first.rank(),
        epoch: first.epoch(),
        physical_experts,
        resident_order,
        shape: [expert_count, n13, k13, n2, k2],
        payload_bytes,
        codes13: concatenate_plane!(codes13),
        scales13: concatenate_plane!(scales13),
        gscales13: concatenate_plane!(gscales13),
        codes2: concatenate_plane!(codes2),
        scales2: concatenate_plane!(scales2),
        gscales2: concatenate_plane!(gscales2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn streamed_canonical_check_matches_full_encoding_for_mutations() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/resident_import_qwen_expert0.json");
        let request: crate::checkpoint::Request =
            serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let stack = crate::checkpoint::load(&request).unwrap().stack;
        let original = encode(&stack).unwrap();
        assert!(canonical_encoding_matches(&stack, &original).unwrap());
        let header = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
        for index in (0..8 + header).chain((8 + header..original.len()).step_by(137)) {
            let mut altered = original.clone();
            altered[index] ^= 1;
            assert_eq!(
                canonical_encoding_matches(&stack, &altered).unwrap(),
                encode(&stack).unwrap() == altered,
                "byte {index}"
            );
        }
        assert!(!canonical_encoding_matches(&stack, &original[..original.len() - 1]).unwrap());
        let mut appended = original.clone();
        appended.push(0);
        assert!(!canonical_encoding_matches(&stack, &appended).unwrap());
    }

    #[test]
    fn failed_partial_staging_never_publishes_final_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vf-a19-unit-{}-{nonce}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let final_path = dir.join("artifact.safetensors");
        let result = publish_with(&final_path, |stage| {
            fs::write(stage, b"partial")
                .map_err(|e| io_error("ARTIFACT_IO", "injected write", e))?;
            Err(Error::new("ARTIFACT_SERIALIZE", "injected failure"))
        });
        assert!(result.is_err());
        assert!(!final_path.exists());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        fs::remove_dir(dir).unwrap();
    }

    fn file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut raw = (header.len() as u64).to_le_bytes().to_vec();
        raw.extend_from_slice(header.as_bytes());
        raw.extend_from_slice(payload);
        raw
    }

    #[test]
    fn safetensors_output_is_made_canonical_without_a_second_file_buffer() {
        let (a, b) = ([1u8, 2, 3], [4u8; 5]);
        let tensors = vec![
            ("b", TensorView::new(Dtype::U8, vec![5], &b).unwrap()),
            ("a", TensorView::new(Dtype::U8, vec![3], &a).unwrap()),
        ];
        let metadata = (0..16)
            .map(|i| (format!("k{i}"), format!("v{i}")))
            .collect::<HashMap<_, _>>();
        let raw = serialize(tensors, Some(metadata)).unwrap();
        let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let (len, at, payload) = (raw.len(), raw.as_ptr(), raw[8 + n..].to_vec());
        let out = canonical(raw).unwrap();
        assert_eq!((out.len(), out.as_ptr()), (len, at));
        assert_eq!(u64::from_le_bytes(out[..8].try_into().unwrap()), n as u64);
        assert_eq!(out[8 + n..], payload[..]);
        let text = std::str::from_utf8(&out[8..8 + n]).unwrap().trim_end();
        let header: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(header["__metadata__"].as_object().unwrap().len(), 16);
        assert_eq!(text, serde_json::to_string(&header).unwrap());
    }

    #[test]
    fn a_header_that_changes_length_is_rebuilt_with_its_new_length() {
        let payload = b"PAYLOAD";
        let sorted = file("{\"a\":2,\"z\":1}   ", payload);
        let same = canonical(file("{\"z\":1,\"a\":2}   ", payload)).unwrap();
        let spaced = canonical(file("{ \"z\" : 1 , \"a\" : 2 }   ", payload)).unwrap();
        assert_eq!(same, sorted);
        assert_eq!(spaced, sorted);

        let nested = canonical(file("{\"z\":1,\"a\":{\"z\":0,\"b\":1}}", payload)).unwrap();
        let header_len = u64::from_le_bytes(nested[..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&nested[8..8 + header_len])
            .unwrap()
            .trim_end();
        assert_eq!(header, "{\"a\":{\"b\":1,\"z\":0},\"z\":1}");
    }
}
