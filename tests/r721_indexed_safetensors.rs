use safetensors::tensor::{serialize, Dtype, TensorView};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use voltafabric::{checkpoint, sha256};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "voltafabric-r721-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn shard(name: &str, payload: &[u8]) -> Vec<u8> {
    let view = TensorView::new(Dtype::F32, vec![1], payload).unwrap();
    let unrelated = TensorView::new(Dtype::F32, vec![1], &[0; 4]).unwrap();
    serialize(
        vec![
            (name.to_owned(), view),
            ("unselected.weight".to_owned(), unrelated),
        ],
        None,
    )
    .unwrap()
}

fn spec(name: &str, payload: &[u8]) -> checkpoint::NamedTensorSpec {
    checkpoint::NamedTensorSpec {
        name: name.to_owned(),
        dtype: Dtype::F32,
        shape: vec![1],
        bytes: payload.len() as u64,
        sha256: sha256::hex(payload),
    }
}

fn index() -> Vec<u8> {
    br#"{"metadata":{"total_size":8},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#.to_vec()
}

fn write_two_shards(root: &Path, a: &[u8], b: &[u8]) {
    fs::write(
        root.join("model-00001-of-00002.safetensors"),
        shard("a.weight", a),
    )
    .unwrap();
    fs::write(
        root.join("model-00002-of-00002.safetensors"),
        shard("b.weight", b),
    )
    .unwrap();
}

#[test]
fn selected_tensors_load_across_indexed_shards() {
    let root = TempDirectory::new();
    let a = 1.25f32.to_le_bytes();
    let b = (-2.5f32).to_le_bytes();
    write_two_shards(root.path(), &a, &b);
    let index_bytes = index();

    let imported = checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &sha256::hex(&index_bytes),
        &[spec("a.weight", &a), spec("b.weight", &b)],
        8,
    )
    .unwrap();

    assert_eq!(imported.tensors.len(), 2);
    assert_eq!(imported.tensors["a.weight"].bytes, a);
    assert_eq!(imported.tensors["b.weight"].bytes, b);
    assert_eq!(imported.plan.tensors_by_shard.len(), 2);
    assert!(imported.file_bytes > imported.io_bytes);
}

#[test]
fn invalid_second_shard_returns_no_partial_import() {
    let root = TempDirectory::new();
    let a = 3.0f32.to_le_bytes();
    let b = 4.0f32.to_le_bytes();
    write_two_shards(root.path(), &a, &b);
    let index_bytes = index();
    let mut invalid_b = spec("b.weight", &b);
    invalid_b.sha256 = "0".repeat(64);

    let result = checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &sha256::hex(&index_bytes),
        &[spec("a.weight", &a), invalid_b],
        8,
    );

    assert!(result.is_err());
}

#[test]
fn missing_shard_and_bad_index_digest_fail_closed() {
    let root = TempDirectory::new();
    let a = 5.0f32.to_le_bytes();
    let b = 6.0f32.to_le_bytes();
    write_two_shards(root.path(), &a, &b);
    let index_bytes = index();
    fs::remove_file(root.path().join("model-00002-of-00002.safetensors")).unwrap();

    assert!(checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &sha256::hex(&index_bytes),
        &[spec("a.weight", &a), spec("b.weight", &b)],
        8,
    )
    .is_err());
    assert!(checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &"0".repeat(64),
        &[spec("a.weight", &a)],
        4,
    )
    .is_err());
}

#[cfg(unix)]
#[test]
fn symlink_shard_is_refused() {
    use std::os::unix::fs::symlink;

    let root = TempDirectory::new();
    let elsewhere = TempDirectory::new();
    let a = 7.0f32.to_le_bytes();
    let b = 8.0f32.to_le_bytes();
    fs::write(
        elsewhere.path().join("outside.safetensors"),
        shard("b.weight", &b),
    )
    .unwrap();
    fs::write(
        root.path().join("model-00001-of-00002.safetensors"),
        shard("a.weight", &a),
    )
    .unwrap();
    symlink(
        elsewhere.path().join("outside.safetensors"),
        root.path().join("model-00002-of-00002.safetensors"),
    )
    .unwrap();
    let index_bytes = index();

    assert!(checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &sha256::hex(&index_bytes),
        &[spec("a.weight", &a), spec("b.weight", &b)],
        8,
    )
    .is_err());
}

#[test]
fn total_payload_cap_is_checked_before_opening_shards() {
    let root = TempDirectory::new();
    let a = 9.0f32.to_le_bytes();
    let b = 10.0f32.to_le_bytes();
    write_two_shards(root.path(), &a, &b);
    let index_bytes = index();

    let result = checkpoint::read_indexed_named_tensors(
        root.path(),
        &index_bytes,
        &sha256::hex(&index_bytes),
        &[spec("a.weight", &a), spec("b.weight", &b)],
        7,
    );

    assert!(result.is_err());
}
