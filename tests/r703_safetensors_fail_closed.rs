use safetensors::tensor::Dtype;
use std::io::Cursor;
use voltafabric::checkpoint::{read_named_tensors, NamedTensorSpec, MAX_HEADER_BYTES};

const PAYLOAD: &[u8] = &[0x00, 0x00, 0x80, 0x3f];

fn selection() -> NamedTensorSpec {
    NamedTensorSpec {
        name: "tiny".to_owned(),
        dtype: Dtype::F32,
        shape: vec![1],
        bytes: PAYLOAD.len() as u64,
        sha256: voltafabric::sha256::hex(PAYLOAD),
    }
}

fn shard(header: &str, payload: &[u8]) -> Cursor<Vec<u8>> {
    let mut header = header.as_bytes().to_vec();
    header.resize((header.len() + 7) & !7, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(payload);
    Cursor::new(bytes)
}

fn single_tensor_header(offsets: &str) -> String {
    format!(r#"{{"tiny":{{"dtype":"F32","shape":[1],"data_offsets":{offsets}}}}}"#)
}

#[test]
fn accepts_one_exact_named_tensor_and_returns_no_unselected_bytes() {
    let mut input = shard(&single_tensor_header("[0,4]"), PAYLOAD);
    let result = read_named_tensors(&mut input, &[selection()], 4).unwrap();
    let tensor = result.tensors.get("tiny").unwrap();
    assert_eq!(tensor.bytes, PAYLOAD);
    assert_eq!(tensor.sha256, voltafabric::sha256::hex(PAYLOAD));
    assert_eq!(
        result.io_bytes,
        result.header_bytes + 8 + PAYLOAD.len() as u64
    );
}

#[test]
fn rejects_duplicate_tensor_and_metadata_keys_before_import() {
    let duplicate_tensor = r#"{"tiny":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"tiny":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut input = shard(duplicate_tensor, PAYLOAD);
    assert!(read_named_tensors(&mut input, &[selection()], 4).is_err());

    let duplicate_metadata = r#"{"__metadata__":{"model":"one","model":"two"},"tiny":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut input = shard(duplicate_metadata, PAYLOAD);
    assert!(read_named_tensors(&mut input, &[selection()], 4).is_err());
}

#[test]
fn rejects_overlapping_offsets_and_trailing_unindexed_data() {
    let overlapping = r#"{"tiny":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"shadow":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut input = shard(overlapping, PAYLOAD);
    assert!(read_named_tensors(&mut input, &[selection()], 4).is_err());

    let mut input = shard(&single_tensor_header("[0,4]"), &[0, 0, 0x80, 0x3f, 0xaa]);
    assert!(read_named_tensors(&mut input, &[selection()], 4).is_err());
}

#[test]
fn rejects_oversized_declared_header_without_reading_a_header_body() {
    let mut input = Cursor::new((MAX_HEADER_BYTES + 1).to_le_bytes().to_vec());
    assert!(read_named_tensors(&mut input, &[selection()], 4).is_err());
}

#[test]
fn rejects_duplicate_selection_names_before_reading_source() {
    let mut duplicate = selection();
    duplicate.name = "tiny".to_owned();
    let selected = [selection(), duplicate];
    let mut input = Cursor::new(Vec::new());
    assert!(read_named_tensors(&mut input, &selected, 8).is_err());
}
