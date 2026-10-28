use serde_json::{json, Value};
use std::io::{Cursor, Read, Seek, SeekFrom};
use voltafabric::{
    checkpoint::{self, Selection, Spec},
    moe,
};

fn spec() -> Spec {
    Spec {
        model_revision: "rev".into(),
        layer: 0,
        rank: 2,
        registry: moe::Registry {
            epoch: 7,
            copies: vec![moe::Replica {
                expert: 99,
                rank: 2,
                local: 8,
            }],
        },
        resident_order: vec![8],
        experts: vec![Selection {
            id: 99,
            local: 8,
            source_expert: 0,
        }],
        expected_metadata: [("revision".into(), "rev".into())].into_iter().collect(),
    }
}
fn fixture() -> Vec<u8> {
    let mut h = serde_json::Map::new();
    h.insert("__metadata__".into(), json!({"revision":"rev"}));
    let mut data = Vec::new();
    for plane in ["gate_proj", "up_proj", "down_proj"] {
        for (suffix, dtype, shape, bytes) in [
            ("weight", "U8", vec![64, 32], vec![0x22; 2048]),
            ("weight_scale", "F8_E4M3", vec![64, 4], vec![0x38; 256]),
            (
                "weight_scale_2",
                "F32",
                vec![],
                0.25f32.to_le_bytes().to_vec(),
            ),
        ] {
            let start = data.len();
            data.extend(bytes);
            h.insert(
                format!("model.language_model.layers.0.mlp.experts.0.{plane}.{suffix}"),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
    }
    encode(Value::Object(h), data)
}
fn encode(h: Value, data: Vec<u8>) -> Vec<u8> {
    let mut header = serde_json::to_vec(&h).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}
fn edit(mut bytes: Vec<u8>, f: impl FnOnce(&mut Value)) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data = bytes.split_off(8 + n);
    let mut h: Value = serde_json::from_slice(&bytes[8..]).unwrap();
    f(&mut h);
    encode(h, data)
}
const W: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
#[test]
fn import_sparse_identity_exact_factors() {
    let bytes = fixture();
    let result = checkpoint::read(&mut Cursor::new(&bytes), &spec()).unwrap();
    assert_eq!(result.stack.shape(), (1, 128, 64, 64, 64));
    assert_eq!(result.stack.rank(), 2);
    assert_eq!(result.stack.physical_experts()[0].id(), 99);
    assert_eq!(result.stack.physical_experts()[0].local(), 8);
    assert!(result.stack.codes13().iter().all(|&x| x == 0x22));
    assert!(result
        .stack
        .gscales13()
        .iter()
        .chain(result.stack.gscales2())
        .all(|&x| x.to_bits() == 0.25f32.to_bits()));
    assert_eq!(result.io_bytes, bytes.len() as u64);
}
#[test]
fn malformed_header_offsets_dtype_shape_and_identity() {
    let bytes = fixture();
    for bad in [
        edit(bytes.clone(), |h| h[W]["data_offsets"] = json!([1, 2049])),
        edit(bytes.clone(), |h| h[W]["dtype"] = json!("I8")),
        edit(bytes.clone(), |h| h[W]["shape"] = json!([32, 64])),
        edit(bytes.clone(), |h| {
            h["__metadata__"]["revision"] = json!("other")
        }),
        edit(bytes.clone(), |h| {
            h.as_object_mut().unwrap().remove(W);
        }),
        bytes[..bytes.len() - 1].to_vec(),
        [bytes.clone(), vec![0]].concat(),
        vec![255; 8],
    ] {
        assert!(checkpoint::read(&mut Cursor::new(bad), &spec()).is_err());
    }
}
#[test]
fn invalid_f32_and_scale_bits_rejected() {
    let bytes = fixture();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let f = h["model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_2"]
        ["data_offsets"][0]
        .as_u64()
        .unwrap() as usize
        + 8
        + n;
    for value in [0f32, -0f32, -1f32, f32::NAN, f32::INFINITY] {
        let mut b = bytes.clone();
        b[f..f + 4].copy_from_slice(&value.to_le_bytes());
        assert!(checkpoint::read(&mut Cursor::new(b), &spec()).is_err());
    }
    let s = h["model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale"]["data_offsets"][0]
        .as_u64()
        .unwrap() as usize
        + 8
        + n;
    for value in [127, 128, 255] {
        let mut b = bytes.clone();
        b[s] = value;
        assert!(checkpoint::read(&mut Cursor::new(b), &spec()).is_err());
    }
}
#[test]
fn invalid_registry_rejected_before_payload_read() {
    let mut s = spec();
    s.resident_order = vec![900];
    let b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap());
    let len = b.len() as u64;
    let mut guarded = Guarded {
        cursor: Cursor::new(b),
        end: len,
        max_read_end: 8 + n,
    };
    assert!(checkpoint::read(&mut guarded, &s).is_err());
}
struct Guarded {
    cursor: Cursor<Vec<u8>>,
    end: u64,
    max_read_end: u64,
}
impl Read for Guarded {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        assert!(
            self.cursor.position() + b.len() as u64 <= self.max_read_end,
            "unselected tensor was read"
        );
        self.cursor.read(b)
    }
}
impl Seek for Guarded {
    fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
        match p {
            SeekFrom::End(0) => Ok(self.end),
            _ => self.cursor.seek(p),
        }
    }
}
#[test]
fn large_unselected_tensor_never_read() {
    let mut b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let mut h: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let data = b.split_off(8 + n);
    let start = data.len();
    h["unselected"] =
        json!({"dtype":"U8","shape":[1_000_000_000],"data_offsets":[start,start+1_000_000_000]});
    let bytes = encode(h, data);
    let length = bytes.len() as u64;
    let mut reader = Guarded {
        cursor: Cursor::new(bytes),
        end: length + 1_000_000_000,
        max_read_end: length,
    };
    let result = checkpoint::read(&mut reader, &spec()).unwrap();
    assert_eq!(result.io_bytes, length);
}
#[test]
fn duplicate_tensor_key_rejected() {
    let b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let mut header = serde_json::to_string(&h).unwrap();
    header.pop();
    header.push_str(&format!(
        ",{k}:{v}}}",
        k = serde_json::to_string(W).unwrap(),
        v = h[W]
    ));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.bytes());
    bytes.extend(&b[8 + n..]);
    assert!(checkpoint::read(&mut Cursor::new(bytes), &spec()).is_err());
}

#[test]
fn empty_rank_reads_header_only() {
    let b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap());
    let mut s = spec();
    s.rank = 17;
    s.resident_order.clear();
    s.experts.clear();
    let mut guarded = Guarded {
        end: b.len() as u64,
        cursor: Cursor::new(b),
        max_read_end: 8 + n,
    };
    let imported = checkpoint::read(&mut guarded, &s).unwrap();
    assert!(imported.stack.is_empty());
    assert_eq!(imported.io_bytes, 8 + n);
}

#[test]
fn duplicate_metadata_label_rejected() {
    let b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let header = serde_json::to_string(&h).unwrap().replace(
        r#"{"revision":"rev"}"#,
        r#"{"revision":"other","revision":"rev"}"#,
    );
    assert!(header.contains(r#""revision":"other","revision":"rev""#));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.bytes());
    bytes.extend(&b[8 + n..]);
    assert!(checkpoint::read(&mut Cursor::new(bytes), &spec()).is_err());
}
#[test]
fn payload_cap_precedes_selected_reads() {
    let b = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/checkpoint_qwen36_expert0_full.safetensors"
    ))
    .unwrap();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap());
    let mut s = spec();
    s.expected_metadata.clear();
    s.experts = (0..76)
        .map(|i| Selection {
            id: i,
            local: i,
            source_expert: 0,
        })
        .collect();
    s.registry.copies = (0..76)
        .map(|i| moe::Replica {
            expert: i,
            rank: 2,
            local: i,
        })
        .collect();
    s.resident_order = (0..76).collect();
    let mut guarded = Guarded {
        end: b.len() as u64,
        cursor: Cursor::new(b),
        max_read_end: 8 + n,
    };
    let err = checkpoint::read(&mut guarded, &s).unwrap_err();
    assert_eq!(err.code, "CAPACITY_EXCEEDED");
}
#[test]
fn local_file_api_refuses_directory() {
    // A40: the post-open is_file check refuses a directory too, so is_err() alone
    // held with the pre-open check removed; require the pre-open reason.
    let error = checkpoint::load(&checkpoint::Request {
        path: env!("CARGO_MANIFEST_DIR").into(),
        spec: spec(),
    })
    .unwrap_err();
    assert!(
        error.message.contains("path must be a regular file"),
        "{error}"
    );
}

// A40 guard sweep: each input below violates exactly one header or selection
// clause. Every one of them was accepted, or refused only by a later check, when
// that clause alone was disabled, so each must be refused for its own reason.
const SCALE: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
fn refused(bytes: Vec<u8>, s: &Spec, reason: &str) {
    let error = checkpoint::read(&mut Cursor::new(bytes), s).unwrap_err();
    assert!(
        error.message.contains(reason),
        "expected {reason:?}, got {error}"
    );
}
fn with_header_padding(bytes: &[u8], extra: usize) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header = bytes[8..8 + n].to_vec();
    header.extend(std::iter::repeat_n(b' ', extra));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(&bytes[8 + n..]);
    out
}
#[test]
fn metadata_label_bounds_are_enforced_one_by_one() {
    let b = fixture();
    let label = "duplicate/oversized metadata label";
    refused(
        edit(b.clone(), |h| {
            h["__metadata__"]["x".repeat(257)] = json!("v")
        }),
        &spec(),
        label,
    );
    refused(
        edit(b.clone(), |h| {
            for i in 0..64 {
                h["__metadata__"][format!("k{i}")] = json!("v");
            }
        }),
        &spec(),
        label,
    );
    refused(
        edit(b, |h| h["__metadata__"]["k"] = json!("v".repeat(4097))),
        &spec(),
        "metadata value exceeds bound",
    );
}
#[test]
fn tensor_name_length_and_count_are_bounded() {
    let b = fixture();
    let reason = "duplicate/oversized safetensors name or tensor count";
    refused(
        edit(b.clone(), |h| {
            h["y".repeat(513)] = json!({"dtype":"U8","shape":[0],"data_offsets":[0,0]})
        }),
        &spec(),
        reason,
    );
    refused(
        edit(b, |h| {
            for i in 0..65536 {
                h[format!("t{i}")] = json!({"dtype":"U8","shape":[0],"data_offsets":[0,0]});
            }
        }),
        &spec(),
        reason,
    );
}
#[test]
fn scale_tensors_must_have_their_exact_dtype_and_shape() {
    // Same byte count each time: only the declared layout is wrong.
    let b = fixture();
    let reason = "wrong dtype/shape for";
    refused(
        edit(b.clone(), |h| h[SCALE]["dtype"] = json!("U8")),
        &spec(),
        reason,
    );
    refused(
        edit(b.clone(), |h| h[SCALE]["shape"] = json!([32, 8])),
        &spec(),
        reason,
    );
    refused(
        edit(b, |h| h[SCALE]["shape"] = json!([256])),
        &spec(),
        reason,
    );
}
#[test]
fn packed_weight_must_be_two_dimensional() {
    refused(
        edit(fixture(), |h| h[W]["shape"] = json!([64, 32, 1])),
        &spec(),
        "weight must be packed U8 rows by columns/2",
    );
}
#[test]
fn header_length_is_checked_before_it_is_read() {
    let b = fixture();
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let bound = "checkpoint header exceeds 8 MiB bound or is empty";
    // A syntactically valid header one padding block over 8 MiB.
    refused(
        with_header_padding(&b, 8 * 1024 * 1024 + 8 - n),
        &spec(),
        bound,
    );
    refused([vec![0; 8], b[8..].to_vec()].concat(), &spec(), bound);
    refused(b[..n].to_vec(), &spec(), "truncated checkpoint header");
}
#[test]
fn oversized_selection_is_refused_before_the_file_is_touched() {
    let reason = "checkpoint selection or expected metadata too large";
    let mut s = spec();
    s.experts = vec![s.experts[0].clone(); 1025];
    refused(fixture(), &s, reason);
    let mut s = spec();
    s.expected_metadata = (0..65).map(|i| (format!("k{i}"), "v".into())).collect();
    refused(fixture(), &s, reason);
    let mut s = spec();
    s.expected_metadata = [("k".repeat(257), "v".into())].into_iter().collect();
    refused(fixture(), &s, reason);
    let mut s = spec();
    s.expected_metadata = [("k".into(), "v".repeat(4097))].into_iter().collect();
    refused(fixture(), &s, reason);
}

#[test]
fn a_selected_tensor_absent_from_the_header_is_named_in_the_refusal() {
    // A41 reason sweep: deleting a tensor also breaks the exact-length check, which
    // refused first; renaming it keeps the file valid and leaves only the lookup.
    for missing in [W, SCALE] {
        let bytes = edit(fixture(), |h| {
            let tensor = h.as_object_mut().unwrap().remove(missing).unwrap();
            h[format!("{missing}.renamed")] = tensor;
        });
        let error = checkpoint::read(&mut Cursor::new(bytes), &spec()).unwrap_err();
        assert_eq!(error.message, format!("missing selected tensor {missing}"));
    }
}
