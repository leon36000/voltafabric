#![cfg(feature = "native-ggml-cpu")]
use std::io::{Read, Seek, SeekFrom};
use voltafabric::{
    native_ggml_cpu::{Binding, Resident},
    strata_native_layout::NativeLayout,
};
const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const BLOB_SHA: &str = "50eff9144c635807dfdc0cafd1c3bb978d0202d537ae892ec8ebed90dc13e172";
const SIZE: u64 = 50_292_326_400;
struct Zeros(u64);
impl Seek for Zeros {
    fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
        self.0 = match s {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (SIZE as i64 + n) as u64,
            SeekFrom::Current(n) => (self.0 as i64 + n) as u64,
        };
        Ok(self.0)
    }
}
impl Read for Zeros {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        b.fill(0);
        self.0 += b.len() as u64;
        Ok(b.len())
    }
}
fn binding() -> Binding {
    Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert: 0,
        local: 0,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: BLOB_SHA.into(),
    }
}
#[test]
fn verified_native_expert_loads_once_and_executes_twice_in_rust() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    let mut resident = Resident::load(&layout, blob, binding()).unwrap();
    assert_eq!(resident.binding().expert, 0);
    let x = vec![0.125; 2560];
    let first = resident.forward(&x).unwrap();
    let second = resident.forward(&x).unwrap();
    assert_eq!(first, vec![0.0; 2560]);
    assert_eq!(first, second);
    assert!(resident.forward(&[f32::NAN; 2560]).is_err());
}
#[test]
fn native_resident_refuses_changed_identity_or_mutated_payload() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    blob.bytes[0] = 1;
    assert!(Resident::load(&layout, blob, binding()).is_err());
    let blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    let mut wrong = binding();
    wrong.expert = 1;
    assert!(Resident::load(&layout, blob, wrong).is_err());
}

struct Pattern {
    pos: u64,
    data: Vec<u8>,
}
impl Seek for Pattern {
    fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
        self.pos = match s {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (SIZE as i64 + n) as u64,
            SeekFrom::Current(n) => (self.pos as i64 + n) as u64,
        };
        Ok(self.pos)
    }
}
impl Read for Pattern {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let start = self.pos as usize;
        if start >= self.data.len() {
            return Ok(0);
        }
        let n = buf.len().min(self.data.len() - start);
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}
#[test]
fn native_owner_processes_nonzero_verified_synthetic_expert() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut data = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        data[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        data[i + 1] = 0x28;
    }
    let digest = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
    let blob = layout
        .read_verified_expert(&mut Pattern { pos: 0, data }, 0, 0, INDEX_SHA, digest)
        .unwrap();
    let mut identity = binding();
    identity.blob_sha256 = digest.into();
    let mut owner = Resident::load(&layout, blob, identity).unwrap();
    let out = owner.forward(&vec![0.125; 2560]).unwrap();
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(out.iter().any(|x| x.abs() > 1e-8));
    assert_eq!(out, owner.forward(&vec![0.125; 2560]).unwrap());
}
