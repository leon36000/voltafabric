#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    gdn_cpu::Identity,
    sc117_head_cpu::{Head, Manifest},
};
#[test]
fn rejects_wrong_model_layer_geometry_and_source_before_load() {
    let make =
        || serde_json::from_str::<Manifest>(include_str!("../fixtures/r429/head.json")).unwrap();
    let id = Identity {
        run_id: "run".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 48,
        epoch: 1,
    };
    let mut d = std::fs::File::open("Cargo.toml").unwrap();
    let mut s = std::fs::File::open("Cargo.lock").unwrap();
    let mut wrong = id.clone();
    wrong.layer = 47;
    assert!(Head::load(make(), wrong, &mut d, &mut s).is_err());
    let mut wrong = id.clone();
    wrong.model_revision = "wrong".into();
    assert!(Head::load(make(), wrong, &mut d, &mut s).is_err());
    let mut m = make();
    m.vocab = 1;
    assert!(Head::load(m, id.clone(), &mut d, &mut s).is_err());
    let mut m = make();
    m.chunks[1].offset += 1;
    let err = match Head::load(m, id.clone(), &mut d, &mut s) {
        Ok(_) => panic!("noncontiguous chunk accepted"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("chunk geometry"));
    assert!(Head::load(make(), id, &mut d, &mut s).is_err());
}
