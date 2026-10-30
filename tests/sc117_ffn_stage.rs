#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    gdn_cpu::Identity,
    sc117_ffn_stage::{Manifest, Stage},
};
#[test]
fn rejects_wrong_identity_and_source_before_load() {
    let m =
        || serde_json::from_str::<Manifest>(include_str!("../fixtures/r431/layer3.json")).unwrap();
    let id = Identity {
        run_id: "run".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 3,
        epoch: 1,
    };
    let mut d = std::fs::File::open("Cargo.toml").unwrap();
    let mut x = id.clone();
    x.layer = 2;
    assert!(Stage::load(m(), x, &mut d).is_err());
    let mut x = id.clone();
    x.model_revision = "other".into();
    assert!(Stage::load(m(), x, &mut d).is_err());
    assert!(Stage::load(m(), id, &mut d).is_err());
}
