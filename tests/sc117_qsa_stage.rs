#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    gdn_cpu::Identity,
    sc117_qsa_stage::{Manifest, Stage},
};
#[test]
fn reject_wrong_layer_model_and_sources() {
    let manifest =
        || serde_json::from_str::<Manifest>(include_str!("../fixtures/r430/layer3.json")).unwrap();
    let id = Identity {
        run_id: "run".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 3,
        epoch: 1,
    };
    let mut d = std::fs::File::open("Cargo.toml").unwrap();
    let mut s = std::fs::File::open("Cargo.lock").unwrap();
    let mut bad = id.clone();
    bad.layer = 2;
    assert!(Stage::load(manifest(), bad, &mut d, &mut s, 2).is_err());
    let mut bad = id.clone();
    bad.model_revision = "wrong".into();
    assert!(Stage::load(manifest(), bad, &mut d, &mut s, 2).is_err());
    assert!(Stage::load(manifest(), id, &mut d, &mut s, 2).is_err());
}
