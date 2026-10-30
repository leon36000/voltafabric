#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    gdn_cpu::Identity,
    sc117_shared_stage::{Manifest, Stage},
};
#[test]
fn rejects_wrong_identity_type_and_source_without_loading_weights() {
    let make =
        || serde_json::from_str::<Manifest>(include_str!("../fixtures/r424/layer2.json")).unwrap();
    let id = Identity {
        run_id: "run".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 2,
        epoch: 1,
    };
    let mut f = std::fs::File::open("Cargo.toml").unwrap();
    let mut wrong = id.clone();
    wrong.layer = 1;
    assert!(Stage::load(make(), wrong, &mut f).is_err());
    let mut wrong = id.clone();
    wrong.model_revision = "other".into();
    assert!(Stage::load(make(), wrong, &mut f).is_err());
    let mut m = make();
    m.gate_type = 42;
    assert!(Stage::load(m, id.clone(), &mut f).is_err());
    assert!(Stage::load(make(), id, &mut f).is_err());
}
