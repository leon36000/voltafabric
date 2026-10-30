#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    gdn_cpu::Identity,
    sc117_gdn_stage::{Manifest, Stage},
};
#[test]
fn rejects_wrong_layer_model_and_file_sizes_before_admission() {
    let manifest =
        || serde_json::from_str::<Manifest>(include_str!("../fixtures/r418/layer2.json")).unwrap();
    let id = Identity {
        run_id: "run".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 2,
        epoch: 1,
    };
    let mut dense = std::fs::File::open("Cargo.toml").unwrap();
    let mut shard = std::fs::File::open("Cargo.lock").unwrap();
    let mut wrong = id.clone();
    wrong.layer = 1;
    assert!(Stage::load(manifest(), wrong, &mut dense, &mut shard).is_err());
    let mut wrong = id.clone();
    wrong.model_revision = "other".into();
    assert!(Stage::load(manifest(), wrong, &mut dense, &mut shard).is_err());
    assert!(Stage::load(manifest(), id, &mut dense, &mut shard).is_err());
}
