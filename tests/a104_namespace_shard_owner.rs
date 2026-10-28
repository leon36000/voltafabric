//! Backbone and MTP expert identities must remain distinct in a shared shard.
use safetensors::tensor::{serialize, SafeTensors};
use voltafabric::{
    deepseek_v41::{Mxfp4ExpertNamespace, Mxfp4ExpertShard},
    sha256,
};

fn namespace_shard(namespaces: &[&str], omit: Option<(&str, &str)>) -> Vec<u8> {
    let original = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let tensors = SafeTensors::deserialize(original).unwrap();
    let mut entries = Vec::with_capacity(6 * namespaces.len());
    for namespace in namespaces {
        for name in tensors.names() {
            let suffix = name
                .strip_prefix("layers.0.ffn.experts.0.")
                .expect("pinned fixture has one backbone expert prefix");
            if omit == Some((namespace, suffix)) {
                continue;
            }
            entries.push((
                format!("{namespace}.0.ffn.experts.0.{suffix}"),
                tensors.tensor(name).unwrap(),
            ));
        }
    }
    serialize(entries, None).unwrap()
}

#[test]
fn same_layer_and_expert_numbers_are_distinct_between_backbone_and_mtp() {
    let bytes = namespace_shard(&["mtp", "layers"], None);
    let digest = sha256::hex(&bytes);
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
            .unwrap();

    assert_eq!(shard.expert_count(), 2);
    assert_eq!(shard.backing_bytes(), bytes.len());
    assert_eq!(shard.artifact_sha256(), digest);

    let backbone = shard.expert(0, 0).unwrap();
    let mtp = shard.expert_in(Mxfp4ExpertNamespace::Mtp, 0, 0).unwrap();
    assert_eq!(backbone.namespace(), Mxfp4ExpertNamespace::Backbone);
    assert_eq!(mtp.namespace(), Mxfp4ExpertNamespace::Mtp);
    assert_eq!((backbone.layer(), backbone.expert()), (0, 0));
    assert_eq!((mtp.layer(), mtp.expert()), (0, 0));
    assert!(std::ptr::eq(
        backbone,
        shard
            .expert_in(Mxfp4ExpertNamespace::Backbone, 0, 0)
            .unwrap()
    ));
    assert!(shard.expert_in(Mxfp4ExpertNamespace::Mtp, 1, 0).is_err());
    assert!(shard
        .expert_in(Mxfp4ExpertNamespace::Backbone, 0, 1)
        .is_err());
}

#[test]
fn mtp_only_shard_has_no_backbone_alias_and_rejects_incomplete_groups() {
    let bytes = namespace_shard(&["mtp"], None);
    let digest = sha256::hex(&bytes);
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
            .unwrap();
    assert_eq!(shard.expert_count(), 1);
    assert!(shard.expert(0, 0).is_err());
    assert_eq!(
        shard
            .expert_in(Mxfp4ExpertNamespace::Mtp, 0, 0)
            .unwrap()
            .namespace(),
        Mxfp4ExpertNamespace::Mtp
    );

    let incomplete = namespace_shard(&["mtp"], Some(("mtp", "w2.scale")));
    let digest = sha256::hex(&incomplete);
    let error = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        incomplete.clone(),
        &digest,
        incomplete.len(),
    )
    .unwrap_err();
    assert!(error.message.contains("missing tensor w2.scale"));
}

#[test]
fn unknown_or_noncanonical_expert_namespaces_are_not_silently_ignored() {
    let fixture = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let tensors = SafeTensors::deserialize(fixture).unwrap();
    let mut entries = Vec::new();
    for name in tensors.names() {
        let suffix = name.strip_prefix("layers.0.ffn.experts.0.").unwrap();
        entries.push((
            format!("other.0.ffn.experts.0.{suffix}"),
            tensors.tensor(name).unwrap(),
        ));
    }
    let unknown = serialize(entries, None).unwrap();
    let digest = sha256::hex(&unknown);
    assert!(Mxfp4ExpertShard::from_safetensors_shard_artifact(
        unknown.clone(),
        &digest,
        unknown.len()
    )
    .is_err());

    let mut noncanonical_entries = Vec::new();
    for name in tensors.names() {
        let suffix = name.strip_prefix("layers.0.ffn.experts.0.").unwrap();
        noncanonical_entries.push((
            format!("mtp.00.ffn.experts.0.{suffix}"),
            tensors.tensor(name).unwrap(),
        ));
    }
    let noncanonical = serialize(noncanonical_entries, None).unwrap();
    let digest = sha256::hex(&noncanonical);
    assert!(Mxfp4ExpertShard::from_safetensors_shard_artifact(
        noncanonical.clone(),
        &digest,
        noncanonical.len()
    )
    .is_err());
}
