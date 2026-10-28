//! CPU contract for loading several compressed experts from one Safetensors backing.
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
use serde_json::Value;
use voltafabric::{deepseek_v41::Mxfp4ExpertShard, sha256};

fn synthetic_two_expert_shard() -> Vec<u8> {
    synthetic_shard(&["1", "0"], None)
}

fn synthetic_shard(expert_ids: &[&str], omitted_suffix: Option<&str>) -> Vec<u8> {
    let original = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let tensors = SafeTensors::deserialize(original).unwrap();
    let names = tensors.names();
    assert_eq!(names.len(), 6);
    let mut entries = Vec::with_capacity(names.len() * expert_ids.len() + 1);
    for expert in expert_ids {
        for name in &names {
            let suffix = name
                .strip_prefix("layers.0.ffn.experts.0.")
                .expect("pinned fixture names have one exact expert prefix");
            if omitted_suffix == Some(suffix) {
                continue;
            }
            entries.push((
                format!("layers.0.ffn.experts.{expert}.{suffix}"),
                tensors.tensor(name).unwrap(),
            ));
        }
    }
    // A mixed real shard may contain non-MoE tensors; the expert owner ignores these.
    let ancillary = [0u8; 4];
    entries.push((
        "layers.0.ffn.router.weight".to_owned(),
        TensorView::new(Dtype::F32, vec![1], &ancillary).unwrap(),
    ));
    serialize(entries, None).unwrap()
}

fn append_fixture_expert(
    entries: &mut Vec<(String, TensorView<'static>)>,
    fixture: &'static [u8],
    source_expert: u32,
    target_expert: u32,
) {
    let tensors = SafeTensors::deserialize(fixture).unwrap();
    let source_prefix = format!("layers.0.ffn.experts.{source_expert}.");
    for name in tensors.names() {
        let suffix = name
            .strip_prefix(&source_prefix)
            .expect("fixture contains only its declared expert tensors");
        entries.push((
            format!("layers.0.ffn.experts.{target_expert}.{suffix}"),
            tensors.tensor(name).unwrap(),
        ));
    }
}

fn real_two_expert_shard(alias_expert383_to_expert0: bool) -> Vec<u8> {
    let expert0 = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let expert383 = if alias_expert383_to_expert0 {
        expert0.as_slice()
    } else {
        include_bytes!("../fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors").as_slice()
    };
    let source_expert383 = if alias_expert383_to_expert0 { 0 } else { 383 };
    let mut entries: Vec<(String, TensorView<'static>)> = Vec::with_capacity(12);
    append_fixture_expert(&mut entries, expert0, 0, 0);
    append_fixture_expert(&mut entries, expert383, source_expert383, 383);
    serialize(entries, None).unwrap()
}

fn golden_input_and_output() -> (Vec<f32>, Vec<f32>, f32) {
    let golden: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"
    ))
    .unwrap();
    let input = golden["input_f32_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| {
            let word = word.as_str().unwrap();
            let (negative, magnitude) = word
                .strip_prefix('-')
                .map_or((false, word), |magnitude| (true, magnitude));
            let bits = u32::from_str_radix(magnitude, 16).unwrap();
            f32::from_bits(if negative { bits.wrapping_neg() } else { bits })
        })
        .collect();
    let expected = golden["output_f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let weight_bits =
        u32::from_str_radix(golden["routing_weight_f32_bits"].as_str().unwrap(), 16).unwrap();
    (input, expected, f32::from_bits(weight_bits))
}

fn pinned_expert_golden(expert_id: u32) -> (Vec<f32>, Vec<f32>, f32, String) {
    let data: &[u8] = match expert_id {
        0 => include_bytes!("../fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"),
        383 => include_bytes!("../fixtures/deepseek_v41_expert383_forward_bf16_314113f8.json"),
        _ => panic!("no pinned expert golden for expert {expert_id}"),
    };
    let golden: Value = serde_json::from_slice(data).unwrap();
    if let Some(golden_expert_id) = golden["expert_id"].as_u64() {
        assert_eq!(golden_expert_id, expert_id as u64);
    } else {
        assert_eq!(
            expert_id, 0,
            "new expert goldens must record their identity"
        );
    }
    let input = golden["input_f32_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| {
            let word = word.as_str().unwrap();
            let (negative, magnitude) = word
                .strip_prefix('-')
                .map_or((false, word), |magnitude| (true, magnitude));
            let bits = u32::from_str_radix(magnitude, 16).unwrap();
            f32::from_bits(if negative { bits.wrapping_neg() } else { bits })
        })
        .collect();
    let expected = golden["output_f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let weight_bits =
        u32::from_str_radix(golden["routing_weight_f32_bits"].as_str().unwrap(), 16).unwrap();
    (
        input,
        expected,
        f32::from_bits(weight_bits),
        golden["artifact_sha256"].as_str().unwrap().to_owned(),
    )
}

#[test]
fn two_experts_from_one_artifact_match_the_pinned_oracle_in_reversed_id_order() {
    let bytes = synthetic_two_expert_shard();
    let digest = sha256::hex(&bytes);
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
            .unwrap();
    assert_eq!(shard.expert_count(), 2);
    assert_eq!(shard.backing_bytes(), bytes.len());
    assert_eq!(shard.artifact_sha256(), digest);

    let (input, expected, routing_weight) = golden_input_and_output();
    let expert1 = shard.expert(0, 1).unwrap();
    let expert0 = shard.expert(0, 0).unwrap();
    assert_eq!((expert1.layer(), expert1.expert()), (0, 1));
    assert_eq!((expert0.layer(), expert0.expert()), (0, 0));
    for expert in [&expert0, &expert1] {
        let output = expert
            .forward_reference(&input, Some(routing_weight))
            .unwrap();
        assert_eq!(output.len(), expected.len());
        let max_abs = output
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 2.0e-5, "oracle max abs error {max_abs}");
    }
}

#[test]
fn distinct_official_expert_ids_do_not_alias_the_same_weights() {
    let bytes = real_two_expert_shard(false);
    let digest = sha256::hex(&bytes);
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
            .unwrap();
    let (input, expected0, routing_weight0, fixture_sha0) = pinned_expert_golden(0);
    let (input383, expected383, routing_weight383, fixture_sha383) = pinned_expert_golden(383);
    assert_eq!(input, input383);
    assert_eq!(routing_weight0, routing_weight383);
    let fixture0 = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let fixture383 =
        include_bytes!("../fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors");
    assert_eq!(sha256::hex(fixture0), fixture_sha0);
    assert_eq!(sha256::hex(fixture383), fixture_sha383);
    let output0 = shard
        .expert(0, 0)
        .unwrap()
        .forward_reference(&input, Some(routing_weight0))
        .unwrap();
    let output383 = shard
        .expert(0, 383)
        .unwrap()
        .forward_reference(&input, Some(routing_weight383))
        .unwrap();

    for (expert_id, output, expected) in
        [(0, &output0, &expected0), (383, &output383, &expected383)]
    {
        assert_eq!(output.len(), expected.len(), "expert {expert_id}");
        let max_abs = output
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 2.0e-5,
            "expert {expert_id} oracle max abs {max_abs}"
        );
    }
    assert_ne!(
        output0, output383,
        "official experts 0 and 383 must remain distinct"
    );
}

#[test]
fn shard_owner_checks_digest_budget_and_missing_expert_identity() {
    let bytes = synthetic_two_expert_shard();
    let digest = sha256::hex(&bytes);
    assert_eq!(
        Mxfp4ExpertShard::from_safetensors_shard_artifact(
            bytes.clone(),
            &"0".repeat(64),
            bytes.len(),
        )
        .unwrap_err()
        .code,
        "ARTIFACT_SHA256_MISMATCH"
    );
    assert!(Mxfp4ExpertShard::from_safetensors_shard_artifact(
        bytes.clone(),
        &digest,
        bytes.len() - 1,
    )
    .is_err());
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes.clone(), &digest, bytes.len())
            .unwrap();
    assert!(shard.expert(1, 0).is_err());
    assert!(shard.expert(0, 2).is_err());
    assert!(shard.expert(0, 0).is_ok());
    assert!(shard.expert(0, 1).is_ok());
}

#[test]
fn shard_owner_rejects_partial_and_noncanonical_expert_groups() {
    let partial = synthetic_shard(&["0"], Some("w2.scale"));
    let partial_digest = sha256::hex(&partial);
    let error = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        partial.clone(),
        &partial_digest,
        partial.len(),
    )
    .unwrap_err();
    assert!(error.message.contains("missing tensor w2.scale"));

    let aliases = synthetic_shard(&["0", "00"], None);
    let aliases_digest = sha256::hex(&aliases);
    let error = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        aliases.clone(),
        &aliases_digest,
        aliases.len(),
    )
    .unwrap_err();
    assert!(error.message.contains("canonical decimal spelling"));
}

#[test]
fn shard_owner_rejects_empty_or_unrelated_only_artifacts() {
    assert!(
        Mxfp4ExpertShard::from_safetensors_shard_artifact(Vec::new(), &"0".repeat(64), 1).is_err()
    );
    let unrelated = synthetic_shard(&[], None);
    let digest = sha256::hex(&unrelated);
    let error = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        unrelated.clone(),
        &digest,
        unrelated.len(),
    )
    .unwrap_err();
    assert!(error.message.contains("no recognized expert tensors"));
}
