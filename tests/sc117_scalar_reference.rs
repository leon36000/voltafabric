#![cfg(feature = "native-scalar-reference")]
use std::collections::BTreeMap;
use voltafabric::{
    completion::{self},
    moe::{Choice, Collector, Registry, Replica, Stamp},
    native_ggml_cpu::{scalar_reference, Binding, RankWorker, Resident},
    routing::{self, RoutingProblem},
    sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
};
const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
#[test]
fn scalar_reference_preserves_binding_and_collector_weight_once() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, 50_292_326_400).unwrap();
    let span = layout.expert(2, 255).unwrap();
    assert_eq!((span.gate_up_type, span.down_type), (22, 20));
    let mut bytes = vec![0u8; span.bytes as usize];
    let up = 2 * 640 * (2560 / 256) * 82;
    for i in (0..up).step_by(82) {
        bytes[i + 1] = 0x28;
    }
    for i in (up..bytes.len()).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let binding = Binding {
        model_revision: "test-scalar".into(),
        rank: 2,
        layer: 2,
        expert: 255,
        local: 0,
        epoch: 1,
        index_sha256: layout.index_sha256().into(),
        blob_sha256: sha256::hex(&bytes),
    };
    let expert = VerifiedNativeExpert {
        layer: 2,
        expert: 255,
        span,
        bytes,
    };
    let x = vec![0.001_f32; 2560];
    let row = scalar_reference(&layout, &expert, &binding, &x).unwrap();
    assert_eq!(row.len(), 2560);
    assert!(row.iter().any(|v| v.abs() > 1e-12));
    let mut wrong = binding.clone();
    wrong.expert = 254;
    assert!(scalar_reference(&layout, &expert, &wrong, &x).is_err());
    wrong = binding.clone();
    wrong.blob_sha256 = "0".repeat(64);
    assert!(scalar_reference(&layout, &expert, &wrong, &x).is_err());
    let mut invalid = x.clone();
    invalid[0] = f32::NAN;
    assert!(scalar_reference(&layout, &expert, &binding, &invalid).is_err());
    let invalid_expert = VerifiedNativeExpert {
        layer: expert.layer,
        expert: expert.expert,
        span: expert.span.clone(),
        bytes: expert.bytes.clone(),
    };
    let mut invalid_binding = binding.clone();
    invalid_binding.blob_sha256 = "0".repeat(64);
    assert!(Resident::load_scalar(&layout, invalid_expert, invalid_binding).is_err());
    let mut resident = Resident::load_scalar(&layout, expert, binding.clone()).unwrap();
    assert_eq!(resident.binding(), &binding);
    let resident_row = resident.forward(&x).unwrap();
    assert_eq!(resident_row.len(), row.len());
    for (got, expected) in resident_row.iter().zip(&row) {
        assert_eq!(got.to_bits(), (*expected as f32).to_bits());
    }
    assert!(resident.forward(&vec![f32::NAN; 2560]).is_err());
    let stamp = Stamp {
        run_id: "scalar-test".into(),
        model_revision: "test-scalar".into(),
        plan_id: "no-worker".into(),
        layer: 2,
        batch: 0,
        epoch: 1,
    };
    let problem = RoutingProblem {
        hidden: 2560,
        experts: vec![255],
        routes: vec![vec![Choice {
            expert: 255,
            weight: 0.25,
        }]],
        registry: Registry {
            epoch: 1,
            copies: vec![Replica {
                expert: 255,
                rank: 2,
                local: 0,
            }],
        },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0])]),
    };
    let bits: Vec<u16> = row
        .iter()
        .map(|v| completion::encode_fp16(f64::from(*v as f32)).unwrap())
        .collect();
    let dispatch = routing::dispatch(&problem).unwrap();
    let mut worker = RankWorker::new(vec![resident]).unwrap();
    let raw = worker
        .execute(&dispatch.ranks[0], &[(0, x.clone())], 1)
        .unwrap();
    assert_eq!(raw.fp16_bits, bits);
    assert!(worker
        .execute(&dispatch.ranks[0], &[(0, x.clone())], 2)
        .is_err());
    let mut invalid_dispatch = dispatch.ranks[0].clone();
    invalid_dispatch.stamp.batch = 1;
    assert!(worker
        .execute(&invalid_dispatch, &[(0, vec![1e20; 2560])], 2)
        .is_err());
    invalid_dispatch.stamp.batch = 2;
    assert!(worker.execute(&invalid_dispatch, &[(0, x)], 3).is_err());
    let mut collector = Collector::from_routing(&problem).unwrap();
    completion::accept_routed_output(1, raw, &mut collector).unwrap();
    let result = collector.finish().unwrap();
    for (got, bits) in result[0].iter().zip(bits) {
        assert_eq!(*got, completion::decode_fp16(bits).unwrap() * 0.25);
    }
}

#[test]
fn scalar_boundary_accepts_each_observed_format_pair() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, 50_292_326_400).unwrap();
    let mut formats = std::collections::BTreeSet::new();
    for layer in [0, 12, 17, 21, 2, 1, 47] {
        let span = layout.expert(layer, 0).unwrap();
        formats.insert((span.gate_up_type, span.down_type));
        let bytes = vec![0; span.bytes as usize];
        let binding = Binding {
            model_revision: "test-scalar".into(),
            rank: 2,
            layer: layer as u32,
            expert: 0,
            local: 0,
            epoch: 1,
            index_sha256: layout.index_sha256().into(),
            blob_sha256: sha256::hex(&bytes),
        };
        let expert = VerifiedNativeExpert {
            layer,
            expert: 0,
            span,
            bytes,
        };
        let result = scalar_reference(&layout, &expert, &binding, &vec![0.125; 2560]).unwrap();
        assert_eq!(result, vec![0.0; 2560]);
    }
    assert_eq!(
        formats,
        std::collections::BTreeSet::from([
            (18, 20),
            (18, 42),
            (21, 20),
            (21, 42),
            (22, 20),
            (22, 42),
            (23, 20)
        ])
    );
}

#[test]
fn worker_rejects_mixed_numerical_backends() {
    use voltafabric::native_ggml_cpu::Fleet;
    let layout = NativeLayout::parse_sc117_v3(INDEX, 50_292_326_400).unwrap();
    let owner = |expert_id: u32, scalar: bool| {
        let span = layout.expert(2, u64::from(expert_id)).unwrap();
        let bytes = vec![0; span.bytes as usize];
        let binding = Binding {
            model_revision: "backend-test".into(),
            rank: 2,
            layer: 2,
            expert: expert_id,
            local: expert_id,
            epoch: 1,
            index_sha256: layout.index_sha256().into(),
            blob_sha256: sha256::hex(&bytes),
        };
        let expert = VerifiedNativeExpert {
            layer: 2,
            expert: u64::from(expert_id),
            span,
            bytes,
        };
        if scalar {
            Resident::load_scalar(&layout, expert, binding).unwrap()
        } else {
            Resident::load(&layout, expert, binding).unwrap()
        }
    };
    assert!(RankWorker::new(vec![owner(0, true), owner(1, false)]).is_err());
    assert!(RankWorker::new(vec![owner(0, false), owner(1, true)]).is_err());
    assert!(Fleet::new(vec![owner(0, true), owner(1, false)]).is_err());
    assert!(Fleet::new(vec![owner(0, false), owner(1, true)]).is_err());
    assert!(Fleet::new(vec![owner(0, true), owner(1, true)]).is_ok());
    assert!(Fleet::new(vec![owner(0, false), owner(1, false)]).is_ok());
    let mut vector_worker = RankWorker::new(vec![owner(0, false)]).unwrap();
    assert!(vector_worker.add_owner(owner(1, true)).is_err());
    assert_eq!(vector_worker.owner_count(), 1);
    let mut worker = RankWorker::new(vec![owner(0, true)]).unwrap();
    assert!(worker.add_owner(owner(1, false)).is_err());
    assert_eq!(worker.owner_count(), 1);
    assert!(worker.add_owner(owner(1, true)).is_ok());
}
