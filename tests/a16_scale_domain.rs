use std::process::{Command, Stdio};

use voltafabric::{checkpoint, launch, moe, quant, resident, routing};

fn plane(rows: u64, columns: u64, code: u8, scale: u8, factor: f64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: vec![code; (rows * columns / 2) as usize],
        scales: vec![scale; (rows * columns / 16) as usize],
        global_dequant: factor,
    }
}

fn weights(scale: u8, factor: f64) -> resident::ResidentInput {
    resident::ResidentInput {
        model_revision: "m".into(),
        layer: 3,
        rank: 17,
        registry: moe::Registry {
            epoch: 12,
            copies: vec![
                moe::Replica {
                    expert: 23,
                    rank: 17,
                    local: 3,
                },
                moe::Replica {
                    expert: 99,
                    rank: 17,
                    local: 900,
                },
            ],
        },
        resident_order: vec![900, 3],
        experts: vec![
            resident::ResidentExpert {
                id: 23,
                local: 3,
                gate: plane(128, 64, 0x22, scale, factor),
                up: plane(128, 64, 0x22, scale, factor),
                down: plane(64, 128, 0x22, scale, factor),
            },
            resident::ResidentExpert {
                id: 99,
                local: 900,
                gate: plane(128, 64, 0x22, scale, factor),
                up: plane(128, 64, 0x22, scale, factor),
                down: plane(64, 128, 0x22, scale, factor),
            },
        ],
    }
}

fn dispatch() -> routing::RankDispatch {
    let stamp = moe::Stamp {
        run_id: "r".into(),
        model_revision: "m".into(),
        plan_id: "p".into(),
        layer: 3,
        batch: 4,
        epoch: 12,
    };
    let slots = [
        (0, 0, 23, 3, 1, 0.25),
        (0, 1, 99, 900, 0, 0.0),
        (1, 0, 99, 900, 0, 1.0),
        (1, 1, 23, 3, 1, 2.0),
    ]
    .into_iter()
    .map(
        |(token, slot, expert, local, physical_expert, weight)| routing::RankSlot {
            slot: moe::Slot {
                token,
                slot,
                expert,
                rank: 17,
                local,
                weight,
            },
            physical_expert,
        },
    )
    .collect();
    routing::RankDispatch {
        stamp,
        rank: 17,
        hidden: 64,
        tokens: 2,
        resident_order: vec![900, 3],
        slots,
    }
}

fn options() -> launch::Options {
    launch::Options {
        splitk13: 4,
        splitk2: 8,
        max_workspace_bytes: 128 * 1024 * 1024,
    }
}

fn resident_launch_cli(scale: u8, factor: f64) -> std::process::Output {
    let input = weights(scale, factor);
    for expert in &input.experts {
        for p in [&expert.gate, &expert.up, &expert.down] {
            assert!(quant::dequantize(p).is_ok());
        }
    }
    let d = dispatch();
    let stamp = d.stamp.clone();
    let request = serde_json::json!({
        "weights": input,
        "dispatch": d,
        "expected_stamp": stamp,
        "options": options(),
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["resident-launch", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &request).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn resident_launch_cli_accepts_finite_control() {
    let output = resident_launch_cli(0x38, 1.0);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn resident_launch_cli_rejects_scale_one_times_factor_four() {
    // Exact kernel order: RN-half(gscale * 16384), then RN-half(E4M3_scale * gm).
    // 0x38 decodes to 1: factor 4 makes gm overflow half although final FP4
    // values (max 6 * 1 * 4 = 24) are finite.
    let output = resident_launch_cli(0x38, 4.0);
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("SM70_SCALE_RANGE"), "{text}");
}

#[test]
fn resident_launch_cli_rejects_scale_448_times_factor_half() {
    // 0x7e decodes to 448: factor 0.5 gives gm=8192, then 448*8192
    // overflows half even though the largest FP4 result (224) is finite.
    let output = resident_launch_cli(0x7e, 0.5);
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("SM70_SCALE_RANGE"), "{text}");
}

#[test]
fn imported_qwen_expert0_admits_all_96_scale_tiles() {
    let request: checkpoint::Request = serde_json::from_str(include_str!(
        "../fixtures/resident_import_qwen_expert0.json"
    ))
    .unwrap();
    let imported = checkpoint::load(&request).unwrap();
    let stack = imported.stack;

    assert_eq!(stack.gscales13().len(), 32, "16 gate + 16 up tiles");
    assert_eq!(stack.gscales2().len(), 64, "down tiles");
    let gate_up_factor = 3.215_245e-5_f32;
    let down_factor = 3.233_41e-5_f32;
    assert!(stack
        .gscales13()
        .iter()
        .all(|v| v.to_bits() == gate_up_factor.to_bits()));
    assert!(stack
        .gscales2()
        .iter()
        .all(|v| v.to_bits() == down_factor.to_bits()));

    quant::validate_sm70_scale_domain(stack.scales13(), stack.gscales13(), 2048).unwrap();
    quant::validate_sm70_scale_domain(stack.scales2(), stack.gscales2(), 512).unwrap();
}

#[test]
fn zero_local_scales_and_underflowed_positive_global_are_admitted() {
    let bytes_per_tile = (64 / 16 * 32) as usize;
    let zero_local = vec![0_u8; bytes_per_tile];
    quant::validate_sm70_scale_domain(&zero_local, &[1.0], 64).unwrap();

    // The smallest positive F32 times 16384 underflows to binary32 zero;
    // the kernel's subsequent half intermediate therefore remains finite.
    let largest_local = vec![0x7e_u8; bytes_per_tile];
    quant::validate_sm70_scale_domain(&largest_local, &[f32::from_bits(1)], 64).unwrap();
}
