#![cfg(feature = "c-abi-validation")]
use vf_qpn2_ffi::{DeviceBuffer, DeviceBuffers};
use voltafabric::{checkpoint, launch, moe, routing};

fn request() -> checkpoint::Request {
    let mut r: checkpoint::Request = serde_json::from_str(include_str!(
        "../fixtures/resident_import_qwen_expert0.json"
    ))
    .unwrap();
    r.path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&r.path);
    r
}
fn dispatch() -> routing::RankDispatch {
    let r = request();
    let stamp = moe::Stamp {
        run_id: "ffi".into(),
        model_revision: r.spec.model_revision,
        plan_id: "p".into(),
        layer: 0,
        batch: 8,
        epoch: 1,
    };
    routing::RankDispatch {
        stamp,
        rank: 0,
        hidden: 2048,
        tokens: 2,
        resident_order: vec![900],
        slots: (0..2)
            .map(|token| routing::RankSlot {
                slot: moe::Slot {
                    token,
                    slot: 0,
                    expert: 0,
                    rank: 0,
                    local: 900,
                    weight: if token == 0 { 0.25 } else { 0.75 },
                },
                physical_expert: 0,
            })
            .collect(),
    }
}
fn options() -> launch::Options {
    launch::Options {
        splitk13: 4,
        splitk2: 4,
        max_workspace_bytes: 128 * 1024 * 1024,
    }
}
fn buffer(address: usize, capacity: usize) -> DeviceBuffer {
    DeviceBuffer { address, capacity }
}
fn buffers(down: bool) -> DeviceBuffers {
    let (codes, scales, globals, x, y) = if down {
        (524288, 65536, 64, 1024, 4096)
    } else {
        (1048576, 131072, 32, 4096, 2048)
    };
    DeviceBuffers {
        codes: buffer(if down { 0xa0000000 } else { 0x10000000 }, codes),
        scales: buffer(if down { 0xb0000000 } else { 0x20000000 }, scales),
        gscales: buffer(if down { 0xc0000000 } else { 0x30000000 }, globals),
        x: buffer(if down { 0x80000000 } else { 0x40000000 }, x),
        y: buffer(if down { 0x90000000 } else { 0x50000000 }, y),
        block_expert: buffer(0x60000000, 1),
        sorted_slots: buffer(0x60001000, 8),
        ntpp: buffer(0x60002000, 1),
    }
}
#[test]
fn actual_native_import_plans_validate_through_real_c_abi() {
    let imported = checkpoint::load(&request()).unwrap();
    let d = dispatch();
    let prepared = launch::prepare(&imported.stack, &d, &d.stamp, &options()).unwrap();
    let before = serde_json::to_value(&prepared).unwrap();
    prepared
        .validate_c(&d.stamp, &buffers(false), &buffers(true))
        .unwrap();
    assert_eq!(serde_json::to_value(&prepared).unwrap(), before);
    assert!(std::ptr::eq(prepared.weights(), &imported.stack));
    assert_eq!(prepared.completion_slots()[0].weight, 0.25);
    assert_eq!(prepared.completion_slots()[1].weight, 0.75);
}
#[test]
fn stale_stamp_short_buffers_and_host_output_alias_rejected() {
    let imported = checkpoint::load(&request()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&imported.stack, &d, &d.stamp, &options()).unwrap();
    let mut stamp = d.stamp.clone();
    stamp.epoch += 1;
    assert_eq!(
        p.validate_c(&stamp, &buffers(false), &buffers(true))
            .unwrap_err()
            .code,
        "STALE_EXECUTION"
    );
    for down in [false, true] {
        let mut a = buffers(false);
        let mut b = buffers(true);
        if down {
            b.codes.capacity -= 1;
        } else {
            a.codes.capacity -= 1;
        }
        let err = p.validate_c(&d.stamp, &a, &b).unwrap_err();
        assert_eq!(err.code, "C_ABI_VALIDATION");
        assert!(err.message.contains("-3"));
    }
    let mut a = buffers(false);
    a.y.address = p.sorted_slots().as_ptr() as usize;
    assert!(p
        .validate_c(&d.stamp, &a, &buffers(true))
        .unwrap_err()
        .message
        .contains("-4"));
}
#[test]
fn empty_rank_skips_c_validator_but_still_requires_current_identity() {
    let imported = checkpoint::load(&request()).unwrap();
    let mut d = dispatch();
    d.slots.clear();
    let p = launch::prepare(&imported.stack, &d, &d.stamp, &options()).unwrap();
    assert!(p.skip());
    let zero = || buffer(0, 0);
    let b = DeviceBuffers {
        codes: zero(),
        scales: zero(),
        gscales: zero(),
        x: zero(),
        y: zero(),
        block_expert: zero(),
        sorted_slots: zero(),
        ntpp: zero(),
    };
    p.validate_c(&d.stamp, &b, &b).unwrap();
    let mut old = d.stamp.clone();
    old.batch += 1;
    assert_eq!(
        p.validate_c(&old, &b, &b).unwrap_err().code,
        "STALE_EXECUTION"
    );
}

#[test]
fn layer_outputs_cannot_clobber_other_projection_weights_or_reuse_activation_spans() {
    let imported = checkpoint::load(&request()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&imported.stack, &d, &d.stamp, &options()).unwrap();
    for case in 0..4 {
        let mut a = buffers(false);
        let mut b = buffers(true);
        match case {
            0 => a.y.address = b.scales.address,
            1 => b.y.address = a.gscales.address,
            2 => b.x.address = a.y.address,
            _ => b.y.address = a.x.address,
        }
        // A49 return-Err sweep: each alias passes both per-projection C validators
        // and is refused by the Rust cross-layer check, for that reason.
        let error = p.validate_c(&d.stamp, &a, &b).unwrap_err();
        assert_eq!(error.code, "C_ABI_VALIDATION", "case {case}");
        assert_eq!(
            error.message,
            "four-buffer layer workspace overlaps activation or persistent weight/metadata span: -4",
            "cross-projection alias case {case}"
        );
    }
}

#[test]
fn a_buffer_span_that_wraps_the_address_space_is_refused() {
    // A41 reason sweep: addresses come from the caller, so end = address + length
    // can overflow. The per-projection C validator refuses it (-4) before the Rust
    // cross-layer overlap check, whose own wrap guard is therefore a second line.
    let imported = checkpoint::load(&request()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&imported.stack, &d, &d.stamp, &options()).unwrap();
    let mut a = buffers(false);
    a.y.address = usize::MAX - 1;
    let error = p.validate_c(&d.stamp, &a, &buffers(true)).unwrap_err();
    assert_eq!(error.code, "C_ABI_VALIDATION");
    assert_eq!(error.message, "gate/up rejected by C CPU validator: -4");
}
