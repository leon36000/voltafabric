use voltafabric::memory::*;
#[test]
fn layout_counts_codes_scales_and_global() {
    let x = nvfp4_layout(3 * 4096, 2048, 1, 0).unwrap();
    assert_eq!(x.codes, 12_582_912);
    assert_eq!(x.scales, 1_572_864);
    assert_eq!(x.total, 14_155_776);
}
#[test]
fn padding_is_per_row_not_whole_tensor() {
    let x = nvfp4_layout(3, 17, 1, 4).unwrap();
    assert_eq!(x.padded_columns, 32);
    assert_eq!(x.codes, 48);
    assert_eq!(x.scales, 6);
    assert_eq!(x.total, 58);
}
#[test]
fn planes_are_aligned_separately() {
    assert_eq!(nvfp4_layout(1, 16, 256, 4).unwrap().total, 768);
}
#[test]
fn invalid_layouts_are_rejected() {
    for (r, c, a) in [
        (0, 16, 1),
        (1, 0, 1),
        (1, 16, 0),
        (1, 16, 3),
        (u64::MAX, 16, 1),
        (1, u64::MAX, 1),
    ] {
        assert!(nvfp4_layout(r, c, a, 0).is_err());
    }
}
#[test]
fn capacity_counts_load_peak_and_all_reservations() {
    let b = Budget {
        capacity: 100,
        weights: 30,
        kv: 10,
        recurrent: 4,
        workspace: 5,
        graphs: 6,
        communication: 7,
        margin: 8,
        load_temporary: 20,
    };
    let r = b.check().unwrap();
    assert_eq!(r.steady_bytes, 70);
    assert_eq!(r.loading_peak_bytes, 90);
    assert_eq!(r.headroom_bytes, 10);
}
#[test]
fn fit_at_runtime_but_not_during_load_is_rejected() {
    let b = Budget {
        capacity: 100,
        weights: 90,
        load_temporary: 20,
        ..Budget::default()
    };
    assert!(b.check().is_err());
}
#[test]
fn capacity_overflow_cannot_wrap() {
    let b = Budget {
        capacity: u64::MAX,
        weights: u64::MAX,
        kv: 1,
        ..Budget::default()
    };
    assert!(b.check().is_err());
}

#[test]
fn zero_capacity_budget_is_refused_even_when_nothing_is_requested() {
    // A40 guard sweep: an all-zero budget fits in zero capacity, so only the
    // positive-capacity clause refuses it.
    let error = Budget::default().check().unwrap_err();
    assert_eq!(error.message, "capacity must be positive");
}

#[test]
fn over_capacity_names_the_loading_peak() {
    // A41 reason sweep: reached before, but only through is_err().
    let error = Budget {
        capacity: 10,
        weights: 8,
        load_temporary: 3,
        ..Budget::default()
    }
    .check()
    .unwrap_err();
    assert_eq!(error.code, "CAPACITY_EXCEEDED");
    assert_eq!(error.message, "loading peak 11 exceeds 10");
}
