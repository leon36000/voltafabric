use voltafabric::strata_native_layout::NativeLayout;

const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
const SIZE: u64 = 50_292_326_400;

#[test]
fn actual_sc117_index_locates_bounded_mixed_format_experts() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    assert_eq!(layout.layers().len(), 48);
    let first = layout.expert(0, 0).unwrap();
    assert_eq!(
        (
            first.offset,
            first.bytes,
            first.gate_up_type,
            first.down_type
        ),
        (0, 2_176_000, 18, 20)
    );
    let last = layout.expert(47, 511).unwrap();
    assert_eq!(last.offset + last.bytes, SIZE);
    assert_eq!((last.gate_up_type, last.down_type), (23, 20));
    assert!(layout.expert(48, 0).is_err());
    assert!(layout.expert(0, 512).is_err());
}

#[test]
fn malformed_and_inconsistent_indices_fail_before_any_access() {
    assert!(NativeLayout::parse_sc117_v3(INDEX, SIZE - 1).is_err());
    assert!(NativeLayout::parse_sc117_v3(
        &INDEX.replace("0 18 20 0 2176000", "0 39 20 0 2176000"),
        SIZE
    )
    .is_err());
    assert!(NativeLayout::parse_sc117_v3(
        &INDEX.replace("0 18 20 0 2176000", "0 18 20 1 2176000"),
        SIZE
    )
    .is_err());
    assert!(NativeLayout::parse_sc117_v3(
        &INDEX.replace("0 18 20 0 2176000", "0 18 20 0 2176001"),
        SIZE
    )
    .is_err());
    assert!(
        NativeLayout::parse_sc117_v3(&INDEX.replace("experts v3:", "experts v4:"), SIZE).is_err()
    );
    assert!(NativeLayout::parse_sc117_v3(
        &INDEX.lines().take(48).collect::<Vec<_>>().join("\n"),
        SIZE
    )
    .is_err());
    let duplicate = format!("{INDEX}{}\n", INDEX.lines().nth(1).unwrap());
    assert!(NativeLayout::parse_sc117_v3(&duplicate, SIZE).is_err());
}
