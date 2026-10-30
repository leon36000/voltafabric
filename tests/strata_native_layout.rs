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

struct SparseSynthetic {
    pos: u64,
    len: u64,
    calls: usize,
    short_at: Option<u64>,
}
impl std::io::Seek for SparseSynthetic {
    fn seek(&mut self, whence: std::io::SeekFrom) -> std::io::Result<u64> {
        let p = match whence {
            std::io::SeekFrom::Start(n) => i128::from(n),
            std::io::SeekFrom::End(n) => i128::from(self.len) + i128::from(n),
            std::io::SeekFrom::Current(n) => i128::from(self.pos) + i128::from(n),
        };
        self.pos =
            u64::try_from(p).map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
        Ok(self.pos)
    }
}
impl std::io::Read for SparseSynthetic {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.calls += 1;
        let end = self.short_at.unwrap_or(self.len).min(self.len);
        let n = buf.len().min(end.saturating_sub(self.pos) as usize);
        buf[..n].fill(0);
        self.pos += n as u64;
        Ok(n)
    }
}

#[test]
fn native_reader_accepts_only_pinned_index_and_complete_digest_verified_blob() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let index_sha = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
    let blob_sha = "50eff9144c635807dfdc0cafd1c3bb978d0202d537ae892ec8ebed90dc13e172";
    let mut source = SparseSynthetic {
        pos: 0,
        len: SIZE,
        calls: 0,
        short_at: None,
    };
    assert!(layout
        .read_verified_expert(&mut source, 0, 0, &"0".repeat(64), blob_sha)
        .is_err());
    assert_eq!(source.calls, 0);
    let expert = layout
        .read_verified_expert(&mut source, 0, 0, index_sha, blob_sha)
        .unwrap();
    assert_eq!(expert.span.bytes, 2_176_000);
    assert_eq!(expert.bytes.len(), 2_176_000);
    assert!(expert.bytes.iter().all(|x| *x == 0));
    assert!(layout
        .read_verified_expert(&mut source, 0, 0, index_sha, &"0".repeat(64))
        .is_err());
    let mut short = SparseSynthetic {
        pos: 0,
        len: SIZE,
        calls: 0,
        short_at: Some(1),
    };
    assert!(layout
        .read_verified_expert(&mut short, 0, 0, index_sha, blob_sha)
        .is_err());
    let mut wrong_size = SparseSynthetic {
        pos: 0,
        len: SIZE - 1,
        calls: 0,
        short_at: None,
    };
    assert!(layout
        .read_verified_expert(&mut wrong_size, 0, 0, index_sha, blob_sha)
        .is_err());
    assert_eq!(wrong_size.calls, 0);
}
