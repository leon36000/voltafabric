use voltafabric::sc117_embedding::{decode_row, ROW_BYTES, WIDTH};
#[test]
fn layout_scale_nibbles_and_codebook() {
    let mut row = vec![0u8; ROW_BYTES];
    for block in row.chunks_exact_mut(136) {
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0xaaaau16.to_le_bytes()); // high bits 2 for all groups
        block[4..8].fill(0x21); // low 1,2 alternating => scales 1,2
        block[8..].fill(0xf0); // code 0 in first half, 15 in second
    }
    let out = decode_row(&row).unwrap();
    assert_eq!(out.len(), WIDTH);
    assert_eq!(
        (out[0], out[16], out[32], out[48]),
        (-127.0, 113.0, -254.0, 226.0)
    );
    assert!(decode_row(&row[..ROW_BYTES - 1]).is_err());
    row[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(decode_row(&row).is_err());
}
#[cfg(unix)]
#[test]
fn pinned_table_checks_bounds_and_mutations() {
    use std::{
        fs::OpenOptions,
        io::{Seek, SeekFrom, Write},
        time::{SystemTime, UNIX_EPOCH},
    };
    use voltafabric::sc117_embedding::{Table, VOCAB};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r452-embed-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let size = u64::from(VOCAB) * ROW_BYTES as u64;
    f.set_len(size).unwrap();
    let table = Table::open(f.try_clone().unwrap(), size, 0).unwrap();
    assert_eq!(table.gather(0).unwrap(), vec![0.0; WIDTH]);
    assert!(table.gather(VOCAB).is_err());
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(&[1]).unwrap();
    f.sync_all().unwrap();
    assert!(table.gather(0).is_err());
    drop(table);
    drop(f);
    std::fs::remove_file(path).unwrap();
}
