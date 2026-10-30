use std::io::{Seek, SeekFrom, Write};
use voltafabric::sc117_ple::{self, Table};

// Golden rows from the independently generated Strata ple_oracle_vectors.inc,
// MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
#[test]
fn hash_matches_window_direction_eos_and_zero_golden_vectors() {
    assert_eq!(
        sc117_ple::rows(11, [-1, -1]).unwrap(),
        [
            13555483, 24877159, 54278900, 63631058, 98394898, 117447587, 136774765, 157274238,
            173487882, 192109366, 213364329, 236577563, 258758149, 262164274, 283572184, 312653920
        ]
    );
    assert_eq!(
        sc117_ple::rows(8, [7, 248044]).unwrap(),
        [
            7508793, 29969020, 54939975, 74089003, 91535754, 104816064, 125484693, 154439441,
            164805529, 181662903, 214193203, 238193346, 259492009, 262016223, 296463606, 310742083
        ]
    );
    assert_eq!(
        sc117_ple::rows(100, [0, 0]).unwrap(),
        [
            223356, 29869136, 44693816, 65450370, 83243562, 109175069, 138072664, 149936704,
            175869089, 193667153, 214433731, 233779437, 242682975, 271587098, 294555343, 309396483
        ]
    );
    assert_eq!(
        sc117_ple::rows(248044, [5, 248044]).unwrap(),
        [
            9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725,
            170054832, 192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548
        ]
    );
    assert!(sc117_ple::rows(-1, [-1, -1]).is_err());
    assert!(sc117_ple::rows(0, [-2, -1]).is_err());
    assert!(sc117_ple::rows(sc117_ple::VOCAB, [-1, -1]).is_err());
}

#[test]
fn iq4nl_decoding_uses_split_halves_and_rejects_nonfinite_scales() {
    let mut row = [0u8; 90];
    for block in row.chunks_exact_mut(18) {
        block[0] = 0;
        block[1] = 0x3c; // half 1.0
        block[2..].fill(0xf0); // lower nibble -127, upper nibble 113
    }
    let out = sc117_ple::decode_row(&row).unwrap();
    for block in out.chunks_exact(32) {
        assert_eq!(&block[..16], &[-127.0; 16]);
        assert_eq!(&block[16..], &[113.0; 16]);
    }
    row[1] = 0x7c;
    assert!(sc117_ple::decode_row(&row).is_err());
    assert!(sc117_ple::decode_row(&row[..89]).is_err());
}

#[cfg(unix)]
#[test]
fn sparse_table_reads_only_selected_rows_and_refuses_modified_source() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r441-ple-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    let mut writer = std::fs::File::create(&path).unwrap();
    let size = 192 + sc117_ple::TABLE_ROWS * sc117_ple::ROW_BYTES as u64;
    writer.set_len(size).unwrap();
    let selected = sc117_ple::rows(11, [-1, -1]).unwrap();
    let mut row = [0u8; 90];
    for block in row.chunks_exact_mut(18) {
        block[1] = 0x3c;
        block[2..].fill(0xf0);
    }
    for &idx in &selected {
        writer.seek(SeekFrom::Start(192 + idx as u64 * 90)).unwrap();
        writer.write_all(&row).unwrap();
    }
    writer.sync_all().unwrap();
    let table = Table::open(std::fs::File::open(&path).unwrap(), size, 192).unwrap();
    let (actual, embedding) = table.gather(11, [-1, -1]).unwrap();
    assert_eq!(actual, selected);
    assert_eq!(embedding.len(), 2560);
    for head in embedding.chunks_exact(160) {
        assert_eq!(head[0], -127.0);
        assert_eq!(head[16], 113.0);
    }
    writer.seek(SeekFrom::Start(0)).unwrap();
    writer.write_all(b"x").unwrap();
    writer.sync_all().unwrap();
    assert!(table.gather(11, [-1, -1]).is_err());
    drop(table);
    drop(writer);
    std::fs::remove_file(path).unwrap();
}
