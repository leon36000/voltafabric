use voltafabric::{moe, quant, resident};

fn plane(rows: u64, columns: u64, seed: u8, factor: f64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: (0..rows * columns / 2)
            .map(|i| (i as u8).wrapping_mul(73).wrapping_add(seed))
            .collect(),
        scales: (0..rows * columns / 16)
            .map(|i| ((i + u64::from(seed)) % 127) as u8)
            .collect(),
        global_dequant: factor,
    }
}
// Coordinate-forward oracle, independent of the implementation's lane traversal.
fn fragment(p: &quant::PackedRows) -> (Vec<u8>, Vec<u8>) {
    let columns = p.columns as usize;
    let groups = columns / 16;
    let k_to_slot = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];
    let mut codes = vec![0; p.codes.len()];
    let mut scales = vec![0; p.scales.len()];
    for row in 0..p.rows as usize {
        let lane = (row % 32 / 8) * 4 + (row % 8 / 4) * 16 + row % 4;
        for column in 0..columns {
            let group = column / 16;
            let slot = k_to_slot[column % 16];
            let source = row * columns + column;
            let value = (p.codes[source / 2] >> (4 * (source % 2))) & 15;
            let target = ((row / 32 * groups + group) * 32 + lane) * 16 + slot;
            codes[target / 2] |= value << (4 * (target % 2));
            if column % 16 == 0 {
                scales[(row / 32 * groups + group) * 32 + lane] = p.scales[row * groups + group];
            }
        }
    }
    (codes, scales)
}

#[test]
fn direct_resident_packing_matches_forward_coordinates_and_identity_permutations() {
    for hidden in [64, 128, 192] {
        for intermediate in [64, 128, 192] {
            for order in [vec![3, 900], vec![900, 3]] {
                let experts: Vec<_> = [(7, 3, 5), (99, 900, 11)]
                    .into_iter()
                    .map(|(id, local, seed)| resident::ResidentExpert {
                        id,
                        local,
                        gate: plane(intermediate, hidden, seed, 0.5),
                        up: plane(intermediate, hidden, seed + 1, 0.25),
                        down: plane(hidden, intermediate, seed + 2, 0.125),
                    })
                    .collect();
                let mut input = resident::ResidentInput {
                    model_revision: "a18-coordinate-oracle".into(),
                    layer: 5,
                    rank: 17,
                    registry: moe::Registry {
                        epoch: 19,
                        copies: experts
                            .iter()
                            .map(|e| moe::Replica {
                                expert: e.id,
                                rank: 17,
                                local: e.local,
                            })
                            .collect(),
                    },
                    resident_order: order,
                    experts,
                };
                let (mut c13, mut s13, mut c2, mut s2, mut g13, mut g2) =
                    (vec![], vec![], vec![], vec![], vec![], vec![]);
                for local in &input.resident_order {
                    let e = input.experts.iter().find(|e| e.local == *local).unwrap();
                    for p in [&e.gate, &e.up] {
                        let (c, s) = fragment(p);
                        c13.extend(c);
                        s13.extend(s);
                        g13.extend(std::iter::repeat_n(
                            (p.global_dequant as f32).to_bits(),
                            (p.rows / 32) as usize,
                        ));
                    }
                    let (c, s) = fragment(&e.down);
                    c2.extend(c);
                    s2.extend(s);
                    g2.extend(std::iter::repeat_n(
                        (e.down.global_dequant as f32).to_bits(),
                        (e.down.rows / 32) as usize,
                    ));
                }
                let actual = resident::pack(&input).unwrap();
                assert_eq!(actual.codes13(), c13);
                assert_eq!(actual.scales13(), s13);
                assert_eq!(actual.codes2(), c2);
                assert_eq!(actual.scales2(), s2);
                assert_eq!(
                    actual
                        .gscales13()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    g13
                );
                assert_eq!(
                    actual
                        .gscales2()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    g2
                );
                assert_eq!(
                    actual.shape(),
                    (2, 2 * intermediate, hidden, hidden, intermediate)
                );
                assert_eq!(
                    (
                        actual.model_revision(),
                        actual.layer(),
                        actual.rank(),
                        actual.epoch()
                    ),
                    ("a18-coordinate-oracle", 5, 17, 19)
                );
                assert_eq!(actual.resident_order(), input.resident_order);
                let expected = serde_json::to_value(&actual).unwrap();
                input.experts.reverse();
                input.registry.copies.reverse();
                assert_eq!(
                    serde_json::to_value(resident::pack(&input).unwrap()).unwrap(),
                    expected
                );
            }
        }
    }
}
