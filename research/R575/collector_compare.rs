//! Isolated R428 slot contribution through the actual unweighted FP16 Collector.
use std::{collections::BTreeMap, error::Error};
use voltafabric::{
    completion::{self, OutputSlot, RoutedRawOutput},
    moe::{Choice, Collector, Registry, Replica, Stamp},
    routing::RoutingProblem,
    sha256,
};
const H: usize = 2560;
const IDS: [u32; 10] = [75, 12, 209, 181, 257, 272, 480, 159, 492, 255];
const WEIGHTS: [f64; 10] = [
    0.3420320749282837,
    0.1739722639322281,
    0.09913824498653412,
    0.08050233870744705,
    0.06733449548482895,
    0.061283160001039505,
    0.05222782492637634,
    0.042672742158174515,
    0.04069479927420616,
    0.04014204069972038,
];
fn row(path: &str) -> Result<(Vec<f32>, String), Box<dyn Error>> {
    let b = std::fs::read(path)?;
    if b.len() != H * 4 {
        return Err("wrong row size".into());
    }
    let hash = sha256::hex(&b);
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if v.iter().any(|v| !v.is_finite()) {
        return Err("non-finite row".into());
    }
    Ok((v, hash))
}
fn collect(v: &[f32]) -> Result<(Vec<u16>, Vec<f64>), Box<dyn Error>> {
    let stamp = Stamp {
        run_id: "r575-collector-isolation".into(),
        model_revision: "sc117-local-observation".into(),
        plan_id: "r575-no-worker".into(),
        layer: 2,
        batch: 0,
        epoch: 1,
    };
    let problem = RoutingProblem {
        hidden: H,
        experts: IDS.to_vec(),
        routes: vec![IDS
            .iter()
            .zip(WEIGHTS)
            .map(|(&expert, weight)| Choice { expert, weight })
            .collect()],
        registry: Registry {
            epoch: 1,
            copies: IDS
                .iter()
                .enumerate()
                .map(|(i, &expert)| Replica {
                    expert,
                    rank: 2,
                    local: i as u32,
                })
                .collect(),
        },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, (0..10).collect())]),
    };
    let bits: Vec<u16> = v
        .iter()
        .map(|&x| completion::encode_fp16(f64::from(x)))
        .collect::<voltafabric::Result<_>>()?;
    let mut rows = vec![0; 9 * H];
    rows.extend_from_slice(&bits);
    let slots = IDS
        .iter()
        .enumerate()
        .map(|(slot, &expert)| OutputSlot {
            token: 0,
            slot,
            expert,
            local: slot as u32,
        })
        .collect();
    let raw = RoutedRawOutput {
        stamp,
        rank: 2,
        attempt: 1,
        rows: 10,
        columns: H,
        slots,
        fp16_bits: rows,
    };
    let mut collector = Collector::from_routing(&problem)?;
    completion::accept_routed_output(1, raw, &mut collector)?;
    Ok((bits, collector.finish()?.remove(0)))
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: collector_compare PC1_F32 PC6_F32".into());
    }
    let (one, one_hash) = row(&a[1])?;
    let (six, six_hash) = row(&a[2])?;
    let (b1, c1) = collect(&one)?;
    let (b6, c6) = collect(&six)?;
    let mut raw_max = 0f64;
    let mut collect_max = 0f64;
    let mut fp16_diff = 0;
    let mut collection_diff = 0;
    let mut oracle_max = 0f64;
    for i in 0..H {
        raw_max = raw_max.max((f64::from(one[i]) - f64::from(six[i])).abs());
        fp16_diff += usize::from(b1[i] != b6[i]);
        collection_diff += usize::from(c1[i].to_bits() != c6[i].to_bits());
        collect_max = collect_max.max((c1[i] - c6[i]).abs());
        oracle_max = oracle_max
            .max((c1[i] - completion::decode_fp16(b1[i])? * WEIGHTS[9]).abs())
            .max((c6[i] - completion::decode_fp16(b6[i])? * WEIGHTS[9]).abs());
    }
    if oracle_max != 0.0 {
        return Err("coefficient/order check failed".into());
    }
    println!("pc1_sha={one_hash} pc6_sha={six_hash} raw_max={raw_max:.17} fp16_different={fp16_diff} collected_different={collection_diff} collected_max={collect_max:.17} oracle_max={oracle_max:.17}");
    Ok(())
}
