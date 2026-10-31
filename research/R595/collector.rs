//! Three captured SC117 experts, seven zero rows, actual Rust Collector.
use std::{collections::BTreeMap, error::Error};
use voltafabric::{
    completion::{self, OutputSlot, RoutedRawOutput},
    moe::{Choice, Collector, Registry, Replica, Stamp},
    native_ggml_cpu::{Binding, Resident},
    routing::RoutingProblem,
    sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
};
const H: usize = 2560;
const IDS: [u32; 10] = [75, 12, 209, 181, 257, 272, 480, 159, 492, 255];
const W: [f64; 10] = [
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
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("expected fresh output file".into());
    }
    let fixture = std::fs::read_to_string("fixtures/strata_sc117_native_experts_v3.txt")?;
    let layout = NativeLayout::parse_sc117_v3(&fixture, 50_292_326_400)?;
    let input = std::fs::read(".runs/vf-r574-input/activation0.f32")?;
    if input.len() != H * 4
        || sha256::hex(&input) != "c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a"
    {
        return Err("input identity mismatch".into());
    }
    let x: Vec<f32> = input
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut rows = vec![0u16; 10 * H];
    for (slot, path, expected, capture) in [
        (
            0,
            ".runs/vf-r590-work/expert75.bin",
            "a312f304226bc4cfb7590ea90e9d91babbba67ef0fe4686a0eae85e26a9ed74d",
            ".runs/vf-r590-work/pc6-75.bin",
        ),
        (
            1,
            ".runs/vf-r590-work/expert12.bin",
            "658b2b7b9c7b5055667577babea78964084e1a1dad958c546c46d513249f578e",
            ".runs/vf-r590-work/pc6-12.bin",
        ),
        (
            9,
            ".runs/vf-r574-input/expert255.bin",
            "8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5",
            ".runs/vf-r590-work/pc6.bin",
        ),
    ] {
        let bytes = std::fs::read(path)?;
        if sha256::hex(&bytes) != expected {
            return Err("expert hash mismatch".into());
        }
        let span = layout.expert(2, IDS[slot] as u64)?;
        let expert = VerifiedNativeExpert {
            layer: 2,
            expert: IDS[slot] as u64,
            span,
            bytes,
        };
        let binding = Binding {
            model_revision: "r595-sc117".into(),
            rank: 2,
            layer: 2,
            expert: IDS[slot],
            local: slot as u32,
            epoch: 1,
            index_sha256: layout.index_sha256().into(),
            blob_sha256: expected.into(),
        };
        let mut resident = Resident::load_scalar(&layout, expert, binding)?;
        let out = resident.forward(&x)?;
        let captured = std::fs::read(capture)?;
        if captured.len() != 31760 {
            return Err("capture size mismatch".into());
        }
        for (i, y) in out.iter().enumerate() {
            let prior = f64::from_le_bytes(captured[11280 + i * 8..11288 + i * 8].try_into()?);
            if y.to_bits() != (prior as f32).to_bits() {
                return Err(format!("capture mismatch expert {} row {}", IDS[slot], i).into());
            }
            rows[slot * H + i] = completion::encode_fp16(f64::from(*y))?;
        }
    }
    let stamp = Stamp {
        run_id: "r595-diagnostic".into(),
        model_revision: "r595-sc117".into(),
        plan_id: "local-no-worker".into(),
        layer: 2,
        batch: 0,
        epoch: 1,
    };
    let p = RoutingProblem {
        hidden: H,
        experts: IDS.to_vec(),
        routes: vec![IDS
            .iter()
            .zip(W)
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
    let mut c = Collector::from_routing(&p)?;
    completion::accept_routed_output(1, raw, &mut c)?;
    let result = c.finish()?;
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    for value in &result[0] {
        file.write_all(&value.to_le_bytes())?
    }
    file.sync_all()?;
    println!(
        "three resident FP32 rows match; Collector output {} rows",
        result[0].len()
    );
    Ok(())
}
