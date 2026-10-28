//! A125: compressed MXFP4 -> the existing Skinny QPN2 resident format.
//! This is a CPU byte/layout/numeric proof, not a GPU or full-model qualification.

use std::path::Path;
use voltafabric::{
    deepseek_v41::{Mxfp4Expert, Mxfp4ExpertShard, Mxfp4Matrix},
    quant,
    resident::ResidentStack,
};

const FIXTURE_SHA256: &str = "e6d39f1821c0e3f7f1cc4827d206391aa428a6f3944731069d45dee4135f8c00";
const K_ORDER: [usize; 16] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15];

fn resident_plane_value(
    codes: &[u8],
    scales: &[u8],
    factors: &[f32],
    rows: usize,
    columns: usize,
    row: usize,
    col: usize,
) -> f64 {
    assert!(row < rows && col < columns);
    let tile = row / 32;
    let r = row % 32;
    let lane = (r & 3) | (((r >> 3) & 3) << 2) | (((r >> 2) & 1) << 4);
    let group = col / 16;
    let logical_k = col % 16;
    let slot = K_ORDER.iter().position(|&k| k == logical_k).unwrap();
    let fragment = (tile * (columns / 16) + group) * 32 + lane;
    let packed = codes[fragment * 8 + slot / 2];
    let nibble = if slot % 2 == 0 {
        packed & 15
    } else {
        packed >> 4
    };
    quant::decode_fp4(nibble).unwrap()
        * quant::decode_scale(scales[fragment]).unwrap()
        * f64::from(factors[tile])
}

fn assert_sampled_projection_exact(
    expert: &Mxfp4Expert,
    stack: &ResidentStack,
    projection: &str,
    matrix: &Mxfp4Matrix,
    code_offset: usize,
    scale_offset: usize,
    factor_offset: usize,
) {
    let rows = matrix.rows() as usize;
    let cols = matrix.cols() as usize;
    assert_eq!(rows % 32, 0);
    assert_eq!(cols % 64, 0);

    // Exercise every N32 tile edge and spread K across all K16/K32 groups. A
    // seeded affine sequence gives reproducible coordinates without a dependency.
    let mut coordinates = Vec::new();
    for tile in 0..rows / 32 {
        for lane in [0usize, 1, 7, 15, 16, 23, 31] {
            for col in [0usize, 1, 7, 15, 16, 31, cols / 2, cols - 2, cols - 1] {
                coordinates.push((tile * 32 + lane, col));
            }
        }
    }
    let resident_codes = if projection == "w1" || projection == "w3" {
        &stack.codes13()[code_offset..code_offset + rows * cols / 2]
    } else {
        &stack.codes2()[code_offset..code_offset + rows * cols / 2]
    };
    let resident_scales = if projection == "w1" || projection == "w3" {
        &stack.scales13()[scale_offset..scale_offset + rows * cols / 16]
    } else {
        &stack.scales2()[scale_offset..scale_offset + rows * cols / 16]
    };
    let resident_factors = if projection == "w1" || projection == "w3" {
        &stack.gscales13()[factor_offset..factor_offset + rows / 32]
    } else {
        &stack.gscales2()[factor_offset..factor_offset + rows / 32]
    };

    for (row, col) in coordinates {
        let expected = matrix.value(row as u64, col as u64).unwrap();
        let actual = resident_plane_value(
            resident_codes,
            resident_scales,
            resident_factors,
            rows,
            cols,
            row,
            col,
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "expert {} {projection}[{row},{col}] differs",
            expert.expert()
        );
    }
}

fn check_expert(shard: &Mxfp4ExpertShard, id: u32) {
    let expert = shard.expert(0, id).unwrap();
    let stack = expert.to_qpn2_resident_stack(0, 0, 9).unwrap();
    assert_eq!(stack.model_revision(), voltafabric::deepseek_v41::REVISION);
    assert_eq!(stack.layer(), 0);
    assert_eq!(stack.rank(), 0);
    assert_eq!(stack.epoch(), 9);
    assert_eq!(stack.physical_experts().len(), 1);
    assert_eq!(stack.physical_experts()[0].id(), id);
    assert_eq!(stack.physical_experts()[0].local(), 0);
    assert_eq!(stack.resident_order(), &[0]);
    assert_eq!(stack.shape(), (1, 4608, 5120, 5120, 2304));

    let plane_codes = 2304usize * 5120 / 2;
    let plane_scales = 2304usize * 5120 / 16;
    let plane_factors = 2304usize / 32;
    assert_sampled_projection_exact(expert, &stack, "w1", expert.w1(), 0, 0, 0);
    assert_sampled_projection_exact(
        expert,
        &stack,
        "w3",
        expert.w3(),
        plane_codes,
        plane_scales,
        plane_factors,
    );
    assert_sampled_projection_exact(expert, &stack, "w2", expert.w2(), 0, 0, 0);

    assert_eq!(stack.codes13().len(), plane_codes * 2);
    assert_eq!(stack.scales13().len(), plane_scales * 2);
    assert_eq!(stack.gscales13().len(), plane_factors * 2);
    assert_eq!(stack.codes2().len(), 5120 * 2304 / 2);
    assert_eq!(stack.scales2().len(), 5120 * 2304 / 16);
    assert_eq!(stack.gscales2().len(), 5120 / 32);
    assert_eq!(stack.payload_bytes(), 19_907_776);
}

#[test]
fn real_experts_96_and_192_convert_to_exact_compressed_qpn2_samples() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes =
        std::fs::read(root.join("fixtures/deepseek_v41_layer0_experts96-192_314113f8.safetensors"))
            .unwrap();
    let shard =
        Mxfp4ExpertShard::from_safetensors_shard_artifact(bytes, FIXTURE_SHA256, 64 * 1024 * 1024)
            .unwrap();
    assert_eq!(shard.expert_count(), 2);
    check_expert(&shard, 96);
    check_expert(&shard, 192);
}
