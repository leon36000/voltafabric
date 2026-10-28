use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub codes: u64,
    pub scales: u64,
    pub total: u64,
    pub padded_columns: u64,
}
/// Storage contract: 4-bit codes and one 8-bit scale per row-local block of 16.
/// Each plane is independently aligned. This is accounting, not a CUDA packer.
pub fn nvfp4_layout(
    rows: u64,
    columns: u64,
    alignment: u64,
    global_scale_bytes: u64,
) -> Result<Layout> {
    if rows == 0 || columns == 0 {
        return Err(invalid("matrix dimensions must be positive"));
    }
    let padded_columns = align(columns, 16)?;
    let n = mul(rows, padded_columns)?;
    let codes = align(n / 2, alignment)?;
    let scales = align(n / 16, alignment)?;
    let total = add(add(codes, scales)?, align(global_scale_bytes, alignment)?)?;
    Ok(Layout {
        codes,
        scales,
        total,
        padded_columns,
    })
}
pub(crate) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::new("OVERFLOW", "u64 addition"))
}
pub(crate) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::new("OVERFLOW", "u64 multiplication"))
}
pub(crate) fn align(n: u64, a: u64) -> Result<u64> {
    if !a.is_power_of_two() {
        return Err(invalid("alignment must be a nonzero power of two"));
    }
    let rem = n % a;
    if rem == 0 {
        Ok(n)
    } else {
        add(n, a - rem)
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub capacity: u64,
    pub weights: u64,
    pub kv: u64,
    pub recurrent: u64,
    pub workspace: u64,
    pub graphs: u64,
    pub communication: u64,
    pub margin: u64,
    pub load_temporary: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReport {
    pub steady_bytes: u64,
    pub loading_peak_bytes: u64,
    pub headroom_bytes: u64,
}
impl Budget {
    pub fn check(&self) -> Result<BudgetReport> {
        if self.capacity == 0 {
            return Err(invalid("capacity must be positive"));
        }
        let mut steady = 0;
        for x in [
            self.weights,
            self.kv,
            self.recurrent,
            self.workspace,
            self.graphs,
            self.communication,
            self.margin,
        ] {
            steady = add(steady, x)?;
        }
        let peak = add(steady, self.load_temporary)?;
        let headroom = self.capacity.checked_sub(peak).ok_or_else(|| {
            Error::new(
                "CAPACITY_EXCEEDED",
                format!("loading peak {peak} exceeds {}", self.capacity),
            )
        })?;
        Ok(BudgetReport {
            steady_bytes: steady,
            loading_peak_bytes: peak,
            headroom_bytes: headroom,
        })
    }
}
