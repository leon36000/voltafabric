//! Deterministic inspection budget, not a wall-clock or memory measurement.
use crate::{invalid, Error, Result};
pub(crate) const DEFAULT_WORK_LIMIT: u64 = 1_000_000;
pub(crate) struct WorkBudget {
    limit: u64,
    used: u64,
}
impl WorkBudget {
    pub(crate) fn new(limit: u64) -> Result<Self> {
        if limit == 0 || limit > DEFAULT_WORK_LIMIT {
            return Err(invalid("work limit must be within 1..=1000000"));
        }
        Ok(Self { limit, used: 0 })
    }
    pub(crate) fn standard() -> Self {
        Self {
            limit: DEFAULT_WORK_LIMIT,
            used: 0,
        }
    }
    pub(crate) fn charge(&mut self, units: usize) -> Result<()> {
        let units = u64::try_from(units).map_err(|_| invalid("work count overflow"))?;
        let next = self.used.checked_add(units);
        if next.is_none_or(|n| n > self.limit) {
            return Err(Error::new("WORK_BUDGET_EXCEEDED", format!(
                "inspection budget exhausted: used={}, requested={}, limit={}; no partial placement",
                self.used, units, self.limit
            )));
        }
        self.used = next.ok_or_else(|| invalid("work count overflow"))?;
        Ok(())
    }
    pub(crate) fn used(&self) -> u64 {
        self.used
    }
}
