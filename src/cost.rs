//! Provenance labels describe supplied costs; they do not certify measurements.
use crate::{invalid, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostOrigin {
    #[default]
    Assumed,
    Synthetic,
    ImportedMeasurement,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProvenance {
    pub origin: CostOrigin,
    #[serde(default)]
    pub source: Option<String>,
}
impl CostProvenance {
    pub(crate) fn validate(&self) -> Result<()> {
        if self
            .source
            .as_ref()
            .is_some_and(|s| s.len() > 4096 || s.trim().is_empty())
            || (self.origin == CostOrigin::ImportedMeasurement && self.source.is_none())
        {
            return Err(invalid(
                "cost evidence reference missing, empty or oversized",
            ));
        }
        Ok(())
    }
}
