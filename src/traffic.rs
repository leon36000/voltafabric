use crate::memory::{add, align, mul};
use crate::{invalid, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireFormat {
    pub input_bytes: u64,
    pub expert_output_bytes: u64,
    pub partial_output_bytes: u64,
    pub header_bytes: u64,
    pub slot_metadata_bytes: u64,
    pub alignment: u64,
}
impl Default for WireFormat {
    fn default() -> Self {
        Self {
            input_bytes: 2,
            expert_output_bytes: 2,
            partial_output_bytes: 4,
            header_bytes: 0,
            slot_metadata_bytes: 0,
            alignment: 1,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Traffic {
    pub baseline_bytes: u64,
    pub grouped_bytes: u64,
    pub adaptive_bytes: u64,
    pub baseline_messages: u64,
    pub grouped_messages: u64,
    pub adaptive_messages: u64,
}
/// Wire accounting only. FP32 partial sums are not silently treated as FP16.
pub fn message_sizes(
    hidden: u64,
    count: u64,
    f: WireFormat,
    grouped: bool,
) -> Result<(u64, u64, u64)> {
    if hidden == 0
        || f.input_bytes == 0
        || f.expert_output_bytes == 0
        || f.partial_output_bytes == 0
    {
        return Err(invalid("positive hidden and element sizes required"));
    }
    align(0, f.alignment)?;
    if count == 0 {
        return Ok((0, 0, 0));
    }
    let slots = if grouped { count } else { 1 };
    let metadata = mul(slots, f.slot_metadata_bytes)?;
    let header = add(f.header_bytes, metadata)?;
    let input = align(add(header, mul(hidden, f.input_bytes)?)?, f.alignment)?;
    let output = align(
        add(
            header,
            mul(
                hidden,
                if grouped {
                    f.partial_output_bytes
                } else {
                    f.expert_output_bytes
                },
            )?,
        )?,
        f.alignment,
    )?;
    Ok((input, output, if grouped { 1 } else { count }))
}
pub fn wire_bytes(hidden: u64, counts: &[u64], f: WireFormat) -> Result<Traffic> {
    message_sizes(hidden, 0, f, false)?;
    let mut t = Traffic {
        baseline_bytes: 0,
        grouped_bytes: 0,
        adaptive_bytes: 0,
        baseline_messages: 0,
        grouped_messages: 0,
        adaptive_messages: 0,
    };
    for &c in counts {
        let (bi, bo, bn) = message_sizes(hidden, c, f, false)?;
        let (gi, go, gn) = message_sizes(hidden, c, f, true)?;
        let b = mul(add(bi, bo)?, bn)?;
        let g = mul(add(gi, go)?, gn)?;
        t.baseline_bytes = add(t.baseline_bytes, b)?;
        t.grouped_bytes = add(t.grouped_bytes, g)?;
        t.adaptive_bytes = add(t.adaptive_bytes, b.min(g))?;
        t.baseline_messages = add(t.baseline_messages, mul(2, bn)?)?;
        t.grouped_messages = add(t.grouped_messages, mul(2, gn)?)?;
        t.adaptive_messages = add(t.adaptive_messages, mul(2, if g < b { gn } else { bn })?)?;
    }
    Ok(t)
}
