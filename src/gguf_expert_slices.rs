//! Bounded GGUF expert-axis selection for rank-local resident matrix stacks.
//!
//! A GGUF tensor stores all checkpoint experts, while a VoltaFabric rank owns only
//! its registered residents. This module validates that mapping and streams selected
//! physical slices in resident order without materializing the checkpoint tensor.

use std::io::{self, Read, Seek, SeekFrom};

pub const MAX_SELECTED_EXPERTS: usize = 512;

/// Explicit mapping between a VoltaFabric resident entry and one checkpoint expert.
/// The order of this slice is the physical matrix order loaded into the rank worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCheckpointSlice {
    pub resident_expert: u32,
    pub resident_local: u32,
    pub checkpoint_expert: u32,
}

/// Validate the plan and return checkpoint indices in the exact resident order.
pub fn validate_checkpoint_slices(
    residents: &[(u32, u32)],
    slices: &[ExpertCheckpointSlice],
    checkpoint_expert_count: usize,
    tensor_bytes: u64,
    max_selected_bytes: u64,
) -> Result<(Vec<u32>, u64), &'static str> {
    use std::collections::BTreeSet;

    if residents.is_empty()
        || residents.len() > MAX_SELECTED_EXPERTS
        || slices.len() != residents.len()
        || checkpoint_expert_count == 0
        || tensor_bytes == 0
        || tensor_bytes
            % u64::try_from(checkpoint_expert_count)
                .map_err(|_| "GGUF expert count does not fit u64")?
            != 0
    {
        return Err("invalid GGUF expert slice plan or tensor span");
    }
    let expert_bytes = tensor_bytes / checkpoint_expert_count as u64;
    if expert_bytes == 0 {
        return Err("empty GGUF checkpoint expert slice");
    }

    let mut seen = BTreeSet::new();
    let mut indices = Vec::with_capacity(slices.len());
    for (resident, mapping) in residents.iter().zip(slices) {
        if resident != &(mapping.resident_expert, mapping.resident_local)
            || mapping.checkpoint_expert as usize >= checkpoint_expert_count
            || !seen.insert(mapping.checkpoint_expert)
        {
            return Err("GGUF expert mapping differs from resident identity or order");
        }
        indices.push(mapping.checkpoint_expert);
    }
    let selected_bytes = expert_bytes
        .checked_mul(u64::try_from(indices.len()).map_err(|_| "resident count overflow")?)
        .filter(|&bytes| bytes != 0 && bytes <= max_selected_bytes)
        .ok_or("selected GGUF resident slices exceed payload bound")?;
    Ok((indices, selected_bytes))
}

/// Seek-and-read adapter over a tensor's expert axis. Each read exposes only the
/// selected slices, concatenated in the caller's planned resident order.
pub struct SelectedExpertSliceReader<'a, R> {
    source: &'a mut R,
    span_offset: u64,
    expert_bytes: u64,
    checkpoint_indices: &'a [u32],
    selected_index: usize,
    consumed_in_expert: u64,
}

impl<'a, R: Read + Seek> SelectedExpertSliceReader<'a, R> {
    pub fn new(
        source: &'a mut R,
        span_offset: u64,
        tensor_bytes: u64,
        checkpoint_expert_count: usize,
        checkpoint_indices: &'a [u32],
    ) -> io::Result<Self> {
        use std::collections::BTreeSet;

        let expert_count = u64::try_from(checkpoint_expert_count)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "expert count overflow"))?;
        if expert_count == 0
            || tensor_bytes == 0
            || tensor_bytes % expert_count != 0
            || checkpoint_indices.is_empty()
            || checkpoint_indices.len() > MAX_SELECTED_EXPERTS
            || checkpoint_indices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != checkpoint_indices.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid selected GGUF expert slices",
            ));
        }
        let expert_bytes = tensor_bytes / expert_count;
        if expert_bytes == 0
            || checkpoint_indices
                .iter()
                .any(|&index| u64::from(index) >= expert_count)
            || span_offset.checked_add(tensor_bytes).is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "selected GGUF expert slice range overflow",
            ));
        }
        Ok(Self {
            source,
            span_offset,
            expert_bytes,
            checkpoint_indices,
            selected_index: 0,
            consumed_in_expert: 0,
        })
    }
}

impl<R: Read + Seek> Read for SelectedExpertSliceReader<'_, R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() || self.selected_index == self.checkpoint_indices.len() {
            return Ok(0);
        }

        let checkpoint_expert = u64::from(self.checkpoint_indices[self.selected_index]);
        let relative = checkpoint_expert
            .checked_mul(self.expert_bytes)
            .and_then(|offset| offset.checked_add(self.consumed_in_expert))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "slice offset overflow"))?;
        let offset = self
            .span_offset
            .checked_add(relative)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "slice offset overflow"))?;
        let remaining = self.expert_bytes - self.consumed_in_expert;
        let wanted = output
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        self.source.seek(SeekFrom::Start(offset))?;
        let read = self.source.read(&mut output[..wanted])?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "GGUF expert slice ended before its declared size",
            ));
        }
        self.consumed_in_expert += read as u64;
        if self.consumed_in_expert == self.expert_bytes {
            self.selected_index += 1;
            self.consumed_in_expert = 0;
        }
        Ok(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn residents() -> [(u32, u32); 2] {
        [(40, 0), (3, 1)]
    }

    fn mapping() -> [ExpertCheckpointSlice; 2] {
        [
            ExpertCheckpointSlice {
                resident_expert: 40,
                resident_local: 0,
                checkpoint_expert: 3,
            },
            ExpertCheckpointSlice {
                resident_expert: 3,
                resident_local: 1,
                checkpoint_expert: 1,
            },
        ]
    }

    #[test]
    fn validates_and_streams_noncontiguous_checkpoint_slices_in_resident_order() {
        let (indices, selected_bytes) =
            validate_checkpoint_slices(&residents(), &mapping(), 4, 24, 12).unwrap();
        assert_eq!(indices, [3, 1]);
        assert_eq!(selected_bytes, 12);

        let source_bytes: Vec<_> = (0_u8..24).collect();
        let mut source = Cursor::new(source_bytes.clone());
        let mut reader = SelectedExpertSliceReader::new(&mut source, 0, 24, 4, &indices).unwrap();
        let mut selected = Vec::new();
        reader.read_to_end(&mut selected).unwrap();
        assert_eq!(&selected[..6], &source_bytes[18..24]);
        assert_eq!(&selected[6..], &source_bytes[6..12]);
        assert_eq!(selected.len() as u64, selected_bytes);
    }

    #[test]
    fn rejects_duplicate_out_of_range_or_reordered_checkpoint_mapping() {
        let mut duplicate = mapping();
        duplicate[1].checkpoint_expert = 3;
        assert!(validate_checkpoint_slices(&residents(), &duplicate, 4, 24, 24).is_err());

        let mut out_of_range = mapping();
        out_of_range[0].checkpoint_expert = 4;
        assert!(validate_checkpoint_slices(&residents(), &out_of_range, 4, 24, 24).is_err());

        let mut reordered = mapping();
        reordered.swap(0, 1);
        assert!(validate_checkpoint_slices(&residents(), &reordered, 4, 24, 24).is_err());
    }

    #[test]
    fn rejects_malformed_span_and_payload_over_cap_before_reading() {
        assert!(validate_checkpoint_slices(&residents(), &mapping(), 4, 25, 12).is_err());
        assert!(validate_checkpoint_slices(&residents(), &mapping(), 4, 24, 11).is_err());
        assert!(
            SelectedExpertSliceReader::new(&mut Cursor::new(vec![0; 24]), 0, 24, 4, &[4]).is_err()
        );
    }

    #[test]
    fn reports_short_selected_slice_instead_of_silent_partial_completion() {
        let indices = [3, 1];
        let mut source = Cursor::new(vec![0_u8; 22]);
        let mut reader = SelectedExpertSliceReader::new(&mut source, 0, 24, 4, &indices).unwrap();
        let mut selected = Vec::new();
        assert_eq!(
            reader.read_to_end(&mut selected).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(selected.len(), 4);
    }
}
