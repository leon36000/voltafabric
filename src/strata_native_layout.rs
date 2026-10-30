//! Bounded metadata admission for the observed SC117 native-expert pack.
//! Index admission and bounded digest-verified reads for the observed pack.
//! It does not decode or execute model payloads, or establish checkpoint identity.
use crate::{invalid, sha256, Result};
use std::io::{Read, Seek, SeekFrom};

const LAYERS: usize = 48;
const EXPERTS: u64 = 512;
const HIDDEN: u64 = 2560;
const INTERMEDIATE: u64 = 640;
const TOTAL: u64 = 50_292_326_400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLayer {
    pub offset: u64,
    pub bytes: u64,
    pub gate_up_type: u32,
    pub down_type: u32,
    pub gguf_offsets: [u64; 3],
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSpan {
    pub offset: u64,
    pub bytes: u64,
    pub gate_up_type: u32,
    pub down_type: u32,
}
#[derive(Debug)]
pub struct NativeLayout {
    layers: Vec<NativeLayer>,
    index_sha256: String,
}

/// An exact blob verified against a caller-supplied digest. Its format remains
/// the native GGML format named in `span`; it is not QPN2 or MXFP4.
pub struct VerifiedNativeExpert {
    pub layer: usize,
    pub expert: u64,
    pub span: ExpertSpan,
    pub bytes: Vec<u8>,
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn block(kind: u32) -> Result<(u64, u64)> {
    match kind {
        18 => Ok((256, 98)),
        21 => Ok((256, 110)),
        22 => Ok((256, 82)),
        23 => Ok((256, 136)),
        20 => Ok((32, 18)),
        42 => Ok((64, 18)),
        _ => Err(invalid("unsupported native expert format")),
    }
}
fn row_bytes(kind: u32, width: u64) -> Result<u64> {
    let (elements, bytes) = block(kind)?;
    if !width.is_multiple_of(elements) {
        return Err(invalid("native format block alignment"));
    }
    (width / elements)
        .checked_mul(bytes)
        .ok_or_else(|| invalid("native row overflow"))
}
impl NativeLayout {
    /// SC117 48-layer, 512-expert v3 index; a declared file length is checked,
    /// but the caller must separately bind the index and actual file identity.
    pub fn parse_sc117_v3(index: &str, file_len: u64) -> Result<Self> {
        if index.len() > 65_536 || file_len != TOTAL {
            return Err(invalid("native index or pack size"));
        }
        let mut lines = index.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid("missing native index header"))?;
        if !header.starts_with("# strata native experts v3:")
            || !header.contains("(n_expert 512, total 50292326400;")
        {
            return Err(invalid("unsupported native index header"));
        }
        let mut layers = Vec::with_capacity(LAYERS);
        let mut next_offset = 0_u64;
        for line in lines {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if !(fields.len() == 8 || fields.len() == 9) || layers.len() >= LAYERS {
                return Err(invalid("native index field or layer count"));
            }
            let parse = |i: usize| {
                fields[i]
                    .parse::<u64>()
                    .map_err(|_| invalid("invalid native index number"))
            };
            let layer = parse(0)?;
            if layer != layers.len() as u64 {
                return Err(invalid("native index layer order"));
            }
            let gu = u32::try_from(parse(1)?).map_err(|_| invalid("native gate type"))?;
            let down = u32::try_from(parse(2)?).map_err(|_| invalid("native down type"))?;
            if !matches!(gu, 18 | 21 | 22 | 23) || !matches!(down, 20 | 42) {
                return Err(invalid("unsupported SC117 native format"));
            }
            let offset = parse(3)?;
            let bytes = parse(4)?;
            let gu_row = row_bytes(gu, HIDDEN)?;
            let down_row = row_bytes(down, INTERMEDIATE)?;
            let expected = gu_row
                .checked_mul(INTERMEDIATE)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| down_row.checked_mul(HIDDEN).and_then(|d| n.checked_add(d)))
                .ok_or_else(|| invalid("native blob geometry overflow"))?;
            if bytes != expected || offset != next_offset {
                return Err(invalid("native blob layout mismatch"));
            }
            next_offset = bytes
                .checked_mul(EXPERTS)
                .and_then(|n| offset.checked_add(n))
                .filter(|&n| n <= file_len)
                .ok_or_else(|| invalid("native layer span exceeds pack"))?;
            layers.push(NativeLayer {
                offset,
                bytes,
                gate_up_type: gu,
                down_type: down,
                gguf_offsets: [parse(5)?, parse(6)?, parse(7)?],
            });
        }
        if layers.len() != LAYERS || next_offset != file_len {
            return Err(invalid("incomplete native index"));
        }
        Ok(Self {
            layers,
            index_sha256: sha256::hex(index.as_bytes()),
        })
    }
    pub fn layers(&self) -> &[NativeLayer] {
        &self.layers
    }
    /// Maximum new expert-payload bytes for one token when every layer
    /// requests `routed_per_layer` previously absent experts. Does not include
    /// cache state, activations, allocator overhead or transient copies.
    pub fn worst_case_admission_bytes(&self, routed_per_layer: usize) -> Result<u64> {
        if routed_per_layer == 0 || routed_per_layer as u64 > EXPERTS {
            return Err(invalid("invalid native routed expert count"));
        }
        self.layers.iter().try_fold(0u64, |total, layer| {
            layer
                .bytes
                .checked_mul(routed_per_layer as u64)
                .and_then(|amount| total.checked_add(amount))
                .ok_or_else(|| invalid("native admission budget overflow"))
        })
    }
    pub fn index_sha256(&self) -> &str {
        &self.index_sha256
    }
    pub fn expert(&self, layer: usize, expert: u64) -> Result<ExpertSpan> {
        let entry = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid("native layer out of range"))?;
        if expert >= EXPERTS {
            return Err(invalid("native expert out of range"));
        }
        let offset = expert
            .checked_mul(entry.bytes)
            .and_then(|n| entry.offset.checked_add(n))
            .ok_or_else(|| invalid("native expert offset overflow"))?;
        Ok(ExpertSpan {
            offset,
            bytes: entry.bytes,
            gate_up_type: entry.gate_up_type,
            down_type: entry.down_type,
        })
    }

    /// Read a single bounded expert only when both expected hashes are supplied
    /// by a trusted manifest. The descriptor must refer to the intended pack;
    /// per-blob SHA binds the returned bytes, not the rest of the pack or model.
    pub fn read_verified_expert<R: Read + Seek>(
        &self,
        reader: &mut R,
        layer: usize,
        expert: u64,
        expected_index_sha256: &str,
        expected_blob_sha256: &str,
    ) -> Result<VerifiedNativeExpert> {
        if !valid_sha(expected_index_sha256)
            || !valid_sha(expected_blob_sha256)
            || expected_index_sha256 != self.index_sha256
        {
            return Err(invalid("native index or blob digest not pinned"));
        }
        let span = self.expert(layer, expert)?;
        if span.bytes > 4 * 1024 * 1024 {
            return Err(invalid("native blob exceeds read bound"));
        }
        let len = reader
            .seek(SeekFrom::End(0))
            .map_err(|_| invalid("native pack size unavailable"))?;
        if len != TOTAL {
            return Err(invalid("native pack size changed"));
        }
        let pos = reader
            .seek(SeekFrom::Start(span.offset))
            .map_err(|_| invalid("native expert seek failed"))?;
        if pos != span.offset {
            return Err(invalid("native expert seek mismatch"));
        }
        let n = usize::try_from(span.bytes).map_err(|_| invalid("native blob size overflow"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(n)
            .map_err(|_| invalid("native blob allocation failed"))?;
        bytes.resize(n, 0);
        reader
            .read_exact(&mut bytes)
            .map_err(|_| invalid("native blob read incomplete"))?;
        if sha256::hex(&bytes) != expected_blob_sha256 {
            return Err(invalid("native blob digest mismatch"));
        }
        Ok(VerifiedNativeExpert {
            layer,
            expert,
            span,
            bytes,
        })
    }
}
