//! Bounded SC117 IQ4_XS token rows, adapted from GGML/llama.cpp's MIT format.
//! The caller validates checkpoint provenance and tensor metadata before opening.
use crate::{completion::decode_fp16, invalid, Result};
use std::fs::File;
pub const WIDTH: usize = 2560;
pub const VOCAB: u32 = 248_320;
pub const ROW_BYTES: usize = 1360;
const VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Decode ten 256-element IQ4_XS blocks. This is not IQ4_NL, MXFP4 or NVFP4.
pub fn decode_row(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() != ROW_BYTES {
        return Err(invalid("IQ4_XS embedding row size"));
    }
    let mut out = vec![0.0; WIDTH];
    for (block, raw) in bytes.chunks_exact(136).enumerate() {
        let d = decode_fp16(u16::from_le_bytes([raw[0], raw[1]]))? as f32;
        let high = u16::from_le_bytes([raw[2], raw[3]]);
        for group in 0..8 {
            let low = (raw[4 + group / 2] >> (4 * (group % 2))) & 15;
            let scale = (i32::from(low | (((high >> (2 * group)) as u8 & 3) << 4)) - 32) as f32;
            let factor = d * scale;
            for j in 0..16 {
                let q = raw[8 + group * 16 + j];
                let start = block * 256 + group * 32;
                out[start + j] = factor * f32::from(VALUES[(q & 15) as usize]);
                out[start + 16 + j] = factor * f32::from(VALUES[(q >> 4) as usize]);
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err(invalid("IQ4_XS nonfinite output"));
    }
    Ok(out)
}
#[cfg(unix)]
fn identity(file: &File) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let m = file.metadata().map_err(|e| invalid(e.to_string()))?;
    if !m.is_file() {
        return Err(invalid("embedding source not regular"));
    }
    Ok((
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}
/// Descriptor pinned to a bounded tensor. Origin and metadata must be authenticated externally.
pub struct Table {
    file: File,
    identity: (u64, u64, u64, i64, i64, i64, i64),
    offset: u64,
}
#[cfg(unix)]
impl Table {
    pub fn open(file: File, expected_bytes: u64, offset: u64) -> Result<Self> {
        let id = identity(&file)?;
        if id.2 != expected_bytes
            || offset
                .checked_add(u64::from(VOCAB) * ROW_BYTES as u64)
                .is_none_or(|end| end > expected_bytes)
        {
            return Err(invalid("embedding tensor span"));
        }
        Ok(Self {
            file,
            identity: id,
            offset,
        })
    }
    pub fn gather(&self, token: u32) -> Result<Vec<f32>> {
        use std::os::unix::fs::FileExt;
        if token >= VOCAB || identity(&self.file)? != self.identity {
            return Err(invalid("embedding token or source changed"));
        }
        let at = self
            .offset
            .checked_add(u64::from(token) * ROW_BYTES as u64)
            .ok_or_else(|| invalid("embedding offset overflow"))?;
        let mut row = [0u8; ROW_BYTES];
        self.file
            .read_exact_at(&mut row, at)
            .map_err(|e| invalid(e.to_string()))?;
        let decoded = decode_row(&row)?;
        if identity(&self.file)? != self.identity {
            return Err(invalid("embedding source changed"));
        }
        Ok(decoded)
    }
}
