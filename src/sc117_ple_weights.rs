//! Authenticated SC117 PLE CPU reference weights. Manifest provenance is the caller's responsibility.
//! BF16 weights and round-to-nearest-even BF16 inputs, with scalar f64 accumulation.
//! This follows the GGML CPU activation type, not its exact reduction order.
use crate::{
    completion::decode_fp16,
    invalid,
    sc117_ple_post::{self, Inputs, Output, CHANNELS, WIDTH},
    sha256, Result,
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub offset: u64,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer: u32,
    pub source_bytes: u64,
    pub key: Span,
    pub value: Span,
    pub norm_key: Span,
    pub norm_query: Span,
    pub norm_conv: Span,
    pub kernel: Span,
}
pub struct Weights {
    model_revision: String,
    key: Vec<u8>,
    value: Vec<u8>,
    nk: Vec<f32>,
    nq: Vec<f32>,
    nc: Vec<f32>,
    kernel: Vec<f32>,
}
fn read(file: &mut File, span: &Span, expected: usize, size: u64) -> Result<Vec<u8>> {
    if span.bytes != expected
        || !sha256::is_hex_digest(&span.sha256)
        || span
            .offset
            .checked_add(expected as u64)
            .is_none_or(|end| end > size)
    {
        return Err(invalid("PLE weight span"));
    }
    file.seek(SeekFrom::Start(span.offset))
        .map_err(|e| invalid(e.to_string()))?;
    let mut b = vec![0; expected];
    file.read_exact(&mut b)
        .map_err(|e| invalid(e.to_string()))?;
    if sha256::hex(&b) != span.sha256 {
        return Err(invalid("PLE weight digest"));
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Result<Vec<f32>> {
    let values: Vec<f32> = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(invalid("PLE nonfinite norm"));
    }
    Ok(values)
}
#[cfg(unix)]
fn identity(f: &File) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let m = f.metadata().map_err(|e| invalid(e.to_string()))?;
    if !m.is_file() {
        return Err(invalid("PLE source not regular"));
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
#[cfg(unix)]
impl Weights {
    pub fn load(file: &mut File, m: &Manifest, expected_revision: &str) -> Result<Self> {
        if m.layer != 1
            || m.model_revision != expected_revision
            || expected_revision.trim().is_empty()
            || expected_revision.len() > 256
        {
            return Err(invalid("PLE model identity"));
        }
        let before = identity(file)?;
        if before.2 != m.source_bytes {
            return Err(invalid("PLE source size"));
        }
        let key = read(file, &m.key, CHANNELS * WIDTH * 2, before.2)?;
        let value = read(file, &m.value, WIDTH * WIDTH * 2, before.2)?;
        let nk = floats(&read(file, &m.norm_key, CHANNELS * 4, before.2)?)?;
        let nq = floats(&read(file, &m.norm_query, CHANNELS * 4, before.2)?)?;
        let nc = floats(&read(file, &m.norm_conv, CHANNELS * 4, before.2)?)?;
        let kb = read(file, &m.kernel, CHANNELS * 4 * 2, before.2)?;
        let kernel = kb
            .chunks_exact(2)
            .map(|b| decode_fp16(u16::from_le_bytes([b[0], b[1]])).map(|v| v as f32))
            .collect::<Result<Vec<_>>>()?;
        if identity(file)? != before {
            return Err(invalid("PLE source changed"));
        }
        Ok(Self {
            model_revision: m.model_revision.clone(),
            key,
            value,
            nk,
            nq,
            nc,
            kernel,
        })
    }
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }
    /// The returned candidate is not committed; the caller owns transaction state.
    pub fn evaluate(&self, emb: &[f32], hidden: &[f32], history: &[f32]) -> Result<Output> {
        let key = project(&self.key, emb, CHANNELS)?;
        let value = project(&self.value, emb, WIDTH)?;
        sc117_ple_post::evaluate(Inputs {
            key: &key,
            value: &value,
            hidden,
            norm_key: &self.nk,
            norm_query: &self.nq,
            norm_conv: &self.nc,
            kernel: &self.kernel,
            history,
        })
    }
}
/// Row-major BF16 matrix; finite FP32 inputs are rounded to BF16 nearest-even.
pub fn project(weights: &[u8], input: &[f32], rows: usize) -> Result<Vec<f32>> {
    if input.len() != WIDTH
        || rows == 0
        || rows > CHANNELS
        || weights.len() != rows * WIDTH * 2
        || input.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("PLE projection shape"));
    }
    let mut activations = Vec::with_capacity(WIDTH);
    for &x in input {
        let bits = x.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        let value = f32::from_bits(rounded);
        if !value.is_finite() {
            return Err(invalid("PLE activation BF16 overflow"));
        }
        activations.push(value);
    }
    let mut out = Vec::with_capacity(rows);
    for row in weights.chunks_exact(WIDTH * 2) {
        let mut sum = 0.0f64;
        for (b, &x) in row.chunks_exact(2).zip(&activations) {
            let w = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            if !w.is_finite() {
                return Err(invalid("PLE nonfinite BF16"));
            }
            sum += f64::from(w) * f64::from(x);
        }
        let v = sum as f32;
        if !v.is_finite() {
            return Err(invalid("PLE nonfinite projection"));
        }
        out.push(v);
    }
    Ok(out)
}
