//! Standard RFC 4648 base64 with padding, for carrying artifact bytes inside one JSON
//! frame. Decoding is strict: only canonical text is accepted, so one byte string has
//! exactly one encoding and the artifact SHA-256 names the bytes, not a spelling of them.
use crate::{Error, Result};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i <= chunk.len() {
                ALPHABET[((n >> shift) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

fn value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn malformed() -> Error {
    Error::new(
        "ARTIFACT_ENCODING",
        "artifact is not canonical padded base64",
    )
}

pub fn decode(text: &str) -> Result<Vec<u8>> {
    let text = text.as_bytes();
    if !text.len().is_multiple_of(4) {
        return Err(malformed());
    }
    let mut out = Vec::new();
    out.try_reserve_exact(text.len() / 4 * 3)
        .map_err(|e| Error::new("ALLOCATION_FAILED", e.to_string()))?;
    let quads = text.len() / 4;
    for (q, quad) in text.chunks_exact(4).enumerate() {
        let pad = if q + 1 == quads {
            quad.iter().rev().take_while(|&&c| c == b'=').count()
        } else {
            0
        };
        if pad > 2 {
            return Err(malformed());
        }
        let mut n = 0_u32;
        for &c in &quad[..4 - pad] {
            n = (n << 6) | value(c).ok_or_else(malformed)?;
        }
        n <<= 6 * pad as u32;
        // Bits below the last encoded byte must be zero in canonical text.
        if n & ((1_u32 << (8 * pad)) - 1) != 0 {
            return Err(malformed());
        }
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - pad]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rfc4648_vectors() {
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode(plain.as_bytes()), coded);
            assert_eq!(decode(coded).unwrap(), plain.as_bytes());
        }
    }
}
