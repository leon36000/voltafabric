#![forbid(unsafe_code)]
//! CPU-only contracts and planning helpers for VoltaFabric.
pub mod memory;

use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub code: String,
    pub message: String,
}
impl Error {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new("INVALID_INPUT", message)
}
pub mod moe;
pub mod pipeline;
pub mod planner;
pub mod preweighted;
pub mod quant;
pub mod schedule;
pub mod topology;
pub mod traffic;

pub mod cost;
pub(crate) mod work;

pub mod qpn2;
pub mod routing;

pub mod base64;
pub mod process;
pub mod session;
pub mod sha256;

pub mod resident;
pub mod resident_artifact;

pub mod launch;

pub mod completion;

pub mod checkpoint;

pub mod deepseek_v41;
pub mod deepseek_v41_attention;
pub mod deepseek_v41_attention_output;
pub mod deepseek_v41_gate;
pub mod deepseek_v41_projection;
pub mod deepseek_v41_rope;
pub mod deepseek_v41_window_block;
pub mod engram;
