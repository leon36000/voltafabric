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

pub mod gdn_cpu;
pub mod hyperconnection_cpu;
#[cfg(feature = "native-ggml-cpu")]
pub mod native_ggml_cpu;
#[cfg(feature = "native-ggml-cpu")]
pub mod native_ledger;
pub mod qpn2;
pub mod qsa_cpu;
pub mod routing;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_bound_layer;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_ffn_stage;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_gdn_stage;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_head_cpu;
pub mod sc117_layer_completion;
pub mod sc117_ple;
pub mod sc117_ple_post;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_layer_chain;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_post_attention;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_qsa_bound_layer;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_qsa_stage;
pub mod sc117_router_cpu;
#[cfg(feature = "native-ggml-cpu")]
pub mod sc117_shared_stage;
pub mod shared_expert_cpu;
pub mod strata_ingress;
#[cfg(feature = "native-ggml-cpu")]
pub mod strata_native_bridge;
pub mod strata_native_layout;

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
pub mod engram;
