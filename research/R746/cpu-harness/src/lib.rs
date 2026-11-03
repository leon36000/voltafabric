#![forbid(unsafe_code)]
pub mod moe {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct Stamp {
        pub run_id: String,
        pub model_revision: String,
        pub plan_id: String,
        pub layer: u32,
        pub batch: u64,
        pub epoch: u64,
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
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
pub type Result<T> = std::result::Result<T, Error>;
#[path = "../../../../src/deepseek_v41_head.rs"]
pub mod deepseek_v41_head;
#[path = "../../../../src/deepseek_v41_prompt.rs"]
pub mod deepseek_v41_prompt;
#[path = "../../../../src/deepseek_v41_sampler.rs"]
pub mod deepseek_v41_sampler;
#[path = "../../../../src/sha256.rs"]
pub mod sha256;
