use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer};
use tokenizers::Tokenizer;

use crate::sha256;

pub const MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const MODEL_REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
pub const TOKENIZER_JSON_BYTES: usize = 6_367_257;
pub const TOKENIZER_SHA256: &str =
    "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
pub const MAX_TOKENIZER_JSON_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_USER_TEXT_BYTES: usize = 1024 * 1024;
pub const MAX_PROMPT_TOKENS: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPrompt {
    pub token_ids: Vec<u32>,
    pub tokenizer_sha256: String,
    pub context_limit: usize,
    pub reserved_completion_tokens: usize,
}

#[derive(Clone)]
pub struct DeepSeekV41PromptEncoder {
    encoding: DeepseekV41Encoding,
    tokenizer_sha256: String,
}

impl DeepSeekV41PromptEncoder {
    /// Load the caller-owned tokenizer bytes only when they match the exact
    /// pinned checkpoint artifact. This adapter does not download or cache
    /// model files.
    pub fn from_pinned_tokenizer_bytes(tokenizer_json: &[u8]) -> Result<Self, String> {
        if tokenizer_json.len() != TOKENIZER_JSON_BYTES
            || tokenizer_json.len() > MAX_TOKENIZER_JSON_BYTES
        {
            return Err("tokenizer JSON size does not match the pinned artifact".to_owned());
        }
        let actual_sha256 = sha256::hex(tokenizer_json);
        if actual_sha256 != TOKENIZER_SHA256 {
            return Err("tokenizer SHA-256 does not match the pinned artifact".to_owned());
        }
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|error| format!("tokenizer JSON is invalid: {error}"))?;
        let encoding = DeepseekV41Encoding::new().with_tokenizer(PromptTokenizer::new(&tokenizer));
        Ok(Self {
            encoding,
            tokenizer_sha256: actual_sha256,
        })
    }

    /// Encode exactly one plain user turn. Rich histories, tools, images,
    /// schemas, and multimodal inputs require separate reference qualification
    /// and are intentionally outside this first boundary.
    pub fn encode_user_turn(
        &self,
        user_text: &str,
        thinking_mode: bool,
        reasoning_effort: Option<ReasoningEffort>,
        context_limit: usize,
        reserved_completion_tokens: usize,
    ) -> Result<EncodedPrompt, String> {
        if user_text.is_empty() || user_text.len() > MAX_USER_TEXT_BYTES {
            return Err("user text must be nonempty and within the admitted byte bound".to_owned());
        }
        if context_limit == 0 || context_limit > MAX_PROMPT_TOKENS {
            return Err("context limit is outside the admitted token bound".to_owned());
        }
        if reserved_completion_tokens == 0 || reserved_completion_tokens > context_limit {
            return Err("completion reservation must be nonzero and within context".to_owned());
        }

        let conversation = Conversation {
            messages: vec![InputMessage::User {
                content: Prompt::plain(user_text),
                image_sources: Vec::new(),
            }],
            thinking_mode,
            reasoning_effort,
            ..Conversation::default()
        };
        let token_ids = self
            .encoding
            .encode(&conversation)
            .map_err(|error| format!("DeepSeek V4.1 prompt encoding failed: {error}"))?;
        if token_ids.is_empty() || token_ids.len() > MAX_PROMPT_TOKENS {
            return Err("encoded prompt token count is outside the admitted bound".to_owned());
        }
        let total = token_ids
            .len()
            .checked_add(reserved_completion_tokens)
            .ok_or_else(|| "prompt plus completion reservation overflowed".to_owned())?;
        if total > context_limit {
            return Err(format!(
                "prompt and completion reservation require {total} tokens; context limit is {context_limit}"
            ));
        }
        Ok(EncodedPrompt {
            token_ids,
            tokenizer_sha256: self.tokenizer_sha256.clone(),
            context_limit,
            reserved_completion_tokens,
        })
    }
}
