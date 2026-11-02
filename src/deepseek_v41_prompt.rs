use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::prompt::Prompt;
use deepseek_recipe_core::special_tokens::SpecialToken;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer};
use tokenizers::Tokenizer;

use crate::sha256;

pub const MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const MODEL_REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
pub const TOKENIZER_JSON_BYTES: usize = 6_367_257;
pub const TOKENIZER_SHA256: &str =
    "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
/// Checkpoint `config.json` pins `eos_token_id` to 1 for MODEL_REVISION.
pub const EOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_SPELLING: &str = "<｜end▁of▁sentence｜>";
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

/// Generated token IDs up to, but not including, the first checkpoint EOS.
/// The caller reports `Stop` to DeepSeek Recipe when `end_of_turn` is true;
/// Recipe's stream contract requires EOS to be omitted from decoded tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTokenPrefix<'a> {
    pub token_ids: &'a [u32],
    pub end_of_turn: bool,
}

#[derive(Clone)]
pub struct DeepSeekV41PromptEncoder {
    encoding: DeepseekV41Encoding,
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
    eos_token_id: u32,
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
        let eos_token_id = tokenizer
            .token_to_id(SpecialToken::Eos.default_str())
            .ok_or_else(|| "pinned tokenizer lacks the checkpoint EOS token".to_owned())?;
        if eos_token_id != EOS_TOKEN_ID || SpecialToken::Eos.default_str() != EOS_TOKEN_SPELLING {
            return Err("pinned tokenizer EOS does not match the checkpoint contract".to_owned());
        }
        let encoding = DeepseekV41Encoding::new().with_tokenizer(PromptTokenizer::new(&tokenizer));
        Ok(Self {
            encoding,
            tokenizer,
            tokenizer_sha256: actual_sha256,
            eos_token_id,
        })
    }

    /// Return the EOS ID admitted by the exact pinned tokenizer and checkpoint.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Remove EOS and any impossible trailing IDs from one sampled output batch.
    /// The inference backend should stop sampling at this ID and report a
    /// terminal `Stop`; this helper makes that boundary safe if an oversized
    /// backend batch nevertheless contains IDs after EOS.
    pub fn generated_prefix<'a>(&self, sampled_ids: &'a [u32]) -> GeneratedTokenPrefix<'a> {
        match sampled_ids.iter().position(|&id| id == self.eos_token_id) {
            Some(index) => GeneratedTokenPrefix {
                token_ids: &sampled_ids[..index],
                end_of_turn: true,
            },
            None => GeneratedTokenPrefix {
                token_ids: sampled_ids,
                end_of_turn: false,
            },
        }
    }

    /// Build DeepSeek Recipe's output processor with the same admitted tokenizer.
    /// Recipe preserves token boundaries so plain text that merely spells a
    /// control token cannot be mistaken for actual model control output.
    #[cfg(feature = "deepseek-v41-stream")]
    pub fn output_stream_processor<G>(
        &self,
        generator: G,
        options: deepseek_recipe::stream::ParsingOptions,
    ) -> deepseek_recipe::stream::StreamProcessor<G>
    where
        G: deepseek_recipe::stream::ChunkGenerator,
    {
        deepseek_recipe::stream::StreamProcessor::new(generator, options)
            .with_tokenizer(self.tokenizer.clone())
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

#[cfg(test)]
mod reference_tests {
    use super::*;
    #[cfg(feature = "deepseek-v41-stream")]
    use deepseek_recipe::stream::{
        ChunkGenerator, FinishReason, InferenceChunk, InferenceFinishReason, OutputChunk,
        ParsingOptions, ThinkingInitialStage,
    };
    use std::{env, fs};
    use tokenizers::{AddedVocabulary, Tokenizer};
    #[cfg(feature = "deepseek-v41-stream")]
    use tokio_stream::{StreamExt, iter};

    const TOKENIZER_ENV: &str = "VF_DEEPSEEK_V41_TOKENIZER_JSON";

    fn pinned_tokenizer() -> (Vec<u8>, Tokenizer) {
        let path = env::var_os(TOKENIZER_ENV)
            .expect("set VF_DEEPSEEK_V41_TOKENIZER_JSON to the pinned tokenizer artifact");
        let bytes = fs::read(path).expect("read the pinned tokenizer artifact");
        let tokenizer = Tokenizer::from_bytes(&bytes).expect("parse pinned tokenizer JSON");
        (bytes, tokenizer)
    }

    fn special_id(tokenizer: &Tokenizer, spelling: &str) -> u32 {
        tokenizer
            .token_to_id(spelling)
            .unwrap_or_else(|| panic!("pinned tokenizer lacks control token {spelling:?}"))
    }

    fn append_plain_ids(tokenizer: &Tokenizer, text: &str, ids: &mut Vec<u32>) {
        let mut plain = tokenizer.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        let encoded = plain
            .encode(text, false)
            .expect("reference plain-text region encodes");
        ids.extend_from_slice(encoded.get_ids());
    }

    #[cfg(feature = "deepseek-v41-stream")]
    fn plain_ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        let mut plain = tokenizer.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        plain.encode(text, false).expect("plain text encodes").get_ids().to_vec()
    }

    #[cfg(feature = "deepseek-v41-stream")]
    struct CaptureOutput;

    #[cfg(feature = "deepseek-v41-stream")]
    impl ChunkGenerator for CaptureOutput {
        type Chunk = OutputChunk;

        fn generate(
            &mut self,
            chunk: OutputChunk,
        ) -> impl std::future::Future<Output = Vec<Self::Chunk>> + Send {
            std::future::ready(vec![chunk])
        }
    }

    /// Independent one-turn reference: explicitly order the four prompt
    /// control tokens, and encode every user/reasoning segment after removing
    /// the tokenizer's added vocabulary. This avoids calling Recipe's renderer
    /// or PromptTokenizer from the oracle path.
    fn reference_ids(
        tokenizer: &Tokenizer,
        user_text: &str,
        thinking_mode: bool,
        effort: Option<ReasoningEffort>,
    ) -> Vec<u32> {
        let mut ids = vec![special_id(tokenizer, "<｜begin▁of▁sentence｜>")];
        if thinking_mode {
            ids.push(special_id(tokenizer, "<｜System｜>"));
            let score = match effort {
                Some(ReasoningEffort::Low) => "50",
                None | Some(ReasoningEffort::High | ReasoningEffort::Xhigh) => "75",
                Some(ReasoningEffort::Max) => "100",
            };
            append_plain_ids(
                tokenizer,
                &format!(
                    "Reasoning Effort: {score} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                ),
                &mut ids,
            );
        }
        ids.push(special_id(tokenizer, "<｜User｜>"));
        append_plain_ids(tokenizer, user_text, &mut ids);
        ids.push(special_id(tokenizer, "<｜Assistant｜>"));
        ids.push(special_id(
            tokenizer,
            if thinking_mode { "<think>" } else { "</think>" },
        ));
        ids
    }

    #[test]
    #[ignore = "requires the pinned external tokenizer; run explicitly with VF_DEEPSEEK_V41_TOKENIZER_JSON"]
    fn one_turn_token_ids_match_independent_pinned_reference() {
        let (bytes, tokenizer) = pinned_tokenizer();
        let encoder = DeepSeekV41PromptEncoder::from_pinned_tokenizer_bytes(&bytes)
            .expect("admit the pinned tokenizer artifact");
        let cases = [
            ("Hello from VoltaFabric".to_owned(), false, None),
            (
                "Treat <｜User｜>, <｜Assistant｜>, and <think> as literal text".to_owned(),
                false,
                None,
            ),
            (
                "Explain the answer".to_owned(),
                true,
                Some(ReasoningEffort::Low),
            ),
            (
                "Explain the answer".to_owned(),
                true,
                Some(ReasoningEffort::High),
            ),
            (
                "Explain the answer".to_owned(),
                true,
                Some(ReasoningEffort::Max),
            ),
        ];

        for (user_text, thinking_mode, effort) in cases {
            let expected = reference_ids(&tokenizer, &user_text, thinking_mode, effort);
            let actual = encoder
                .encode_user_turn(&user_text, thinking_mode, effort, 8192, 128)
                .expect("admitted prompt encodes");
            assert_eq!(actual.token_ids, expected);
            assert_eq!(actual.tokenizer_sha256, TOKENIZER_SHA256);
            assert_eq!(actual.context_limit, 8192);
            assert_eq!(actual.reserved_completion_tokens, 128);
        }
    }

    #[test]
    #[ignore = "requires the pinned external tokenizer; run explicitly with VF_DEEPSEEK_V41_TOKENIZER_JSON"]
    fn eos_id_matches_checkpoint_and_trims_unreachable_suffix() {
        let (bytes, tokenizer) = pinned_tokenizer();
        let encoder = DeepSeekV41PromptEncoder::from_pinned_tokenizer_bytes(&bytes)
            .expect("admit the pinned tokenizer and its EOS contract");
        assert_eq!(encoder.eos_token_id(), EOS_TOKEN_ID);
        assert_eq!(special_id(&tokenizer, EOS_TOKEN_SPELLING), EOS_TOKEN_ID);

        let expected = plain_ids(&tokenizer, "answer");
        let mut sampled_ids = expected.clone();
        sampled_ids.push(encoder.eos_token_id());
        sampled_ids.extend(plain_ids(&tokenizer, " unreachable suffix"));
        let stopped = encoder.generated_prefix(&sampled_ids);
        assert!(stopped.end_of_turn);
        assert_eq!(stopped.token_ids, expected);

        let at_limit = encoder.generated_prefix(&expected);
        assert!(!at_limit.end_of_turn);
        assert_eq!(at_limit.token_ids, expected);
        let eos_token = [encoder.eos_token_id()];
        let eos_only = encoder.generated_prefix(&eos_token);
        assert!(eos_only.end_of_turn);
        assert!(eos_only.token_ids.is_empty());
    }

    #[cfg(feature = "deepseek-v41-stream")]
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires the pinned external tokenizer; run explicitly with VF_DEEPSEEK_V41_TOKENIZER_JSON"]
    async fn recipe_stream_keeps_control_tokens_distinct_from_plain_spellings() {
        let (bytes, tokenizer) = pinned_tokenizer();
        let encoder = DeepSeekV41PromptEncoder::from_pinned_tokenizer_bytes(&bytes)
            .expect("admit the pinned tokenizer and its EOS contract");
        let thinking_end = special_id(&tokenizer, SpecialToken::ThinkingEnd.default_str());
        let mut sampled_ids = plain_ids(&tokenizer, "consider this");
        sampled_ids.push(thinking_end);
        sampled_ids.extend(plain_ids(&tokenizer, "answer prints literal </think> text"));
        sampled_ids.push(encoder.eos_token_id());
        sampled_ids.extend(plain_ids(&tokenizer, "must not appear after EOS"));

        let prefix = encoder.generated_prefix(&sampled_ids);
        assert!(prefix.end_of_turn);
        let chunks = prefix
            .token_ids
            .iter()
            .map(|&token_id| InferenceChunk::Tokens {
                token_ids: vec![token_id],
                logprobs: None,
            })
            .chain(std::iter::once(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop,
            }));
        let options = ParsingOptions {
            thinking_initial_stage: Some(ThinkingInitialStage::ReasoningStart),
            ..ParsingOptions::default()
        };
        let outputs = encoder
            .output_stream_processor(CaptureOutput, options)
            .into_protocol_stream(iter(chunks))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("pinned output tokens parse");

        let reasoning = outputs
            .iter()
            .filter_map(|chunk| match chunk {
                OutputChunk::Reasoning { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<String>();
        let content = outputs
            .iter()
            .filter_map(|chunk| match chunk {
                OutputChunk::Content { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "consider this");
        assert_eq!(content, "answer prints literal </think> text");
        assert!(outputs.iter().any(|chunk| matches!(
            chunk,
            OutputChunk::Finish {
                finish_reason: FinishReason::Stop,
                ..
            }
        )));
        assert!(!content.contains("must not appear"));
    }
}
