use crate::moe::Stamp;
use crate::{Error, Result};

/// Why this deterministic decode sequence ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyFinishReason {
    EndOfSequence,
    LengthLimit,
}

/// One token selected from the exact logits row admitted for this decode identity.
#[derive(Debug, Clone)]
pub struct GreedySampledToken {
    pub stamp: Stamp,
    pub attempt: u64,
    /// Zero-based position in this completion.
    pub token_index: usize,
    pub token_id: u32,
    /// Set on the terminal token; no following sample is accepted.
    pub finish_reason: Option<GreedyFinishReason>,
}

/// CPU reference sampler for deterministic argmax decoding.
///
/// It preserves llama.cpp's ordered-candidate tie behavior: a candidate replaces the
/// current winner only on a strict greater-than comparison, so the first (lowest token
/// ID) wins a tie. The caller supplies logits produced by its model backend; this type
/// does not implement the LM head or claim a backend sampling kernel.
#[derive(Debug)]
pub struct DeepSeekV41GreedyDecoder {
    stamp: Stamp,
    attempt: u64,
    tokenizer_sha256: String,
    vocab_size: usize,
    eos_token_id: u32,
    reserved_completion_tokens: usize,
    max_new_tokens: usize,
    sampled_ids: Vec<u32>,
    finish_reason: Option<GreedyFinishReason>,
}

impl DeepSeekV41GreedyDecoder {
    /// Constructed by the pinned prompt encoder after tokenizer, EOS, identity,
    /// context, and completion-budget admission.
    pub(crate) fn new(
        stamp: Stamp,
        attempt: u64,
        tokenizer_sha256: String,
        vocab_size: usize,
        eos_token_id: u32,
        reserved_completion_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Self> {
        if attempt == 0
            || stamp.run_id.is_empty()
            || stamp.model_revision.is_empty()
            || stamp.plan_id.is_empty()
        {
            return Err(Error::new(
                "INVALID_INPUT",
                "greedy decode requires a complete run/model/plan identity and nonzero attempt",
            ));
        }
        if tokenizer_sha256.len() != 64
            || !tokenizer_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || vocab_size == 0
            || u32::try_from(vocab_size - 1).is_err()
            || eos_token_id as usize >= vocab_size
        {
            return Err(Error::new(
                "INVALID_INPUT",
                "greedy decode tokenizer digest, vocabulary, or EOS ID is invalid",
            ));
        }
        if reserved_completion_tokens == 0
            || max_new_tokens == 0
            || max_new_tokens > reserved_completion_tokens
        {
            return Err(Error::new(
                "INVALID_INPUT",
                "greedy decode length must be positive and fit the admitted completion reservation",
            ));
        }
        Ok(Self {
            stamp,
            attempt,
            tokenizer_sha256,
            vocab_size,
            eos_token_id,
            reserved_completion_tokens,
            max_new_tokens,
            sampled_ids: Vec::with_capacity(max_new_tokens),
            finish_reason: None,
        })
    }

    /// Select one token. Identity, shape, and numerical failures leave decoder state
    /// untouched; a terminal decoder cannot be sampled or replayed.
    pub fn sample_next(
        &mut self,
        stamp: &Stamp,
        attempt: u64,
        logits: &[f32],
    ) -> Result<GreedySampledToken> {
        if self.finish_reason.is_some() {
            return Err(Error::new(
                "DECODE_FINISHED",
                "greedy decoder already reached EOS or its admitted token limit",
            ));
        }
        if !same_stamp(stamp, &self.stamp) {
            return Err(Error::new(
                "IDENTITY_MISMATCH",
                "logits stamp does not match the admitted decode run/model/plan/layer/batch/epoch",
            ));
        }
        if attempt == 0 || attempt != self.attempt {
            return Err(Error::new(
                "ATTEMPT_MISMATCH",
                "logits attempt does not match the admitted decode attempt",
            ));
        }
        if logits.len() != self.vocab_size {
            return Err(Error::new(
                "INVALID_LOGITS",
                "logits row length does not match the pinned tokenizer vocabulary",
            ));
        }

        let mut best: Option<(usize, f32)> = None;
        for (token_index, &logit) in logits.iter().enumerate() {
            if logit.is_nan() || logit == f32::INFINITY {
                return Err(Error::new(
                    "INVALID_LOGITS",
                    "logits contain NaN or positive infinity",
                ));
            }
            if !logit.is_finite() {
                // Negative infinity is a valid masked-out candidate.
                continue;
            }
            if best.map_or(true, |(_, best_logit)| logit > best_logit) {
                best = Some((token_index, logit));
            }
        }
        let (token_index, _) =
            best.ok_or_else(|| Error::new("INVALID_LOGITS", "logits row has no finite candidate"))?;
        let token_id = u32::try_from(token_index)
            .map_err(|_| Error::new("INVALID_LOGITS", "selected token ID exceeds u32"))?;
        let completion_index = self.sampled_ids.len();
        self.sampled_ids.push(token_id);
        let finish_reason = if token_id == self.eos_token_id {
            Some(GreedyFinishReason::EndOfSequence)
        } else if self.sampled_ids.len() == self.max_new_tokens {
            Some(GreedyFinishReason::LengthLimit)
        } else {
            None
        };
        self.finish_reason = finish_reason;

        Ok(GreedySampledToken {
            stamp: self.stamp.clone(),
            attempt: self.attempt,
            token_index: completion_index,
            token_id,
            finish_reason,
        })
    }

    /// Includes EOS when it was sampled; pass these IDs through the matching prompt
    /// encoder's output-prefix method before sending them to the stream processor.
    pub fn sampled_ids(&self) -> &[u32] {
        &self.sampled_ids
    }

    pub fn finish_reason(&self) -> Option<GreedyFinishReason> {
        self.finish_reason
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    pub fn reserved_completion_tokens(&self) -> usize {
        self.reserved_completion_tokens
    }

    pub(crate) fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    pub(crate) fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

fn same_stamp(left: &Stamp, right: &Stamp) -> bool {
    left.run_id == right.run_id
        && left.model_revision == right.model_revision
        && left.plan_id == right.plan_id
        && left.layer == right.layer
        && left.batch == right.batch
        && left.epoch == right.epoch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> Stamp {
        Stamp {
            run_id: "run-r744".into(),
            model_revision: "deepseek-v4.1-flash-pinned".into(),
            plan_id: "plan-r744".into(),
            layer: 60,
            batch: 9,
            epoch: 3,
        }
    }

    fn decoder(max: usize, reservation: usize) -> DeepSeekV41GreedyDecoder {
        DeepSeekV41GreedyDecoder::new(
            stamp(),
            4,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            5,
            4,
            reservation,
            max,
        )
        .unwrap()
    }

    #[test]
    fn greedy_ties_keep_the_lowest_token_id() {
        let mut decoder = decoder(3, 3);
        let step = decoder
            .sample_next(&stamp(), 4, &[0.0, 7.0, 7.0, -1.0, -2.0])
            .unwrap();
        assert_eq!(step.token_id, 1);
        assert_eq!(step.token_index, 0);
        assert_eq!(step.finish_reason, None);
        assert_eq!(decoder.sampled_ids(), &[1]);
    }

    #[test]
    fn eos_is_returned_once_and_marks_the_terminal_step() {
        let mut decoder = decoder(3, 3);
        let step = decoder
            .sample_next(&stamp(), 4, &[0.0, 1.0, 0.0, 0.0, 2.0])
            .unwrap();
        assert_eq!(step.token_id, 4);
        assert_eq!(step.finish_reason, Some(GreedyFinishReason::EndOfSequence));
        assert_eq!(decoder.sampled_ids(), &[4]);
        assert!(decoder.is_finished());
        let err = decoder
            .sample_next(&stamp(), 4, &[0.0, 1.0, 0.0, 0.0, 2.0])
            .unwrap_err();
        assert_eq!(err.code, "DECODE_FINISHED");
        assert_eq!(decoder.sampled_ids(), &[4]);
    }

    #[test]
    fn length_limit_is_terminal_and_fits_the_reserved_budget() {
        let mut decoder = decoder(2, 2);
        decoder
            .sample_next(&stamp(), 4, &[0.0, 4.0, 0.0, 0.0, -1.0])
            .unwrap();
        let last = decoder
            .sample_next(&stamp(), 4, &[0.0, 0.0, 3.0, 0.0, -1.0])
            .unwrap();
        assert_eq!(last.finish_reason, Some(GreedyFinishReason::LengthLimit));
        assert_eq!(decoder.sampled_ids(), &[1, 2]);
    }

    #[test]
    fn wrong_identity_attempt_or_shape_rejects_without_partial_token() {
        let mut decoder = decoder(3, 3);
        let mut stale = stamp();
        stale.epoch += 1;
        assert_eq!(
            decoder
                .sample_next(&stale, 4, &[0.0, 1.0, 0.0, 0.0, 0.0])
                .unwrap_err()
                .code,
            "IDENTITY_MISMATCH"
        );
        assert_eq!(
            decoder
                .sample_next(&stamp(), 5, &[0.0, 1.0, 0.0, 0.0, 0.0])
                .unwrap_err()
                .code,
            "ATTEMPT_MISMATCH"
        );
        assert_eq!(
            decoder
                .sample_next(&stamp(), 4, &[0.0, 1.0])
                .unwrap_err()
                .code,
            "INVALID_LOGITS"
        );
        assert!(decoder.sampled_ids().is_empty());
    }

    #[test]
    fn invalid_logits_are_refused_atomically_but_negative_infinity_can_be_masked() {
        let mut decoder = decoder(3, 3);
        for bad in [
            vec![0.0, f32::NAN, 0.0, 0.0, 0.0],
            vec![0.0, f32::INFINITY, 0.0, 0.0, 0.0],
            vec![f32::NEG_INFINITY; 5],
        ] {
            assert_eq!(
                decoder.sample_next(&stamp(), 4, &bad).unwrap_err().code,
                "INVALID_LOGITS"
            );
            assert!(decoder.sampled_ids().is_empty());
        }
        let step = decoder
            .sample_next(
                &stamp(),
                4,
                &[0.0, f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, -1.0],
            )
            .unwrap();
        assert_eq!(step.token_id, 2);
    }

    #[test]
    fn decoder_requires_real_identity_and_bounded_eos_vocabulary() {
        let mut invalid_stamp = stamp();
        invalid_stamp.plan_id.clear();
        assert_eq!(
            DeepSeekV41GreedyDecoder::new(
                invalid_stamp,
                4,
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                5,
                4,
                4,
                3,
            )
            .unwrap_err()
            .code,
            "INVALID_INPUT"
        );
        assert_eq!(
            DeepSeekV41GreedyDecoder::new(
                stamp(),
                4,
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                5,
                5,
                4,
                3,
            )
            .unwrap_err()
            .code,
            "INVALID_INPUT"
        );
    }
}
