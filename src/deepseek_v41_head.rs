//! Greedy reduction for vocabulary-sharded DeepSeek V4.1 output logits.
//!
//! The backend supplies one FP32 logit slice per explicit, contiguous vocabulary
//! range. This reducer keeps only one `(score, token_id)` candidate and emits it
//! only after every planned range has arrived. It does not compute the output
//! projection and it does not authenticate a remote transport or worker.

use crate::moe::Stamp;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_VOCAB_SHARDS: usize = 4096;
const PLAN_DOMAIN: &[u8] = b"VF-DEEPSEEK-V41-GREEDY-VOCAB-PLAN-V1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabRange {
    pub rank: u32,
    pub start: usize,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct ShardedGreedyPlan {
    vocab_size: usize,
    tokenizer_sha256: String,
    ranges: Vec<VocabRange>,
    ranges_by_rank: BTreeMap<u32, VocabRange>,
    sha256: String,
}

impl ShardedGreedyPlan {
    /// Admit an explicit partition. Ranges must cover `[0, vocab_size)` exactly
    /// once; rank order may differ from token order, but range boundaries may not.
    pub fn new(
        vocab_size: usize,
        tokenizer_sha256: &str,
        mut ranges: Vec<VocabRange>,
    ) -> Result<Self> {
        if vocab_size == 0
            || u32::try_from(vocab_size - 1).is_err()
            || !crate::sha256::is_hex_digest(tokenizer_sha256)
            || tokenizer_sha256 != tokenizer_sha256.to_ascii_lowercase()
            || ranges.is_empty()
            || ranges.len() > MAX_VOCAB_SHARDS
            || ranges.len() > vocab_size
        {
            return Err(invalid("vocabulary partition profile is invalid"));
        }

        ranges.sort_by_key(|range| (range.start, range.rank));
        let mut cursor = 0usize;
        let mut ranges_by_rank = BTreeMap::new();
        for range in &ranges {
            if range.count == 0
                || range.start != cursor
                || range.start.checked_add(range.count).is_none()
                || range.start + range.count > vocab_size
                || ranges_by_rank.insert(range.rank, *range).is_some()
            {
                return Err(invalid(
                    "vocabulary ranges contain a gap, overlap, duplicate rank, or overflow",
                ));
            }
            cursor += range.count;
        }
        if cursor != vocab_size {
            return Err(invalid(
                "vocabulary ranges do not cover the pinned vocabulary",
            ));
        }

        let mut canonical = Vec::with_capacity(PLAN_DOMAIN.len() + 8 + 64 + ranges.len() * 16);
        canonical.extend_from_slice(PLAN_DOMAIN);
        canonical.extend_from_slice(&(vocab_size as u64).to_le_bytes());
        canonical.extend_from_slice(tokenizer_sha256.as_bytes());
        for range in &ranges {
            canonical.extend_from_slice(&range.rank.to_le_bytes());
            canonical.extend_from_slice(&(range.start as u64).to_le_bytes());
            canonical.extend_from_slice(&(range.count as u64).to_le_bytes());
        }
        let sha256 = crate::sha256::hex(&canonical);

        Ok(Self {
            vocab_size,
            tokenizer_sha256: tokenizer_sha256.to_owned(),
            ranges,
            ranges_by_rank,
            sha256,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn rank_count(&self) -> usize {
        self.ranges.len()
    }

    /// Compute one rank-local winner next to the rank-local logits. The returned
    /// value carries only identity, rank, and an optional `(token_id, score)` pair;
    /// a future transport can send this small record instead of the full slice.
    /// This method does not implement or authenticate that transport.
    pub fn local_argmax(
        &self,
        stamp: &Stamp,
        attempt: u64,
        token_index: usize,
        rank: u32,
        logits: &[f32],
    ) -> Result<ShardedGreedyPartial> {
        validate_stamp(stamp)?;
        if attempt == 0 {
            return Err(invalid("rank-local argmax requires a nonzero attempt"));
        }
        let range = *self
            .ranges_by_rank
            .get(&rank)
            .ok_or_else(|| invalid("rank-local logits came from an unplanned rank"))?;
        if logits.len() != range.count {
            return Err(Error::new(
                "INVALID_LOGITS",
                "rank-local vocabulary slice length differs from its admitted range",
            ));
        }

        let mut best: Option<(u32, f32)> = None;
        for (offset, &score) in logits.iter().enumerate() {
            if score.is_nan() || score == f32::INFINITY {
                return Err(Error::new(
                    "INVALID_LOGITS",
                    "rank-local vocabulary slice contains NaN or positive infinity",
                ));
            }
            if score == f32::NEG_INFINITY {
                continue;
            }
            let token_id = u32::try_from(range.start + offset)
                .map_err(|_| invalid("vocabulary token ID exceeds u32"))?;
            if best.is_none_or(|(best_id, best_score)| {
                score > best_score || (score == best_score && token_id < best_id)
            }) {
                best = Some((token_id, score));
            }
        }

        Ok(ShardedGreedyPartial {
            stamp: stamp.clone(),
            attempt,
            token_index,
            tokenizer_sha256: self.tokenizer_sha256.clone(),
            plan_sha256: self.sha256.clone(),
            rank,
            best,
        })
    }
}

/// Rank-local reduction result. Its private fields prevent accidental mutation
/// between local argmax and collector admission inside one process. A serialized
/// transport must still authenticate and revalidate its decoded fields.
#[derive(Debug)]
pub struct ShardedGreedyPartial {
    stamp: Stamp,
    attempt: u64,
    token_index: usize,
    tokenizer_sha256: String,
    plan_sha256: String,
    rank: u32,
    best: Option<(u32, f32)>,
}

impl ShardedGreedyPartial {
    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn local_best(&self) -> Option<(u32, f32)> {
        self.best
    }
}

const MAX_GREEDY_PARTIAL_FRAME_BYTES: usize = 4096;
const GREEDY_PARTIAL_WIRE_KIND: &str = "deepseek_v41_greedy_partial";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardedGreedyPartialWireV1 {
    version: u8,
    kind: String,
    stamp: Stamp,
    attempt: u64,
    token_index: u64,
    tokenizer_sha256: String,
    plan_sha256: String,
    rank: u32,
    best_token_id: Option<u32>,
    best_score_bits: Option<u32>,
}

impl ShardedGreedyPartial {
    /// Encode one bounded UTF-8 JSON line. This is a versioned payload codec;
    /// the current session::Reply enum still needs a dedicated variant before
    /// this line is accepted by a VoltaFabric Session.
    pub fn encode_wire_frame(&self) -> Result<Vec<u8>> {
        let token_index = u64::try_from(self.token_index)
            .map_err(|_| invalid_wire("token position exceeds the wire integer range"))?;
        let (best_token_id, best_score_bits) = match self.best {
            Some((token_id, score)) => (Some(token_id), Some(score.to_bits())),
            None => (None, None),
        };
        let wire = ShardedGreedyPartialWireV1 {
            version: 1,
            kind: GREEDY_PARTIAL_WIRE_KIND.to_owned(),
            stamp: self.stamp.clone(),
            attempt: self.attempt,
            token_index,
            tokenizer_sha256: self.tokenizer_sha256.clone(),
            plan_sha256: self.plan_sha256.clone(),
            rank: self.rank,
            best_token_id,
            best_score_bits,
        };
        let mut frame = serde_json::to_vec(&wire)
            .map_err(|_| invalid_wire("rank-local winner could not be encoded"))?;
        if frame
            .len()
            .checked_add(1)
            .is_none_or(|n| n > MAX_GREEDY_PARTIAL_FRAME_BYTES)
        {
            return Err(invalid_wire(
                "rank-local winner exceeds the bounded wire frame",
            ));
        }
        frame.push(b'\n');
        Ok(frame)
    }

    /// Decode exactly one version-1 JSON line, rejecting oversized input before
    /// JSON parsing. The peer rank is checked separately by the reducer caller.
    pub fn decode_wire_frame(frame: &[u8]) -> Result<Self> {
        if frame.len() < 2
            || frame.len() > MAX_GREEDY_PARTIAL_FRAME_BYTES
            || frame.last() != Some(&b'\n')
            || frame[..frame.len() - 1]
                .iter()
                .any(|b| *b == b'\n' || *b == b'\r')
        {
            return Err(invalid_wire(
                "rank-local winner is not one bounded JSON line",
            ));
        }
        let wire: ShardedGreedyPartialWireV1 = serde_json::from_slice(&frame[..frame.len() - 1])
            .map_err(|_| {
                invalid_wire("rank-local winner JSON is malformed or has unknown fields")
            })?;
        if wire.version != 1 || wire.kind != GREEDY_PARTIAL_WIRE_KIND {
            return Err(invalid_wire(
                "unsupported rank-local winner wire kind or version",
            ));
        }
        validate_stamp(&wire.stamp)
            .map_err(|_| invalid_wire("rank-local winner identity is incomplete"))?;
        if wire.attempt == 0
            || !crate::sha256::is_hex_digest(&wire.tokenizer_sha256)
            || wire.tokenizer_sha256 != wire.tokenizer_sha256.to_ascii_lowercase()
            || !crate::sha256::is_hex_digest(&wire.plan_sha256)
            || wire.plan_sha256 != wire.plan_sha256.to_ascii_lowercase()
        {
            return Err(invalid_wire(
                "rank-local winner identity or digest is invalid",
            ));
        }
        let best = match (wire.best_token_id, wire.best_score_bits) {
            (None, None) => None,
            (Some(token_id), Some(bits)) => {
                let score = f32::from_bits(bits);
                if !score.is_finite() {
                    return Err(invalid_wire("rank-local winner score is not finite"));
                }
                Some((token_id, score))
            }
            _ => {
                return Err(invalid_wire(
                    "rank-local winner score fields are incomplete",
                ))
            }
        };
        let token_index = usize::try_from(wire.token_index)
            .map_err(|_| invalid_wire("token position exceeds the host integer range"))?;
        Ok(Self {
            stamp: wire.stamp,
            attempt: wire.attempt,
            token_index,
            tokenizer_sha256: wire.tokenizer_sha256,
            plan_sha256: wire.plan_sha256,
            rank: wire.rank,
            best,
        })
    }
}

impl ShardedGreedyReducer {
    /// Bind the claimed rank to the caller's trusted Session endpoint identity,
    /// then pass through the reducer's complete identity and atomicity checks.
    /// This comparison does not authenticate the transport or peer.
    pub fn accept_wire_frame(&mut self, frame: &[u8], expected_session_rank: u32) -> Result<()> {
        let partial = ShardedGreedyPartial::decode_wire_frame(frame)?;
        if partial.rank != expected_session_rank {
            return Err(Error::new(
                "SESSION_RANK_MISMATCH",
                "rank-local winner does not match the Session endpoint rank",
            ));
        }
        self.accept_partial(partial)
    }
}

fn invalid_wire(message: &str) -> Error {
    Error::new("INVALID_GREEDY_FRAME", message)
}

/// Collects local argmax values from all head shards without retaining or
/// all-gathering their full vocabulary logits.
#[derive(Debug)]
pub struct ShardedGreedyReducer {
    plan: ShardedGreedyPlan,
    stamp: Stamp,
    attempt: u64,
    token_index: usize,
    seen_ranks: BTreeSet<u32>,
    best: Option<(u32, f32)>,
}

impl ShardedGreedyReducer {
    pub fn new(
        plan: ShardedGreedyPlan,
        stamp: Stamp,
        attempt: u64,
        token_index: usize,
    ) -> Result<Self> {
        validate_stamp(&stamp)?;
        if attempt == 0 {
            return Err(invalid(
                "sharded greedy reduction requires a nonzero attempt",
            ));
        }
        Ok(Self {
            plan,
            stamp,
            attempt,
            token_index,
            seen_ranks: BTreeSet::new(),
            best: None,
        })
    }

    /// Merge a small rank-local top-1 record. All identity, partition, duplicate,
    /// token-range, and score checks finish before the receiver state is mutated.
    pub fn accept_partial(&mut self, partial: ShardedGreedyPartial) -> Result<()> {
        if !same_stamp(&partial.stamp, &self.stamp) {
            return Err(Error::new(
                "IDENTITY_MISMATCH",
                "rank-local winner does not match the admitted run/model/plan/layer/batch/epoch",
            ));
        }
        if partial.attempt == 0 || partial.attempt != self.attempt {
            return Err(Error::new(
                "ATTEMPT_MISMATCH",
                "rank-local winner attempt differs from the admitted decode attempt",
            ));
        }
        if partial.token_index != self.token_index {
            return Err(Error::new(
                "TOKEN_INDEX_MISMATCH",
                "rank-local winner belongs to another completion position",
            ));
        }
        if partial.tokenizer_sha256 != self.plan.tokenizer_sha256
            || partial.plan_sha256 != self.plan.sha256
        {
            return Err(Error::new(
                "PLAN_MISMATCH",
                "rank-local winner tokenizer or vocabulary plan differs from the admitted plan",
            ));
        }
        let range = *self
            .plan
            .ranges_by_rank
            .get(&partial.rank)
            .ok_or_else(|| invalid("rank-local winner came from an unplanned rank"))?;
        if self.seen_ranks.contains(&partial.rank) {
            return Err(Error::new(
                "DUPLICATE_SHARD",
                "vocabulary rank submitted more than one winner",
            ));
        }
        if let Some((token_id, score)) = partial.best {
            let token_id = token_id as usize;
            if token_id < range.start || token_id >= range.start + range.count || !score.is_finite()
            {
                return Err(Error::new(
                    "INVALID_LOGITS",
                    "rank-local winner token or score falls outside its admitted range",
                ));
            }
        }

        self.seen_ranks.insert(partial.rank);
        if let Some(candidate) = partial.best {
            if self.best.is_none_or(|(best_id, best_score)| {
                candidate.1 > best_score || (candidate.1 == best_score && candidate.0 < best_id)
            }) {
                self.best = Some(candidate);
            }
        }
        Ok(())
    }

    /// Return a candidate only after every range was received and at least one
    /// finite logit exists. Consuming `self` makes incomplete reduction terminal.
    pub fn finish(self) -> Result<ShardedGreedyCandidate> {
        if self.seen_ranks.len() != self.plan.ranges.len() {
            return Err(Error::new(
                "PARTIAL_LOGITS",
                "not every planned vocabulary range completed",
            ));
        }
        let (token_id, score) = self
            .best
            .ok_or_else(|| invalid("all vocabulary logits are masked"))?;
        Ok(ShardedGreedyCandidate {
            stamp: self.stamp,
            attempt: self.attempt,
            token_index: self.token_index,
            tokenizer_sha256: self.plan.tokenizer_sha256,
            plan_sha256: self.plan.sha256,
            vocab_size: self.plan.vocab_size,
            token_id,
            score,
        })
    }
}

/// A non-forgeable top-1 result emitted only by a complete shard reduction.
#[derive(Debug)]
pub struct ShardedGreedyCandidate {
    stamp: Stamp,
    attempt: u64,
    token_index: usize,
    tokenizer_sha256: String,
    plan_sha256: String,
    vocab_size: usize,
    token_id: u32,
    score: f32,
}

impl ShardedGreedyCandidate {
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    pub fn score(&self) -> f32 {
        self.score
    }

    pub(crate) fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    pub(crate) fn attempt(&self) -> u64 {
        self.attempt
    }

    pub(crate) fn token_index(&self) -> usize {
        self.token_index
    }

    pub(crate) fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    pub(crate) fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
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

fn validate_stamp(stamp: &Stamp) -> Result<()> {
    if stamp.run_id.is_empty() || stamp.model_revision.is_empty() || stamp.plan_id.is_empty() {
        return Err(invalid(
            "sharded greedy reduction requires a complete run/model/plan identity",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_INPUT", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v41_sampler::DeepSeekV41GreedyDecoder;

    const TOKENIZER: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn stamp() -> Stamp {
        Stamp {
            run_id: "run-r745".into(),
            model_revision: "deepseek-v4.1-flash-pinned".into(),
            plan_id: "plan-r745".into(),
            layer: 40,
            batch: 8,
            epoch: 2,
        }
    }

    fn plan() -> ShardedGreedyPlan {
        ShardedGreedyPlan::new(
            5,
            TOKENIZER,
            vec![
                VocabRange {
                    rank: 7,
                    start: 0,
                    count: 2,
                },
                VocabRange {
                    rank: 3,
                    start: 2,
                    count: 3,
                },
            ],
        )
        .unwrap()
    }

    fn reducer(plan: ShardedGreedyPlan) -> ShardedGreedyReducer {
        ShardedGreedyReducer::new(plan, stamp(), 9, 0).unwrap()
    }

    fn partial(
        plan: &ShardedGreedyPlan,
        stamp: &Stamp,
        attempt: u64,
        token_index: usize,
        rank: u32,
        logits: &[f32],
    ) -> ShardedGreedyPartial {
        plan.local_argmax(stamp, attempt, token_index, rank, logits)
            .unwrap()
    }

    fn flat_argmax(logits: &[f32]) -> Option<(u32, f32)> {
        let mut best = None;
        for (token_id, score) in logits.iter().copied().enumerate() {
            if !score.is_finite() {
                continue;
            }
            let token_id = u32::try_from(token_id).ok()?;
            if best.is_none_or(|(best_id, best_score)| {
                score > best_score || (score == best_score && token_id < best_id)
            }) {
                best = Some((token_id, score));
            }
        }
        best
    }

    #[test]
    fn versioned_wire_roundtrip_preserves_exact_fp32_score_bits() {
        let plan = plan();
        let score = f32::from_bits(0x3f80_0001);
        let partial = plan.local_argmax(&stamp(), 9, 0, 7, &[0.0, score]).unwrap();
        let frame = partial.encode_wire_frame().unwrap();
        assert_eq!(frame.last(), Some(&b'\n'));
        assert_eq!(frame.iter().filter(|b| **b == b'\n').count(), 1);
        let decoded = ShardedGreedyPartial::decode_wire_frame(&frame).unwrap();
        assert_eq!(decoded.rank(), 7);
        assert_eq!(decoded.local_best().unwrap().1.to_bits(), score.to_bits());
    }

    #[test]
    fn versioned_wire_rejects_bad_versions_unknown_fields_and_bad_framing() {
        let plan = plan();
        let partial_value = partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]);
        let frame = partial_value.encode_wire_frame().unwrap();

        let json = std::str::from_utf8(&frame[..frame.len() - 1]).unwrap();
        let bad_version = json.replace("\"version\":1", "\"version\":2");
        let mut bad_version = bad_version.into_bytes();
        bad_version.push(b'\n');
        assert_eq!(
            ShardedGreedyPartial::decode_wire_frame(&bad_version)
                .unwrap_err()
                .code,
            "INVALID_GREEDY_FRAME"
        );

        let mut wrong_kind: serde_json::Value =
            serde_json::from_slice(&frame[..frame.len() - 1]).unwrap();
        wrong_kind.as_object_mut().unwrap().insert(
            "kind".to_owned(),
            serde_json::Value::String("worker_event".to_owned()),
        );
        let mut wrong_kind = serde_json::to_vec(&wrong_kind).unwrap();
        wrong_kind.push(b'\n');
        assert!(ShardedGreedyPartial::decode_wire_frame(&wrong_kind).is_err());

        let mut value: serde_json::Value =
            serde_json::from_slice(&frame[..frame.len() - 1]).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("extra".to_owned(), serde_json::Value::Bool(true));
        let mut unknown = serde_json::to_vec(&value).unwrap();
        unknown.push(b'\n');
        assert_eq!(
            ShardedGreedyPartial::decode_wire_frame(&unknown)
                .unwrap_err()
                .code,
            "INVALID_GREEDY_FRAME"
        );

        assert!(ShardedGreedyPartial::decode_wire_frame(&frame[..frame.len() - 1]).is_err());
        let mut multiline = frame.clone();
        multiline.extend_from_slice(b"{}\n");
        assert!(ShardedGreedyPartial::decode_wire_frame(&multiline).is_err());
        assert!(ShardedGreedyPartial::decode_wire_frame(&vec![
            b' ';
            MAX_GREEDY_PARTIAL_FRAME_BYTES + 1
        ])
        .is_err());
    }

    #[test]
    fn versioned_wire_rejects_incomplete_or_nonfinite_winner() {
        let plan = plan();
        let partial = partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]);
        let frame = partial.encode_wire_frame().unwrap();

        let mut missing_score: serde_json::Value =
            serde_json::from_slice(&frame[..frame.len() - 1]).unwrap();
        missing_score
            .as_object_mut()
            .unwrap()
            .insert("best_score_bits".to_owned(), serde_json::Value::Null);
        let mut missing_score = serde_json::to_vec(&missing_score).unwrap();
        missing_score.push(b'\n');
        assert!(ShardedGreedyPartial::decode_wire_frame(&missing_score).is_err());

        let mut nonfinite: serde_json::Value =
            serde_json::from_slice(&frame[..frame.len() - 1]).unwrap();
        nonfinite.as_object_mut().unwrap().insert(
            "best_score_bits".to_owned(),
            serde_json::Value::from(f32::NAN.to_bits()),
        );
        let mut nonfinite = serde_json::to_vec(&nonfinite).unwrap();
        nonfinite.push(b'\n');
        assert!(ShardedGreedyPartial::decode_wire_frame(&nonfinite).is_err());
    }

    #[test]
    fn wire_rank_mismatch_is_rejected_without_consuming_shard() {
        let plan = plan();
        let mut reducer = reducer(plan.clone());
        let rank7 = partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0])
            .encode_wire_frame()
            .unwrap();
        assert_eq!(
            reducer.accept_wire_frame(&rank7, 3).unwrap_err().code,
            "SESSION_RANK_MISMATCH"
        );
        reducer.accept_wire_frame(&rank7, 7).unwrap();

        let rank3 = partial(&plan, &stamp(), 9, 0, 3, &[3.0, 4.0, 0.0])
            .encode_wire_frame()
            .unwrap();
        reducer.accept_wire_frame(&rank3, 3).unwrap();
        assert_eq!(reducer.finish().unwrap().token_id(), 3);
    }

    #[test]
    fn complete_shards_reduce_to_first_global_argmax_and_feed_decoder() {
        let plan = plan();
        let plan_sha = plan.sha256().to_owned();
        let mut reduce = reducer(plan.clone());
        // Deliberately accept the later vocabulary range first. Equal maxima at
        // IDs 1 and 2 must still choose the lowest global token ID.
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 3, &[5.0, 4.0, 5.0]))
            .unwrap();
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[0.0, 5.0]))
            .unwrap();
        let candidate = reduce.finish().unwrap();
        assert_eq!(candidate.token_id(), 1);
        assert_eq!(candidate.score(), 5.0);

        let mut decoder =
            DeepSeekV41GreedyDecoder::new(stamp(), 9, TOKENIZER.to_owned(), 5, 4, 3, 3).unwrap();
        let selected = decoder
            .sample_next_from_shards(&candidate, &plan_sha)
            .unwrap();
        assert_eq!(selected.token_id, 1);
        assert_eq!(selected.token_index, 0);
        assert_eq!(decoder.sampled_ids(), &[1]);
        assert_eq!(
            decoder
                .sample_next_from_shards(&candidate, &plan_sha)
                .unwrap_err()
                .code,
            "TOKEN_INDEX_MISMATCH"
        );
    }

    #[test]
    fn incomplete_partition_never_emits_a_candidate() {
        let plan = plan();
        let mut reduce = reducer(plan.clone());
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]))
            .unwrap();
        assert_eq!(reduce.finish().unwrap_err().code, "PARTIAL_LOGITS");
    }

    #[test]
    fn invalid_shard_is_atomic_and_fully_masked_shard_is_allowed() {
        let plan = plan();
        let mut reduce = reducer(plan.clone());
        assert_eq!(
            plan.local_argmax(&stamp(), 9, 0, 7, &[1.0, f32::NAN])
                .unwrap_err()
                .code,
            "INVALID_LOGITS"
        );
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[f32::NEG_INFINITY; 2]))
            .unwrap();
        reduce
            .accept_partial(partial(
                &plan,
                &stamp(),
                9,
                0,
                3,
                &[f32::NEG_INFINITY, 2.0, 1.0],
            ))
            .unwrap();
        assert_eq!(reduce.finish().unwrap().token_id(), 3);
    }

    #[test]
    fn identity_attempt_position_plan_and_duplicate_rank_are_checked() {
        let plan = plan();
        let mut reduce = reducer(plan.clone());
        let mut stale = stamp();
        stale.epoch += 1;
        assert_eq!(
            reduce
                .accept_partial(partial(&plan, &stale, 9, 0, 7, &[1.0, 2.0]))
                .unwrap_err()
                .code,
            "IDENTITY_MISMATCH"
        );
        assert_eq!(
            reduce
                .accept_partial(partial(&plan, &stamp(), 8, 0, 7, &[1.0, 2.0]))
                .unwrap_err()
                .code,
            "ATTEMPT_MISMATCH"
        );
        assert_eq!(
            reduce
                .accept_partial(partial(&plan, &stamp(), 9, 1, 7, &[1.0, 2.0]))
                .unwrap_err()
                .code,
            "TOKEN_INDEX_MISMATCH"
        );
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]))
            .unwrap();
        assert_eq!(
            reduce
                .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]))
                .unwrap_err()
                .code,
            "DUPLICATE_SHARD"
        );
        assert_eq!(reduce.finish().unwrap_err().code, "PARTIAL_LOGITS");
    }

    #[test]
    fn partition_rejects_gaps_overlaps_and_reused_ranks() {
        for ranges in [
            vec![
                VocabRange {
                    rank: 0,
                    start: 0,
                    count: 1,
                },
                VocabRange {
                    rank: 1,
                    start: 2,
                    count: 3,
                },
            ],
            vec![
                VocabRange {
                    rank: 0,
                    start: 0,
                    count: 3,
                },
                VocabRange {
                    rank: 1,
                    start: 2,
                    count: 3,
                },
            ],
            vec![
                VocabRange {
                    rank: 0,
                    start: 0,
                    count: 2,
                },
                VocabRange {
                    rank: 0,
                    start: 2,
                    count: 3,
                },
            ],
        ] {
            assert_eq!(
                ShardedGreedyPlan::new(5, TOKENIZER, ranges)
                    .unwrap_err()
                    .code,
                "INVALID_INPUT"
            );
        }
    }

    #[test]
    fn all_masked_vocabulary_and_positive_infinity_fail_closed() {
        let plan = plan();
        let mut reduce = reducer(plan.clone());
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[f32::NEG_INFINITY; 2]))
            .unwrap();
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 3, &[f32::NEG_INFINITY; 3]))
            .unwrap();
        assert_eq!(reduce.finish().unwrap_err().code, "INVALID_INPUT");

        assert_eq!(
            plan.local_argmax(&stamp(), 9, 0, 7, &[0.0, f32::INFINITY])
                .unwrap_err()
                .code,
            "INVALID_LOGITS"
        );
    }

    #[test]
    fn rank_local_top1_merge_matches_independent_flat_argmax() {
        let ranges = vec![
            VocabRange {
                rank: 7,
                start: 0,
                count: 5,
            },
            VocabRange {
                rank: 3,
                start: 5,
                count: 7,
            },
            VocabRange {
                rank: 9,
                start: 12,
                count: 5,
            },
        ];
        let plan = ShardedGreedyPlan::new(17, TOKENIZER, ranges).unwrap();

        for seed in 0..97usize {
            let mut flat: Vec<f32> = (0..17)
                .map(|token| {
                    let value = (seed * 29 + token * 17 + token * token % 11) % 23;
                    (value as i32 - 11) as f32 / 3.0
                })
                .collect();
            if seed % 7 == 0 {
                flat[2] = f32::NEG_INFINITY;
                flat[15] = f32::NEG_INFINITY;
            }
            if seed % 3 == 0 {
                flat[4] = flat[6];
            }

            let expected = flat_argmax(&flat).unwrap();
            let mut reduce = reducer(plan.clone());
            for rank in [9u32, 7, 3] {
                let range = plan.ranges_by_rank.get(&rank).unwrap();
                let slice = &flat[range.start..range.start + range.count];
                reduce
                    .accept_partial(partial(&plan, &stamp(), 9, 0, rank, slice))
                    .unwrap();
            }
            let actual = reduce.finish().unwrap();
            assert_eq!((actual.token_id(), actual.score()), expected, "seed={seed}");
        }
    }

    #[test]
    fn malformed_partial_is_atomic_and_cannot_claim_another_ranks_token() {
        let plan = plan();
        let mut reduce = reducer(plan.clone());
        let invalid = ShardedGreedyPartial {
            stamp: stamp(),
            attempt: 9,
            token_index: 0,
            tokenizer_sha256: plan.tokenizer_sha256.clone(),
            plan_sha256: plan.sha256.clone(),
            rank: 7,
            best: Some((3, 8.0)),
        };
        assert_eq!(
            reduce.accept_partial(invalid).unwrap_err().code,
            "INVALID_LOGITS"
        );
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]))
            .unwrap();
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 3, &[3.0, 2.0, 1.0]))
            .unwrap();
        assert_eq!(reduce.finish().unwrap().token_id(), 2);
    }

    #[test]
    fn partial_from_another_vocabulary_plan_is_rejected_without_consuming_rank() {
        let plan = plan();
        let other = ShardedGreedyPlan::new(5, &"f".repeat(64), plan.ranges.clone()).unwrap();
        let mut reduce = reducer(plan.clone());
        assert_eq!(
            reduce
                .accept_partial(partial(&other, &stamp(), 9, 0, 7, &[1.0, 2.0]))
                .unwrap_err()
                .code,
            "PLAN_MISMATCH"
        );
        reduce
            .accept_partial(partial(&plan, &stamp(), 9, 0, 7, &[1.0, 2.0]))
            .unwrap();
    }
}
