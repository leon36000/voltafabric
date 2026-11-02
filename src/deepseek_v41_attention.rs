//! Bounded CPU reference for DeepSeek-V4.1's one-token sliding-window attention.
//!
//! The reference follows the pinned MIT donor's KV ring ordering and sink-softmax
//! semantics. Q/KV projection and RoPE are separate boundaries: callers provide
//! finite BF16 values after those operations. Compressed-KV selection and the
//! TileLang reduction order are deliberately outside this module.

use crate::{deepseek_v41::cast_bf16_rne_reference, invalid, Error, Result};
use crate::{deepseek_v41_gate::PINNED_MODEL_REVISION, moe::Stamp};

const MAX_WINDOW_SIZE: usize = 4096;
const MAX_CACHE_ELEMENTS: usize = 2_097_152;
const MAX_ATTENTION_MACS: usize = 8_388_608;

/// A single-token attention result, tagged with the exact request identity.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41WindowAttentionOutput {
    stamp: Stamp,
    attempt: u64,
    position: u64,
    context_slots: Vec<i32>,
    output_bf16_f32: Vec<f32>,
}

impl DeepSeekV41WindowAttentionOutput {
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    /// The source-compatible ring indices used for this attention request;
    /// `-1` means the slot has not been written yet.
    pub fn context_slots(&self) -> &[i32] {
        &self.context_slots
    }

    /// One BF16-rounded output row per query head, flattened head-major.
    pub fn output_bf16_f32(&self) -> &[f32] {
        &self.output_bf16_f32
    }
}

/// One-sequence sliding KV ring for the pinned sparse-attention window path.
/// It is bound at construction to one full VoltaFabric token identity and
/// attempt; each append must be the next absolute token position.
#[derive(Debug, Clone)]
pub struct DeepSeekV41WindowCache {
    stamp: Stamp,
    attempt: u64,
    layer: u32,
    query_heads: usize,
    head_dim: usize,
    window_size: usize,
    next_position: u64,
    kv_bf16_f32: Vec<f32>,
}

/// A one-row KV update kept private until its enclosing transformer block succeeds.
/// It owns only the new head row, never a copy of the full sliding-window cache.
#[derive(Debug)]
pub struct DeepSeekV41WindowCacheUpdate {
    stamp: Stamp,
    attempt: u64,
    layer: u32,
    expected_position: u64,
    next_position: u64,
    slot: usize,
    key_value_bf16_f32: Box<[f32]>,
}

impl DeepSeekV41WindowCacheUpdate {
    pub fn expected_position(&self) -> u64 {
        self.expected_position
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }
}

struct WindowKvView<'a> {
    slots: &'a [f32],
    staged_slot: usize,
    staged_key_value: &'a [f32],
}

/// Borrowed inputs for one identity-bound query and KV append.
pub struct DeepSeekV41WindowAttentionRequest<'a> {
    pub stamp: &'a Stamp,
    pub attempt: u64,
    pub position: u64,
    pub query_bf16_f32: &'a [f32],
    pub key_value_bf16_f32: &'a [f32],
    pub attention_sink_f32: &'a [f32],
    pub softmax_scale: f32,
}

impl DeepSeekV41WindowCache {
    pub fn new(
        stamp: Stamp,
        attempt: u64,
        query_heads: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<Self> {
        let cache_elements = window_size
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("DeepSeek window cache size overflow"))?;
        if stamp.model_revision != PINNED_MODEL_REVISION
            || stamp.run_id.is_empty()
            || stamp.plan_id.is_empty()
            || stamp.layer >= 40
            || attempt == 0
            || query_heads == 0
            || query_heads > 256
            || head_dim == 0
            || head_dim > 4096
            || window_size == 0
            || window_size > MAX_WINDOW_SIZE
            || cache_elements > MAX_CACHE_ELEMENTS
            || query_heads
                .checked_mul(head_dim)
                .is_none_or(|elements| elements > MAX_CACHE_ELEMENTS)
        {
            return Err(invalid(
                "DeepSeek window cache identity or dimensions are outside the bounded contract",
            ));
        }
        let mut kv_bf16_f32 = Vec::new();
        kv_bf16_f32.try_reserve_exact(cache_elements).map_err(|_| {
            Error::new("ARTIFACT_BUDGET", "DeepSeek window cache allocation failed")
        })?;
        kv_bf16_f32.resize(cache_elements, 0.0);
        Ok(Self {
            layer: stamp.layer,
            stamp,
            attempt,
            query_heads,
            head_dim,
            window_size,
            next_position: 0,
            kv_bf16_f32,
        })
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Compute against the old ring plus a borrowed replacement row. The returned
    /// update owns one head row and can be committed after a larger block succeeds.
    pub fn prepare_append_and_attend(
        &self,
        request: DeepSeekV41WindowAttentionRequest<'_>,
    ) -> Result<(DeepSeekV41WindowAttentionOutput, DeepSeekV41WindowCacheUpdate)> {
        validate_request_identity(
            &self.stamp,
            self.attempt,
            self.layer,
            request.stamp,
            request.attempt,
        )?;
        if request.position != self.next_position {
            return Err(invalid(
                "DeepSeek window append must match the next absolute token position",
            ));
        }
        let query_elements = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("DeepSeek query size overflow"))?;
        if request.query_bf16_f32.len() != query_elements
            || request.key_value_bf16_f32.len() != self.head_dim
            || request.attention_sink_f32.len() != self.query_heads
            || !request.softmax_scale.is_finite()
            || request.softmax_scale <= 0.0
        {
            return Err(invalid(
                "DeepSeek sparse-window attention input shapes or scale are invalid",
            ));
        }
        validate_exact_bf16("query", request.query_bf16_f32)?;
        validate_exact_bf16("window key/value", request.key_value_bf16_f32)?;
        if request
            .attention_sink_f32
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::new(
                "NONFINITE_INPUT",
                "DeepSeek attention sink contains NaN or infinity",
            ));
        }
        let macs = self
            .query_heads
            .checked_mul(self.head_dim)
            .and_then(|value| value.checked_mul(self.window_size))
            .ok_or_else(|| invalid("DeepSeek attention work count overflow"))?;
        if macs > MAX_ATTENTION_MACS {
            return Err(Error::new(
                "ARTIFACT_BUDGET",
                "DeepSeek sparse-window attention exceeds its CPU reference work limit",
            ));
        }
        let next_position = request
            .position
            .checked_add(1)
            .ok_or_else(|| invalid("DeepSeek token position overflow"))?;
        let position_usize = usize::try_from(request.position)
            .map_err(|_| invalid("DeepSeek token position does not fit usize"))?;
        let context_slots = source_window_indices(self.window_size, position_usize)?;
        let slot = position_usize % self.window_size;
        let output_bf16_f32 = sparse_window_softmax_reference(
            self.query_heads,
            self.head_dim,
            &WindowKvView {
                slots: &self.kv_bf16_f32,
                staged_slot: slot,
                staged_key_value: request.key_value_bf16_f32,
            },
            &context_slots,
            request.query_bf16_f32,
            request.attention_sink_f32,
            request.softmax_scale,
        )?;
        let mut key_value = Vec::new();
        key_value.try_reserve_exact(self.head_dim).map_err(|error| {
            Error::new(
                "ALLOCATION_FAILED",
                format!("could not stage DeepSeek KV row: {error}"),
            )
        })?;
        key_value.extend_from_slice(request.key_value_bf16_f32);
        let update = DeepSeekV41WindowCacheUpdate {
            stamp: self.stamp.clone(),
            attempt: self.attempt,
            layer: self.layer,
            expected_position: request.position,
            next_position,
            slot,
            key_value_bf16_f32: key_value.into_boxed_slice(),
        };
        Ok((
            DeepSeekV41WindowAttentionOutput {
                stamp: self.stamp.clone(),
                attempt: self.attempt,
                position: request.position,
                context_slots,
                output_bf16_f32,
            },
            update,
        ))
    }

    /// Commit one previously prepared row only if the cache is still at the exact
    /// position and identity from which that row was computed.
    pub fn commit_prepared(&mut self, update: DeepSeekV41WindowCacheUpdate) -> Result<()> {
        let expected_next = update
            .expected_position
            .checked_add(1)
            .ok_or_else(|| invalid("DeepSeek staged cache position overflow"))?;
        let expected_slot = usize::try_from(update.expected_position)
            .ok()
            .map(|position| position % self.window_size);
        if update.stamp != self.stamp
            || update.attempt != self.attempt
            || update.layer != self.layer
            || update.expected_position != self.next_position
            || update.next_position != expected_next
            || expected_slot != Some(update.slot)
            || update.key_value_bf16_f32.len() != self.head_dim
        {
            return Err(invalid(
                "DeepSeek staged KV update is stale or belongs to another cache identity",
            ));
        }
        validate_exact_bf16("staged window key/value", &update.key_value_bf16_f32)?;
        let start = update
            .slot
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("DeepSeek staged cache slot offset overflow"))?;
        let end = start
            .checked_add(self.head_dim)
            .ok_or_else(|| invalid("DeepSeek staged cache slot end overflow"))?;
        let target = self
            .kv_bf16_f32
            .get_mut(start..end)
            .ok_or_else(|| invalid("DeepSeek staged cache slot is outside the ring"))?;
        target.copy_from_slice(&update.key_value_bf16_f32);
        self.next_position = update.next_position;
        Ok(())
    }

    /// Preserve the direct append API while using the same staged single-row commit.
    pub fn append_and_attend(
        &mut self,
        request: DeepSeekV41WindowAttentionRequest<'_>,
    ) -> Result<DeepSeekV41WindowAttentionOutput> {
        let (output, update) = self.prepare_append_and_attend(request)?;
        self.commit_prepared(update)?;
        Ok(output)
    }

}

fn validate_request_identity(
    bound: &Stamp,
    bound_attempt: u64,
    bound_layer: u32,
    request: &Stamp,
    attempt: u64,
) -> Result<()> {
    if request != bound
        || attempt != bound_attempt
        || attempt == 0
        || request.layer != bound_layer
        || request.model_revision != PINNED_MODEL_REVISION
    {
        return Err(invalid(
            "DeepSeek attention request does not match its cache run/model/plan/layer/batch/epoch/attempt identity",
        ));
    }
    Ok(())
}

fn validate_exact_bf16(label: &str, values: &[f32]) -> Result<()> {
    for &value in values {
        if !value.is_finite() || cast_bf16_rne_reference(value)?.to_bits() != value.to_bits() {
            return Err(invalid(format!(
                "DeepSeek {label} must contain finite exact-BF16 values"
            )));
        }
    }
    Ok(())
}

/// Translate the pinned decode ring ordering into physical cache slots.
/// Position zero uses prefill's single causal slot; later positions expose the
/// current and prior keys, mask unwritten slots with `-1`, and wrap oldest-first.
fn source_window_indices(window_size: usize, position: usize) -> Result<Vec<i32>> {
    if window_size == 0 || window_size > MAX_WINDOW_SIZE || window_size > i32::MAX as usize {
        return Err(invalid("DeepSeek attention window size is invalid"));
    }
    if position == 0 {
        return Ok(vec![0]);
    }
    let oldest = position % window_size + 1;
    let mut indices = Vec::with_capacity(window_size);
    for slot in oldest..window_size {
        indices.push(if slot > position { -1 } else { slot as i32 });
    }
    for slot in 0..oldest {
        indices.push(if slot > position { -1 } else { slot as i32 });
    }
    Ok(indices)
}

fn sparse_window_softmax_reference(
    query_heads: usize,
    head_dim: usize,
    kv_view: &WindowKvView<'_>,
    context_slots: &[i32],
    query: &[f32],
    attention_sink: &[f32],
    softmax_scale: f32,
) -> Result<Vec<f32>> {
    let kv_slots = kv_view.slots;
    let expected_cache = kv_slots
        .len()
        .checked_div(head_dim)
        .filter(|_| head_dim > 0)
        .and_then(|slot_count| slot_count.checked_mul(head_dim))
        .filter(|length| *length == kv_slots.len())
        .ok_or_else(|| invalid("DeepSeek window cache has an invalid shape"))?;
    if query.len() != query_heads.saturating_mul(head_dim)
        || attention_sink.len() != query_heads
        || context_slots.is_empty()
        || expected_cache != kv_slots.len()
    {
        return Err(invalid("DeepSeek sparse-window attention shape mismatch"));
    }
    let slot_count = kv_slots.len() / head_dim;
    if kv_view.staged_slot >= slot_count || kv_view.staged_key_value.len() != head_dim {
        return Err(invalid("DeepSeek staged window row is outside the cache shape"));
    }
    if context_slots
        .iter()
        .any(|index| *index < -1 || (*index >= 0 && *index as usize >= slot_count))
    {
        return Err(invalid("DeepSeek window index is outside the cache"));
    }

    let mut output = Vec::with_capacity(query.len());
    for head in 0..query_heads {
        let query_row = &query[head * head_dim..(head + 1) * head_dim];
        let sink = f64::from(attention_sink[head]);
        let mut scores = Vec::with_capacity(context_slots.len());
        let mut maximum = sink;
        for &index in context_slots {
            if index == -1 {
                scores.push(None);
                continue;
            }
            let key_row = if index as usize == kv_view.staged_slot {
                kv_view.staged_key_value
            } else {
                &kv_slots[index as usize * head_dim..(index as usize + 1) * head_dim]
            };
            let dot = query_row
                .iter()
                .zip(key_row)
                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                .sum::<f64>();
            let score = dot * f64::from(softmax_scale);
            if !score.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "DeepSeek sparse attention score became nonfinite",
                ));
            }
            maximum = maximum.max(score);
            scores.push(Some((index as usize, score)));
        }

        let sink_weight = (sink - maximum).exp();
        let mut denominator = sink_weight;
        let mut numerator = vec![0.0f64; head_dim];
        for entry in scores.into_iter().flatten() {
            let (index, score) = entry;
            let weight = (score - maximum).exp();
            denominator += weight;
            let key_row = if index == kv_view.staged_slot {
                kv_view.staged_key_value
            } else {
                &kv_slots[index * head_dim..(index + 1) * head_dim]
            };
            for (sum, &value) in numerator.iter_mut().zip(key_row) {
                *sum += weight * f64::from(value);
            }
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek sparse attention denominator is invalid",
            ));
        }
        for value in numerator {
            let rounded = cast_bf16_rne_reference((value / denominator) as f32)?;
            output.push(rounded);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> Stamp {
        Stamp {
            run_id: "r733-run".into(),
            model_revision: PINNED_MODEL_REVISION.into(),
            plan_id: "r733-window-plan".into(),
            layer: 0,
            batch: 4,
            epoch: 2,
        }
    }

    fn request<'a>(
        stamp: &'a Stamp,
        attempt: u64,
        position: u64,
        query_bf16_f32: &'a [f32],
        key_value_bf16_f32: &'a [f32],
        attention_sink_f32: &'a [f32],
        softmax_scale: f32,
    ) -> DeepSeekV41WindowAttentionRequest<'a> {
        DeepSeekV41WindowAttentionRequest {
            stamp,
            attempt,
            position,
            query_bf16_f32,
            key_value_bf16_f32,
            attention_sink_f32,
            softmax_scale,
        }
    }

    #[test]
    fn window_positions_follow_pinned_ring_and_sink_formula() {
        let identity = stamp();
        let mut cache = DeepSeekV41WindowCache::new(identity.clone(), 17, 1, 2, 4).unwrap();
        let query = [1.0f32, 0.0];
        let sink = [0.0f32];
        let mut observed = Vec::new();
        let mut final_output = Vec::new();
        for position in 0..5 {
            let key = [1.0f32, (position as f32) * 0.25];
            let out = cache
                .append_and_attend(request(
                    &identity,
                    17,
                    position,
                    &query,
                    &key,
                    &sink,
                    2.0f32.sqrt().recip(),
                ))
                .unwrap();
            observed.push(out.context_slots().to_vec());
            final_output = out.output_bf16_f32().to_vec();
            assert_eq!(out.stamp(), &identity);
            assert_eq!(out.attempt(), 17);
            assert_eq!(out.position(), position);
            assert_eq!(out.output_bf16_f32().len(), 2);
        }
        assert_eq!(observed[0], [0]);
        assert_eq!(observed[1], [-1, -1, 0, 1]);
        assert_eq!(observed[2], [-1, 0, 1, 2]);
        assert_eq!(observed[3], [0, 1, 2, 3]);
        assert_eq!(observed[4], [1, 2, 3, 0]);

        // For position zero, one key of logit 1/sqrt(2) competes with a
        // denominator-only sink at logit 0. The independent closed form below
        // checks the numerical output, including the BF16 RNE cast.
        let expected = f64::from(1.0f32 / 2.0f32.sqrt()).exp();
        let weight = expected / (expected + 1.0);
        let raw = (weight * 1.0) as f32;
        assert_eq!(observed.len(), 5);
        let first = cache_output_for_first_position(&identity, 17);
        assert_eq!(first, cast_bf16_rne_reference(raw).unwrap());

        // After wrapping, the cache contains positions 4, 1, 2, 3 in slots
        // 0..3. The attention result is order-independent; four equal key
        // logits compete with the sink and their values average to 0.625.
        let key_weight = expected / (1.0 + 4.0 * expected);
        assert_eq!(
            final_output,
            [
                cast_bf16_rne_reference((4.0 * key_weight) as f32).unwrap(),
                cast_bf16_rne_reference((0.625 * 4.0 * key_weight) as f32).unwrap(),
            ]
        );
    }

    fn cache_output_for_first_position(identity: &Stamp, attempt: u64) -> f32 {
        let mut cache = DeepSeekV41WindowCache::new(identity.clone(), attempt, 1, 2, 4).unwrap();
        cache
            .append_and_attend(request(
                identity,
                attempt,
                0,
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[0.0],
                2.0f32.sqrt().recip(),
            ))
            .unwrap()
            .output_bf16_f32()[0]
    }

    #[test]
    fn changed_identity_and_skipped_position_do_not_advance_cache() {
        let identity = stamp();
        let mut cache = DeepSeekV41WindowCache::new(identity.clone(), 8, 1, 2, 4).unwrap();
        let query = [1.0, 0.0];
        let key = [1.0, 0.0];
        let sink = [0.0];
        assert!(cache
            .append_and_attend(request(&identity, 8, 1, &query, &key, &sink, 1.0))
            .is_err());
        assert_eq!(cache.next_position(), 0);

        let mut foreign = identity.clone();
        foreign.epoch += 1;
        assert!(cache
            .append_and_attend(request(&foreign, 8, 0, &query, &key, &sink, 1.0))
            .is_err());
        assert_eq!(cache.next_position(), 0);

        let first = cache
            .append_and_attend(request(&identity, 8, 0, &query, &key, &sink, 1.0))
            .unwrap();
        assert_eq!(first.position(), 0);
        assert_eq!(cache.next_position(), 1);
    }

    #[test]
    fn malformed_bf16_or_attention_inputs_fail_atomically() {
        let identity = stamp();
        let mut cache = DeepSeekV41WindowCache::new(identity.clone(), 3, 1, 2, 4).unwrap();
        let not_bf16 = [1.001, 0.0];
        assert!(cache
            .append_and_attend(request(
                &identity,
                3,
                0,
                &not_bf16,
                &[1.0, 0.0],
                &[0.0],
                1.0,
            ))
            .is_err());
        assert_eq!(cache.next_position(), 0);
        assert!(cache
            .append_and_attend(request(
                &identity,
                3,
                0,
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[f32::NAN],
                1.0,
            ))
            .is_err());
        assert_eq!(cache.next_position(), 0);
        assert!(cache
            .append_and_attend(request(
                &identity,
                3,
                0,
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[0.0],
                0.0,
            ))
            .is_err());
        assert_eq!(cache.next_position(), 0);
    }

    #[test]
    fn sink_only_window_returns_zero_without_nan() {
        let identity = stamp();
        let mut cache = DeepSeekV41WindowCache::new(identity.clone(), 6, 1, 2, 4).unwrap();
        let output = cache
            .append_and_attend(request(
                &identity,
                6,
                0,
                &[1.0, 0.0],
                &[0.0, 0.0],
                &[1000.0],
                1.0,
            ))
            .unwrap();
        assert_eq!(output.output_bf16_f32(), [0.0, 0.0]);
        assert!(output
            .output_bf16_f32()
            .iter()
            .all(|value| value.is_finite()));
    }
}
