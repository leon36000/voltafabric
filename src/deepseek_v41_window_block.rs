//! One-token composition of the pinned pure-window attention and routed FFN paths.
//!
//! Admitted resident weights enter through existing VoltaFabric objects; the route
//! executor is an explicit one-shot trust boundary to rank Sessions. This module
//! neither reads model payloads nor retries an ambiguous route. It keeps a candidate
//! KV cache private until the complete attention+FFN block and both HC joins succeed.

use crate::deepseek_v41::PinnedMxfp4RouteExecutionReceipt;
use crate::deepseek_v41_attention::DeepSeekV41WindowCache;
use crate::deepseek_v41_attention_output::{
    DeepSeekV41AttentionLayerOutput, DeepSeekV41AttentionOutputProjection,
};
use crate::deepseek_v41_gate::{
    combine_pinned_moe_route_with_shared, hc::DeepSeekV41HcProjection, DeepSeekV41IndexedMoeLayer,
    DeepSeekV41IndexedMoeTokenPlan, DeepSeekV41MoeTokenOutput, DeepSeekV41ResidualStream,
    TokenKind, PINNED_MODEL_REVISION,
};
use crate::deepseek_v41_projection::DeepSeekV41AttentionProjection;
use crate::deepseek_v41_rope::DeepSeekV41WindowRopeConfig;
use crate::{invalid, Result};
use std::collections::BTreeMap;

/// Inputs whose immutable weights and checkpoint identities were admitted earlier.
/// The FFN route registry and owner policy are moved into the gate-created plan.
pub struct DeepSeekV41WindowBlockRequest<'request, 'source> {
    pub residual: &'request DeepSeekV41ResidualStream,
    pub attention_hc: &'request DeepSeekV41HcProjection,
    pub ffn_hc: &'request DeepSeekV41HcProjection,
    pub attention_qkv: &'request DeepSeekV41AttentionProjection,
    pub attention_output: &'request DeepSeekV41AttentionOutputProjection,
    pub rope: &'request DeepSeekV41WindowRopeConfig,
    pub cache: &'request mut DeepSeekV41WindowCache,
    pub position: u64,
    pub ffn_layer: &'request DeepSeekV41IndexedMoeLayer<'source>,
    pub token_kind: TokenKind,
    pub registry: crate::moe::Registry,
    pub preferred_owners: BTreeMap<u32, u32>,
    pub resident_order: BTreeMap<u32, Vec<u32>>,
    pub max_total_selected_payload_bytes: u64,
}

/// Completed output of one pure-window transformer block.
#[derive(Debug)]
pub struct DeepSeekV41WindowBlockOutput {
    attention: DeepSeekV41AttentionLayerOutput,
    route: PinnedMxfp4RouteExecutionReceipt,
    moe: DeepSeekV41MoeTokenOutput,
    residual: DeepSeekV41ResidualStream,
}

impl DeepSeekV41WindowBlockOutput {
    pub fn attention(&self) -> &DeepSeekV41AttentionLayerOutput {
        &self.attention
    }

    pub fn route(&self) -> &PinnedMxfp4RouteExecutionReceipt {
        &self.route
    }

    pub fn moe(&self) -> &DeepSeekV41MoeTokenOutput {
        &self.moe
    }

    pub fn residual(&self) -> &DeepSeekV41ResidualStream {
        &self.residual
    }

    pub fn next_position(&self) -> u64 {
        self.attention.next_position()
    }
}

/// Compose one attention+FFN block for a single identified token.
///
/// `execute_route_once` receives the already gate-selected, identity-bound plan.
/// The caller must execute that exact route once against resident rank owners and
/// return its terminal receipt. It must not reload weights on the token path or retry
/// after an UNKNOWN. Residency is an executor precondition; this closure signature
/// does not independently attest it. The route's existing FP32-preweighted Collector
/// remains the sole place where the six route coefficients are applied.
///
/// All local normalization, projections, the shared expert, identities and budgets
/// are checked before the caller's cache is changed. An Aborted/Unknown receipt or
/// any other error withholds every block output and leaves the original KV position
/// untouched. The route executor itself remains responsible for its Session leases
/// and for preserving the terminal UNKNOWN receipt.
pub fn forward_token_with_route<'request, 'source, F>(
    request: DeepSeekV41WindowBlockRequest<'request, 'source>,
    execute_route_once: F,
) -> Result<DeepSeekV41WindowBlockOutput>
where
    F: FnOnce(&DeepSeekV41IndexedMoeTokenPlan) -> Result<PinnedMxfp4RouteExecutionReceipt>,
{
    let stamp = request.residual.stamp();
    let attempt = request.residual.attempt();
    if stamp.model_revision != PINNED_MODEL_REVISION
        || stamp.layer >= 40
        || attempt == 0
        || request.position != request.cache.next_position()
        || request.max_total_selected_payload_bytes == 0
        || request.attention_hc.receipt().model_revision != PINNED_MODEL_REVISION
        || request.attention_hc.receipt().layer != stamp.layer
        || request.attention_hc.receipt().branch
            != crate::deepseek_v41_gate::hc::HcProjectionBranch::Attention
        || request.attention_hc.receipt().index_sha256
            != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
        || request.ffn_hc.receipt().model_revision != PINNED_MODEL_REVISION
        || request.ffn_hc.receipt().layer != stamp.layer
        || request.ffn_hc.receipt().branch != crate::deepseek_v41_gate::hc::HcProjectionBranch::Ffn
        || request.ffn_hc.receipt().index_sha256 != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
        || request.ffn_layer.gate_receipt().model_revision != PINNED_MODEL_REVISION
        || request.ffn_layer.gate_receipt().layer != stamp.layer
        || request.ffn_layer.ffn_norm_receipt().is_none()
        || request.ffn_layer.shared_expert_receipt().is_none()
    {
        return Err(invalid(
            "DeepSeek window block inputs are not one admitted pure-window layer and token",
        ));
    }
    let qkv_receipt = request
        .attention_qkv
        .receipt()
        .ok_or_else(|| invalid("DeepSeek block QKV weights are not index-loaded"))?;
    if qkv_receipt.model_revision != PINNED_MODEL_REVISION
        || qkv_receipt.layer != stamp.layer
        || qkv_receipt.index_sha256 != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
    {
        return Err(invalid(
            "DeepSeek block QKV receipt differs from its pinned layer and index",
        ));
    }

    // Stage only the new KV row. The original ring remains untouched until FFN,
    // routed Session results and both HC joins have all completed.
    let initial_position = request.cache.next_position();
    let expected_next_position = initial_position
        .checked_add(1)
        .ok_or_else(|| invalid("DeepSeek block position overflow"))?;
    let attention_mixes = request.attention_hc.project(request.residual)?;
    let (attention, cache_update) = request.attention_output.forward_token_staged(
        request.residual,
        &attention_mixes,
        request.attention_qkv,
        request.rope,
        request.cache,
        request.position,
    )?;

    let ffn_mixes = request.ffn_hc.project(attention.residual())?;
    let plan = request.ffn_layer.prepare_ffn_token_from_hc(
        crate::deepseek_v41_gate::DeepSeekV41IndexedHcFfnTokenRequest {
            mixes: &ffn_mixes,
            residual: attention.residual(),
            token_kind: request.token_kind,
            registry: request.registry,
            preferred_owners: request.preferred_owners,
            resident_order: request.resident_order,
            max_total_selected_payload_bytes: request.max_total_selected_payload_bytes,
        },
    )?;

    // Run all resident local work before consuming a Session route. In particular,
    // the shared expert remains separate from routed choices and is added exactly
    // once only after the route Collector has completed.
    let shared = request.ffn_layer.execute_shared_reference(&plan)?;
    let route = execute_route_once(&plan)?;
    let problem = plan.batch().problem();
    let expected_fingerprint =
        crate::deepseek_v41::pinned_top6_route_fingerprint(problem, plan.batch().attempt())?;
    if route.stamp() != &problem.stamp
        || route.attempt() != plan.batch().attempt()
        || route.route_experts() != problem.experts
        || route.route_fingerprint() != expected_fingerprint
    {
        return Err(invalid(
            "DeepSeek route executor returned a receipt for another plan or token",
        ));
    }
    let moe = combine_pinned_moe_route_with_shared(&route, &shared)?;
    let final_residual = ffn_mixes.apply_ffn_post(&moe, attention.residual())?;
    if attention.next_position() != expected_next_position
        || cache_update.expected_position() != initial_position
        || cache_update.next_position() != expected_next_position
        || final_residual.stamp() != stamp
        || final_residual.attempt() != attempt
    {
        return Err(invalid(
            "DeepSeek complete block output changed token identity or KV position",
        ));
    }

    request.cache.commit_prepared(cache_update)?;
    Ok(DeepSeekV41WindowBlockOutput {
        attention,
        route,
        moe,
        residual: final_residual,
    })
}

/// Execute one token using the already-loaded Sessions for its layer and selected ranks.
/// This adapter issues EXEC only; Session preflight refuses a missing/stale owner before
/// any rank receives bytes. Aborted and UNKNOWN routes remain terminal and do not commit KV.
pub fn forward_token_with_sessions<'request, 'source>(
    request: DeepSeekV41WindowBlockRequest<'request, 'source>,
    sessions_by_layer: &mut BTreeMap<u32, BTreeMap<u32, crate::session::Session>>,
    timeout: std::time::Duration,
) -> Result<DeepSeekV41WindowBlockOutput> {
    if timeout.is_zero() {
        return Err(invalid("DeepSeek Session block timeout must be nonzero"));
    }
    let layer = request.residual.stamp().layer;
    let sessions = sessions_by_layer
        .get_mut(&layer)
        .ok_or_else(|| invalid("DeepSeek block has no persistent Session set for its layer"))?;
    forward_token_with_route(request, |plan| plan.execute_preloaded_mxfp4(sessions, timeout))
}
