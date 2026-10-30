"""Experimental 1Cat 1.2.2 CPU-only external resident MoE method.

Only a local, live DurableSession covering the entire logical expert range can
be bound. This is not a distributed lease or a complete model-loader adapter.
"""
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from .session_bridge import DurableSession
from .onecat_shared_guard import call_after_shared

class ResidentAdmissionError(RuntimeError):
    pass

class ResidentMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, session, *, model_revision, plan_id, layer_id, epoch, layer_name):
        super().__init__(moe)
        if type(session) is not DurableSession:
            raise ResidentAdmissionError('a live local DurableSession is required')
        self.session=session
        self.binding={'model_revision':model_revision,'plan_id':plan_id,'layer':layer_id,'epoch':epoch}
        self.layer_name=layer_name
        self._admitted=False

    def _check_binding(self):
        session=self.session
        if session.terminal or session.proc.poll() is not None:
            raise ResidentAdmissionError('session is terminal or not alive')
        if any(session.identity.get(k)!=v for k,v in self.binding.items()):
            raise ResidentAdmissionError('session identity mismatch')
        if not self.binding['model_revision'] or not self.binding['plan_id'] or not self.layer_name:
            raise ResidentAdmissionError('empty binding identity')

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
        self._check_binding()
        total=extra.get('global_num_experts')
        if (type(total) is not int or total <= 0 or num_experts != total
            or self.session.experts != frozenset(range(total))
            or self.session.hidden != hidden_size or layer.layer_name != self.layer_name):
            raise ResidentAdmissionError('incomplete coverage, shape or layer mismatch')
        # The live session owns all expert weights; do not register weight tensors here.
        # The caller must separately prevent the default model loader from writing them.
        self._admitted=True

    def get_fused_moe_quant_config(self, layer):
        self._check_binding()
        if not self._admitted or layer.layer_name != self.layer_name:
            raise ResidentAdmissionError('unadmitted layer')
        return None  # the resident provider owns its own explicit QPN2 format

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        self._check_binding()
        if not self._admitted or layer.layer_name != self.layer_name:
            raise ResidentAdmissionError('unadmitted layer')
        if shared_experts is None and shared_experts_input is None:
            return self.session(layer,x,topk_weights,topk_ids,None,None)
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts, SharedExpertsOrder
        return call_after_shared(self.session,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input,
                                 shared_type=SharedExperts,no_overlap=SharedExpertsOrder.NO_OVERLAP)
