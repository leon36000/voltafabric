"""Private 1Cat quantization dispatch overlay for explicitly resident layers.

Construction and loading must run inside a private transaction. The caller must
validate require_complete(), load the corresponding weights with the resident
filter, and discard the model on any failure before making it accessible.
"""

class ResidentOverlayError(RuntimeError):
    pass


class ResidentQuantOverlay:
    def __init__(self, delegate, sessions, *, model_revision, plan_id, epoch):
        if delegate is None or not callable(getattr(delegate, 'get_quant_method', None)):
            raise ResidentOverlayError('upstream quantization delegate required')
        if not sessions or any(type(layer) is not int or layer < 0 for layer in sessions):
            raise ResidentOverlayError('invalid resident layer mapping')
        if not model_revision or not plan_id or type(epoch) is not int or epoch < 0:
            raise ResidentOverlayError('invalid identity')
        self._delegate = delegate
        self._sessions = dict(sessions)
        self._identity = (model_revision, plan_id, epoch)
        self._methods = {}
        self._prefixes = {f'model.language_model.layers.{layer}.mlp.experts': layer
                          for layer in sessions}

    def __getattr__(self, name):
        delegate = object.__getattribute__(self, '_delegate')
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    @property
    def packed_modules_mapping(self):
        return self._delegate.packed_modules_mapping

    @packed_modules_mapping.setter
    def packed_modules_mapping(self, value):
        self._delegate.packed_modules_mapping = value

    def get_quant_method(self, layer, prefix):
        layer_id = self._prefixes.get(prefix)
        if layer_id is None:
            return self._delegate.get_quant_method(layer, prefix)
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from .onecat_resident_method import ResidentMoEMethod
        if type(layer) is not FusedMoE or layer_id in self._methods:
            raise ResidentOverlayError('invalid or repeated resident layer construction')
        revision, plan, epoch = self._identity
        method = ResidentMoEMethod(layer.moe_config, self._sessions[layer_id],
                                   model_revision=revision, plan_id=plan,
                                   layer_id=layer_id, epoch=epoch, layer_name=prefix)
        self._methods[layer_id] = method
        return method

    def require_complete(self):
        if self._methods.keys() != self._sessions.keys():
            raise ResidentOverlayError('resident construction incomplete')
        for method in self._methods.values():
            method._check_binding()
            if not method._admitted:
                raise ResidentOverlayError('resident method not admitted')
        return dict(self._methods)
