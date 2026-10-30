"""CPU-only reference dispatch for named Qwen shared-expert projections.

Experimental private-load boundary. Preserve checkpoint identity and discard
any partially loaded model; never substitute this for a GPU production kernel.
"""

class CpuSharedOverlayError(RuntimeError):
    pass


class CpuSharedOverlay:
    def __init__(self, delegate, layer_ids):
        from vllm.platforms import current_platform
        ids = tuple(layer_ids)
        if current_platform.device_type != 'cpu':
            raise CpuSharedOverlayError('CPU platform required')
        if (delegate is None or not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or i < 0 for i in ids)):
            raise CpuSharedOverlayError('invalid delegate or layer ids')
        self._delegate = delegate
        self._prefixes = {f'model.language_model.layers.{i}.mlp.shared_expert.{name}'
                          for i in ids for name in ('gate_up_proj', 'down_proj')}
        self._methods = {}

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_delegate'), name)

    @property
    def packed_modules_mapping(self):
        return self._delegate.packed_modules_mapping

    @packed_modules_mapping.setter
    def packed_modules_mapping(self, mapping):
        self._delegate.packed_modules_mapping = mapping

    def get_quant_method(self, layer, prefix):
        if prefix not in self._prefixes:
            return self._delegate.get_quant_method(layer, prefix)
        from vllm.model_executor.layers.linear import LinearBase
        from .onecat_cpu_nvfp4 import CpuNvfp4Reference
        if not isinstance(layer, LinearBase) or prefix in self._methods:
            raise CpuSharedOverlayError('invalid or duplicate shared projection')
        if self._delegate._resolve_quant_algo(prefix) != 'W4A16_NVFP4':
            raise CpuSharedOverlayError('shared projection is not W4A16 NVFP4')
        method = CpuNvfp4Reference(self._delegate.w4a16_nvfp4_config)
        self._methods[prefix] = method
        return method

    def require_constructed(self):
        if self._methods.keys() != self._prefixes:
            raise CpuSharedOverlayError('shared projections incomplete')
        return dict(self._methods)
