"""Fail-closed 1Cat weight-filter seam for explicitly resident Qwen MoE layers.

This experimental post-iterator filter does not avoid upstream disk reads. It
must run only in a private model-loading transaction; a partial model must never
be published if the iterator, session or loader fails. It is not a shared lease.
"""
import re

_PREFIX=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$')
_FIELDS=frozenset(f'{projection}.{field}' for projection in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2','input_scale'))

class ResidentFilterError(RuntimeError):
    pass

class ResidentWeightFilter:
    def __init__(self, methods, expected_type):
        if not methods or any(type(i) is not int or i<0 for i in methods):
            raise ResidentFilterError('invalid resident layer mapping')
        self.methods=dict(methods);self.expected_type=expected_type;self.finished=False
        self.seen={layer:set() for layer in methods}
        self._check()

    def _check(self):
        for layer,method in self.methods.items():
            if type(method) is not self.expected_type or not method._admitted:
                raise ResidentFilterError('method not admitted')
            if method.binding.get('layer')!=layer or method.layer_name!=f'model.language_model.layers.{layer}.mlp.experts':
                raise ResidentFilterError('layer binding mismatch')
            method._check_binding()
            count=method.moe.num_experts
            if type(count) is not int or count<=0 or method.session.experts!=frozenset(range(count)):
                raise ResidentFilterError('incomplete expert coverage')

    def filter(self, iterator):
        for name,tensor in iterator:
            self._check()
            match=_PREFIX.fullmatch(name)
            if match and int(match.group(1)) in self.methods:
                layer,expert,field=int(match.group(1)),int(match.group(2)),match.group(3)
                if field not in _FIELDS or expert not in self.methods[layer].session.experts:
                    raise ResidentFilterError('unexpected resident tensor')
                key=(expert,field)
                if key in self.seen[layer]:raise ResidentFilterError('duplicate resident tensor')
                self.seen[layer].add(key)
                continue
            yield name,tensor
        self._check()
        for layer,method in self.methods.items():
            expected={(expert,field) for expert in method.session.experts for field in _FIELDS}
            if self.seen[layer]!=expected:raise ResidentFilterError('missing resident tensor')
        self.finished=True

    def load(self, loader, model_config, model):
        """Call real loader with an instance-local, temporary iterator wrapper."""
        if 'get_all_weights' in vars(loader):raise ResidentFilterError('loader already customized')
        original=loader.get_all_weights
        def get_all_weights(config, target):
            if getattr(loader,'local_expert_ids',None) is not None:
                raise ResidentFilterError('upstream global expert filter conflicts')
            return self.filter(original(config,target))
        loader.get_all_weights=get_all_weights
        try:
            loader.load_weights(model,model_config)
            self._check()
            if not self.finished:raise ResidentFilterError('loader did not consume full stream')
        finally:
            del loader.get_all_weights
