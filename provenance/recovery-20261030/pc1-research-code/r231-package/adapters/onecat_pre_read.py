"""Experimental local, lazy safetensors path for admitted resident experts.

The caller must discard the model on any failure and prevent publication until
load() succeeds. This instance-scoped adapter is not a distributed lease.
"""
import os
import time
from .onecat_weight_filter import ResidentFilterError, ResidentWeightFilter


class ResidentPreReadFilter(ResidentWeightFilter):
    def __init__(self, methods, expected_type, *, opener=None, is_local=None):
        super().__init__(methods, expected_type)
        self._opener = opener
        self._is_local = is_local or os.path.isdir

    def load(self, loader, model_config, model):
        if self.finished or any(self.seen.values()):
            raise ResidentFilterError('filter already used')
        if any(name in vars(loader) for name in ('get_all_weights', '_get_weights_iterator')):
            raise ResidentFilterError('loader already customized')
        config = loader.load_config
        if (config.load_format not in ('auto', 'safetensors')
                or config.safetensors_load_strategy is not None
                or config.model_loader_extra_config):
            raise ResidentFilterError('unsupported loading strategy')
        if self._opener is None:
            from safetensors.torch import safe_open
            opener = safe_open
        else:
            opener = self._opener
        original = loader.get_all_weights

        def lazy_iterator(source):
            self._check()
            if getattr(loader, 'local_expert_ids', None) is not None:
                raise ResidentFilterError('upstream global expert filter conflicts')
            if not self._is_local(source.model_or_path):
                raise ResidentFilterError('only existing local checkpoints are supported')
            _, paths, is_safe = loader._prepare_weights(
                source.model_or_path, source.subfolder, source.revision,
                source.fall_back_to_pt, source.allow_patterns_overrides)
            if not is_safe or not paths:
                raise ResidentFilterError('safetensors required')
            if loader.counter_before_loading_weights == 0.0:
                loader.counter_before_loading_weights = time.perf_counter()
            for path in sorted(paths):
                with opener(path, framework='pt') as f:
                    for name in f.keys():
                        yield source.prefix + name, lambda name=name: f.get_tensor(name)

        def selected_weights(config_arg, target):
            for name, read in self.filter(original(config_arg, target)):
                self._check()
                yield name, read()

        loader._get_weights_iterator = lazy_iterator
        loader.get_all_weights = selected_weights
        try:
            loader.load_weights(model, model_config)
            self._check()
            if not self.finished:
                raise ResidentFilterError('loader did not consume full stream')
        finally:
            del loader.get_all_weights
            del loader._get_weights_iterator
