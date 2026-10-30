"""Process-local staged KV for private Qwen attention experiments.

The enclosing decoder must commit only after expert execution and collection
succeed. This state is neither durable nor coupled to the distributed ledger.
"""
import threading
import torch


class KvTransactionError(RuntimeError):
    pass


class TransactionalKvAttention(torch.nn.Module):
    _BOUND = ('run_id', 'model_revision', 'plan_id', 'layer', 'epoch')

    def __init__(self, reference, identity):
        super().__init__()
        if (not callable(reference) or not isinstance(identity, dict)
                or set(identity) != set(self._BOUND)
                or any(not isinstance(identity[k], str) or not identity[k] for k in self._BOUND[:3])
                or any(type(identity[k]) is not int or identity[k] < 0 for k in self._BOUND[3:])):
            raise KvTransactionError('invalid bound identity')
        self._reference = reference
        self._identity = dict(identity)
        self._cache = None
        self._next_batch = 0
        self._seen = set()
        self._pending = None
        self._unknown = False
        self._lock = threading.RLock()

    @property
    def committed_batch(self):
        with self._lock:
            return self._next_batch

    def begin(self, identity):
        with self._lock:
            if self._unknown or self._pending is not None:
                raise KvTransactionError('state unknown or transaction pending')
            if (not isinstance(identity, dict) or set(identity) != set(self._BOUND) | {'batch', 'attempt'}
                    or any(type(identity.get(k)) is not type(v) or identity.get(k) != v
                           for k, v in self._identity.items())
                    or type(identity.get('batch')) is not int or identity['batch'] != self._next_batch
                    or type(identity.get('attempt')) is not int or identity['attempt'] < 0):
                raise KvTransactionError('identity or batch mismatch')
            key = (identity['batch'], identity['attempt'])
            if key in self._seen:
                raise KvTransactionError('attempt already consumed')
            self._seen.add(key)
            self._pending = {'owner': threading.get_ident(), 'state': None}

    def _check_pending(self):
        if (self._unknown or self._pending is None
                or self._pending['owner'] != threading.get_ident()):
            raise KvTransactionError('no current transaction on this thread')

    def forward(self, hidden_states, output):
        with self._lock:
            self._check_pending()
            if self._pending['state'] is not None:
                raise KvTransactionError('attention already staged')
            if (not isinstance(hidden_states, torch.Tensor) or not isinstance(output, torch.Tensor)
                    or hidden_states.device.type != 'cpu' or output.device.type != 'cpu'
                    or hidden_states.dtype != torch.float16 or output.dtype != hidden_states.dtype
                    or hidden_states.ndim != 2 or not 0 < hidden_states.shape[0] <= 64
                    or hidden_states.shape[1] != 2048 or output.shape != hidden_states.shape
                    or output.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr()
                    or not torch.isfinite(hidden_states).all()):
                raise KvTransactionError('invalid attention input/output')
            old_len = 0 if self._cache is None else self._cache[0].shape[2]
            old = None if self._cache is None else tuple(t.clone() for t in self._cache)
            result, cache = self._reference(hidden_states.float().unsqueeze(0), old)
            shape = (1, 2, old_len + hidden_states.shape[0], 256)
            if (not isinstance(result, torch.Tensor) or result.device.type != 'cpu'
                    or result.dtype != torch.float32 or result.shape != (1, *hidden_states.shape)
                    or not torch.isfinite(result).all() or not isinstance(cache, tuple) or len(cache) != 2
                    or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                           or t.dtype != torch.float32 or t.shape != shape or not torch.isfinite(t).all()
                           for t in cache)):
                raise KvTransactionError('invalid staged attention result')
            cast = result.squeeze(0).to(hidden_states.dtype)
            if not torch.isfinite(cast).all():
                raise KvTransactionError('attention output overflows')
            staged = tuple(t.clone() for t in cache)
            output.copy_(cast)
            self._pending['state'] = staged

    def commit(self):
        with self._lock:
            self._check_pending()
            if self._pending['state'] is None:
                raise KvTransactionError('attention result not staged')
            self._cache = self._pending['state']
            self._next_batch += 1
            self._pending = None

    def abort(self):
        with self._lock:
            self._check_pending()
            self._pending = None

    def mark_unknown(self):
        with self._lock:
            self._check_pending()
            self._unknown = True
            self._pending = None
