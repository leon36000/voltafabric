"""Process-local staged Qwen GDN state for private CPU experiments.

The caller must commit only after the whole enclosing decoder operation has
succeeded. This is not durable storage, a distributed lease, or crash recovery.
"""
import threading
import torch


class GdnTransactionError(RuntimeError):
    pass


class TransactionalGdnAttention(torch.nn.Module):
    _BOUND = ('run_id', 'model_revision', 'plan_id', 'layer', 'epoch')

    def __init__(self, reference, identity):
        super().__init__()
        if (not callable(reference) or set(identity) != set(self._BOUND)
                or any(not isinstance(identity[k], str) or not identity[k] for k in self._BOUND[:3])
                or any(type(identity[k]) is not int or identity[k] < 0 for k in self._BOUND[3:])):
            raise GdnTransactionError('invalid bound identity')
        self._reference = reference
        self._identity = dict(identity)
        self._history = torch.zeros((1,8192,3),dtype=torch.float32)
        self._state = None
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
                raise GdnTransactionError('state unknown or transaction pending')
            if (set(identity) != set(self._BOUND) | {'batch','attempt'}
                    or any(type(identity.get(k)) is not type(v) or identity.get(k) != v
                           for k,v in self._identity.items())
                    or type(identity.get('batch')) is not int or identity['batch'] != self._next_batch
                    or type(identity.get('attempt')) is not int or identity['attempt'] < 0):
                raise GdnTransactionError('identity or batch mismatch')
            key = (identity['batch'],identity['attempt'])
            if key in self._seen:
                raise GdnTransactionError('attempt already consumed')
            self._seen.add(key)
            self._pending = {'owner':threading.get_ident(), 'state':None}

    def _check_pending(self):
        if (self._unknown or self._pending is None
                or self._pending['owner'] != threading.get_ident()):
            raise GdnTransactionError('no current transaction on this thread')

    def forward(self, hidden_states, output):
        with self._lock:
            self._check_pending()
            if self._pending['state'] is not None:
                raise GdnTransactionError('attention already staged')
            if (not isinstance(hidden_states,torch.Tensor) or not isinstance(output,torch.Tensor)
                    or hidden_states.device.type != 'cpu' or output.device.type != 'cpu'
                    or hidden_states.dtype != torch.float16 or output.dtype != hidden_states.dtype
                    or hidden_states.ndim != 2 or not 0 < hidden_states.shape[0] <= 64
                    or hidden_states.shape[1] != 2048 or output.shape != hidden_states.shape
                    or output.untyped_storage().data_ptr() == hidden_states.untyped_storage().data_ptr()
                    or not torch.isfinite(hidden_states).all()):
                raise GdnTransactionError('invalid attention input/output')
            old_state = None if self._state is None else self._state.clone()
            result, history, state = self._reference(hidden_states.float().unsqueeze(0),self._history.clone(),old_state)
            if (not isinstance(result,torch.Tensor) or result.device.type != 'cpu' or result.dtype != torch.float32
                    or result.shape != (1,*hidden_states.shape) or not torch.isfinite(result).all()
                    or not isinstance(history,torch.Tensor) or history.device.type != 'cpu' or history.dtype != torch.float32
                    or history.shape != self._history.shape or not torch.isfinite(history).all()
                    or not isinstance(state,torch.Tensor) or state.device.type != 'cpu' or state.dtype != torch.float32
                    or state.shape != (1,32,128,128) or not torch.isfinite(state).all()):
                raise GdnTransactionError('invalid staged attention result')
            cast = result.squeeze(0).to(hidden_states.dtype)
            if not torch.isfinite(cast).all():
                raise GdnTransactionError('attention output overflows')
            output.copy_(cast)
            self._pending['state'] = (history.clone(),state.clone())

    def commit(self):
        with self._lock:
            self._check_pending()
            if self._pending['state'] is None:
                raise GdnTransactionError('attention result not staged')
            self._history,self._state=self._pending['state']
            self._next_batch+=1
            self._pending=None

    def abort(self):
        with self._lock:
            self._check_pending()
            self._pending=None

    def mark_unknown(self):
        with self._lock:
            self._check_pending()
            self._unknown=True
            self._pending=None

    def invalidate(self):
        """Fence this local state even if a multi-layer commit partly succeeded."""
        with self._lock:
            self._unknown=True
            self._pending=None
