"""Single-use sequential CPU decoding over staged attention states.

The callbacks own model execution and prediction; this controller provides no
network lease, persistence, streaming API, GPU qualification or crash recovery.
"""
import threading

class DecodeError(RuntimeError):
    pass

class TransactionalDecoder:
    def __init__(self, attentions, identities, forward, predict, *, context_limit=None, eos_token_ids=()):
        self._attentions = tuple(attentions)
        self._identities = tuple(dict(x) for x in identities)
        keys = {'run_id', 'model_revision', 'plan_id', 'layer', 'epoch'}
        if not self._attentions or len(self._attentions) != len(self._identities):
            raise DecodeError('attention mapping invalid')
        if not callable(forward) or not callable(predict):
            raise DecodeError('callbacks required')
        if context_limit is not None and (type(context_limit) is not int or context_limit < 1):
            raise DecodeError('invalid context limit')
        if (not isinstance(eos_token_ids, (tuple, list, set, frozenset))
                or any(type(token) is not int or token < 0 for token in eos_token_ids)):
            raise DecodeError('invalid end-of-sequence tokens')
        self._context_limit = context_limit
        self._eos_token_ids = frozenset(eos_token_ids)
        shared = None
        layers = set()
        for attention, identity in zip(self._attentions, self._identities):
            if (set(identity) != keys or any(not isinstance(identity[k], str) or not identity[k] for k in ('run_id','model_revision','plan_id'))
                or type(identity['layer']) is not int or identity['layer'] < 0
                or type(identity['epoch']) is not int or identity['epoch'] < 0
                or any(not callable(getattr(attention, k, None)) for k in ('begin','commit','invalidate'))):
                raise DecodeError('invalid identity or attention')
            if identity['layer'] in layers: raise DecodeError('duplicate layer')
            layers.add(identity['layer'])
            current = tuple(identity[k] for k in ('run_id','model_revision','plan_id','epoch'))
            if shared is None: shared = current
            elif shared != current: raise DecodeError('mixed execution identities')
        self._forward, self._predict = forward, predict
        self._lock = threading.Lock()
        self._terminal = False

    def generate(self, prompt_tokens, max_new_tokens):
        with self._lock:
            if self._terminal: raise DecodeError('decoder is terminal; no replay')
            if (not isinstance(prompt_tokens, (tuple,list)) or not prompt_tokens
                or any(type(t) is not int or t < 0 for t in prompt_tokens)
                or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128):
                raise DecodeError('invalid prompt or token limit')
            # The final generated token does not require another model forward.
            if self._context_limit is not None and len(prompt_tokens) + max_new_tokens - 1 > self._context_limit:
                raise DecodeError('requested generation exceeds context limit')
            self._terminal = True
            generated=[]
            inputs=list(prompt_tokens)
            batch=0
            try:
                while batch < len(inputs):
                    for attention, identity in zip(self._attentions, self._identities):
                        attention.begin({**identity,'batch':batch,'attempt':1})
                    hidden=self._forward(inputs[batch],batch)
                    next_token=None
                    if batch >= len(prompt_tokens)-1:
                        next_token=self._predict(hidden)
                        if type(next_token) is not int or next_token < 0:
                            raise DecodeError('invalid predicted token')
                    for attention in self._attentions: attention.commit()
                    if next_token is not None:
                        generated.append(next_token)
                        if next_token in self._eos_token_ids or len(generated) == max_new_tokens: break
                        inputs.append(next_token)
                    batch+=1
            except BaseException as exc:
                for attention in self._attentions:
                    try: attention.invalidate()
                    except BaseException as fence_error:
                        exc.add_note('attention fence failed: '+type(fence_error).__name__)
                raise
            return generated
