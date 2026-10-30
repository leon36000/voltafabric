"""Experimental CPU session bridge: bounded pipes and durable no-replay identities.

A directory claim is local to the configured ledger; this is not a distributed lease.
Unknown calls are never automatically replayed or killed; reconcile the process first.
"""
import hashlib
import json
import os
import pathlib
import selectors
import struct
import subprocess
import threading
import time
import torch

class DurableHookError(RuntimeError):
    pass

def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def _exclusive(path, data):
    with open(path, 'xb') as f:
        f.write(json.dumps(data, separators=(',', ':'), allow_nan=False).encode())
        f.flush()
        os.fsync(f.fileno())
    _sync_dir(path.parent)

class DurableSession:
    MAX = 16 * 1024 * 1024
    def __init__(self, binary, init, ledger, deadline_seconds=10):
        if not 0 < deadline_seconds <= 180:
            raise ValueError('deadline out of range')
        stamp = init['routing']['stamp']
        identity = {k: stamp[k] for k in ('run_id', 'model_revision', 'plan_id', 'layer', 'epoch')}
        if not all(isinstance(identity[k], str) and identity[k] for k in ('run_id','model_revision','plan_id')):
            raise ValueError('missing identity')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        root = pathlib.Path(ledger)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / key
        self.directory.mkdir()  # exclusive claim: a previously used identity cannot be reopened
        _sync_dir(root)
        _exclusive(self.directory / 'identity.json', identity)
        self.identity, self.lock = identity, threading.Lock()
        self.experts = frozenset(init['routing']['experts'])
        self.hidden = len(init['inputs'][0])
        if not self.experts or not 0 < self.hidden <= 65536:
            raise ValueError('invalid resident experts or hidden size')
        self.batch, self.terminal, self.deadline = stamp['batch'], False, deadline_seconds
        self.pending = b''
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        _exclusive(self.directory / 'process.json', {'pid': self.proc.pid})
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)
        try:
            response = self._exchange(init)
            if response.get('state') != 'ready':
                raise DurableHookError('initialization not admitted')
        except BaseException:
            self.terminal = True
            raise

    def _exchange(self, value):
        data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(data) > self.MAX:
            raise DurableHookError('request too large')
        end = time.monotonic() + self.deadline
        sent = 0
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            selector.register(self.proc.stdin, selectors.EVENT_WRITE)
            while True:
                remain = end - time.monotonic()
                if remain <= 0:
                    raise DurableHookError('deadline exceeded; outcome UNKNOWN')
                events = selector.select(remain)
                if not events:
                    raise DurableHookError('deadline exceeded; outcome UNKNOWN')
                for key, _ in events:
                    if key.fileobj is self.proc.stdin:
                        try: written = os.write(key.fd, data[sent:])
                        except BlockingIOError: continue
                        if written <= 0: raise DurableHookError('write failed; outcome UNKNOWN')
                        sent += written
                        if sent == len(data): selector.unregister(self.proc.stdin)
                    else:
                        try: part = os.read(key.fd, 65536)
                        except BlockingIOError: continue
                        if not part: raise DurableHookError('session ended; outcome UNKNOWN')
                        self.pending += part
                        if len(self.pending) > self.MAX: raise DurableHookError('response exceeds bound')
                        if b'\n' in self.pending:
                            raw, self.pending = self.pending.split(b'\n', 1)
                            reply = json.loads(raw)
                            if sent != len(data) or not reply.get('ok'):
                                raise DurableHookError('session refused or invalid response')
                            return reply

    def __call__(self, layer, x, weights, ids, shared_experts, shared_experts_input):
        with self.lock:
            if self.terminal: raise DurableHookError('terminal session; no replay')
            if shared_experts is not None or shared_experts_input is not None: raise DurableHookError('shared experts unsupported')
            if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16 or x.ndim!=2
                or not 0 < x.shape[0] <= 64 or x.shape[1]!=self.hidden or not torch.isfinite(x).all()
                or not isinstance(ids,torch.Tensor) or ids.device.type!='cpu' or ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=2
                or ids.shape[0]!=x.shape[0] or not 0 < ids.shape[1] <= 64
                or not isinstance(weights,torch.Tensor) or weights.device.type!='cpu' or not weights.is_floating_point()
                or weights.shape!=ids.shape or not torch.isfinite(weights).all() or (weights<0).any()):
                raise DurableHookError('invalid CPU activation, routes or weights')
            id_rows=ids.tolist(); weight_rows=weights.tolist()
            if any(len(row)!=len(set(row)) or any(expert not in self.experts for expert in row) for row in id_rows):
                raise DurableHookError('duplicate or nonresident expert')
            req={'batch':self.batch,'routes':[[{'expert':int(e),'weight':float(w)} for e,w in zip(row,weight_rows[i])] for i,row in enumerate(id_rows)],
                 'inputs_fp16_bits':[[struct.unpack('<H',struct.pack('<e',v))[0] for v in row] for row in x.tolist()]}
            batch_identity = {**self.identity, 'batch': self.batch, 'attempt': 'assigned-by-rust'}
            # Exclusive fsync before sending; without a done file, outcome is UNKNOWN.
            claim = self.directory / ('batch-%020d.claim.json' % self.batch)
            self.terminal = True
            _exclusive(claim, batch_identity)
            response = self._exchange(req)
            result = response['result']
            output = torch.tensor(result['output'], dtype=x.dtype)
            if response['batch']!=self.batch or not result['ok'] or output.shape!=x.shape or not torch.isfinite(output).all():
                raise DurableHookError('invalid response')
            # Persist output itself with its hash: a done marker never certifies unrecorded output.
            payload = output.contiguous().numpy().tobytes()
            output_path = self.directory / ('batch-%020d.fp16' % self.batch)
            with open(output_path,'xb') as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            _sync_dir(self.directory)
            _exclusive(self.directory / ('batch-%020d.done.json' % self.batch),
                       {**batch_identity, 'attempt': result['attempt'], 'sha256':hashlib.sha256(payload).hexdigest()})
            self.batch += 1
            self.terminal = False
            return output

    def close(self):
        with self.lock:
            if self.terminal: raise DurableHookError('UNKNOWN requires reconciliation')
            self.terminal = True
            response = self._exchange({'op':'shutdown'})
            code = self.proc.wait(timeout=self.deadline)
            if code or response.get('state')!='closed' or any(c['exit_code']!=0 for c in response['close']):
                raise DurableHookError('close failed')
            _exclusive(self.directory / 'closed.json', {'exit_code':code})
            self.proc.stdin.close()
            self.proc.stdout.close()
            return response
