# SPDX-License-Identifier: Apache-2.0
"""Restricted CPU research seam: donor MoE call to actual Rust compressed Session."""
import copy
import hashlib
import json
import pathlib
import os
import struct
import subprocess
import threading
import uuid
import torch

class SessionHookError(RuntimeError):
    pass

class SessionHook:
    def __init__(self, *, template, cli, worker, python, operation_dir):
        self.template = copy.deepcopy(template)
        self.cli, self.worker, self.python = str(cli), str(worker), str(python)
        self.operation_dir = pathlib.Path(operation_dir)
        self.terminal = False
        self.lock = threading.Lock()

    def __call__(self, layer, x, weights, ids, shared_experts, shared_experts_input):
        with self.lock:
            if self.terminal:
                raise SessionHookError('consumed session: no replay')
            if shared_experts is not None or shared_experts_input is not None:
                raise SessionHookError('shared experts are not supported')
            if (not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.dtype != torch.float16
                or tuple(x.shape) != (1, 2048) or not torch.isfinite(x).all()
                or torch.any(x.abs() > 1)
                or not isinstance(ids, torch.Tensor) or ids.device.type != 'cpu'
                or ids.dtype not in (torch.int32, torch.int64) or tuple(ids.shape) != (1, 2)
                or ids.tolist() != [[0, 1]]
                or not isinstance(weights, torch.Tensor) or weights.device.type != 'cpu'
                or not weights.is_floating_point() or tuple(weights.shape) != (1, 2)
                or not torch.isfinite(weights).all() or torch.any(weights < 0) or torch.any(weights > 1)):
                raise SessionHookError('input outside the two-owner CPU fixture')
            run_id = 'r180-' + uuid.uuid4().hex
            p = copy.deepcopy(self.template)
            p['routing']['stamp']['run_id'] = run_id
            p['routing']['stamp']['plan_id'] = 'donor-session-cpu-fixture'
            p['routing']['routes'] = [[{'expert': int(e), 'weight': float(w)}
                                       for e, w in zip(ids[0].tolist(), weights[0].tolist())]]
            p.pop('inputs', None)
            p['inputs_fp16_bits'] = [[struct.unpack('<H', struct.pack('<e', v))[0] for v in x[0].tolist()]]
            p['fp16_completions'] = True
            p['batches'] = 2
            p['record'] = [0, 1]
            p['tolerance_abs'] = 0.01  # fixture oracle guard, not model quality threshold
            p['workers'] = [{'rank': r, 'argv': [self.python, '-B', self.worker, run_id]} for r in (2, 9)]
            data = json.dumps(p, allow_nan=False, separators=(',', ':')).encode()
            self.terminal = True  # no replay if child outcome is unknown
            # A durable RUNNING receipt predates submission. If this process disappears,
            # RUNNING means UNKNOWN until child processes and outputs are reconciled.
            self.operation_dir.mkdir(parents=True, exist_ok=True)
            receipt = self.operation_dir / (run_id + '.json')
            with receipt.open('x') as f:
                json.dump({'run_id': run_id, 'status': 'RUNNING', 'replay_allowed': False}, f)
                f.flush()
                os.fsync(f.fileno())
            try:
                proc = subprocess.run([self.cli, 'session-run-compressed', '-'], input=data,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if len(proc.stdout) > 16 * 1024 * 1024:
                    raise SessionHookError('session response exceeds research bound')
                response_path = self.operation_dir / (run_id + '.output.json')
                with response_path.open('xb') as f:
                    f.write(proc.stdout)
                    f.flush()
                    os.fsync(f.fileno())
                reply = json.loads(proc.stdout)
                if proc.returncode or not reply.get('ok'):
                    raise SessionHookError('Rust session refused or failed')
                result = reply['result']
                if (not result['ok'] or result['batches_ok'] != 2 or result['exec_requests'] != 4
                    or len(result['recorded']) != 2
                    or result['recorded'][0]['output'] != result['recorded'][1]['output']
                    or any(c['exit_code'] != 0 or c['bye_execs'] != 2 for c in result['close'])
                    or len(result['close']) != 2 or len(result['traffic']) != 2
                    or any(t['traffic']['loads_transferred'] != 1 or t['traffic']['execs'] != 2
                           for t in result['traffic'])):
                    raise SessionHookError('incomplete session or mismatched repeated batch')
                out = torch.tensor(result['recorded'][0]['output'], dtype=torch.float16)
                if out.shape != x.shape or not torch.isfinite(out).all():
                    raise SessionHookError('invalid collected output')
                temp = receipt.with_suffix('.complete')
                with temp.open('x') as f:
                    json.dump({'run_id': run_id, 'status': 'SUCCEEDED', 'replay_allowed': False, 'output_sha256': hashlib.sha256(proc.stdout).hexdigest(), 'batches_ok': 2, 'exec_requests': 4}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, receipt)
                return out
            except SessionHookError:
                raise
            except Exception as exc:
                raise SessionHookError('session outcome unavailable or invalid; no replay') from exc
