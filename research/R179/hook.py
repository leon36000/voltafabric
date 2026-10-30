# SPDX-License-Identifier: Apache-2.0
"""CPU-only research bridge for the pinned donor's six-argument MoE call.

This is an explicit one-batch laboratory adapter, not a production engine hook.
It invokes the Rust CLI for routing and ordered FP16 collection and accepts
preloaded owner objects; it cannot move weights, run CUDA or replay failure.
"""
import copy
import json
import math
import struct
import subprocess
import threading

import torch

class HookError(RuntimeError):
    pass

class CpuHook:
    def __init__(self, *, problem, owners, cli, align):
        self.problem = copy.deepcopy(problem)
        self.owners = dict(owners)
        self.cli = cli
        self.align = align
        self.attempt = 0
        self.terminal = False
        self.lock = threading.Lock()

    def _cli(self, operation, value):
        payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
        if len(payload) > 16 * 1024 * 1024:
            raise HookError('bounded request exceeded')
        proc = subprocess.run([self.cli, operation, '-'], input=payload,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        reply = json.loads(proc.stdout)
        if proc.returncode != 0 or not reply.get('ok'):
            raise HookError('Rust routing or collection refused input')
        return reply['result']

    def __call__(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        with self.lock:
            if self.terminal:
                raise HookError('hook is terminal; do not replay')
            if shared_experts is not None or shared_experts_input is not None:
                raise HookError('shared expert path is not admitted')
            if (not isinstance(x, torch.Tensor) or x.device.type != 'cpu'
                or x.dtype != torch.float16 or x.ndim != 2 or not torch.isfinite(x).all()
                or not isinstance(topk_ids, torch.Tensor) or topk_ids.device.type != 'cpu'
                or topk_ids.dtype not in (torch.int32, torch.int64) or topk_ids.ndim != 2
                or not isinstance(topk_weights, torch.Tensor) or topk_weights.device.type != 'cpu'
                or not topk_weights.is_floating_point() or topk_weights.shape != topk_ids.shape
                or topk_ids.shape[0] != x.shape[0] or not torch.isfinite(topk_weights).all()
                or (topk_weights < 0).any() or x.shape[1] != self.problem.get('hidden')
                or not 0 < x.shape[0] <= 4096 or not 0 < topk_ids.shape[1] <= 64):
                raise HookError('invalid CPU input, route or coefficient')
            p = copy.deepcopy(self.problem)
            ids, weights = topk_ids.tolist(), topk_weights.tolist()
            p['routes'] = [[{'expert': int(e), 'weight': float(w)} for e, w in zip(er, wr)]
                           for er, wr in zip(ids, weights)]
            self.attempt += 1
            attempt = self.attempt
            try:
                routed = self._cli('routing', p)
                outputs = []
                for dispatch in routed['ranks']:
                    rank = dispatch['rank']
                    if rank not in self.owners:
                        raise HookError('no admitted owner for rank')
                    if not dispatch['slots']:
                        continue
                    token_ids = sorted({entry['slot']['token'] for entry in dispatch['slots']})
                    activation = x[token_ids].contiguous()
                    result = self.owners[rank].execute(dispatch, token_ids=token_ids,
                        activations=activation, attempt=attempt, align=self.align)
                    identities = tuple((slot['slot']['token'], slot['slot']['slot'],
                                        slot['slot']['expert'], rank, slot['slot']['local'])
                                       for slot in dispatch['slots'])
                    packed = result.as_raw_output_bytes(identities)
                    bits = struct.unpack('<%dH' % (packed['rows'] * packed['columns']),
                                         packed['fp16_le_bytes'])
                    outputs.append({'stamp':packed['stamp'],'rank':rank,'attempt':attempt,
                        'rows':packed['rows'],'columns':packed['columns'],
                        'slots':[{'token':i[0],'slot':i[1],'expert':i[2],'local':i[4]} for i in identities],
                        'fp16_bits':list(bits)})
                collected = self._cli('collect-fp16', {'problem':p,'expected_attempt':attempt,'outputs':outputs})
                result = torch.tensor(collected['output'], dtype=x.dtype, device='cpu')
                if result.shape != x.shape or not torch.isfinite(result).all():
                    raise HookError('collector returned invalid output')
                self.terminal = True  # one batch per instance: no implicit replay
                return result
            except BaseException:
                self.terminal = True
                raise
