#!/usr/bin/env python3
"""Independent standard-library oracle; calls vf, never imports its implementation.
No timing measured here is a V100 benchmark. Fixed seeds make failures reproducible.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import time


def invoke(binary: Path, command: str, payload: object, *, valid: bool = True) -> object:
    run = subprocess.run([str(binary), command, '-'], input=json.dumps(payload), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    try:
        output = json.loads(run.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f'{command}: invalid CLI JSON: {run.stderr[:300]}') from exc
    if valid:
        assert run.returncode == 0 and output['ok'], (command, output)
        return output['result']
    assert run.returncode != 0 and not output['ok'], ('accepted invalid input', command)
    return output['error']


def expert_value(model: dict, expert: dict, x: list[float]) -> list[float]:
    h, inner = model['hidden'], model['intermediate']
    # Independent summation (math.fsum rather than Rust's sequential adds).
    g = [math.fsum(x[i] * expert['gate'][i * inner + j] for i in range(h)) for j in range(inner)]
    u = [math.fsum(x[i] * expert['up'][i * inner + j] for i in range(h)) for j in range(inner)]
    a = [(v / (1 + math.exp(-v)) if v >= 0 else v * math.exp(v) / (1 + math.exp(v))) * w
         for v, w in zip(g, u)]
    return [math.fsum(a[j] * expert['down'][j * h + i] for j in range(inner)) for i in range(h)]


def oracle(p: dict) -> list[list[float]]:
    m = p['model']
    experts = {e['id']: e for e in m['experts']}
    out = []
    for x, route in zip(p['batch']['inputs'], p['batch']['routes']):
        ys = [(c['weight'], expert_value(m, experts[c['expert']], x)) for c in route]
        out.append([math.fsum(w * y[j] for w, y in ys) for j in range(m['hidden'])])
    return out


def problem(seed: int) -> dict:
    rng = random.Random(seed)
    h, inner, experts_count = rng.randint(1, 7), rng.randint(1, 9), rng.randint(1, 7)
    ids = rng.sample(range(3, 800), experts_count)
    ranks = rng.sample(range(1, 40), rng.randint(1, 4))
    local = {r: 0 for r in ranks}
    experts, copies = [], []
    for eid in ids:
        experts.append({'id': eid, **{k: [rng.uniform(-1, 1) for _ in range(h * inner)]
                                      for k in ('gate', 'up', 'down')}})
        owner = rng.choice(ranks)
        copies.append({'expert': eid, 'rank': owner, 'local': local[owner]})
        local[owner] += 1
    tokens, k = rng.randint(0, 7), rng.randint(1, experts_count)
    routes = [[{'expert': eid, 'weight': rng.choice([0.0, rng.random(), 2.0 * rng.random()])}
               for eid in rng.sample(ids, k)] for _ in range(tokens)]
    return {'model': {'hidden': h, 'intermediate': inner, 'experts': experts},
            'batch': {'inputs': [[rng.uniform(-2, 2) for _ in range(h)] for _ in range(tokens)],
                      'routes': routes}, 'registry': {'epoch': 9, 'copies': copies},
            'stamp': {'run_id': f'seed-{seed}', 'model_revision': 'synthetic', 'plan_id': 'fixed',
                      'layer': 5, 'batch': seed, 'epoch': 9}, 'preferred_owners': {}}


def close(a: list[list[float]], b: list[list[float]]) -> float:
    assert len(a) == len(b)
    maximum = 0.0
    for row, expected in zip(a, b):
        assert len(row) == len(expected)
        for x, y in zip(row, expected):
            assert math.isfinite(x) and math.isclose(x, y, rel_tol=2e-11, abs_tol=2e-11), (x, y)
            maximum = max(maximum, abs(x-y))
    return maximum


def brute_pipeline(p: dict) -> float | None:
    n, m = len(p['layers']), len(p['stages'])
    best = math.inf
    for cuts in itertools.combinations(range(1, n), m-1):
        bounds = (0,) + cuts + (n,)
        costs = []
        for j, stage in enumerate(p['stages']):
            layers = p['layers'][bounds[j]:bounds[j+1]]
            if sum(l['bytes'] for l in layers) > stage['available_bytes']:
                break
            costs.append(sum(l['compute_us_by_stage'][j] for l in layers) + stage['boundary_after_us'])
        if len(costs) == m:
            best = min(best, max(costs))
    return None if best == math.inf else best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--seeds', type=int, default=192)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 10000:
        parser.error('seeds must be within 1..10000')
    binary = args.binary.resolve(strict=True)
    counts = {'moe_random': 0, 'registry_permutation': 0, 'collection_reordered': 0,
              'collection_invalid': 0, 'memory_random': 0, 'pipeline_exhaustive': 0,
              'nvfp4_scale_cases': 0, 'cli_invalid': 0}
    maximum = 0.0
    begin = time.monotonic()
    for seed in range(args.seeds):
        p = problem(seed)
        expected = oracle(p)
        result = invoke(binary, 'moe', p)
        maximum = max(maximum, close(result['reference'], expected), close(result['partitioned'], expected))
        counts['moe_random'] += 1
        if seed % 8 == 0:
            p['registry']['copies'].reverse(); p['model']['experts'].reverse()
            maximum = max(maximum, close(invoke(binary, 'moe', p)['partitioned'], expected))
            counts['registry_permutation'] += 1
            slots = invoke(binary, 'dispatch', p)
            experts = {e['id']: e for e in p['model']['experts']}
            completions = [{'stamp': p['stamp'], **{k: s[k] for k in ('token','slot','expert','rank','local')},
                            'value': expert_value(p['model'], experts[s['expert']], p['batch']['inputs'][s['token']])}
                           for s in reversed(slots)]
            maximum = max(maximum, close(invoke(binary, 'collect', {'problem': p, 'completions': completions}), expected))
            counts['collection_reordered'] += 1
            if completions:
                invoke(binary, 'collect', {'problem': p, 'completions': completions + completions[:1]}, valid=False)
                invoke(binary, 'collect', {'problem': p, 'completions': completions[:-1]}, valid=False)
                counts['collection_invalid'] += 2
        rng = random.Random(seed+100000)
        rows, cols, align, glob = rng.randint(1, 100), rng.randint(1, 8193), 2**rng.randint(0, 12), rng.choice([0,4,8])
        padded = (cols+15)//16*16
        up = lambda n: ((n+align-1)//align)*align
        expected_layout = {'codes': up(rows*padded//2), 'scales': up(rows*padded//16),
                           'total': up(rows*padded//2)+up(rows*padded//16)+up(glob), 'padded_columns': padded}
        assert invoke(binary, 'layout', {'rows': rows, 'columns': cols, 'alignment': align, 'global_scale_bytes': glob}) == expected_layout
        counts['memory_random'] += 1
        n = rng.randint(1,9); m = rng.randint(1,min(n,4))
        pp = {'layers': [{'bytes': rng.randint(1,20),'compute_us_by_stage':[rng.randint(0,10) for _ in range(m)]} for _ in range(n)],
              'stages':[{'rank': 2+j*7,'available_bytes': rng.randint(10,80),'boundary_after_us': rng.randint(0,5) if j<m-1 else 0} for j in range(m)]}
        best = brute_pipeline(pp)
        output = invoke(binary,'pipeline',pp,valid=best is not None)
        if best is not None:
            assert output['bottleneck_proxy_us'] == best
            segments = output['segments']
            assert segments[0]['first_layer']==0 and segments[-1]['end_layer']==n
            assert all(a['end_layer']==b['first_layer'] for a,b in zip(segments,segments[1:]))
        counts['pipeline_exhaustive'] += 1
    lut=[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]
    for scale in range(256):
        req={'rows':1,'columns':16,'codes':[i+(i+1)*16 for i in range(0,16,2)],'scales':[scale],'global_dequant':1.0}
        valid=scale<127
        out=invoke(binary,'dequant',req,valid=valid)
        if valid:
            e, m=scale>>3, scale&7
            s=m/512 if e==0 else (1+m/8)*2**(e-7)
            close(out,[[v*s for v in lut]])
        counts['nvfp4_scale_cases']+=1
    invalid_cases=[('layout',{'rows':1,'columns':2,'alignment':1,'global_scale_bytes':0,'invented':True}),
                   ('layout',{'rows':-1,'columns':2,'alignment':1,'global_scale_bytes':0}),
                   ('memory',{'capacity':1}),('not-a-command',{}),('moe',{})]
    for cmd, payload in invalid_cases:
        invoke(binary,cmd,payload,valid=False);counts['cli_invalid']+=1
    # Sensitivity control: a deliberately corrupted output must be detected.
    detected=False
    try:close([[1.0]],[[1.001]])
    except AssertionError:detected=True
    assert detected
    report={'status':'PASS','evidence_level':'CPU_ONLY','seed_range':[0,args.seeds-1],
            'counts':counts,'total_cases':sum(counts.values()),'max_abs_error':maximum,
            'tolerance':{'atol':2e-11,'rtol':2e-11},'sensitivity_control':'detected deliberate output corruption',
            'elapsed_seconds_cpu_harness':time.monotonic()-begin,'gpu_claim':None}
    text=json.dumps(report,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    print(text)

if __name__=='__main__':
    main()
