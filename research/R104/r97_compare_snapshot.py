#!/usr/bin/env python3
"""Compare declared benchmark contracts; never infer or publish speed ratios."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

LIMIT = 256 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")

class Invalid(ValueError):
    pass


def pairs_unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise Invalid("duplicate JSON key: " + key)
        obj[key] = value
    return obj


def invalid_constant(value):
    raise Invalid("non-finite JSON constant: " + value)


def read_json(path, expected_digest=None):
    with Path(path).open('rb') as file:
        raw = file.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise Invalid("JSON exceeds size cap")
    if expected_digest is not None and hashlib.sha256(raw).hexdigest() != expected_digest:
        raise Invalid("quality evidence SHA mismatch")
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs_unique, parse_constant=invalid_constant)


def obj(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise Invalid(label + " fields differ from schema")
    return value


def string(value, label):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise Invalid(label + " must be a nonempty bounded string")
    return value


def sha(value, label):
    if type(value) is not str or HEX.fullmatch(value) is None:
        raise Invalid(label + " must be lowercase SHA256")
    return value


def positive_int(value, label):
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise Invalid(label + " must be a positive integer")
    return value


def number(value, label, positive=False):
    if type(value) not in (int, float):
        raise Invalid(label + " must be a finite number")
    try:
        valid = math.isfinite(value) and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise Invalid(label + " must be a finite" + (" positive" if positive else "") + " number")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise Invalid(label + " must be a boolean")
    return value


def quality(run):
    ref = obj(run['quality'], ('evidence_path', 'evidence_sha256'), 'quality reference')
    path = string(ref['evidence_path'], 'quality path')
    if not Path(path).is_absolute():
        raise Invalid('quality path must be absolute')
    sha(ref['evidence_sha256'], 'quality evidence sha')
    q = obj(read_json(path, ref['evidence_sha256']), ('schema', 'package_sha256', 'dataset_sha256', 'reference_sha256', 'metric', 'higher_is_better', 'threshold', 'score', 'sample_count'), 'quality evidence')
    if q['schema'] != 'r97-quality-v1':
        raise Invalid('quality evidence schema mismatch')
    for name in ('package_sha256', 'dataset_sha256', 'reference_sha256'):
        sha(q[name], 'quality ' + name)
    if q['package_sha256'] != run['model']['package_sha256']:
        raise Invalid('quality package mismatch')
    string(q['metric'], 'quality metric')
    boolean(q['higher_is_better'], 'quality direction')
    threshold = number(q['threshold'], 'quality threshold')
    score = number(q['score'], 'quality score')
    positive_int(q['sample_count'], 'quality sample count')
    if (q['higher_is_better'] and score < threshold) or (not q['higher_is_better'] and score > threshold):
        raise Invalid('quality threshold not met')
    return q


def validate(run):
    obj(run, ('schema','engine','model','environment','workload','metrics','quality'), 'run')
    if run['schema'] != 'r97-run-v1':
        raise Invalid('run schema mismatch')
    engine = obj(run['engine'], ('name','revision'), 'engine')
    for key in engine: string(engine[key], 'engine ' + key)
    model = obj(run['model'], ('logical_id','base_sha256','tokenizer_sha256','package_sha256','quantization','layout'), 'model')
    for key in ('logical_id','quantization','layout'): string(model[key], 'model ' + key)
    for key in ('base_sha256','tokenizer_sha256','package_sha256'): sha(model[key], 'model ' + key)
    env = obj(run['environment'], ('hardware_id','driver','device_count','parallelism','software_boundary'), 'environment')
    for key in ('hardware_id','driver','parallelism','software_boundary'): string(env[key], 'environment ' + key)
    positive_int(env['device_count'], 'device count')
    work = obj(run['workload'], ('prompt_corpus_sha256','token_counts_sha256','sampling_sha256','concurrency','max_new_tokens','prefix_cache','warmup_policy','measurement_boundary'), 'workload')
    for key in ('prompt_corpus_sha256','token_counts_sha256','sampling_sha256'): sha(work[key], 'workload ' + key)
    for key in ('concurrency','max_new_tokens'): positive_int(work[key], 'workload ' + key)
    for key in ('warmup_policy','measurement_boundary'): string(work[key], 'workload ' + key)
    boolean(work['prefix_cache'], 'prefix cache')
    metrics = obj(run['metrics'], ('requests','output_tokens','output_token_counts_sha256','wall_seconds','ttft_p50_ms','ttft_p95_ms','tpot_p50_ms','tpot_p95_ms'), 'metrics')
    for key in ('requests','output_tokens'): positive_int(metrics[key], 'metrics ' + key)
    sha(metrics['output_token_counts_sha256'], 'metrics output token counts sha')
    for key in ('wall_seconds','ttft_p50_ms','ttft_p95_ms','tpot_p50_ms','tpot_p95_ms'): number(metrics[key], 'metrics ' + key, positive=True)
    if metrics['ttft_p95_ms'] < metrics['ttft_p50_ms'] or metrics['tpot_p95_ms'] < metrics['tpot_p50_ms']:
        raise Invalid('p95 below p50')
    return quality(run)


def compare_files(left, right):
    try:
        a, b = read_json(left), read_json(right)
        qa, qb = validate(a), validate(b)
        for section, fields in (('model', ('logical_id','base_sha256','tokenizer_sha256')), ('environment', tuple(a['environment'])), ('workload', tuple(a['workload']))):
            for key in fields:
                if a[section][key] != b[section][key]:
                    raise Invalid(section + '.' + key + ' differs')
        for key in ('requests','output_tokens','output_token_counts_sha256'):
            if a['metrics'][key] != b['metrics'][key]:
                raise Invalid('metrics.' + key + ' differs')
        for key in ('dataset_sha256','reference_sha256','metric','higher_is_better','threshold','sample_count'):
            if qa[key] != qb[key]: raise Invalid('quality ' + key + ' differs')
        same_package = a['model']['package_sha256'] == b['model']['package_sha256']
        same_format = all(a['model'][key] == b['model'][key] for key in ('quantization','layout'))
        if same_package and same_format:
            verdict = 'ENGINE_COMPARABLE_SAME_PACKAGE'
        elif not same_package and not same_format:
            verdict = 'PACKAGE_COMPARABLE_WITH_QUALITY_EVIDENCE'
        else:
            raise Invalid('package identity and format relationship ambiguous')
        return {'verdict':verdict, 'limitations':['Declarations and quality artifact bytes are not independently authenticated; no speed ratio was computed.']}
    except (Invalid, OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        return {'verdict':'NON_COMPARABLE', 'reasons':[str(exc)]}


def main(argv):
    if len(argv) != 3:
        print(json.dumps({'verdict':'NON_COMPARABLE','reasons':['expected two JSON paths']}))
        return 2
    result = compare_files(argv[1], argv[2])
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 2 if result['verdict'] == 'NON_COMPARABLE' else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
