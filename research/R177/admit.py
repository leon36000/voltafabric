# SPDX-License-Identifier: Apache-2.0
"""Private CPU admission of an expert-placement proposal, not a live migration.

The shape of the proposal is inspired by DeepInfer's physical-to-logical map;
no source code from the donor is included. Resource capacities are caller-owned
byte budgets and are not discovered or reserved by this module.
"""
from copy import deepcopy

class AdmissionError(ValueError):
    pass

def _uint(value, bits):
    return type(value) is int and 0 <= value < (1 << bits)

def admit(problem, proposal):
    if type(problem) is not dict or type(proposal) is not dict:
        raise AdmissionError('problem and proposal must be objects')
    if set(proposal) != {'rank_ids', 'capacity_bytes', 'resident_bytes', 'proposed'}:
        raise AdmissionError('proposal fields differ')
    ranks, capacities, sizes, rows = (proposal[k] for k in ('rank_ids', 'capacity_bytes', 'resident_bytes', 'proposed'))
    experts = problem.get('experts')
    registry = problem.get('registry')
    stamp = problem.get('stamp')
    if (type(ranks) is not list or not ranks or len(ranks) > 1024
        or any(not _uint(r, 32) for r in ranks) or len(set(ranks)) != len(ranks)
        or type(rows) is not list or len(rows) != len(ranks)
        or type(capacities) is not dict or set(capacities) != set(ranks)
        or type(experts) is not list or not experts or len(experts) > 1024
        or any(not _uint(e, 32) for e in experts) or len(set(experts)) != len(experts)
        or type(sizes) is not dict or set(sizes) != set(experts)
        or any(not _uint(n, 64) for n in capacities.values())
        or any(not _uint(n, 64) or n == 0 for n in sizes.values())
        or type(registry) is not dict or set(registry) != {'epoch', 'copies'}
        or type(stamp) is not dict or not _uint(registry['epoch'], 64)
        or not _uint(stamp.get('epoch'), 64) or stamp['epoch'] != registry['epoch']):
        raise AdmissionError('invalid identity, capacity, counts or epoch')
    known = set(experts)
    copies = []
    resident_order = {}
    covered = set()
    for rank, row in zip(ranks, rows):
        if (type(row) is not list or len(row) > 1024
            or any(not _uint(e, 32) or e not in known for e in row)
            or len(set(row)) != len(row)):
            raise AdmissionError('unknown or duplicated same-rank expert')
        if sum(sizes[e] for e in row) > capacities[rank]:
            raise AdmissionError('resident bytes exceed available rank capacity')
        for local, expert in enumerate(row):
            copies.append({'expert': expert, 'rank': rank, 'local': local})
            covered.add(expert)
        resident_order[str(rank)] = list(range(len(row)))
    if covered != known or len(copies) > 65536:
        raise AdmissionError('proposal does not cover experts or exceeds bounds')
    preferred = problem.get('preferred_owners', {})
    if type(preferred) is not dict:
        raise AdmissionError('preferred owners must be an object')
    pairs = {(c['expert'], c['rank']) for c in copies}
    for key, rank in preferred.items():
        if (type(key) is not str or not key.isascii() or not key.isdecimal()
            or len(key) > 10 or str(int(key)) != key or not _uint(rank, 32)
            or (int(key), rank) not in pairs):
            raise AdmissionError('preferred owner has no resident replica')
    result = deepcopy(problem)
    result['registry']['copies'] = copies
    result['resident_order'] = resident_order
    return result
