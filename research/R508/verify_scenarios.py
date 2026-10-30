#!/usr/bin/env python3
"""Independent small exhaustive oracle for two rank-headroom counterexamples."""
from itertools import combinations

def feasible(used, caps, reserve, target, candidates, counts):
    answers=[]
    for count in range(len(candidates)+1):
        for group in combinations(candidates,count):
            left=dict(used); per=dict(counts)
            for layer,rank,size in group:
                left[rank]-=size; per[(layer,rank)]-=1
            if (sum(left.values())<=target and all(n>=1 for n in per.values())
                    and any(caps[r]-left[r]>=reserve for r in caps)):
                answers.append(group)
    return answers

candidate=(0,2,4); caps={2:10,3:12}; counts={(0,2):2,(0,3):1}
a=feasible({2:4,3:12},caps,6,12,[candidate],counts)
b=feasible({2:8,3:10},caps,6,18,[candidate],counts)
assert a==[(candidate,)] and b==[(candidate,)]
assert not feasible({2:8,3:10},caps,6,18,[],counts)
print('PASS: both rank-2 alternatives are feasible and no-candidate case is not')
