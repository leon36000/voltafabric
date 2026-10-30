"""Enumerate feasible one-eviction plans for the R511 witness."""
from itertools import combinations

used, target, budget, residents = 10, 6, 1, 3
candidates = [(1, 2), (2, 4)]  # (expert, accounted bytes), LRU order
feasible = [tuple(e for e, _ in subset)
            for n in range(budget + 1)
            for subset in combinations(candidates, n)
            if residents - len(subset) >= 1
            and used - sum(size for _, size in subset) <= target]
assert feasible == [(2,)], feasible
print('PASS: only expert 2 satisfies the bound and resident constraint')
