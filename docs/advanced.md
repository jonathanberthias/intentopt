# Search options

Sequential search is the default. Binary search can reduce the number of solver
calls when there are many preference groups. Both find the same cutoff, though
solutions can differ when several points are equally good.

These examples use two conflicting preferences for a single variable.

## Binary search and a starting guess

```python
import cvxpy as cp
from intentopt import BinarySearch, Intentions, solve

x = cp.Variable(name="x")
intentions = Intentions.from_priorities({0: [x >= 0], 1: [x >= 10], 2: [x <= 5]})

result = solve(intentions, search=BinarySearch(), solver="CLARABEL")
result = solve(intentions, search=BinarySearch(start=2), solver="CLARABEL")
print([attempt.hard_levels for attempt in result.attempts])
```

`start=2` means two finite constraint-bearing tiers. Required tiers and tiers
containing only objectives do not count. The guess must be in range; it is
tested, not trusted. The search still finds the full feasible prefix.

## A custom search policy

A policy receives the maximum count and a cached Boolean feasibility callback.
This policy works backwards, which can be useful when almost every tier is
expected to be feasible. It costs one probe when every tier is feasible.

```python
class FromStrictest:
    def search(self, max_hard_levels, feasible):
        for count in range(max_hard_levels, -1, -1):
            if feasible(count):
                return count
        return None


result = solve(intentions, search=FromStrictest(), solver="CLARABEL")
```

The callback handles CVXPY solves, status checks, and caching. Custom policies
must find the largest feasible count or return `None` when even zero finite
tiers is infeasible. The solver validates the chosen cutoff and its next
neighbor.
