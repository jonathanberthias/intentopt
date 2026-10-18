# Examples

## Portfolio wishes

This illustrative model expresses all holding values in the same currency.
Return and allocation expressions are scaled by the user. Round lots here are
lots of holding value; use share quantities to express lots of actual shares.

```python
import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.sum import Sum
from intentopt import Intentions, MultipleOf, solve

holdings = cp.Variable(3, name="holdings")
current = np.array([500.0, 300.0, 200.0])
returns = np.array([0.08, 0.04, 0.02])

intentions = Intentions.from_priorities(
    {
        0: [holdings >= 0, Sum(holdings) == 1000],
        1: [Sum(holdings[:2]) >= 600, Sum(holdings[:2]) <= 700],
        2: [Sum(cp.multiply(returns, holdings)) >= float(returns @ current)],
        3: [holdings[0] == 0],
        4: [MultipleOf(holdings, 100)],
        5: [cp.Minimize(cp.norm1(holdings - current))],
    }
)
result = solve(intentions, solver="SCIPY")
print(result.status, holdings.value)
for violation in result.violations:
    print(violation.weight, violation.constraint, violation.amount)
```

The allocation and return requirements prevent fully selling the first holding.
The result keeps those requirements and reports the unmet sale and round-lot
preferences.

## Values that survive another solve

```python
x = cp.Variable()
target = cp.Parameter(value=3)
intentions = Intentions({1: [cp.Minimize(cp.square(x - target))]})
first = solve(intentions, solver="CLARABEL")
target.value = 7
second = solve(intentions, solver="CLARABEL")
print(first.values[x])  # Approximately 3
print(second.values[x])  # Approximately 7
```

Values are keyed by the actual variable, so duplicate variable names do not
collide. CVXPY variables and problems are still mutable; their current values
reflect the latest solve.

Search options and custom policies are described in the
[advanced guide](advanced.md).
