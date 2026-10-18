"""Balance portfolio preferences with round lots and limited turnover."""

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.sum import Sum

from intentopt import Intentions, MultipleOf, solve

holdings = cp.Variable(3, name="holdings")
current = np.array([500.0, 300.0, 200.0])
returns = np.array([0.08, 0.04, 0.02])

priorities = {
    0: [holdings >= 0, Sum(holdings) == 1000],
    1: [Sum(holdings[:2]) >= 600, Sum(holdings[:2]) <= 700],
    2: [Sum(cp.multiply(returns, holdings)) >= float(returns @ current)],
    3: [holdings[0] == 0],
    4: [MultipleOf(holdings, 100)],
    5: [cp.Minimize(cp.norm1(holdings - current))],
}
intentions = Intentions.from_priorities(priorities)
result = solve(intentions, solver="SCIPY")
print(result.status, holdings.value)
for violation in result.violations:
    print(violation.weight, violation.constraint, violation.amount)
