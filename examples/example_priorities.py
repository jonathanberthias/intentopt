"""Express required constraints and preferences with priority labels."""

import cvxpy as cp

from intentopt import Intentions, solve

x = cp.Variable(name="x")
result = solve(
    Intentions.from_priorities(
        {
            0: [x >= 0],
            1: [x >= 10],
            2: [x <= 5],
            3: [cp.Minimize(cp.square(x - 7))],
        }
    ),
    solver="CLARABEL",
)
print(result.status, x.value, result.hard_levels)
for violation in result.violations:
    print(violation.constraint, violation.amount)
