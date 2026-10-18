"""Keep round lots in the same list as the other intentions."""

import cvxpy as cp

from intentopt import Intentions, MultipleOf, solve

shares = cp.Variable(name="shares")
result = solve(
    Intentions.from_priorities(
        {
            0: [shares >= 0, shares <= 1000, MultipleOf(shares, 100)],
            1: [cp.Minimize(cp.abs(shares - 230))],
        }
    ),
    solver="SCIPY",
)
print(result.status, shares.value)
