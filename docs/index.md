# IntentOpt

Optimization problems often ask for more than can be achieved at once. You might
want a portfolio with low risk, little trading, and a particular return, or a
schedule that meets every deadline without overtime. When those requirements
conflict, a solver can simply report that there is no solution.

IntentOpt lets you say which requirements matter most and where you are willing
to compromise. It works with [CVXPY](https://www.cvxpy.org/): you write the
usual constraints, put them in priority order, and get a solution with a report
of which wishes could not be met.

## How it works

Some requirements are mandatory. The others are preferences, ordered by
importance. IntentOpt enforces preferences in that order for as long as they can
all be met. At the first conflict, it keeps the requirements already accepted
and finds a compromise among the remaining preferences, giving more weight to
the more important ones.

Mandatory requirements are never relaxed. If they conflict with each other,
there is no solution.

## Installation

IntentOpt requires Python 3.11+ and CVXPY 1.4.0+. It has not been published to
PyPI yet. From the repository root, install it with a solver such as Clarabel:

```bash
pip install . clarabel
```

## A small example

Suppose you are planning an event. You must stay within the venue's capacity,
would like at least 30 guests, and would prefer to keep the group to 20 so you
have time to talk to everyone. The last two wishes conflict; inviting enough
people matters more.

```python
import cvxpy as cp
from intentopt import Intentions, solve

guests = cp.Variable(name="guests")
intentions = Intentions.from_priorities(
    {
        0: [guests >= 0, guests <= 40],  # Must fit in the venue
        1: [guests >= 30],  # Most important preference
        2: [guests <= 20],  # Prefer a smaller group
    }
)
result = solve(intentions, solver="CLARABEL")

print(round(float(guests.value)))  # 30
for violation in result.violations:
    print(violation.constraint, violation.amount)
```

Priority 0 means mandatory. Smaller positive numbers mean more important
preferences. Here, IntentOpt keeps the minimum of 30 guests and reports that the
smaller-group preference was missed by 10.

See the [user guide](intentopt.md), [examples](examples.md), and
[API reference](api.md) for more.
