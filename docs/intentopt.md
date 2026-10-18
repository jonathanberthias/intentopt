# User guide

Use ordinary CVXPY constraints to describe requirements and preferences. Assign
priorities or weights, then pass them to `solve()`.

## Writing preferences

Use priority numbers when the order is what matters: 0 is mandatory, 1 is the
most important preference, then 2, 3, and so on.

```python
import cvxpy as cp
from intentopt import Intentions, solve

x = cp.Variable()
intentions = Intentions.from_priorities({0: [x >= 0], 1: [x >= 10], 2: [x <= 5]})
result = solve(intentions, solver="CLARABEL")
```

Use explicit weights when you want to choose the strength of each preference.
Larger weights mean more important preferences; infinite weight means mandatory.
A mapping passed directly to `solve()` also uses weights.

```python
from math import inf

result = solve({inf: [x >= 0], 100: [x >= 10], 10: [x <= 5]}, solver="CLARABEL")
```

Constraints with the same priority or weight are considered together. You can
also pass a sequence of `(weight, intention)` pairs to `Intentions`, or
`(priority, intention)` pairs to `Intentions.from_priorities`.

## One number, two roles

A positive weight specifies both enforcement order and soft penalty strength.
Preferences with equal weights form one group, called a tier. Required
constraints have infinite weight.

Priority 0 maps to infinity; a positive priority `p` maps to `2 ** (10 - p)`.
Priorities above 10 keep producing smaller weights. Priorities whose weights
cannot be represented as positive floating-point numbers are rejected. Adding or
removing priorities does not change the weights of existing ones.

## The cutoff

Hard constraints must hold; soft constraints contribute a penalty when violated.

Find the largest number of finite constraint tiers that can all be hard, in
weight order. Keep required constraints hard throughout. Stop at the first
incompatible tier: all its constraints and every lower tier remain soft,
including constraints that could individually be satisfied.

At that cutoff, minimize the weighted sum of soft violations and all objectives.
Inequality losses are one-sided, equality losses absolute, and array losses are
summed entrywise. Lower-weight wishes can collectively outweigh a higher-weight
wish in the soft part. Users control units and normalization.

This is not lexicographic optimization: IntentOpt does not preserve a tier's
best attainable violation or objective value. Express “at least 10, ideally 15”
as two bounds with different priorities. Objectives always remain objective
terms and do not block hardening lower-weight constraints. Infinite-weight
objectives are invalid.

## Compared with CVXPY's targets and priorities

CVXPY's
[`scalarize.targets_and_priorities`](https://www.cvxpy.org/api_reference/cvxpy.transforms.html#cvxpy.transforms.scalarize.targets_and_priorities)
combines several objectives into one. You supply target values, penalty weights
called `priorities`, and optional hard limits. Targets change the penalty slope;
limits remain mandatory. The transform does not search for a feasible set of
limits or relax them when they conflict.

IntentOpt accepts constraints as preferences and chooses which tiers to enforce
before minimizing the remaining weighted loss. For example, with `x >= 10` at
priority 1 and `x <= 5` at priority 2, IntentOpt enforces `x >= 10` and
penalizes `x > 5`. Encoding both as hard limits in CVXPY would make the problem
infeasible; encoding both as targets would let them trade off from the outset.

Use CVXPY's transform when you already know which limits must hold and want to
choose penalties around target values. Use IntentOpt when you want the priority
order to determine which constraints remain mandatory. Neither approach implies
lexicographic optimization: CVXPY's `priorities` are penalty weights, while
IntentOpt's priority numbers determine enforcement order and soft weights.

## Objectives

Add `cp.Minimize(expression)` or `cp.Maximize(expression)` alongside constraints
to express goals without a target value. Their weights scale their contribution
to the compromise. They remain objectives throughout the search; a more
important objective does not prevent a less important constraint from becoming
mandatory for the selected solution.

```python
result = solve(
    Intentions.from_priorities({0: [x >= 0], 1: [x >= 10], 2: [cp.Minimize(x)]}),
    solver="CLARABEL",
)
```

CVXPY's convexity rules still apply. Pass solver options through `solve()` and
choose a solver that supports your model. If `solver` is omitted, CVXPY selects
from the installed solvers.

## Integrality

`Integral(x)` puts integrality in the same list as other intentions.
`MultipleOf(shares, 100)` describes purchases in lots of 100 shares. The step
must be a positive finite scalar; fractional steps such as 0.05 are supported.

An integer auxiliary z allows hard enforcement with `expression == step * z` or
soft penalization with `sum(abs(expression - step * z))`. The optimized soft
penalty is total distance to the nearest multiple in the expression's original
units: 230 has violation 30 for a step of 100. `Integral(x)` is the special case
`MultipleOf(x, 1)`. Both require real affine expressions and a mixed-integer
solver, including when softened.

`Integral(x / 100)` enforces the same hard constraint as `MultipleOf(x, 100)`,
but its soft penalty is in lots: 0.3 instead of 30 at x = 230. Users choose the
expression and weights to match the units of their other intentions.

Fractional values incur a distance penalty, instead of making integrality
disappear without cost. CVXPY's own variable attributes remain structural
requirements.

## Search and statuses

Sequential search is the default. Binary search and a starting guess can reduce
probes. Custom search policies use cached feasibility probes over tier counts.
They must return the largest feasible count; the solver verifies that count and
its next neighbor. Strategies change probe order, not the optimization contract.

An unbounded objective does not mean a prefix is infeasible. A separate
zero-objective solve resolves feasibility when necessary. Conflicting required
constraints return an infeasible result. A final unbounded result has no values.
Indeterminate statuses and solver errors raise rather than selecting a false
cutoff. Inaccurate statuses remain visible; all conclusions depend on the chosen
solver's numerical tolerances.

## Scope

Ordinary CVXPY equalities and inequalities can be softened. Other cone
constraint objects can be required, but unsupported soft forms fail explicitly.
Violation reports cover equalities, inequalities, `Integral`, and `MultipleOf`.
Other required constraint classes are enforced but omitted from these reports;
CVXPY's own diagnostics for them can require additional solver calls. Solver
options are forwarded to CVXPY. Parameters can be changed and the same
intentions solved again. Saved result values and violation magnitudes are
snapshots.

## Reading results

Check `result.has_solution` before using values. `result.status` is the CVXPY
status, and `result.values[x]` contains a saved copy of a variable's value.
`result.violations` lists missed constraints with their weights and amounts.
Only constraint violations above `violation_tolerance` (default `1e-6`) are
reported. This option filters diagnostics; solver options control feasibility
tolerances. Saved values and amounts survive later solves with the same
variables. A variable's saved value can be `None` if CVXPY did not assign it.

`result.hard_levels` counts enforced preference groups, excluding mandatory
constraints and groups containing only objectives. `result.problem` exposes the
selected CVXPY problem, or `None` if mandatory requirements are infeasible.
`result.attempts` reports each search probe's count, status, and feasibility in
the order it was tried. The status describes the weighted objective solve: an
unbounded objective can still have feasible constraints.
