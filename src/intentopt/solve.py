"""Solve the longest feasible prefix of weighted constraint tiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from .integrality import MultipleOf
from .intention import violation
from .model import Intentions, WeightedInput
from .search import SearchPolicy, SequentialSearch, _validate_count
from .solution import IntentSolution, SolveAttempt, Violation

if TYPE_CHECKING:
    from numpy.typing import NDArray

_OPTIMAL = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_INFEASIBLE = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
_UNBOUNDED = {cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE, "infeasible_or_unbounded"}


@dataclass
class _Probe:
    problem: cp.Problem
    feasible: bool
    feasibility_problem: cp.Problem | None = None


def solve(
    intentions: Intentions | WeightedInput,
    *,
    search: SearchPolicy | None = None,
    violation_tolerance: float = 1e-6,
    # CVXPY forwards arbitrary solver-specific option values.
    **solver_options: Any,  # pyrefly: ignore[explicit-any]
) -> IntentSolution:
    """Enforce the highest feasible tiers and minimize the remaining weighted loss.

    Plain mappings contain explicit weights. Use ``Intentions.from_priorities``
    for numbered priorities. Solver options are forwarded to CVXPY, including
    ``solver``, ``warm_start``, and solver-specific settings.

    Required conflicts return an infeasible result. Solver errors and inconclusive
    statuses raise ``cvxpy.error.SolverError`` rather than choosing a false cutoff.
    ``violation_tolerance`` filters reported violations; it does not change the
    solver's feasibility tolerances. Reports cover ordinary equalities,
    inequalities, and explicit integrality; other required cones are enforced
    without computing their numeric violations.
    """
    if not math.isfinite(violation_tolerance) or violation_tolerance < 0:
        msg = "violation_tolerance must be finite and non-negative"
        raise ValueError(msg)
    model = intentions if isinstance(intentions, Intentions) else Intentions(intentions)
    max_hard_levels = len(model._constraint_weights)
    probes: dict[int, _Probe] = {}

    def feasible(count: int) -> bool:
        _validate_count(count, max_hard_levels)
        if count not in probes:
            problem = model._problem(count)
            _ = problem.solve(**solver_options)
            feasibility_problem = None
            status = problem.status
            if status in _UNBOUNDED:
                feasibility_problem = cp.Problem(cp.Minimize(0), problem.constraints)
                _ = feasibility_problem.solve(**solver_options)
                status = feasibility_problem.status
            if status not in _OPTIMAL | _INFEASIBLE:
                msg = f"Cannot establish feasibility for {count} hard tiers: {status}"
                raise cp.error.SolverError(msg)
            probes[count] = _Probe(problem, status in _OPTIMAL, feasibility_problem)
        return probes[count].feasible

    policy = search if search is not None else SequentialSearch()
    hard_levels = policy.search(max_hard_levels, feasible)
    if hard_levels is None:
        if feasible(0):
            msg = "Search returned no cutoff although required constraints are feasible"
            raise ValueError(msg)
    elif not feasible(hard_levels) or (
        hard_levels < max_hard_levels and feasible(hard_levels + 1)
    ):
        msg = "Search must return the largest feasible hard-tier count"
        raise ValueError(msg)

    # Later probes share CVXPY variables and may have cleared their values.
    selected_problem = probes[hard_levels].problem if hard_levels is not None else None
    final_problem = (
        selected_problem if selected_problem is not None else probes[0].problem
    )
    if hard_levels is None and probes[0].feasibility_problem is not None:
        final_problem = probes[0].feasibility_problem
    if final_problem.solution is not None:
        final_problem.unpack(final_problem.solution)
    status = final_problem.status
    if hard_levels is not None and status == "infeasible_or_unbounded":
        # The feasibility solve has excluded infeasibility.
        status = cp.UNBOUNDED
    # Preserve CVXPY's solver-dependent numeric dtype.
    values: dict[cp.Variable, NDArray[Any] | None] = {}  # pyrefly: ignore[explicit-any]
    violations: list[Violation] = []
    if selected_problem is not None and status in _OPTIMAL:
        variables = {
            variable.id: variable
            for items in model.weight_to_intentions.values()
            for item in items
            for variable in (
                item.expression.variables()
                if isinstance(item, MultipleOf)
                else item.variables()
            )
        }
        for variable in variables.values():
            values[variable] = (
                np.array(variable.value, copy=True)
                if variable.value is not None
                else None
            )
        for weight, items in model.weight_to_intentions.items():
            for item in items:
                if not isinstance(item, (cp.Constraint, MultipleOf)):
                    continue
                amount = violation(item)
                if amount is not None and amount > violation_tolerance:
                    violations.append(Violation(item, weight, amount))
    return IntentSolution(
        status,
        selected_problem,
        hard_levels,
        tuple(
            SolveAttempt(count, probe.problem.status, probe.feasible)
            for count, probe in probes.items()
        ),
        tuple(violations),
        values,
    )
