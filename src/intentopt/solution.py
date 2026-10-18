"""Snapshots and diagnostics from an intention solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cvxpy as cp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .integrality import MultipleOf


@dataclass(frozen=True)
class SolveAttempt:
    """A search probe's hard-tier count, objective status, and feasibility."""

    hard_levels: int
    status: str
    feasible: bool


@dataclass(frozen=True)
class Violation:
    """A violated intention, its weight, and measured error.

    Ordinary equalities and inequalities report total entrywise error.
    MultipleOf reports total distance to the nearest multiples. Other required
    constraint classes are enforced but omitted from violation reports.
    """

    constraint: cp.Constraint | MultipleOf
    weight: float
    amount: float


@dataclass(frozen=True)
class IntentSolution:
    """Selected stage and saved values, independent of later variable solves.

    ``problem`` and the variables themselves are mutable CVXPY objects. ``values``
    and violation amounts are copies made when the selected solve finishes.
    """

    status: str
    problem: cp.Problem | None
    hard_levels: int | None
    attempts: tuple[SolveAttempt, ...]
    violations: tuple[Violation, ...]
    # CVXPY does not specify the dtype returned by every solver.
    values: dict[cp.Variable, NDArray[Any] | None]  # pyrefly: ignore[explicit-any]

    @property
    def has_solution(self) -> bool:
        """Whether the solver reports an optimum, including inaccurate results."""
        return self.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
