"""Explicit integrality and round-lot intentions for CVXPY expressions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.elementwise.abs import abs as Abs
from cvxpy.constraints.zero import Equality
from typing_extensions import override


@dataclass(frozen=True, eq=False)
class MultipleOf:
    """Prefer or require multiples of a positive, finite scalar step.

    Use a real affine expression. Finite weights penalize total distance to the
    nearest multiple in the expression's original units: 230 has violation 30
    for a step of 100. Both hard and soft forms need a mixed-integer solver.
    """

    expression: cp.Expression
    step: float
    _integer: cp.Variable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.step) or self.step <= 0:
            msg = "step must be positive and finite"
            raise ValueError(msg)
        if not self.expression.is_affine() or not self.expression.is_real():
            msg = f"{type(self).__name__} requires a real affine CVXPY expression"
            raise TypeError(msg)
        object.__setattr__(
            self, "_integer", cp.Variable(self.expression.shape, integer=True)
        )

    def _constraint(self) -> cp.Constraint:
        return Equality(self.expression, self.step * self._integer)

    def _penalty(self) -> cp.Expression:
        return Sum(Abs(self.expression - self.step * self._integer))

    def _violation(self) -> float:
        value = np.asarray(self.expression.value)
        return float(np.sum(np.abs(value - self.step * np.rint(value / self.step))))

    @override
    def __str__(self) -> str:
        return f"MultipleOf({self.expression}, {self.step})"


class Integral(MultipleOf):
    """Prefer or require integer values, equivalent to MultipleOf(expression, 1).

    Finite weights penalize total distance to the nearest integer. The expression
    must be real and affine. Both hard and soft forms need a mixed-integer solver.
    """

    def __init__(self, expression: cp.Expression) -> None:
        super().__init__(expression, 1)

    @override
    def __str__(self) -> str:
        return f"Integral({self.expression})"
