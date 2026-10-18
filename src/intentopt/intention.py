"""Internal conversion of CVXPY intentions into penalty expressions."""

from __future__ import annotations

from typing import TypeAlias, cast

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.elementwise.abs import abs as Abs
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality

from .integrality import MultipleOf

RawIntention: TypeAlias = cp.Constraint | cp.Minimize | cp.Maximize | MultipleOf


def penalty(intention: RawIntention) -> cp.Expression:
    """Return the scalar loss for an objective or soft constraint."""
    if isinstance(intention, MultipleOf):
        return intention._penalty()
    if isinstance(intention, cp.Minimize):
        return cast("cp.Expression", intention.expr)
    if isinstance(intention, cp.Maximize):
        return cast("cp.Expression", -intention.expr)
    if isinstance(intention, Equality):
        return Sum(Abs(intention.expr))
    if isinstance(intention, Inequality):
        return Sum(maximum(intention.expr, 0))
    msg = (
        f"Cannot soften {type(intention).__name__}; use an infinite weight "
        "or express it as an ordinary equality or inequality"
    )
    raise TypeError(msg)


def violation(intention: cp.Constraint | MultipleOf) -> float | None:
    """Measure total entrywise error, or None for unsupported constraint classes.

    Summing absolute residuals preserves total entrywise error independently
    of CVXPY's ``violation()`` reduction. Other constraint classes can require
    extra projection solves, so their diagnostics are left to the caller.
    """
    if isinstance(intention, MultipleOf):
        return intention._violation()
    if isinstance(intention, (Equality, Inequality)):
        residual = intention.residual
        if residual is None:
            msg = "Cannot compute violation without constraint values"
            raise ValueError(msg)
        return float(np.sum(np.abs(residual)))
    return None
