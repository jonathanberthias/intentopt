import math

import cvxpy as cp
import numpy as np
import pytest
from cvxpy.atoms.elementwise.abs import abs as Abs

from intentopt import (
    BinarySearch,
    Integral,
    Intentions,
    MultipleOf,
    SearchPolicy,
    solve,
)


def test_required_round_lots_are_explicit_intentions() -> None:
    shares = cp.Variable()
    result = solve(
        Intentions.from_priorities(
            {
                0: [shares >= 0, shares <= 1000, Integral(shares / 100)],
                1: [cp.Minimize(Abs(shares - 230))],
            }
        ),
        solver="SCIPY",
    )
    assert result.has_solution
    assert shares.value == pytest.approx(200)
    assert set(result.values) == {shares}  # Auxiliary integers stay internal.


@pytest.mark.parametrize("search", [None, BinarySearch()])
def test_soft_integrality_penalizes_distance_to_nearest_integer(
    search: SearchPolicy | None,
) -> None:
    x = cp.Variable(3)
    integral = Integral(x)
    result = solve(
        {math.inf: [x == [1.2, -1.8, 3.5]], 1: [integral]},
        search=search,
        solver="SCIPY",
    )
    assert result.has_solution
    assert result.hard_levels == 0
    assert result.problem is not None
    assert result.problem.value == pytest.approx(0.9)
    assert len(result.violations) == 1
    assert result.violations[0].constraint is integral
    assert result.violations[0].amount == pytest.approx(0.9)
    result_values = result.values[x]
    assert result_values is not None
    np.testing.assert_allclose(result_values, [1.2, -1.8, 3.5])


def test_integrality_participates_in_prefix_like_other_constraints() -> None:
    x = cp.Variable()
    result = solve(
        {math.inf: [x >= 0, x <= 10], 10: [Integral(x)], 1: [x == 2.3]},
        solver="SCIPY",
    )
    assert result.hard_levels == 1
    assert x.value == pytest.approx(2)
    assert result.violations[0].amount == pytest.approx(0.3)


def test_required_integrality_can_make_required_constraints_infeasible() -> None:
    x = cp.Variable()
    result = solve({math.inf: [x == 1.2, Integral(x)]}, solver="SCIPY")
    assert result.status == cp.INFEASIBLE
    assert not result.has_solution


def test_soft_integrality_influences_continuous_values() -> None:
    x = cp.Variable()
    result = solve(
        {math.inf: [x >= 1.2, x <= 1.3], 10: [Integral(x)], 1: [cp.Maximize(x)]},
        solver="SCIPY",
    )
    assert result.hard_levels == 0
    assert x.value == pytest.approx(1.2)


def test_integrality_requires_affine_expressions() -> None:
    with pytest.raises(TypeError, match="affine"):
        Integral(cp.Variable() ** 2)


def test_integrality_rejects_complex_expressions() -> None:
    with pytest.raises(TypeError, match="real affine"):
        Integral(cp.Variable(complex=True))


def test_required_multiple_of_uses_original_units() -> None:
    shares = cp.Variable()
    result = solve(
        {
            math.inf: [shares >= 0, shares <= 1000, MultipleOf(shares, 100)],
            1: [cp.Minimize(Abs(shares - 230))],
        },
        solver="SCIPY",
    )
    assert result.has_solution
    assert shares.value == pytest.approx(200)
    assert set(result.values) == {shares}


@pytest.mark.parametrize("search", [None, BinarySearch()])
def test_soft_multiple_of_penalizes_distance_in_original_units(
    search: SearchPolicy | None,
) -> None:
    x = cp.Variable(3)
    lots = MultipleOf(x, 100)
    result = solve(
        {math.inf: [x == [230, -170, 50]], 1: [lots]},
        search=search,
        solver="SCIPY",
    )
    assert result.hard_levels == 0
    assert result.problem is not None
    assert result.problem.value == pytest.approx(110)
    assert result.violations[0].constraint is lots
    assert result.violations[0].amount == pytest.approx(110)


def test_multiple_of_accepts_fractional_steps() -> None:
    x = cp.Variable()
    solve(
        {
            math.inf: [x >= 0, x <= 1, MultipleOf(x, 0.05)],
            1: [cp.Minimize(Abs(x - 0.23))],
        },
        solver="SCIPY",
    )
    assert x.value == pytest.approx(0.25)


def test_multiple_of_has_same_tier_semantics_as_integrality() -> None:
    x = cp.Variable()
    result = solve(
        {math.inf: [x >= 0, x <= 1000], 10: [MultipleOf(x, 100)], 1: [x == 230]},
        solver="SCIPY",
    )
    assert result.hard_levels == 1
    assert x.value == pytest.approx(200)
    assert result.violations[0].amount == pytest.approx(30)


@pytest.mark.parametrize("step", [0, -1, math.inf, -math.inf, math.nan])
def test_multiple_of_rejects_invalid_steps(step: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        MultipleOf(cp.Variable(), step)
