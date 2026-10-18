import math
from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pytest
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.elementwise.abs import abs as Abs

from intentopt import BinarySearch, Intentions, SearchPolicy, SequentialSearch, solve


@pytest.mark.parametrize(
    "search", [SequentialSearch(), BinarySearch(), BinarySearch(start=1)]
)
def test_conflict_returns_longest_feasible_prefix_and_restores_values(
    search: SearchPolicy | None,
) -> None:
    x = cp.Variable()
    conflict = x <= 5
    result = solve(
        {100: [x >= 10], 10: [conflict], 1: [cp.Minimize((x - 7) ** 2)]},
        search=search,
        solver="CLARABEL",
    )
    assert result.status == cp.OPTIMAL
    assert result.has_solution
    assert result.hard_levels == 1
    assert x.value == pytest.approx(10, abs=1e-5)
    assert result.values[x] == pytest.approx(10, abs=1e-5)
    assert len(result.violations) == 1
    assert result.violations[0].constraint is conflict
    assert result.violations[0].amount == pytest.approx(5, abs=1e-5)


@pytest.mark.parametrize("search", [SequentialSearch(), BinarySearch()])
def test_required_conflicts_return_no_solution(search: SearchPolicy | None) -> None:
    x = cp.Variable()
    result = solve(
        Intentions.from_priorities({0: [x >= 10, x <= 5], 1: [x >= 0]}),
        search=search,
        solver="CLARABEL",
    )
    assert result.status == cp.INFEASIBLE
    assert not result.has_solution
    assert result.hard_levels is None
    assert result.values == {}
    assert result.violations == ()
    assert x.value is None


def test_conflicting_tier_is_indivisible_and_later_tiers_stay_soft() -> None:
    x, y = cp.Variable(), cp.Variable()
    result = solve(
        {100: [x >= 10, x <= 5], 10: [y >= 20], 1: [cp.Minimize(y**2)]},
        solver="CLARABEL",
    )
    assert result.hard_levels == 0
    assert y.value == pytest.approx(5, abs=1e-4)
    assert any(violation.amount > 14 for violation in result.violations)


def test_equalities_penalize_both_directions_and_vectors_sum_entries() -> None:
    x = cp.Variable(2)
    equality = x == [2, -3]
    result = solve({math.inf: [x == 0], 1: [equality]}, solver="CLARABEL")
    assert result.hard_levels == 0
    assert result.violations[0].constraint is equality
    assert result.violations[0].amount == pytest.approx(5, abs=1e-5)
    assert result.problem is not None
    assert result.problem.value == pytest.approx(5, abs=1e-5)


def test_soft_equalities_actually_affect_the_selected_solution() -> None:
    x = cp.Variable()
    solve(
        {
            math.inf: [x >= -10, x <= 10],
            10: [x == 2, x == 4],
            1: [cp.Minimize(x**2)],
        },
        solver="CLARABEL",
    )
    assert x.value == pytest.approx(2, abs=1e-4)


def test_finite_weights_trade_off_total_violation() -> None:
    x = cp.Variable()
    result = solve(
        {math.inf: [x <= 0], 10: [x >= 2], 1: [cp.Minimize((x + 10) ** 2)]},
        solver="CLARABEL",
    )
    assert result.hard_levels == 0
    assert x.value == pytest.approx(-5, abs=1e-4)


def test_objectives_do_not_block_hardening_lower_weight_constraints() -> None:
    x = cp.Variable()
    result = solve({100: [cp.Maximize(x)], 1: [x <= 12]}, solver="CLARABEL")
    assert result.hard_levels == 1
    assert x.value == pytest.approx(12, abs=1e-4)
    assert result.attempts[0].status == cp.UNBOUNDED
    assert result.attempts[0].feasible


def test_final_unbounded_result_has_no_values_or_violations() -> None:
    x = cp.Variable()
    result = solve({1: [cp.Minimize(x)]}, solver="CLARABEL")
    assert result.status == cp.UNBOUNDED
    assert not result.has_solution
    assert result.values == {}
    assert result.violations == ()
    assert x.value is None


def test_objective_only_and_required_only_models() -> None:
    x = cp.Variable()
    result = solve({1: [cp.Minimize((x - 3) ** 2)]}, solver="CLARABEL")
    assert result.hard_levels == 0
    assert x.value == pytest.approx(3)
    result = solve({math.inf: [x == 7]}, solver="CLARABEL")
    assert result.has_solution
    assert x.value == pytest.approx(7)


def test_snapshot_survives_parameter_changes_and_another_solve() -> None:
    x = cp.Variable(2)
    target = cp.Parameter(2, value=[2, 3])
    intentions = Intentions({1: [cp.Minimize(Sum((x - target) ** 2))]})
    first = solve(intentions, solver="CLARABEL")
    target.value = [5, 6]
    second = solve(intentions, solver="CLARABEL")
    first_values = first.values[x]
    assert first_values is not None
    np.testing.assert_allclose(first_values, [2, 3], atol=1e-5)
    second_values = second.values[x]
    assert second_values is not None
    np.testing.assert_allclose(second_values, [5, 6], atol=1e-5)


def test_required_general_cones_remain_usable() -> None:
    x = cp.Variable(2)
    result = solve(
        Intentions([(math.inf, cp.SOC(1, x)), (1, cp.Maximize(Sum(x)))]),
        solver="CLARABEL",
    )
    assert result.has_solution
    assert x.value is not None
    np.testing.assert_allclose(x.value, [1 / np.sqrt(2)] * 2, atol=1e-4)


def test_required_power_cone_does_not_need_a_projection_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = cp.Variable()
    cone = cp.PowCone3D(x, 1, 1, 0.5)

    def unavailable_projection() -> float:
        msg = "Power-cone diagnostics require a separate projection solver"
        raise NotImplementedError(msg)

    monkeypatch.setattr(cone, "violation", unavailable_projection)
    result = solve({math.inf: [cone], 1: [cp.Minimize(x)]}, solver="CLARABEL")
    assert result.has_solution
    assert x.value == pytest.approx(1, abs=1e-5)
    assert result.violations == ()


def test_integer_variable_domains_remain_enforced() -> None:
    x = cp.Variable(integer=True)
    result = solve(
        {math.inf: [x >= 0, x <= 10], 1: [cp.Minimize(Abs(x - 2.3))]}, solver="SCIPY"
    )
    assert result.has_solution
    assert x.value == pytest.approx(2)


def test_invalid_custom_policy_cannot_silently_return_wrong_cutoff() -> None:
    class WrongPolicy:
        def search(self, max_hard_levels: int, feasible: Callable[[int], bool]) -> int:
            del max_hard_levels, feasible
            return 0

    with pytest.raises(ValueError, match="largest feasible"):
        solve({1: [cp.Variable() >= 0]}, search=WrongPolicy(), solver="CLARABEL")


def test_inconclusive_solver_status_is_not_treated_as_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inconclusive(problem: cp.Problem, **_options: object) -> None:
        problem._status = cp.USER_LIMIT

    monkeypatch.setattr(cp.Problem, "solve", inconclusive)
    with pytest.raises(cp.error.SolverError, match="Cannot establish feasibility"):
        solve({1: [cp.Variable() >= 0]}, solver="CLARABEL")


@pytest.mark.parametrize("feasible", [True, False])
def test_ambiguous_objective_status_is_resolved_before_reporting(
    monkeypatch: pytest.MonkeyPatch, feasible: bool
) -> None:
    calls = 0

    def ambiguous_then_feasibility(problem: cp.Problem, **_options: object) -> None:
        nonlocal calls
        calls += 1
        problem._status = (
            "infeasible_or_unbounded"
            if calls == 1
            else cp.OPTIMAL
            if feasible
            else cp.INFEASIBLE
        )

    monkeypatch.setattr(cp.Problem, "solve", ambiguous_then_feasibility)
    result = solve({1: [cp.Minimize(cp.Variable())]}, solver="CLARABEL")
    assert result.status == (cp.UNBOUNDED if feasible else cp.INFEASIBLE)
    assert not result.has_solution
    assert result.values == {}


def test_saved_violation_amount_survives_changed_parameter() -> None:
    x = cp.Variable()
    target = cp.Parameter(value=2)
    intentions = Intentions({math.inf: [x == 0], 1: [x == target]})
    first = solve(intentions, solver="CLARABEL")
    target.value = 5
    second = solve(intentions, solver="CLARABEL")
    assert first.violations[0].amount == pytest.approx(2)
    assert second.violations[0].amount == pytest.approx(5)


@pytest.mark.parametrize(
    ("point", "expected"),
    [([3, -4], [7, 3, 4]), ([[1, -2], [3, -4]], [10, 4, 6])],
)
def test_cvxpy_residual_diagnostics_preserve_total_entrywise_error(
    point: list[int] | list[list[int]], expected: list[int]
) -> None:
    target = np.asarray(point)
    x = cp.Variable(target.shape)
    constraints = [x == 0, x <= 0, x >= 0]
    result = solve({math.inf: [x == target], 1: constraints}, solver="CLARABEL")
    assert result.hard_levels == 0
    assert [item.amount for item in result.violations] == pytest.approx(
        expected, abs=1e-5
    )
    assert result.problem is not None
    assert result.problem.value == pytest.approx(sum(expected), abs=1e-5)
