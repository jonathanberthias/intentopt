import math

import cvxpy as cp
import pytest

from intentopt import Intentions


def test_priorities_have_fixed_weights_including_p11() -> None:
    x = cp.Variable()
    intentions = Intentions.from_priorities({0: [x >= 0], 1: [x >= 10], 11: [x <= 5]})
    assert list(intentions.weight_to_intentions) == [math.inf, 512, 0.5]
    assert list(Intentions.from_priorities({11: [x >= 0]}).weight_to_intentions) == [
        0.5
    ]


def test_pair_input_groups_equal_weights_and_preserves_raw_objects() -> None:
    x = cp.Variable()
    lower, upper = x >= 0, x <= 1
    intentions = Intentions([(10, lower), (10, upper)])
    assert intentions.weight_to_intentions[10][0] is lower
    assert intentions.weight_to_intentions[10][1] is upper
    with pytest.raises(TypeError):
        # Deliberately exercise immutable mapping behavior at runtime.
        # pyrefly: ignore[unsupported-operation]
        intentions.weight_to_intentions[1] = ()  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("weight", [0, -1, -math.inf, math.nan])
def test_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="Weights"):
        Intentions({weight: [cp.Variable() >= 0]})


def test_rejects_weights_too_large_for_floating_point() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        Intentions({10**1000: [cp.Variable() >= 0]})


def test_smallest_representable_priority_weight_is_preserved() -> None:
    intentions = Intentions.from_priorities({1084: [cp.Variable() >= 0]})
    assert list(intentions.weight_to_intentions) == [math.ulp(0.0)]


@pytest.mark.parametrize("priority", [1085, 10**1000])
def test_rejects_priorities_without_a_representable_weight(priority: int) -> None:
    with pytest.raises(ValueError, match="positive weight"):
        Intentions.from_priorities({priority: [cp.Variable() >= 0]})


@pytest.mark.parametrize("priority", [-1, 1.5, True, 2000])
def test_rejects_invalid_or_unrepresentable_priorities(priority: float) -> None:
    with pytest.raises((ValueError, TypeError), match=r"[Pp]riorit"):
        # Include non-integer input to exercise runtime validation.
        # pyrefly: ignore[bad-assignment]
        Intentions.from_priorities({priority: [cp.Variable() >= 0]})  # ty: ignore[invalid-argument-type]


def test_rejects_required_objectives() -> None:
    with pytest.raises(ValueError, match="Required"):
        Intentions.from_priorities({0: [cp.Minimize(cp.Variable())]})


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        Intentions({})


def test_unsupported_soft_cones_fail_explicitly_but_can_be_required() -> None:
    x = cp.Variable(2)
    cone = cp.SOC(1, x)
    with pytest.raises(TypeError, match="Cannot soften SOC"):
        Intentions({1: [cone]})
    assert len(Intentions({math.inf: [cone]}).weight_to_intentions) == 1
