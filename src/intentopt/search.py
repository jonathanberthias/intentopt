"""Search a nested sequence of constraint tiers without depending on CVXPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


def _validate_count(
    value: object, maximum: int | None = None, *, name: str = "Hard-tier count"
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        msg = (
            f"{name} must be a non-negative integer"
            if maximum is None
            else f"{name} must be an integer between 0 and {maximum}"
        )
        raise ValueError(msg)


class SearchPolicy(Protocol):
    """Locate the largest feasible prefix, or None if no prefix is feasible.

    ``feasible(k)`` tests k finite constraint tiers, plus all required constraints.
    The solver caches probes, so repeated calls do not repeat optimization.
    """

    def search(
        self, max_hard_levels: int, feasible: Callable[[int], bool]
    ) -> int | None:
        """Search counts from zero through max_hard_levels, inclusive."""
        ...


@dataclass(frozen=True)
class SequentialSearch:
    """Add one constraint tier at a time until the first incompatibility."""

    def search(
        self, max_hard_levels: int, feasible: Callable[[int], bool]
    ) -> int | None:
        """Try successive prefixes until the first infeasible one."""
        _validate_count(max_hard_levels, name="max_hard_levels")
        best = None
        for count in range(max_hard_levels + 1):
            if not feasible(count):
                break
            best = count
        return best


@dataclass(frozen=True)
class BinarySearch:
    """Bisect the feasible prefix, optionally probing an expected cutoff first.

    ``start`` is a number of finite constraint tiers, not a priority or weight.
    An incorrect guess only changes probe order, not the selected cutoff.
    """

    start: int | None = None

    def search(
        self, max_hard_levels: int, feasible: Callable[[int], bool]
    ) -> int | None:
        """Locate the last feasible prefix by bisecting the search interval."""
        _validate_count(max_hard_levels, name="max_hard_levels")
        low, high = 0, max_hard_levels
        best = None
        if self.start is not None:
            _validate_count(self.start, max_hard_levels, name="start")
            if feasible(self.start):
                best, low = self.start, self.start + 1
            else:
                high = self.start - 1
        while low <= high:
            middle = (low + high) // 2
            if feasible(middle):
                best, low = middle, middle + 1
            else:
                high = middle - 1
        return best
