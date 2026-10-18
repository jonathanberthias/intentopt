"""Weighted intentions and the priority constructor."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import cvxpy as cp

from .integrality import MultipleOf
from .intention import RawIntention, penalty

# Priority 1084 has weight 2**-1074, the smallest positive Python float.
_MAX_PRIORITY = 1084

WeightedInput: TypeAlias = (
    Mapping[int, Collection[RawIntention]]
    | Mapping[float, Collection[RawIntention]]
    | Collection[tuple[float, RawIntention]]
)
PriorityInput: TypeAlias = (
    Mapping[int, Collection[RawIntention]] | Collection[tuple[int, RawIntention]]
)


@dataclass(frozen=True, init=False)
class Intentions:
    """Group raw CVXPY constraints and objectives by positive weight.

    Larger weights are enforced first and multiply soft losses. Infinite weight
    means required and may only contain constraints. Equal weights form one tier.
    """

    # Frozen dataclass initialization uses object.__setattr__ below.
    weight_to_intentions: Mapping[float, tuple[RawIntention, ...]]  # pyrefly: ignore[uninitialized-instance-variable]

    def __init__(self, intentions: WeightedInput) -> None:
        grouped: dict[float, list[RawIntention]] = defaultdict(list)
        pairs = (
            ((weight, item) for weight, items in intentions.items() for item in items)
            if isinstance(intentions, Mapping)
            else iter(intentions)
        )
        for raw_weight, item in pairs:
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                msg = "Weights must be positive numbers"
                raise TypeError(msg)
            try:
                weight = float(raw_weight)
            except OverflowError:
                msg = "Weights must be representable as floating-point numbers"
                raise ValueError(msg) from None
            if math.isnan(weight) or weight <= 0:
                msg = f"Weights must be positive and not NaN, got {weight}"
                raise ValueError(msg)
            if not isinstance(
                item, (cp.Constraint, cp.Minimize, cp.Maximize, MultipleOf)
            ):
                msg = "Intentions must be CVXPY constraints, objectives, or MultipleOf"
                raise TypeError(msg)
            if math.isinf(weight):
                if not isinstance(item, (cp.Constraint, MultipleOf)):
                    msg = "Required tiers may only contain constraints"
                    raise ValueError(msg)
            else:
                _ = penalty(item)  # Reject unsupported soft constraints early.
            grouped[weight].append(item)
        if len(grouped) == 0:
            msg = "At least one intention is required"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "weight_to_intentions",
            MappingProxyType(
                {
                    weight: tuple(grouped[weight])
                    for weight in sorted(grouped, reverse=True)
                }
            ),
        )

    @classmethod
    def from_priorities(cls, intentions: PriorityInput) -> Intentions:
        """Map priority 0 to infinity and positive p to ``2 ** (10 - p)``.

        Priorities 11 and higher are valid. Reject priorities whose weights cannot be
        represented as positive floating-point numbers.
        """
        pairs = (
            (
                (priority, item)
                for priority, items in intentions.items()
                for item in items
            )
            if isinstance(intentions, Mapping)
            else iter(intentions)
        )
        weighted: list[tuple[float, RawIntention]] = []
        for priority, item in pairs:
            if isinstance(priority, bool) or not isinstance(priority, int):
                msg = "Priorities must be non-negative integers"
                raise TypeError(msg)
            if priority < 0:
                msg = "Priorities must be non-negative integers"
                raise ValueError(msg)
            if priority > _MAX_PRIORITY:
                msg = "Priority is too large to represent a positive weight"
                raise ValueError(msg)
            weight = math.inf if priority == 0 else 2.0 ** (10 - priority)
            weighted.append((weight, item))
        return cls(weighted)

    @property
    def _constraint_weights(self) -> tuple[float, ...]:
        return tuple(
            weight
            for weight, items in self.weight_to_intentions.items()
            if math.isfinite(weight)
            and any(isinstance(item, (cp.Constraint, MultipleOf)) for item in items)
        )

    def _problem(self, hard_levels: int) -> cp.Problem:
        hard_weights = set(self._constraint_weights[:hard_levels])
        constraints: list[cp.Constraint] = []
        objective: cp.Expression = cp.Constant(0.0)
        for weight, items in self.weight_to_intentions.items():
            for item in items:
                if isinstance(item, (cp.Constraint, MultipleOf)) and (
                    math.isinf(weight) or weight in hard_weights
                ):
                    constraints.append(
                        item._constraint() if isinstance(item, MultipleOf) else item
                    )
                else:
                    # CVXPY's operator decorators erase the annotated return type.
                    objective = cast(
                        "cp.Expression", objective + weight * penalty(item)
                    )
        return cp.Problem(cp.Minimize(objective), constraints)
