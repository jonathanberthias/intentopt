import pytest

from intentopt import BinarySearch, SearchPolicy, SequentialSearch


@pytest.mark.parametrize("search", [SequentialSearch(), BinarySearch()])
@pytest.mark.parametrize("size", [-1, 1.5, True])
def test_searches_reject_invalid_sizes(search: SearchPolicy, size: float) -> None:
    with pytest.raises(ValueError, match="max_hard_levels"):
        # Non-integer sizes deliberately exercise runtime validation.
        # pyrefly: ignore[bad-argument-type]
        search.search(size, lambda _count: True)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("size", [0, 1, 2, 9, 32])
def test_searches_find_every_possible_frontier(size: int) -> None:
    for frontier in range(-1, size + 1):

        def feasible(count: int, cutoff: int = frontier) -> bool:
            return count <= cutoff

        policies: list[SearchPolicy] = [SequentialSearch(), BinarySearch()]
        policies.extend(BinarySearch(start=start) for start in range(size + 1))
        for policy in policies:
            assert policy.search(size, feasible) == (
                frontier if frontier >= 0 else None
            )


def test_binary_search_takes_logarithmically_many_probes() -> None:
    probes = []

    def feasible(count: int) -> bool:
        probes.append(count)
        return count <= 700

    assert BinarySearch().search(1024, feasible) == 700
    assert len(probes) <= 11


def test_start_guess_is_probed_first() -> None:
    probes = []

    def feasible(count: int) -> bool:
        probes.append(count)
        return count <= 6

    assert BinarySearch(start=8).search(10, feasible) == 6
    assert probes[0] == 8


@pytest.mark.parametrize("start", [-1, 4, 1.5, True])
def test_invalid_start_is_rejected(start: float) -> None:
    with pytest.raises(ValueError, match="start"):
        # Deliberately pass non-integer guesses to exercise runtime validation.
        # pyrefly: ignore[bad-argument-type]
        BinarySearch(start=start).search(3, lambda _count: True)  # ty: ignore[invalid-argument-type]
