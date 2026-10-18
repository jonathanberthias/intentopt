# Examples

Run an example from the repository root with `uv run python examples/<name>.py`.

- `example_priorities.py`: required constraints and conflicting preferences.
- `example_portfolio.py`: allocations, returns, round lots, and turnover.
- `example_integrality.py`: purchases in round lots.

The portfolio and round-lot examples use SciPy's mixed-integer solver. See the
[user guide](../docs/intentopt.md) for the compromise rules and the
[advanced guide](../docs/advanced.md) for search options.
