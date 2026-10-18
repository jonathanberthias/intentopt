# Tests

Run `uv run pytest` from the repository root.

Test CVXPY 1.4.0 on Python 3.11 in an isolated environment:

```bash
uv run --isolated --python 3.11 \
  --with-requirements tests/requirements-minimum.txt \
  pytest -W 'ignore:invalid escape sequence:DeprecationWarning'
```

Its historical NumPy/SciPy requirements are in `requirements-minimum.txt`; these
constraints do not apply to current CVXPY versions. This command ignores only
invalid-escape deprecations from the old CVXPY source docstrings. The production
code and tests are the same as for current dependencies.

- `test_intentions.py`: input validation, grouping, and fixed priority weights.
- `test_solve.py`: cutoff semantics, compromise losses, statuses, and snapshots.
- `test_search.py`: every cutoff across sequential, binary, and guessed
  searches.
- `test_integrality.py`: required and softened integer and multiple-of
  intentions, round lots, and penalties in the original units.

Continuous tests use the installed CVXPY solvers. Mixed-integer tests explicitly
select SciPy's MILP solver.
