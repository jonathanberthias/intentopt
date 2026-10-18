# Contributing

Use Python 3.11 or later. Install the development dependencies and run the
tests:

```bash
uv sync --group dev
uv run pytest
```

The behavior contract and design decisions live in
[the user guide](../docs/intentopt.md). Production code belongs in
`src/intentopt/`; tests belong in `tests/`. Include coverage for changed
behavior and update examples or documentation when the API changes.

Run the formatting, lint, typing, and configuration checks with:

```bash
uvx prek run --all-files
```

Run the type checkers directly with:

```bash
uv run ty check
uv run pyrefly check
```

Both check the library, tests, and examples with all diagnostic rules selected
and documented exceptions. Keep any exceptions specific and explain upstream
typing limitations.

CI tests Python 3.11 and 3.14 on the configured platforms. See
[the test guide](../tests/README.md) for running against the minimum supported
CVXPY version.

Build or serve documentation with:

```bash
uv sync --group docs
uv run zensical build --clean --strict
uv run zensical serve
```

Build the source distribution and wheel with `uv build`.

Pull request descriptions should state the affected scenario and explain the
resulting behavior. Publishing is not configured yet.
