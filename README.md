# EddyNet

## Run

This project uses Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python eddynet/train.py
```

Add runtime dependencies with `uv add <package>`. Add development-only tools with
`uv add --dev <package>`. Both commands update `pyproject.toml` and `uv.lock`.

## Simulation data

Run the simulation with:

```sh
uv run python eddynet/simulation.py
```

Set the sample count and worker count in the script's `generate_test_data` call:

```python
generate_test_data(size=20_000, workers=4)
```

The default uses at most four worker processes. Each worker computes all
frequencies for one crack and saves its files in `data/`. Workers share read-only
grid caches through memory-mapped files, so each worker does not need a full copy
of those arrays. These temporary cache files are removed when the run exits.
Use `workers=1` to process one crack at a time; more workers require more memory
for the per-crack matrix solves.

## Code quality

Ruff checks and formats Python code. In Cursor or VS Code, install the recommended
Ruff extension to format, apply fixes, and sort imports when you save a Python file.

```sh
uv run ruff check .
uv run ruff format .
```
