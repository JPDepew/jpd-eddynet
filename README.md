# EddyNet

## Run

This project uses Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python eddynet/train.py
```

Add runtime dependencies with `uv add <package>`. Add development-only tools with
`uv add --dev <package>`. Both commands update `pyproject.toml` and `uv.lock`.

## Code quality

Ruff checks and formats Python code. In Cursor or VS Code, install the recommended
Ruff extension to format, apply fixes, and sort imports when you save a Python file.

```sh
uv run ruff check .
uv run ruff format .
```
