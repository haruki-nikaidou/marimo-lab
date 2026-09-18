# marimo-lab

Reactive research workspace built on [marimo](https://docs.marimo.io/), with
dependencies managed by [uv](https://docs.astral.sh/uv/) and the toolchain
pinned by a Nix flake.

## Quickstart

```bash
nix develop            # or: direnv allow  (uses .envrc -> `use flake`)
make edit              # opens notebooks/ in the marimo editor
```

`nix develop` provides `uv` plus CPython 3.13 from the flake lock and runs
`uv sync`, so `.venv` matches `uv.lock` on shell entry. Set
`MARIMO_LAB_NO_SYNC=1` to skip that sync.

Without Nix, CPython 3.13 and `uv` are enough: `uv sync && uv run marimo edit notebooks`.
The project is pinned to `requires-python = ">=3.13,<3.14"`, so `uv` refuses any other
interpreter rather than resolving a set of wheels the lock file was not built for.

## Layout

```
flake.nix / flake.lock   pinned dev shell (uv, python, make)
pyproject.toml           project deps + ruff/pytest/marimo settings
uv.lock                  exact resolved dependency versions
notebooks/               marimo notebooks (plain .py, git-diffable)
src/marimo_lab/          reusable code imported by notebooks
tests/                   pytest tests for src/
data/raw, data/processed gitignored data (scaffolding tracked)
exports/                 generated HTML exports (gitignored)
```

## Tasks

| Command | Purpose |
| --- | --- |
| `make edit` | Open the notebook workspace in the browser |
| `make new N=name` | Create/open `notebooks/name.py` |
| `make run N=00_welcome` | Serve a notebook as a read-only app |
| `make script N=00_welcome` | Execute a notebook headlessly as a script |
| `make export` | Export all notebooks to `exports/*.html` |
| `make check` | `marimo check` + ruff lint/format check + pytest |
| `make fmt` | Autofix lint and format |
| `make sync` / `make lock` | Install from lock / re-resolve dependencies |

## Working notes

- **Dependencies:** `uv add <pkg>` (and `uv add --group dev <pkg>` for tooling).
  The editor's package UI is wired to uv via `[tool.marimo.package_management]`,
  so packages added in the browser land in `pyproject.toml` too.
- **Paths:** import `data_path` from `marimo_lab` instead of relative strings —
  marimo runs a notebook with the notebook's directory as the CWD.
- **Notebooks are Python.** They are lint-checked and formatted with ruff;
  `B018` is ignored under `notebooks/` because a cell renders its last
  expression. Avoid implicit string concatenation inside `mo.md(...)`: the
  editor's format-on-save rewrites markdown cells and mangles it.
- **Self-contained notebooks:** for a notebook with dependencies pinned inline
  (PEP 723) rather than in this project, use `uv run marimo edit --sandbox path.py`.
- **NixOS note:** `UV_PYTHON_DOWNLOADS=never` keeps uv on the Nix interpreter,
  and `LD_LIBRARY_PATH` in the dev shell supplies `libstdc++`/`libz` that
  manylinux wheels (duckdb, pyarrow, scipy) dlopen at import time.
