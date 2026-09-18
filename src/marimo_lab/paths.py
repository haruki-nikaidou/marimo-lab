"""Project path resolution that does not depend on the current directory.

marimo runs notebooks with the notebook's directory as the working directory,
so notebooks must not build data paths from relative strings.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DATA_DIR", "PROCESSED_DIR", "PROJECT_ROOT", "RAW_DIR", "data_path"]


def _find_project_root() -> Path:
    """Nearest ancestor holding ``pyproject.toml``, from the package or the CWD."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for directory in (start, *start.parents):
            if (directory / "pyproject.toml").is_file():
                return directory
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def data_path(*parts: str | Path) -> Path:
    """Resolve ``parts`` under ``data/``, rejecting anything that escapes it."""
    root = DATA_DIR.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(f"path escapes the data directory: {candidate}")
    return candidate
