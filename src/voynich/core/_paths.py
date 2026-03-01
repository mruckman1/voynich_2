"""Centralized path resolution for data and results directories."""
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where pyproject.toml lives)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def data_dir(subdir: str = "") -> Path:
    """Return path to data directory (or a subdirectory of it)."""
    base = PROJECT_ROOT / "data"
    return base / subdir if subdir else base


def results_dir() -> Path:
    """Return path to results directory, creating it if needed."""
    d = PROJECT_ROOT / "results"
    d.mkdir(exist_ok=True)
    return d
