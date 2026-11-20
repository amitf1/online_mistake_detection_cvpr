from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()


def egoper_root() -> Path:
    default = data_root() / "EgoPER"
    return Path(os.environ.get("EGOPER_ROOT", default)).expanduser().resolve()


def output_root() -> Path:
    return Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()


def require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved
