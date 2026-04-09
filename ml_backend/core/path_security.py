"""Path validation and sanitization helpers for the ML backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List

_ALLOWED_INPUT_ENV = "DEFECTDETECT_ALLOWED_PATHS"
_ALLOWED_OUTPUT_ENV = "DEFECTDETECT_ALLOWED_OUTPUT_PATHS"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT_ROOTS = (_REPO_ROOT, Path.cwd(), Path(tempfile.gettempdir()))
_DEFAULT_OUTPUT_ROOTS = (
    _REPO_ROOT / "output",
    Path.cwd() / "output",
    Path(tempfile.gettempdir()),
)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _parse_allowed_roots(env_name: str, default_roots: Iterable[Path]) -> List[Path]:
    configured = os.getenv(env_name, "")
    if not configured.strip():
        return _unique_paths(default_roots)

    roots = [Path(part.strip()) for part in configured.split(os.pathsep) if part.strip()]
    return _unique_paths(roots)


def _resolve_path(path_value: str | os.PathLike[str]) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (Path.cwd() / raw_path).resolve()


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_within_allowed_roots(path: Path, allowed_roots: Iterable[Path], *, path_kind: str) -> Path:
    allowed = list(allowed_roots)
    if any(_is_within_root(path, root) for root in allowed):
        return path

    allowed_display = ", ".join(str(root) for root in allowed)
    raise PermissionError(f"{path_kind} path is outside allowed roots: {allowed_display}")


def get_allowed_input_roots() -> List[Path]:
    """Return allowed roots for reading image files."""
    return _parse_allowed_roots(_ALLOWED_INPUT_ENV, _DEFAULT_INPUT_ROOTS)


def get_allowed_output_roots() -> List[Path]:
    """Return allowed roots for writing generated output."""
    return _parse_allowed_roots(_ALLOWED_OUTPUT_ENV, _DEFAULT_OUTPUT_ROOTS)


def normalize_request_image_path(path_value: str) -> str:
    """Normalize and validate request image paths without requiring existence."""
    if not path_value or not path_value.strip():
        raise ValueError("image_path cannot be empty")
    resolved = _resolve_path(path_value)
    _ensure_within_allowed_roots(resolved, get_allowed_input_roots(), path_kind="Input")
    return str(resolved)


def resolve_existing_input_path(path_value: str | os.PathLike[str]) -> Path:
    """Resolve an input file path and ensure it exists under allowed roots."""
    resolved = _resolve_path(path_value)
    _ensure_within_allowed_roots(resolved, get_allowed_input_roots(), path_kind="Input")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved


def resolve_allowed_output_dir(path_value: str | os.PathLike[str]) -> Path:
    """Resolve an output directory and ensure it is under allowed roots."""
    if not str(path_value).strip():
        raise ValueError("output_dir cannot be empty")
    resolved = _resolve_path(path_value)
    return _ensure_within_allowed_roots(resolved, get_allowed_output_roots(), path_kind="Output")


def safe_display_path(path_value: str | os.PathLike[str]) -> str:
    """Return a sanitized path string suitable for logs."""
    try:
        resolved = _resolve_path(path_value)
    except Exception:
        return Path(path_value).name if str(path_value) else "<unknown>"

    for root in _unique_paths([*get_allowed_input_roots(), *get_allowed_output_roots(), Path.cwd()]):
        try:
            relative = resolved.relative_to(root)
            return str(relative) if str(relative) != "." else resolved.name
        except ValueError:
            continue

    return resolved.name
