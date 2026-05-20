"""Unit tests for path security helper functions."""

import os
import tempfile
from pathlib import Path
import pytest

from core import path_security


def test_unique_paths():
    p1 = Path("/tmp/../tmp")
    p2 = Path("/tmp")
    unique = path_security._unique_paths([p1, p2])
    assert len(unique) == 1


def test_is_within_root():
    root = Path("/tmp")
    path = Path("/tmp/subdir/file.txt")
    assert path_security._is_within_root(path, root) is True

    path_outside = Path("/usr/bin/python")
    assert path_security._is_within_root(path_outside, root) is False


def test_resolve_allowed_output_dir():
    temp_dir = Path(tempfile.gettempdir()).resolve()
    # Should resolve correctly and not raise
    resolved = path_security.resolve_allowed_output_dir(temp_dir)
    assert resolved == temp_dir

    # Passing empty path should raise ValueError
    with pytest.raises(ValueError):
        path_security.resolve_allowed_output_dir("")


def test_safe_display_path():
    temp_dir = Path(tempfile.gettempdir()).resolve()
    test_path = temp_dir / "some_file.png"
    display = path_security.safe_display_path(test_path)
    assert display == "some_file.png"
