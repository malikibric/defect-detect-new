"""Shared application state for ML model registry.

This module provides a single source of truth for loaded models and runtime
configuration used by services and the FastAPI application.
"""

from typing import Any, Dict


model_registry: Dict[str, Any] = {}
