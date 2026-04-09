"""Shared application state for ML model registry.

This module provides a typed single source of truth for loaded models and
runtime configuration used by services and the FastAPI application.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelRegistry(MutableMapping[str, Any]):
	"""Typed runtime registry with dict-like access for existing callers."""

	device: str = "cpu"
	sam_model: Any = None
	sam_predictor: Any = None
	sam_mask_generator: Any = None
	yolo_model: Any = None
	clip_model: Any = None
	clip_processor: Any = None
	diffusion_pipeline: Any = None
	startup_status: Dict[str, str] = field(default_factory=dict)

	_KNOWN_KEYS = {
		"device",
		"sam_model",
		"sam_predictor",
		"sam_mask_generator",
		"yolo_model",
		"clip_model",
		"clip_processor",
		"diffusion_pipeline",
		"startup_status",
	}

	def __getitem__(self, key: str) -> Any:
		if key not in self._KNOWN_KEYS:
			raise KeyError(f"Unsupported model_registry key: {key}")
		return getattr(self, key)

	def __setitem__(self, key: str, value: Any) -> None:
		if key not in self._KNOWN_KEYS:
			raise KeyError(f"Unsupported model_registry key: {key}")
		setattr(self, key, value)

	def __delitem__(self, key: str) -> None:
		if key not in self._KNOWN_KEYS:
			raise KeyError(f"Unsupported model_registry key: {key}")
		if key == "device":
			self.device = "cpu"
		elif key == "startup_status":
			self.startup_status = {}
		else:
			setattr(self, key, None)

	def __iter__(self) -> Iterator[str]:
		return iter(self._KNOWN_KEYS)

	def __len__(self) -> int:
		return len(self._KNOWN_KEYS)

	def clear(self) -> None:
		"""Reset all registry entries to their default values."""
		self.device = "cpu"
		self.sam_model = None
		self.sam_predictor = None
		self.sam_mask_generator = None
		self.yolo_model = None
		self.clip_model = None
		self.clip_processor = None
		self.diffusion_pipeline = None
		self.startup_status = {}


model_registry = ModelRegistry()
