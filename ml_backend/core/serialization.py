"""Serialization helpers shared by ML backend routers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from models.schemas import Annotation, PatchMetadata


def annotation_to_dict(annotation: Annotation) -> Dict[str, Any]:
    """Convert an annotation model to a plain dictionary."""
    return annotation.model_dump(exclude_none=True)


def annotations_to_dicts(annotations: Iterable[Annotation]) -> List[Dict[str, Any]]:
    """Convert annotation models to dictionaries."""
    return [annotation_to_dict(annotation) for annotation in annotations]


def annotation_from_dict(payload: Dict[str, Any]) -> Annotation:
    """Convert a service payload to an annotation model."""
    return Annotation(**payload)


def annotations_from_dicts(payloads: Iterable[Dict[str, Any]]) -> List[Annotation]:
    """Convert service annotation payloads to models."""
    return [annotation_from_dict(payload) for payload in payloads]


def patch_to_dict(patch: PatchMetadata) -> Dict[str, Any]:
    """Convert patch metadata model to a plain dictionary."""
    return patch.model_dump()


def patches_to_dicts(patches: Iterable[PatchMetadata]) -> List[Dict[str, Any]]:
    """Convert patch metadata models to dictionaries."""
    return [patch_to_dict(patch) for patch in patches]


def patch_from_dict(payload: Dict[str, Any]) -> PatchMetadata:
    """Convert a service payload to patch metadata."""
    return PatchMetadata(**payload)


def patches_from_dicts(payloads: Iterable[Dict[str, Any]]) -> List[PatchMetadata]:
    """Convert service patch payloads to models."""
    return [patch_from_dict(payload) for payload in payloads]
