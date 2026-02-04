from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class AppConfig:
    marker_thickness: int = 20
    patch_width: int = 200
    patch_height: int = 200
    patch_stride_x: int = 50
    patch_stride_y: int = 50
    names: List[str] = field(
        default_factory=lambda: [
            "Defect 1",
            "Defect 2",
            "Defect 3",
            "Defect 4",
            "Defect 5",
            "Eraser",
            "Edge",
            "Surface",
            "Class Correct",
            "Leather",
        ]
    )
    rating1_tolerance: int = 95


_KEY_ALIASES: Dict[str, str] = {
    # Preferred keys
    "MarkerThickness": "marker_thickness",
    "x": "patch_width",
    "y": "patch_height",
    "sx": "patch_stride_x",
    "sy": "patch_stride_y",
    "Marker1Name": "name_0",
    "Marker2Name": "name_1",
    "Marker3Name": "name_2",
    "Marker4Name": "name_3",
    "Marker5Name": "name_4",
    "EraserName": "name_5",
    "EdgeName": "name_6",
    "SurfaceName": "name_7",
    "ClassCorrectName": "name_8",
    "ClassLeatherName": "name_9",
    "Rating1Tolerance": "rating1_tolerance",
    # Legacy keys (from older config.txt)
    "velicinaMarkera": "marker_thickness",
    "nazivMarkera1": "name_0",
    "nazivMarkera2": "name_1",
    "nazivMarkera3": "name_2",
    "nazivMarkera4": "name_3",
    "nazivMarkera5": "name_4",
    "nazivGumice": "name_5",
    "nazivRuba": "name_6",
    "nazivPodloge": "name_7",
    "nazivKlaseIspravno": "name_8",
    "nazivKlaseKoza": "name_9",
}


def load_config(path: Path) -> AppConfig:
    cfg = AppConfig()
    if not path.exists():
        return cfg

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        canonical = _KEY_ALIASES.get(key)
        if not canonical:
            continue
        if canonical.startswith("name_"):
            idx = int(canonical.split("_")[1])
            if 0 <= idx < len(cfg.names):
                cfg.names[idx] = value
            continue
        if canonical in {"marker_thickness", "patch_width", "patch_height", "patch_stride_x", "patch_stride_y", "rating1_tolerance"}:
            try:
                setattr(cfg, canonical, int(float(value)))
            except ValueError:
                pass
    return cfg


def save_config(path: Path, cfg: AppConfig) -> None:
    lines = [
        f"MarkerThickness={cfg.marker_thickness}",
        f"x={cfg.patch_width}",
        f"y={cfg.patch_height}",
        f"sx={cfg.patch_stride_x}",
        f"sy={cfg.patch_stride_y}",
        f"Marker1Name={cfg.names[0]}",
        f"Marker2Name={cfg.names[1]}",
        f"Marker3Name={cfg.names[2]}",
        f"Marker4Name={cfg.names[3]}",
        f"Marker5Name={cfg.names[4]}",
        f"EraserName={cfg.names[5]}",
        f"EdgeName={cfg.names[6]}",
        f"SurfaceName={cfg.names[7]}",
        f"ClassCorrectName={cfg.names[8]}",
        f"ClassLeatherName={cfg.names[9]}",
        f"Rating1Tolerance={cfg.rating1_tolerance}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
