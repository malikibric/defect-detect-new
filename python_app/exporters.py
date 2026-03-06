from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import cv2


@dataclass
class PatchMeta:
    file_name: str
    patch_id: str
    height: int
    width: int
    x: int
    y: int


def save_patches(root: Path, image_name: str, patches: List[List], names: List[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for idx, patch_group in enumerate(patches):
        if not patch_group:
            continue
        group_name = names[idx]
        group_dir = root / f"Patches_{image_name}_{group_name}"
        group_dir.mkdir(parents=True, exist_ok=True)
        for j, patch in enumerate(patch_group, start=1):
            filename = f"{image_name}_Patch{j:04d}_{group_name}.bmp"
            cv2.imwrite(str(group_dir / filename), patch)


def export_json_collective(
    root: Path,
    image_name: str,
    patch_width: int,
    patch_height: int,
    coordinates: List[List[int]],
    class_names: List[str],
    names: List[str],
    ratings: List[List[int]],
    include_masks: bool,
) -> None:
    payload = {}

    org_patches = []
    for j, coord in enumerate(coordinates, start=1):
        org_patches.append(
            {
                "file_name": f"{image_name}_Patch{j:04d}_{names[0]}",
                "id": f"{j:04d}",
                "height": patch_height,
                "width": patch_width,
                "x_koor": coord[0],
                "y_koor": coord[1],
            }
        )
    payload["org_patches"] = org_patches

    payload["classes"] = [
        {"class_id": idx, "name": class_names[idx]} for idx in range(len(class_names))
    ]

    annotations = []
    for i in range(len(coordinates)):
        annotation = {
            "id": f"Ann-{i + 1}",
            "patch_id": f"{i + 1:04d}",
            "class_ratings_ids": [ratings[j][i] for j in range(9)],
        }
        if include_masks:
            annotation["mask_ids"] = [f"{j} - mask {i + 1:04d}" for j in range(1, 10)]
        annotations.append(annotation)
    payload["annotation"] = annotations

    if include_masks:
        masks = []
        for j in range(1, len(names)):
            for i in range(len(coordinates)):
                masks.append(
                    {
                        "file_name": f"{image_name}_Patch{i + 1:04d}_{names[j]}",
                        "id": f"{j} - mask {i + 1:04d}",
                        "patch_id": f"{i + 1:04d}",
                        "rating_id": str(ratings[j - 1][i]),
                    }
                )
        payload["masks"] = masks

    payload["ratings"] = [
        {"rating_id": 0, "name": "No presence"},
        {"ocjena_id": 1, "name": "Partial presence"},
        {"ocjena_id": 2, "name": "Full presence"},
    ]

    output_path = root / "json_output.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_json_individual(
    root: Path,
    image_name: str,
    patch_width: int,
    patch_height: int,
    coordinates: List[List[int]],
    class_names: List[str],
    names: List[str],
    ratings: List[List[int]],
    include_masks: bool,
) -> None:
    json_dir = root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    for i, coord in enumerate(coordinates, start=1):
        payload = {
            "org_patches": [
                {
                    "file_name": f"{image_name}_Patch{i:04d}_{names[0]}",
                    "id": f"{i:04d}",
                    "height": patch_height,
                    "width": patch_width,
                    "x_koor": coord[0],
                    "y_koor": coord[1],
                }
            ],
            "classes": [
                {"class_id": idx, "name": class_names[idx]} for idx in range(len(class_names))
            ],
            "annotation": [
                {
                    "id": f"Ann-{i}",
                    "patch_id": f"{i:04d}",
                    "class_ratings_ids": [ratings[j][i - 1] for j in range(9)],
                    **(
                        {
                            "mask_ids": [
                                f"{j} - mask {i:04d}" for j in range(1, 10)
                            ]
                        }
                        if include_masks
                        else {}
                    ),
                }
            ],
        }

        if include_masks:
            masks = []
            for j in range(1, len(names)):
                masks.append(
                    {
                        "file_name": f"{image_name}_Patch{i:04d}_Mask {names[j]}",
                        "id": f"{j} - mask {i:04d}",
                        "patch_id": f"{i:04d}",
                        "rating_id": str(ratings[j - 1][i - 1]),
                    }
                )
            payload["masks"] = masks

        payload["ratings"] = [
            {"ocjena_id": 0, "name": "No presence"},
            {"ocjena_id": 1, "name": "Partial presence"},
            {"ocjena_id": 2, "name": "Full presence"},
        ]

        output_path = json_dir / f"output_{i:04d}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_yolo(
    root: Path,
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
    coordinates: List[List[int]],
    ratings: List[List[int]],
) -> None:
    output = root / "yolo_output.txt"
    lines = []
    for i, coord in enumerate(coordinates):
        x_center = (coord[0] + patch_width / 2) / width
        y_center = (coord[1] + patch_height / 2) / height
        normalized_width = patch_width / width
        normalized_height = patch_height / height

        class_id = 7
        oi = ratings[7][i]
        if oi != 2:
            if ratings[0][i] != 0:
                class_id = 0
            elif ratings[1][i] != 0:
                class_id = 1
            elif ratings[2][i] != 0:
                class_id = 2
            elif ratings[3][i] != 0:
                class_id = 3
            elif ratings[4][i] != 0:
                class_id = 4
            elif ratings[5][i] != 0:
                class_id = 5
            elif ratings[6][i] != 0:
                class_id = 6

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}"
        )
    with output.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_pascal_voc(
    root: Path,
    image_name: str,
    image_path: str,
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
    coordinates: List[List[int]],
    class_names: List[str],
    ratings: List[List[int]],
) -> None:
    output = root / "Pascal_output.xml"
    lines = []
    lines.append("<annotation>")
    lines.append(f"    <filename>{image_name}</filename>")
    lines.append(f"    <path>{image_path}</path>")
    lines.append("    <size>")
    lines.append(f"        <width>{width}</width>")
    lines.append(f"        <height>{height}</height>")
    lines.append("    </size>")

    for i, coord in enumerate(coordinates, start=1):
        patch_name = f" Patch{i:04d}"
        object_name = f"{class_names[7]}{patch_name}"
        oi = ratings[7][i - 1]
        if oi != 2:
            if ratings[0][i - 1] != 0:
                object_name = class_names[0]
            elif ratings[1][i - 1] != 0:
                object_name = class_names[1] + patch_name
            elif ratings[2][i - 1] != 0:
                object_name = class_names[2] + patch_name
            elif ratings[3][i - 1] != 0:
                object_name = class_names[3] + patch_name
            elif ratings[4][i - 1] != 0:
                object_name = class_names[4] + patch_name
            elif ratings[5][i - 1] != 0:
                object_name = class_names[5] + patch_name
            elif ratings[6][i - 1] != 0:
                object_name = class_names[6] + patch_name

        xmin, ymin = coord[0], coord[1]
        xmax, ymax = coord[0] + patch_width, coord[1] + patch_height

        lines.append("    <object>")
        lines.append(f"        <name>{object_name}</name>")
        lines.append("        <bndbox>")
        lines.append(f"            <xmin>{xmin}</xmin>")
        lines.append(f"            <ymin>{ymin}</ymin>")
        lines.append(f"            <xmax>{xmax}</xmax>")
        lines.append(f"            <ymax>{ymax}</ymax>")
        lines.append("        </bndbox>")
        lines.append("    </object>")

    lines.append("</annotation>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
