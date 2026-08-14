#!/usr/bin/env python3
"""Check owned and foreign dimension text in generated view crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def local_box(box: list[float], crop_box: list[float], size: tuple[int, int]) -> tuple[int, ...]:
    left = math.floor(box[0] * 2) - math.floor(crop_box[0] * 2)
    top = math.floor(box[1] * 2) - math.floor(crop_box[1] * 2)
    right = math.ceil(box[2] * 2) - math.floor(crop_box[0] * 2)
    bottom = math.ceil(box[3] * 2) - math.floor(crop_box[1] * 2)
    return max(0, left), max(0, top), min(size[0], right), min(size[1], bottom)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    failures: list[list[str]] = []
    for drawing_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        result_path = drawing_dir / "view-regions.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        owned_visible = foreign_clear = 0
        for page in result["pages"]:
            views = page["views"]
            owners = {dimension_id: view for view in views for dimension_id in view["dimension_ids"]}
            boxes = {
                dimension_id: box
                for view in views
                for dimension_id, box in zip(
                    view["dimension_ids"], view["dimension_text_boxes"]
                )
            }
            for view in views:
                pixels = np.asarray(
                    Image.open(drawing_dir / "crops" / f"{view['id']}.png").convert("RGBA")
                )
                alpha = pixels[:, :, 3]
                for dimension_id, box in zip(
                    view["dimension_ids"], view["dimension_text_boxes"]
                ):
                    x0, y0, x1, y1 = local_box(box, view["bbox"], (pixels.shape[1], pixels.shape[0]))
                    visible = (
                        x1 > x0
                        and y1 > y0
                        and bool(np.any((alpha[y0:y1, x0:x1] > 0) & (np.min(pixels[y0:y1, x0:x1, :3], axis=2) < 245)))
                    )
                    owned_visible += int(visible)
                    if not visible:
                        failures.append([drawing_dir.name, view["id"], "owned_missing", dimension_id])
                for dimension_id, owner in owners.items():
                    if owner is view or intersection_area(
                        view["provisional_bbox"], owner["provisional_bbox"]
                    ) <= 0:
                        continue
                    x0, y0, x1, y1 = local_box(
                        boxes[dimension_id], view["bbox"], (pixels.shape[1], pixels.shape[0])
                    )
                    clear = x1 <= x0 or y1 <= y0 or not np.any(alpha[y0:y1, x0:x1])
                    foreign_clear += int(clear)
                    if not clear:
                        failures.append([drawing_dir.name, view["id"], "foreign_visible", dimension_id])
        summary = result["summary"]
        rows.append(
            {
                "drawing": drawing_dir.name,
                "views": summary["views"],
                "rectangular": summary["views"] - summary["fitted_views"],
                "fitted": summary["fitted_views"],
                "excluded_3d": summary["excluded_embedded_3d_models"],
                "foreign_dimension_collisions": summary[
                    "foreign_dimension_text_collisions"
                ],
                "owned_text_boxes_visible": owned_visible,
                "foreign_text_boxes_clear": foreign_clear,
            }
        )
    payload = {"rows": rows, "failures": failures}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
