#!/usr/bin/env python3
"""Detect drawing-view regions in a vector engineering PDF.

Dimension annotations from ``extract_dimension_ledger.py`` act as view seeds.
Rendered line art supplies the core regions, including views with no dimensions.
The detector is deliberately heuristic and writes an annotated PDF for review.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24 compatibility
    import fitz  # type: ignore[no-redef]

import extract_dimension_ledger as dimension_ledger


VIEW_LABEL_RE = re.compile(
    r"^(?:SECTION\s+|DETAIL\s+|VIEW\s+)?[A-Z0-9]{1,3}\s*[-–—]\s*[A-Z0-9]{1,3}$",
    re.IGNORECASE,
)


def rounded(value: float, digits: int = 3) -> float:
    value = round(float(value), digits)
    return 0.0 if value == -0.0 else value


def bbox_union(boxes: Iterable[Sequence[float]]) -> list[float]:
    values = list(boxes)
    return [
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    ]


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def bbox_gap(first: Sequence[float], second: Sequence[float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def bbox_intersection(first: Sequence[float], second: Sequence[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = bbox_intersection(first, second)
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union else 0.0


def point_rect_distance(point: Sequence[float], box: Sequence[float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def expand_bbox(
    box: Sequence[float], padding: float, page_rect: Sequence[float]
) -> list[float]:
    return [
        max(page_rect[0], box[0] - padding),
        max(page_rect[1], box[1] - padding),
        min(page_rect[2], box[2] + padding),
        min(page_rect[3], box[3] + padding),
    ]


def _mask_rect(mask: np.ndarray, box: Sequence[float], scale: float, padding: float = 0.0) -> None:
    height, width = mask.shape
    x0 = max(0, int(math.floor((box[0] - padding) * scale)))
    y0 = max(0, int(math.floor((box[1] - padding) * scale)))
    x1 = min(width, int(math.ceil((box[2] + padding) * scale)))
    y1 = min(height, int(math.ceil((box[3] + padding) * scale)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = False


def _remove_dimension_geometry(
    mask: np.ndarray,
    dimensions: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
    scale: float,
) -> None:
    linked_ids = {
        segment_id
        for record in dimensions
        for key in ("line_segment_ids", "extension_segment_ids")
        for segment_id in record.get("geometry", {}).get(key, [])
    }
    by_id = {segment["id"]: segment for segment in geometry.segments}
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    draw = ImageDraw.Draw(image)
    width = max(2, int(round(scale * 2.2)))
    for segment_id in linked_ids:
        segment = by_id.get(segment_id)
        if segment:
            draw.line(
                tuple(value * scale for value in segment["p1"])
                + tuple(value * scale for value in segment["p2"]),
                fill=0,
                width=width,
            )
    mask[:] = np.asarray(image) > 0


def _remove_page_frames(
    mask: np.ndarray,
    geometry: dimension_ledger.PageGeometry,
    page_rect: Sequence[float],
    scale: float,
) -> None:
    """Remove sparse vector paths that span almost the full sheet."""
    by_path: dict[str, list[dict[str, Any]]] = {}
    for segment in geometry.segments:
        by_path.setdefault(segment["path_id"], []).append(segment)
    page_width = page_rect[2] - page_rect[0]
    page_height = page_rect[3] - page_rect[1]
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    draw = ImageDraw.Draw(image)
    width = max(3, int(round(scale * 3.5)))
    for segments in by_path.values():
        if len(segments) > 40:
            continue
        xs = [value for segment in segments for value in (segment["p1"][0], segment["p2"][0])]
        ys = [value for segment in segments for value in (segment["p1"][1], segment["p2"][1])]
        path_width = max(xs) - min(xs)
        path_height = max(ys) - min(ys)
        if path_width < page_width * 0.78 or path_height < page_height * 0.78:
            continue
        for segment in segments:
            draw.line(
                tuple(value * scale for value in segment["p1"])
                + tuple(value * scale for value in segment["p2"]),
                fill=0,
                width=width,
            )
    mask[:] = np.asarray(image) > 0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 0


def _render_content_mask(
    page: Any,
    page_data: dict[str, Any],
    geometry: dimension_ledger.PageGeometry,
    scale: float,
) -> np.ndarray:
    """Render visible drawing content while excluding sheet furniture."""
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False, annots=False
    )
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
    mask = pixels < 248
    page_rect = page_data["rect"]
    margin = max(8.0, min(page_rect[2], page_rect[3]) * 0.012)
    _mask_rect(mask, [0.0, 0.0, page_rect[2], margin], scale)
    _mask_rect(mask, [0.0, page_rect[3] - margin, page_rect[2], page_rect[3]], scale)
    _mask_rect(mask, [0.0, 0.0, margin, page_rect[3]], scale)
    _mask_rect(mask, [page_rect[2] - margin, 0.0, page_rect[2], page_rect[3]], scale)
    if page_data.get("detected_title_block"):
        _mask_rect(mask, page_data["detected_title_block"], scale, padding=2.0)
    for box in page_data.get("detected_technical_note_blocks", []):
        _mask_rect(mask, box, scale, padding=3.0)
    _remove_page_frames(mask, geometry, page_rect, scale)
    return mask


def _dimension_evidence_boxes(
    records: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
) -> list[list[float]]:
    """Return text and associated-vector extents used to grow a candidate region."""
    boxes = [list(record["bbox"]) for record in records]
    segments_by_id = {segment["id"]: segment for segment in geometry.segments}
    for record in records:
        record_geometry = record.get("geometry", {})
        points = [tuple(point) for point in record_geometry.get("line_endpoints", [])]
        for segment_id in (
            record_geometry.get("line_segment_ids", [])
            + record_geometry.get("extension_segment_ids", [])
        ):
            segment = segments_by_id.get(segment_id)
            if segment:
                points.extend((tuple(segment["p1"]), tuple(segment["p2"])))
        if points:
            boxes.append(
                [
                    min(point[0] for point in points),
                    min(point[1] for point in points),
                    max(point[0] for point in points),
                    max(point[1] for point in points),
                ]
            )
    return boxes


def _dimension_protection_boxes(
    records: Sequence[dict[str, Any]], page_rect: Sequence[float]
) -> list[list[float]]:
    """Return small search halos for owned text and vector punctuation."""
    boxes = []
    for record in records:
        rotation = float(record.get("rotation_deg", 0.0)) % 90.0
        slanted = 4.0 < rotation < 86.0
        padding = (
            max(8.0, min(11.0, float(record.get("font_size", 9.0)) * 0.75))
            if slanted
            else max(4.0, min(7.0, float(record.get("font_size", 9.0)) * 0.45))
        )
        boxes.append(expand_bbox(record["bbox"], padding, page_rect))
    return boxes


def _evidence_allowed_mask(
    shape: tuple[int, int],
    core_box: Sequence[float],
    records: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
    scale: float,
    *,
    include_core: bool = True,
) -> np.ndarray:
    """Build a loose corridor around one view's core and owned annotations."""
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)

    def rectangle(box: Sequence[float], padding: float) -> None:
        draw.rectangle(
            tuple((box[index] + (-padding if index < 2 else padding)) * scale for index in range(4)),
            fill=255,
        )

    if include_core:
        rectangle(core_box, 10.0)
    segments_by_id = {segment["id"]: segment for segment in geometry.segments}
    corridor_width = max(5, int(round(10.0 * scale)))
    for record in records:
        rectangle(
            record["bbox"],
            max(8.0, min(14.0, float(record.get("font_size", 9.0)) * 0.9)),
        )
        record_geometry = record.get("geometry", {})
        endpoints = record_geometry.get("line_endpoints", [])
        if len(endpoints) >= 2:
            draw.line(
                [tuple(value * scale for value in point) for point in endpoints],
                fill=255,
                width=corridor_width,
            )
        for segment_id in (
            record_geometry.get("line_segment_ids", [])
            + record_geometry.get("extension_segment_ids", [])
        ):
            segment = segments_by_id.get(segment_id)
            if segment:
                draw.line(
                    [
                        tuple(value * scale for value in segment["p1"]),
                        tuple(value * scale for value in segment["p2"]),
                    ],
                    fill=255,
                    width=corridor_width,
                )
    return np.asarray(image) > 0


def _distance_score_to_box(
    xs: np.ndarray, ys: np.ndarray, box: Sequence[float], scale: float
) -> np.ndarray:
    scaled = np.asarray(box, dtype=np.float64) * scale
    dx = np.maximum(np.maximum(scaled[0] - xs, xs - scaled[2]), 0.0)
    dy = np.maximum(np.maximum(scaled[1] - ys, ys - scaled[3]), 0.0)
    center_x = (scaled[0] + scaled[2]) / 2.0
    center_y = (scaled[1] + scaled[3]) / 2.0
    # The small center term provides a deterministic split where core boxes overlap.
    return dx * dx + dy * dy + 0.015 * ((xs - center_x) ** 2 + (ys - center_y) ** 2)


def _rdp(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification for one monotone polygon side."""
    if len(points) <= 2:
        return points
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    middle = np.asarray(points[1:-1], dtype=float)
    if length <= 1e-9:
        distances = np.linalg.norm(middle - start, axis=1)
    else:
        delta = middle - start
        distances = np.abs(vector[0] * delta[:, 1] - vector[1] * delta[:, 0]) / length
    farthest = int(np.argmax(distances))
    if float(distances[farthest]) <= tolerance:
        return [points[0], points[-1]]
    split = farthest + 1
    return _rdp(points[: split + 1], tolerance)[:-1] + _rdp(points[split:], tolerance)


def _band_polygon(
    mask: np.ndarray,
    scale: float,
    obstacles: Sequence[Sequence[float]],
    core_box: Sequence[float],
    *,
    protected_boxes: Sequence[Sequence[float]] = (),
    band_pt: float = 9.0,
    padding_pt: float = 5.0,
    collision_gap_pt: float = 4.0,
) -> list[list[float]]:
    """Fit a compact y-monotone envelope around ink, with obstacle notches."""
    padded = _dilate(mask, max(1, int(round(padding_pt * scale))))
    ys, xs = np.nonzero(padded)
    if not len(xs):
        box = expand_bbox(core_box, padding_pt, [0.0, 0.0, mask.shape[1] / scale, mask.shape[0] / scale])
        return [[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]]

    band_px = max(3, int(round(band_pt * scale)))
    first_band = int(ys.min() // band_px)
    last_band = int(ys.max() // band_px)
    spans: dict[int, tuple[float, float]] = {}
    protected_spans: dict[int, tuple[float, float]] = {}
    core_center_x = bbox_center(core_box)[0]
    for band in range(first_band, last_band + 1):
        y0_px = band * band_px
        y1_px = min(mask.shape[0], (band + 1) * band_px)
        band_ys, band_xs = np.nonzero(padded[y0_px:y1_px])
        if not len(band_xs):
            continue
        left = float(band_xs.min()) / scale
        right = float(band_xs.max() + 1) / scale
        y0, y1 = y0_px / scale, y1_px / scale
        band_protected = [
            box for box in protected_boxes if box[3] > y0 and box[1] < y1
        ]
        if band_protected:
            protected_spans[band] = (
                min(box[0] for box in band_protected),
                max(box[2] for box in band_protected),
            )
        for obstacle in obstacles:
            if obstacle[3] <= y0 or obstacle[1] >= y1:
                continue
            if obstacle[2] <= left or obstacle[0] >= right:
                continue
            obstacle_center_x = bbox_center(obstacle)[0]
            if core_center_x <= obstacle_center_x:
                right = min(right, obstacle[0] - collision_gap_pt)
            else:
                left = max(left, obstacle[2] + collision_gap_pt)
        # Collision avoidance must never clip the nominal value, stacked
        # deviations, units, or nearby vector parentheses of an owned dimension.
        if band in protected_spans:
            protected_left, protected_right = protected_spans[band]
            left = min(left, protected_left)
            right = max(right, protected_right)
        if right - left >= max(2.0, padding_pt):
            spans[band] = (left, right)

    if not spans:
        return _band_polygon(mask, scale, [], core_box, band_pt=band_pt, padding_pt=padding_pt)

    # Bridge small blank bands so text and line fragments stay in one crop.
    populated = sorted(spans)
    for first, second in zip(populated, populated[1:]):
        gap = second - first
        if gap <= 1 or gap > 7:
            continue
        first_span, second_span = spans[first], spans[second]
        for offset in range(1, gap):
            fraction = offset / gap
            spans[first + offset] = (
                first_span[0] * (1.0 - fraction) + second_span[0] * fraction,
                first_span[1] * (1.0 - fraction) + second_span[1] * fraction,
            )

    populated = sorted(spans)
    left_values = [spans[band][0] for band in populated]
    right_values = [spans[band][1] for band in populated]
    if len(populated) >= 3:
        left_values = [
            float(np.median(left_values[max(0, index - 1) : index + 2]))
            for index in range(len(left_values))
        ]
        right_values = [
            float(np.median(right_values[max(0, index - 1) : index + 2]))
            for index in range(len(right_values))
        ]
    for index, band in enumerate(populated):
        if band not in protected_spans:
            continue
        protected_left, protected_right = protected_spans[band]
        left_values[index] = min(left_values[index], protected_left)
        right_values[index] = max(right_values[index], protected_right)

    top_y = populated[0] * band_px / scale
    bottom_y = min(mask.shape[0], (populated[-1] + 1) * band_px) / scale
    y_values = [(band + 0.5) * band_px / scale for band in populated]
    left_profile = [(left_values[0], top_y)] + list(zip(left_values, y_values)) + [(left_values[-1], bottom_y)]
    right_profile = [(right_values[0], top_y)] + list(zip(right_values, y_values)) + [(right_values[-1], bottom_y)]
    tolerance = max(2.5, band_pt * 0.6)
    polygon = _rdp(left_profile, tolerance) + list(reversed(_rdp(right_profile, tolerance)))
    compact: list[list[float]] = []
    for point in polygon:
        rounded_point = [rounded(point[0]), rounded(point[1])]
        if not compact or rounded_point != compact[-1]:
            compact.append(rounded_point)
    return compact


def _polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    return abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(polygon, list(polygon[1:]) + [polygon[0]])
        )
    ) / 2.0


def _polygon_bbox(polygon: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    ]


def _rasterize_polygon(
    polygon: Sequence[Sequence[float]], shape: tuple[int, int], scale: float
) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(image).polygon(
        [tuple(value * scale for value in point) for point in polygon], fill=255
    )
    return np.asarray(image) > 0


def _rectangle_polygon(box: Sequence[float]) -> list[list[float]]:
    return [
        [rounded(box[0]), rounded(box[1])],
        [rounded(box[0]), rounded(box[3])],
        [rounded(box[2]), rounded(box[3])],
        [rounded(box[2]), rounded(box[1])],
    ]


def _fit_collision_polygons(
    content_mask: np.ndarray,
    regions: list[dict[str, Any]],
    dimensions: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
    page_data: dict[str, Any],
    scale: float,
    excluded_boxes: Sequence[Sequence[float]] = (),
) -> None:
    """Keep clean rectangles; fit polygons only where another view collides."""
    dimensions_by_id = {record["id"]: record for record in dimensions}
    page_rect = page_data["rect"]
    fixed_obstacles = [
        box
        for box in [
            page_data.get("detected_title_block"),
            *page_data.get("detected_technical_note_blocks", []),
        ]
        if box
    ]
    polygon_masks: list[np.ndarray] = []
    for region_index, region in enumerate(regions):
        records = [dimensions_by_id[value] for value in region["dimension_ids"] if value in dimensions_by_id]
        provisional = region["provisional_bbox"]
        region["dimension_text_boxes"] = [list(record["bbox"]) for record in records]
        region["dimension_text_colors"] = [record.get("_text_color") for record in records]
        region["dimension_protection_boxes"] = _dimension_protection_boxes(records, page_rect)
        foreign_records = [
            dimensions_by_id[dimension_id]
            for other_index, other in enumerate(regions)
            if other_index != region_index
            for dimension_id in other["dimension_ids"]
            if dimension_id in dimensions_by_id
        ]
        foreign_text_obstacles = [
            expand_bbox(dimensions_by_id[dimension_id]["bbox"], 3.0, page_rect)
            for other_index, other in enumerate(regions)
            if other_index != region_index
            for dimension_id in other["dimension_ids"]
            if dimension_id in dimensions_by_id
            and bbox_intersection(provisional, dimensions_by_id[dimension_id]["bbox"]) > 0
        ]
        # View bodies can overlap legitimately in crowded drawings. Treating the
        # neighbouring core as a hard rectangle cuts real section/profile ink;
        # only single-owner dimension labels are hard collision obstacles.
        core_obstacles: list[Sequence[float]] = []
        fixed_collisions = [
            obstacle
            for obstacle in fixed_obstacles
            if bbox_intersection(provisional, obstacle) > 0
            and not any(bbox_intersection(record["bbox"], obstacle) > 0 for record in records)
        ]
        excluded_collisions = [
            obstacle for obstacle in excluded_boxes
            if region.get("detection_basis") != "embedded_image"
            and bbox_intersection(provisional, obstacle) > 0
        ]
        fit_reasons = []
        if foreign_text_obstacles:
            fit_reasons.append("foreign_dimension_text")
        if core_obstacles:
            fit_reasons.append("other_view_core")
        if fixed_collisions:
            fit_reasons.append("sheet_furniture")
        if excluded_collisions:
            fit_reasons.append("excluded_3d_model")

        if not fit_reasons:
            polygon = _rectangle_polygon(provisional)
            region["polygon"] = polygon
            region["bbox"] = [rounded(value) for value in provisional]
            region["polygon_area"] = rounded(bbox_area(provisional))
            region["bbox_area_reduction_pct"] = 0.0
            region["fit_applied"] = False
            region["fit_reasons"] = []
            polygon_masks.append(_rasterize_polygon(polygon, content_mask.shape, scale))
            continue

        allowed = _evidence_allowed_mask(
            content_mask.shape, region["core_bbox"], records, geometry, scale
        )
        owned_evidence = _evidence_allowed_mask(
            content_mask.shape,
            region["core_bbox"],
            records,
            geometry,
            scale,
            include_core=False,
        )
        candidate = content_mask & allowed
        window = np.zeros_like(candidate)
        x0 = max(0, int(math.floor(provisional[0] * scale)))
        y0 = max(0, int(math.floor(provisional[1] * scale)))
        x1 = min(candidate.shape[1], int(math.ceil(provisional[2] * scale)))
        y1 = min(candidate.shape[0], int(math.ceil(provisional[3] * scale)))
        window[y0:y1, x0:x1] = True
        candidate &= window

        # Lines belonging to another view may cross this view legitimately, but
        # they must not pull its crop envelope toward the foreign annotation.
        # Ignore them while fitting the envelope; owned evidence is restored.
        if foreign_records:
            foreign_evidence = _evidence_allowed_mask(
                content_mask.shape,
                region["core_bbox"],
                foreign_records,
                geometry,
                scale,
                include_core=False,
            )
            candidate &= ~foreign_evidence
            candidate |= content_mask & owned_evidence & window

        candidate_ys, candidate_xs = np.nonzero(candidate)
        if len(candidate_xs):
            own_score = _distance_score_to_box(
                candidate_xs, candidate_ys, region["core_bbox"], scale
            )
            keep = np.ones(len(candidate_xs), dtype=bool)
            for other_index, other in enumerate(regions):
                if other_index == region_index:
                    continue
                if bbox_intersection(provisional, other["provisional_bbox"]) <= 0:
                    continue
                other_score = _distance_score_to_box(
                    candidate_xs, candidate_ys, other["core_bbox"], scale
                )
                keep &= own_score <= other_score
            keep |= owned_evidence[candidate_ys, candidate_xs]
            rejected = ~keep
            candidate[candidate_ys[rejected], candidate_xs[rejected]] = False

        # Geometry may overlap legitimately (shared projection / cutting lines), but
        # dimension text has a single hard owner and must never enter another crop.
        protection_boxes = region["dimension_protection_boxes"]
        polygon = _band_polygon(
            candidate,
            scale,
            fixed_obstacles + foreign_text_obstacles + core_obstacles + excluded_collisions,
            region["core_bbox"],
            protected_boxes=[expand_bbox(record["bbox"], 2.0, page_rect) for record in records],
        )
        polygon_mask = _rasterize_polygon(polygon, content_mask.shape, scale)
        polygon_box = bbox_union([_polygon_bbox(polygon), *protection_boxes])
        region["polygon"] = polygon
        region["bbox"] = [rounded(value) for value in polygon_box]
        region["polygon_area"] = rounded(_polygon_area(polygon))
        region["bbox_area_reduction_pct"] = rounded(
            max(0.0, 100.0 * (1.0 - region["polygon_area"] / max(bbox_area(polygon_box), 1.0))),
            2,
        )
        region["fit_applied"] = True
        region["fit_reasons"] = fit_reasons
        region["excluded_model_boxes"] = [list(box) for box in excluded_collisions]
        polygon_masks.append(polygon_mask)

    # A foreign dimension label can sit inside a large connected view envelope,
    # where a simple outer polygon cannot avoid it. Represent those cases as
    # transparent rectangular holes; edge labels are normally handled by notches.
    for index, region in enumerate(regions):
        exclusion_polygons: list[list[list[float]]] = []
        own_records = [
            dimensions_by_id[dimension_id]
            for dimension_id in region["dimension_ids"]
            if dimension_id in dimensions_by_id
        ]
        own_protection_boxes = [
            expand_bbox(record["bbox"], 1.0, page_rect) for record in own_records
        ]
        for other_index, other in enumerate(regions):
            if other_index == index:
                continue
            for dimension_id in other["dimension_ids"]:
                record = dimensions_by_id.get(dimension_id)
                if not record:
                    continue
                box = expand_bbox(record["bbox"], 3.0, page_rect)
                if any(
                    bbox_intersection(box, own_box)
                    / max(bbox_area(box), 1.0)
                    >= 0.20
                    for own_box in own_protection_boxes
                ):
                    continue
                x0 = max(0, int(math.floor(box[0] * scale)))
                y0 = max(0, int(math.floor(box[1] * scale)))
                x1 = min(content_mask.shape[1], int(math.ceil(box[2] * scale)))
                y1 = min(content_mask.shape[0], int(math.ceil(box[3] * scale)))
                if x1 <= x0 or y1 <= y0 or not np.any(polygon_masks[index][y0:y1, x0:x1]):
                    continue
                hole = [
                    [rounded(box[0]), rounded(box[1])],
                    [rounded(box[2]), rounded(box[1])],
                    [rounded(box[2]), rounded(box[3])],
                    [rounded(box[0]), rounded(box[3])],
                ]
                exclusion_polygons.append(hole)
                polygon_masks[index][y0:y1, x0:x1] = False
        region["exclusion_polygons"] = exclusion_polygons
        region["dimension_text_exclusion_count"] = len(exclusion_polygons)

    for index, region in enumerate(regions):
        avoided: list[str] = []
        overlap_pixels = 0
        foreign_dimension_text_ids: list[str] = []
        for other_index, other in enumerate(regions):
            if other_index == index:
                continue
            old_overlap = bbox_intersection(region["provisional_bbox"], other["provisional_bbox"])
            if old_overlap <= 0:
                continue
            overlap = int(np.count_nonzero(polygon_masks[index] & polygon_masks[other_index]))
            overlap_pixels += overlap
            if overlap * 4.0 < old_overlap * scale * scale:
                avoided.append(other["id"])
            for dimension_id in other["dimension_ids"]:
                record = dimensions_by_id.get(dimension_id)
                if not record:
                    continue
                box = record["bbox"]
                x0 = max(0, int(math.floor(box[0] * scale)))
                y0 = max(0, int(math.floor(box[1] * scale)))
                x1 = min(content_mask.shape[1], int(math.ceil(box[2] * scale)))
                y1 = min(content_mask.shape[0], int(math.ceil(box[3] * scale)))
                if x1 > x0 and y1 > y0 and np.any(polygon_masks[index][y0:y1, x0:x1]):
                    foreign_dimension_text_ids.append(dimension_id)
        region["collision_avoided_with"] = sorted(set(avoided))
        region["polygon_overlap_pixels"] = overlap_pixels
        region["foreign_dimension_text_ids"] = sorted(set(foreign_dimension_text_ids))


def _embedded_image_regions(
    page: Any, page_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return tight non-white boxes for substantial embedded drawing images."""
    page_rect = page_data["rect"]
    page_area = bbox_area(page_rect)
    excluded = [
        box
        for box in [
            page_data.get("detected_title_block"),
            *page_data.get("detected_technical_note_blocks", []),
        ]
        if box
    ]
    regions: list[dict[str, Any]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        for rect in page.get_image_rects(xref):
            signature = (xref, rounded(rect.x0, 1), rounded(rect.y0, 1), rounded(rect.x1, 1), rounded(rect.y1, 1))
            if signature in seen:
                continue
            seen.add(signature)
            box = [rect.x0, rect.y0, rect.x1, rect.y1]
            if bbox_area(box) < page_area * 0.008:
                continue
            if any(
                bbox_intersection(box, excluded_box) / max(bbox_area(box), 1.0) > 0.55
                for excluded_box in excluded
            ):
                continue
            try:
                pixmap = fitz.Pixmap(page.parent, xref)
                channels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
            except (ValueError, RuntimeError):
                continue
            color_channels = channels[:, :, : min(3, pixmap.n)]
            ink = np.min(color_channels, axis=2) < 248
            if pixmap.alpha and pixmap.n > 1:
                ink &= channels[:, :, -1] > 8
            ys, xs = np.nonzero(ink)
            if not len(xs):
                continue
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            tight = [
                rect.x0 + rect.width * x0 / pixmap.width,
                rect.y0 + rect.height * y0 / pixmap.height,
                rect.x0 + rect.width * x1 / pixmap.width,
                rect.y0 + rect.height * y1 / pixmap.height,
            ]
            tight = expand_bbox(tight, 3.0, page_rect)
            if bbox_area(tight) < page_area * 0.004:
                continue
            regions.append(
                {
                    "core_bbox": tight,
                    "dilated_pixels": int(bbox_area(tight)),
                    "assigned_dimensions": [],
                    "source_kind": "embedded_image",
                }
            )
    return regions


def _run_components(mask: np.ndarray) -> list[dict[str, int]]:
    """Connected components using row runs, avoiding a pixel-by-pixel Python BFS."""
    parents: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, stop in zip(changes[0::2], changes[1::2]):
            end = int(stop - 1)
            start = int(start)
            label = len(parents)
            parents.append(label)
            runs.append((y, start, end, label))
            current.append((start, end, label))
            while previous_index < len(previous) and previous[previous_index][1] < start - 1:
                previous_index += 1
            probe = previous_index
            while probe < len(previous) and previous[probe][0] <= end + 1:
                union(label, previous[probe][2])
                probe += 1
        previous = current

    components: dict[int, dict[str, int]] = {}
    for y, start, end, label in runs:
        root = find(label)
        item = components.setdefault(
            root,
            {"x0": start, "y0": y, "x1": end + 1, "y1": y + 1, "pixels": 0},
        )
        item["x0"] = min(item["x0"], start)
        item["y0"] = min(item["y0"], y)
        item["x1"] = max(item["x1"], end + 1)
        item["y1"] = max(item["y1"], y + 1)
        item["pixels"] += end - start + 1
    return list(components.values())


def _render_line_art(
    page: Any,
    page_data: dict[str, Any],
    page_tokens: Sequence[dict[str, Any]],
    page_dimensions: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
    scale: float,
) -> np.ndarray:
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False, annots=False
    )
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
    mask = pixels < 245

    page_rect = page_data["rect"]
    margin = max(8.0, min(page_rect[2], page_rect[3]) * 0.012)
    _mask_rect(mask, [0.0, 0.0, page_rect[2], margin], scale)
    _mask_rect(mask, [0.0, page_rect[3] - margin, page_rect[2], page_rect[3]], scale)
    _mask_rect(mask, [0.0, 0.0, margin, page_rect[3]], scale)
    _mask_rect(mask, [page_rect[2] - margin, 0.0, page_rect[2], page_rect[3]], scale)

    if page_data.get("detected_title_block"):
        _mask_rect(mask, page_data["detected_title_block"], scale, padding=2.0)
    for box in page_data.get("detected_technical_note_blocks", []):
        _mask_rect(mask, box, scale, padding=3.0)
    for token in page_tokens:
        _mask_rect(mask, token["bbox"], scale, padding=max(1.5, token.get("size", 8.0) * 0.12))
    _remove_page_frames(mask, geometry, page_rect, scale)
    _remove_dimension_geometry(mask, page_dimensions, geometry, scale)
    return mask


def _core_regions(
    mask: np.ndarray, page_rect: Sequence[float], scale: float, dilation_pt: float
) -> list[dict[str, Any]]:
    dilated = _dilate(mask, max(1, int(round(dilation_pt * scale))))
    page_area = (page_rect[2] - page_rect[0]) * (page_rect[3] - page_rect[1])
    minimum_span = min(page_rect[2], page_rect[3]) * 0.018
    regions: list[dict[str, Any]] = []
    for component in _run_components(dilated):
        box = [
            component["x0"] / scale,
            component["y0"] / scale,
            component["x1"] / scale,
            component["y1"] / scale,
        ]
        width, height = box[2] - box[0], box[3] - box[1]
        area = width * height
        aspect = max(width, height) / max(min(width, height), 0.01)
        if max(width, height) < minimum_span:
            continue
        if area < page_area * 0.00055:
            continue
        if aspect > 3.0 and min(width, height) < minimum_span * 2.4:
            continue
        if aspect > 24.0 and min(width, height) < minimum_span * 1.5:
            continue
        if area > page_area * 0.82:
            continue
        regions.append(
            {
                "core_bbox": box,
                "dilated_pixels": component["pixels"],
                "assigned_dimensions": [],
                "source_kind": "line_art",
            }
        )
    return sorted(regions, key=lambda item: bbox_area(item["core_bbox"]), reverse=True)


def _dimension_points(record: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = record.get("geometry", {})
    points = [tuple(value) for value in geometry.get("line_endpoints", [])]
    if geometry.get("leader_end"):
        points.append(tuple(geometry["leader_end"]))
    return points


def _assign_dimensions(
    cores: list[dict[str, Any]],
    dimensions: Sequence[dict[str, Any]],
    page_rect: Sequence[float],
) -> None:
    maximum_distance = max(34.0, min(page_rect[2], page_rect[3]) * 0.055)
    page_area = bbox_area(page_rect)
    for record in dimensions:
        points = _dimension_points(record)
        choices = []
        for index, core in enumerate(cores):
            box = core["core_bbox"]
            text_distance = bbox_gap(record["bbox"], box)
            if points:
                density = core.get("dilated_pixels", bbox_area(box)) / max(bbox_area(box), 1.0)
                if density < 0.10 and bbox_area(box) > page_area * 0.02:
                    continue
                distances = [point_rect_distance(point, box) for point in points]
                # A two-arrow dimension belongs where the complete measured span
                # lands. One endpoint crossing a neighbouring view is not enough
                # to steal the record from the view closest to both endpoints.
                anchor_distance = (
                    sum(distances) / len(distances)
                    if len(distances) >= 2
                    else distances[0]
                )
                anchor_x = sum(point[0] for point in points) / len(points)
                anchor_y = sum(point[1] for point in points) / len(points)
                width = max(box[2] - box[0], 1.0)
                height = max(box[3] - box[1], 1.0)
                center_x, center_y = bbox_center(box)
                anchor_centrality = (
                    ((anchor_x - center_x) / width) ** 2
                    + ((anchor_y - center_y) / height) ** 2
                )
                choices.append(
                    (anchor_distance, anchor_centrality, text_distance, -bbox_area(box), index)
                )
            else:
                choices.append((text_distance, 0.0, text_distance, -bbox_area(box), index))
        if not choices:
            continue
        best_distance, _, _, _, best_index = min(choices)
        if best_distance <= maximum_distance:
            cores[best_index]["assigned_dimensions"].append(record)


def _merge_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed = True
    while changed:
        changed = False
        for first_index in range(len(regions)):
            for second_index in range(first_index + 1, len(regions)):
                first, second = regions[first_index], regions[second_index]
                if (
                    first.get("source_kind") == "embedded_image"
                    or second.get("source_kind") == "embedded_image"
                ):
                    continue
                first_box, second_box = first["core_bbox"], second["core_bbox"]
                intersection = bbox_intersection(first_box, second_box)
                containment = intersection / max(min(bbox_area(first_box), bbox_area(second_box)), 1.0)
                shared_dimensions = {
                    record["id"] for record in first["assigned_dimensions"]
                } & {record["id"] for record in second["assigned_dimensions"]}
                if bbox_iou(first_box, second_box) < 0.12 and containment < 0.58 and not shared_dimensions:
                    continue
                merged = {
                    "core_bbox": bbox_union([first_box, second_box]),
                    "dilated_pixels": first["dilated_pixels"] + second["dilated_pixels"],
                    "assigned_dimensions": list(
                        {
                            record["id"]: record
                            for record in first["assigned_dimensions"] + second["assigned_dimensions"]
                        }.values()
                    ),
                    "source_kind": "line_art",
                }
                regions[first_index] = merged
                regions.pop(second_index)
                changed = True
                break
            if changed:
                break
    return regions


def _suppress_embedded_duplicates(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge an image with a small adjacent dimension seed for the same view."""
    kept = []
    for region in regions:
        if region.get("source_kind") != "embedded_image":
            kept.append(region)
            continue
        image_box = region["core_bbox"]
        matches = [
            other
            for other in regions
            if other.get("source_kind") == "line_art"
            and other["assigned_dimensions"]
            and (
                bbox_intersection(image_box, other["core_bbox"])
                / max(bbox_area(image_box), 1.0)
                >= 0.35
                or (
                    bbox_gap(image_box, other["core_bbox"]) <= 5.0
                    and bbox_area(other["core_bbox"]) < bbox_area(image_box) * 0.35
                )
            )
        ]
        if matches:
            owner = min(matches, key=lambda value: bbox_gap(image_box, value["core_bbox"]))
            region["core_bbox"] = bbox_union([owner["core_bbox"], image_box])
            region["dilated_pixels"] += owner["dilated_pixels"]
            region["assigned_dimensions"] = list(
                {
                    record["id"]: record
                    for record in region["assigned_dimensions"] + owner["assigned_dimensions"]
                }.values()
            )
            kept = [value for value in kept if value is not owner]
            kept.append(region)
        else:
            kept.append(region)
    return kept


def _labels_for_region(
    tokens: Sequence[dict[str, Any]], box: Sequence[float], page_rect: Sequence[float]
) -> list[str]:
    search_box = expand_bbox(box, max(18.0, min(page_rect[2], page_rect[3]) * 0.025), page_rect)
    labels = []
    for token in tokens:
        text = token.get("normalized_text", "").strip()
        if VIEW_LABEL_RE.fullmatch(text) and point_rect_distance(bbox_center(token["bbox"]), search_box) == 0:
            labels.append(text)
    return list(dict.fromkeys(labels))


def detect_views(
    input_path: Path,
    ledger: dict[str, Any],
    *,
    scale: float = 1.0,
    dilation_pt: float = 3.0,
) -> dict[str, Any]:
    pages_by_number = {page["page"]: page for page in ledger["pages"]}
    tokens_by_page: dict[int, list[dict[str, Any]]] = {}
    dimensions_by_page: dict[int, list[dict[str, Any]]] = {}
    for token in ledger.get("raw_text_tokens", []):
        tokens_by_page.setdefault(token["page"], []).append(token)
    for record in ledger.get("dimensions", []):
        dimensions_by_page.setdefault(record["page"], []).append(record)

    output_pages = []
    with fitz.open(input_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_data = pages_by_number[page_number]
            page_rect = page_data["rect"]
            tokens = tokens_by_page.get(page_number, [])
            dimensions = dimensions_by_page.get(page_number, [])
            tokens_by_id = {token["id"]: token for token in tokens}
            for record in dimensions:
                root_token = tokens_by_id.get(record.get("root_token_id"))
                record["_text_color"] = root_token.get("color") if root_token else None
            geometry = dimension_ledger.extract_vector_geometry(page, page_number)
            content_mask = _render_content_mask(page, page_data, geometry, scale)
            mask = _render_line_art(
                page, page_data, tokens, dimensions, geometry, scale
            )
            page_area = bbox_area(page_rect)
            # Small embedded images can be legitimate enlarged 2D details. Large
            # rendered images in these drawings are presentation-only 3D models.
            all_image_cores = _embedded_image_regions(page, page_data)
            image_cores = [
                core
                for core in all_image_cores
                if bbox_area(core["core_bbox"]) < page_area * 0.025
            ]
            for image_core in all_image_cores:
                _mask_rect(mask, image_core["core_bbox"], scale, padding=2.0)
            cores = _core_regions(mask, page_rect, scale, dilation_pt)
            cores.extend(image_cores)
            _assign_dimensions(cores, dimensions, page_rect)
            cores = _suppress_embedded_duplicates(cores)
            cores = _merge_regions(cores)

            regions = []
            for core in cores:
                assigned = core["assigned_dimensions"]
                core_area = bbox_area(core["core_bbox"])
                source_kind = core.get("source_kind", "line_art")
                line_art_density = core["dilated_pixels"] / max(core_area * scale * scale, 1.0)
                if not assigned and core_area < page_area * 0.0035:
                    continue
                if not assigned and source_kind == "line_art" and line_art_density < 0.30:
                    width = core["core_bbox"][2] - core["core_bbox"][0]
                    height = core["core_bbox"][3] - core["core_bbox"][1]
                    compact_large_core = (
                        core_area >= page_area * 0.08
                        and max(width, height) / max(min(width, height), 0.01) <= 2.5
                        and line_art_density >= 0.12
                    )
                    if not compact_large_core:
                        continue
                if len(assigned) <= 1 and core_area < page_area * 0.004:
                    continue
                if (
                    assigned
                    and core_area < page_area * 0.01
                    and all(record.get("type") == "geometric_tolerance" for record in assigned)
                ):
                    continue
                sort_box = expand_bbox(
                    bbox_union(
                        [core["core_bbox"]] + [record["bbox"] for record in assigned]
                    ),
                    max(5.0, min(page_rect[2], page_rect[3]) * 0.008),
                    page_rect,
                )
                evidence_boxes = [core["core_bbox"]] + _dimension_evidence_boxes(
                    assigned, geometry
                )
                provisional_box = expand_bbox(
                    bbox_union(evidence_boxes),
                    max(5.0, min(page_rect[2], page_rect[3]) * 0.008),
                    page_rect,
                )
                labels = _labels_for_region(tokens, provisional_box, page_rect)
                confidence = min(
                    0.96,
                    0.46
                    + min(len(assigned), 5) * 0.07
                    + (0.12 if labels else 0.0)
                    + min(core_area / page_area, 0.08),
                )
                regions.append(
                    {
                        "id": "",
                        "_sort_key": [sort_box[1], sort_box[0]],
                        "bbox": [rounded(value) for value in provisional_box],
                        "provisional_bbox": [rounded(value) for value in provisional_box],
                        "core_bbox": [rounded(value) for value in core["core_bbox"]],
                        "labels": labels,
                        "view_type": "section_or_detail" if labels else "unclassified",
                        "confidence": rounded(confidence, 2),
                        "dimension_ids": sorted(record["id"] for record in assigned),
                        "dimension_count": len(assigned),
                        "line_art_density": rounded(line_art_density, 3),
                        "detection_basis": (
                            "embedded_image"
                            if source_kind == "embedded_image"
                            else "line_art_and_dimension_seeds"
                            if assigned
                            else "line_art_only"
                        ),
                    }
                )
            regions.sort(key=lambda item: item["_sort_key"])
            for index, region in enumerate(regions, start=1):
                region["id"] = f"P{page_number}-V{index:02d}"
                region.pop("_sort_key", None)
            _fit_collision_polygons(
                content_mask,
                regions,
                dimensions,
                geometry,
                page_data,
                scale,
                [
                    core["core_bbox"]
                    for core in all_image_cores
                    if core not in image_cores
                ],
            )
            output_pages.append(
                {
                    "page": page_number,
                    "rect": page_rect,
                    "view_count": len(regions),
                    "excluded_embedded_3d_models": len(all_image_cores) - len(image_cores),
                    "views": regions,
                }
            )

    return {
        "schema_version": "0.2",
        "source": str(input_path.resolve()),
        "dimension_ledger_source": ledger.get("source"),
        "parameters": {
            "render_scale": scale,
            "dilation_pt": dilation_pt,
            "region_shape": "rectangle_first_collision_aware_polygon",
            "embedded_3d_models": "excluded",
        },
        "summary": {
            "pages": len(output_pages),
            "views": sum(page["view_count"] for page in output_pages),
            "dimension_seeded_views": sum(
                view["detection_basis"] == "line_art_and_dimension_seeds"
                for page in output_pages
                for view in page["views"]
            ),
            "line_art_only_views": sum(
                view["detection_basis"] == "line_art_only"
                for page in output_pages
                for view in page["views"]
            ),
            "embedded_image_views": sum(
                view["detection_basis"] == "embedded_image"
                for page in output_pages
                for view in page["views"]
            ),
            "excluded_embedded_3d_models": sum(
                page["excluded_embedded_3d_models"] for page in output_pages
            ),
            "collision_avoiding_views": sum(
                bool(view["collision_avoided_with"])
                for page in output_pages
                for view in page["views"]
            ),
            "fitted_views": sum(
                bool(view.get("fit_applied"))
                for page in output_pages
                for view in page["views"]
            ),
            "polygon_overlap_pixels": sum(
                view["polygon_overlap_pixels"]
                for page in output_pages
                for view in page["views"]
            )
            // 2,
            "foreign_dimension_text_collisions": sum(
                len(view["foreign_dimension_text_ids"])
                for page in output_pages
                for view in page["views"]
            ),
        },
        "pages": output_pages,
    }


def write_review_pdf(input_path: Path, output_path: Path, result: dict[str, Any]) -> None:
    colors = ((0.85, 0.1, 0.15), (0.0, 0.55, 0.85), (0.1, 0.65, 0.2), (0.7, 0.25, 0.8))
    with fitz.open(input_path) as document:
        for page_data in result["pages"]:
            page = document[page_data["page"] - 1]
            for index, view in enumerate(page_data["views"]):
                color = colors[index % len(colors)]
                rect = fitz.Rect(view["bbox"])
                polygon = [fitz.Point(*point) for point in view.get("polygon", [])]
                if len(polygon) >= 3:
                    page.draw_polyline(polygon + [polygon[0]], color=color, width=2.0, overlay=True)
                else:
                    page.draw_rect(rect, color=color, width=2.0, overlay=True)
                for hole in view.get("exclusion_polygons", []):
                    hole_points = [fitz.Point(*point) for point in hole]
                    page.draw_polyline(
                        hole_points + [hole_points[0]],
                        color=color,
                        width=1.0,
                        dashes="2 2",
                        overlay=True,
                    )
                reduction = view.get("bbox_area_reduction_pct", 0.0)
                label = (
                    f"{view['id']}  {view['dimension_count']} dims  "
                    f"fit {reduction:.0f}%  {view['confidence']:.2f}"
                )
                page.insert_text(
                    (rect.x0 + 2, max(page.rect.y0 + 9, rect.y0 - 3)),
                    label,
                    fontsize=8,
                    color=color,
                    overlay=True,
                )
        document.save(output_path, garbage=3, deflate=True)


def write_polygon_crops(input_path: Path, output_dir: Path, result: dict[str, Any]) -> None:
    """Write transparent PNG crops and one contact sheet for quick visual review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[tuple[str, Image.Image]] = []
    render_scale = 2.0
    with fitz.open(input_path) as document:
        for page_data in result["pages"]:
            page = document[page_data["page"] - 1]
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(render_scale, render_scale), alpha=False, annots=False
            )
            page_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            page_pixels = np.asarray(page_image)
            page_ink = np.min(page_pixels, axis=2) < 248
            page_views_by_id = {view["id"]: view for view in page_data["views"]}
            dimension_owner = {
                dimension_id: view["id"]
                for view in page_data["views"]
                for dimension_id in view["dimension_ids"]
            }
            dimension_boxes = {
                dimension_id: box
                for view in page_data["views"]
                for dimension_id, box in zip(
                    view.get("dimension_ids", []), view.get("dimension_text_boxes", [])
                )
            }
            for view in page_data["views"]:
                polygon = view.get("polygon")
                if not polygon:
                    continue
                box = view["bbox"]
                crop_px = (
                    max(0, int(math.floor(box[0] * render_scale))),
                    max(0, int(math.floor(box[1] * render_scale))),
                    min(page_image.width, int(math.ceil(box[2] * render_scale))),
                    min(page_image.height, int(math.ceil(box[3] * render_scale))),
                )
                crop = page_image.crop(crop_px).convert("RGBA")
                alpha = Image.new("L", crop.size, 0)
                ImageDraw.Draw(alpha).polygon(
                    [
                        (
                            point[0] * render_scale - crop_px[0],
                            point[1] * render_scale - crop_px[1],
                        )
                        for point in polygon
                    ],
                    fill=255,
                )
                alpha_draw = ImageDraw.Draw(alpha)
                for hole in view.get("exclusion_polygons", []):
                    alpha_draw.polygon(
                        [
                            (
                                point[0] * render_scale - crop_px[0],
                                point[1] * render_scale - crop_px[1],
                            )
                            for point in hole
                        ],
                        fill=0,
                    )
                alpha_pixels = np.asarray(alpha).copy()

                # Embedded 3D presentation models never belong to a 2D view.
                # Clear their full tight boxes before restoring owned dimension
                # ink, because a dimension label may legitimately overlap the
                # model's broad image rectangle.
                for model_box in view.get("excluded_model_boxes", []):
                    mx0 = max(0, int(math.floor(model_box[0] * render_scale)) - crop_px[0])
                    my0 = max(0, int(math.floor(model_box[1] * render_scale)) - crop_px[1])
                    mx1 = min(alpha_pixels.shape[1], int(math.ceil(model_box[2] * render_scale)) - crop_px[0])
                    my1 = min(alpha_pixels.shape[0], int(math.ceil(model_box[3] * render_scale)) - crop_px[1])
                    if mx1 > mx0 and my1 > my0:
                        alpha_pixels[my0:my1, mx0:mx1] = 0

                # Restore only connected ink around each owned text box. This is
                # the final hard constraint that preserves split tolerances and
                # vector parentheses without swallowing a neighbour's whitespace.
                for text_box, protection_box, text_color in zip(
                    view.get("dimension_text_boxes", []),
                    view.get("dimension_protection_boxes", []),
                    view.get("dimension_text_colors", []),
                ):
                    hx0 = max(crop_px[0], int(math.floor(protection_box[0] * render_scale)))
                    hy0 = max(crop_px[1], int(math.floor(protection_box[1] * render_scale)))
                    hx1 = min(crop_px[2], int(math.ceil(protection_box[2] * render_scale)))
                    hy1 = min(crop_px[3], int(math.ceil(protection_box[3] * render_scale)))
                    if hx1 <= hx0 or hy1 <= hy0:
                        continue
                    local_ink = page_ink[hy0:hy1, hx0:hx1]
                    bridged = _dilate(local_ink, max(2, int(round(render_scale * 1.5))))
                    tx0 = max(0, int(math.floor(text_box[0] * render_scale)) - hx0)
                    ty0 = max(0, int(math.floor(text_box[1] * render_scale)) - hy0)
                    tx1 = min(bridged.shape[1], int(math.ceil(text_box[2] * render_scale)) - hx0)
                    ty1 = min(bridged.shape[0], int(math.ceil(text_box[3] * render_scale)) - hy0)
                    seed = np.zeros_like(bridged)
                    if tx1 > tx0 and ty1 > ty0:
                        seed[ty0:ty1, tx0:tx1] = bridged[ty0:ty1, tx0:tx1]
                    connected = seed
                    while True:
                        grown = _dilate(connected, 1) & bridged
                        if np.array_equal(grown, connected):
                            break
                        connected = grown
                    ay0, ay1 = hy0 - crop_px[1], hy1 - crop_px[1]
                    ax0, ax1 = hx0 - crop_px[0], hx1 - crop_px[0]
                    source_pixels = page_pixels[hy0:hy1, hx0:hx1]
                    restore = connected & local_ink
                    if text_color is not None:
                        target = np.asarray(
                            [
                                (int(text_color) >> 16) & 255,
                                (int(text_color) >> 8) & 255,
                                int(text_color) & 255,
                            ],
                            dtype=np.int16,
                        )
                        dominant = int(np.argmax(target))
                        if target.max() >= 180 and target.max() - np.partition(target, -2)[-2] >= 80:
                            others = [index for index in range(3) if index != dominant]
                            color_match = (
                                source_pixels[:, :, dominant] >= 180
                            ) & (
                                source_pixels[:, :, dominant]
                                >= np.maximum(
                                    source_pixels[:, :, others[0]],
                                    source_pixels[:, :, others[1]],
                                )
                                + 18
                            )
                        elif target.max() <= 80:
                            color_match = np.max(source_pixels, axis=2) <= 150
                        else:
                            delta = source_pixels.astype(np.int16) - target
                            color_match = np.sum(delta * delta, axis=2) <= 90 * 90
                        restore &= color_match
                    alpha_pixels[ay0:ay1, ax0:ax1][restore] = 255
                alpha = Image.fromarray(alpha_pixels, mode="L")
                alpha_pixels = np.asarray(alpha).copy()
                # Final single-owner audit: no crop may retain ink inside another
                # view's dimension label box. Owned-label restoration above has
                # already completed before this exclusion pass.
                for dimension_id, owner_id in dimension_owner.items():
                    if owner_id == view["id"]:
                        continue
                    text_box = dimension_boxes.get(dimension_id)
                    owner = page_views_by_id.get(owner_id)
                    if not text_box or not owner:
                        continue
                    # Only clear labels involved in an actual view collision; a
                    # distant label cannot occur in this crop and needs no work.
                    if bbox_intersection(view["provisional_bbox"], owner["provisional_bbox"]) <= 0:
                        continue
                    tx0 = max(0, int(math.floor(text_box[0] * render_scale)) - crop_px[0])
                    ty0 = max(0, int(math.floor(text_box[1] * render_scale)) - crop_px[1])
                    tx1 = min(alpha_pixels.shape[1], int(math.ceil(text_box[2] * render_scale)) - crop_px[0])
                    ty1 = min(alpha_pixels.shape[0], int(math.ceil(text_box[3] * render_scale)) - crop_px[1])
                    if tx1 > tx0 and ty1 > ty0:
                        alpha_pixels[ty0:ty1, tx0:tx1] = 0
                alpha = Image.fromarray(alpha_pixels, mode="L")
                crop.putalpha(alpha)
                crop_path = output_dir / f"{view['id']}.png"
                crop.save(crop_path)

                preview = Image.new("RGB", crop.size, "white")
                preview.paste(crop, mask=crop.getchannel("A"))
                preview.thumbnail((430, 310), Image.Resampling.LANCZOS)
                cards.append((view["id"], preview))

    if not cards:
        return
    columns = 2
    card_width, card_height = 460, 350
    rows = math.ceil(len(cards) / columns)
    sheet = Image.new("RGB", (columns * card_width, rows * card_height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for index, (view_id, preview) in enumerate(cards):
        column, row = index % columns, index // columns
        x0, y0 = column * card_width, row * card_height
        draw.rectangle((x0 + 8, y0 + 8, x0 + card_width - 8, y0 + card_height - 8), fill="white")
        draw.text((x0 + 18, y0 + 16), view_id, fill=(25, 25, 25))
        sheet.paste(preview, (x0 + 18, y0 + 36))
    sheet.save(output_dir / "contact-sheet.png")


def load_or_build_ledger(
    input_path: Path, ledger_path: Path | None, unit: str | None
) -> dict[str, Any]:
    if ledger_path:
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    return dimension_ledger.public_result(dimension_ledger.analyze_pdf(input_path, unit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and box engineering drawing views from dimension seeds and line art."
    )
    parser.add_argument("input", type=Path, help="source vector PDF")
    parser.add_argument("-l", "--ledger", type=Path, help="existing dimension-ledger.json")
    parser.add_argument("-o", "--output", type=Path, help="output directory")
    parser.add_argument("--unit", help="default unit when building a dimension ledger")
    parser.add_argument("--scale", type=float, default=1.0, help="render pixels per PDF point")
    parser.add_argument("--dilation", type=float, default=3.0, help="line-art dilation in PDF points")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 2
    ledger_path = args.ledger.resolve() if args.ledger else None
    if ledger_path and not ledger_path.is_file():
        print(f"error: ledger not found: {ledger_path}", file=sys.stderr)
        return 2
    if args.scale <= 0 or args.dilation < 0:
        print("error: scale must be positive and dilation cannot be negative", file=sys.stderr)
        return 2
    output_dir = (
        args.output or input_path.with_name(f"{input_path.stem}-view-regions")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ledger = load_or_build_ledger(input_path, ledger_path, args.unit)
        result = detect_views(
            input_path, ledger, scale=args.scale, dilation_pt=args.dilation
        )
        (output_dir / "view-regions.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_review_pdf(input_path, output_dir / "view-review.pdf", result)
        write_polygon_crops(input_path, output_dir / "crops", result)
    except (ValueError, KeyError, fitz.FileDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = result["summary"]
    print(f"Output: {output_dir}")
    print(
        f"Views: {summary['views']} across {summary['pages']} page(s); "
        f"dimension-seeded {summary['dimension_seeded_views']}, "
        f"line-art-only {summary['line_art_only_views']}, "
        f"embedded-image {summary['embedded_image_views']}"
    )
    print(
        f"Collision-avoiding views: {summary['collision_avoiding_views']}; "
        f"polygon overlap pixels: {summary['polygon_overlap_pixels']}; "
        f"foreign dimension texts: {summary['foreign_dimension_text_collisions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
