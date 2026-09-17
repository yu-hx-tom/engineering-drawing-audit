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
    r"^(?:SECTION\s+|DETAIL\s+|VIEW\s+)?(?P<label>[A-Z]{1,3}|[0-9]{1,2})\s*[-–—]\s*(?P=label)$",
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
            + record.get("_view_extension_ids", [])
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


def _local_obstacle_polygon(
    box: Sequence[float], obstacles: Sequence[Sequence[float]],
    protected_boxes: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Cut only edge collisions, never shrink an obstacle-free band to its ink.

    Step edges (not interpolated/smoothed vertices) keep thin vector strokes
    inside their protected rectangles. Interior collisions use explicit holes.
    """
    obstacles = [o for o in obstacles if bbox_intersection(box, o) > 0]
    cuts = sorted({box[1], box[3], *(max(box[1], min(box[3], y)) for o in obstacles for y in (o[1], o[3]))})
    left_profile, right_profile = [], []
    for y0, y1 in zip(cuts, cuts[1:]):
        left, right = box[0], box[2]
        for obstacle in obstacles:
            if obstacle[1] >= y1 or obstacle[3] <= y0:
                continue
            protected = [p for p in protected_boxes if p[1] < y1 and p[3] > y0]
            # A full-width obstacle (typically a title block at the bottom)
            # can remove a blank band, but cannot erase a protected annotation.
            if obstacle[0] <= left and obstacle[2] >= right:
                if not protected:
                    left = right
                continue
            if obstacle[0] <= box[0] < obstacle[2]:
                limit = min((p[0] for p in protected), default=right)
                left = max(left, min(obstacle[2], limit))
            if obstacle[0] < box[2] <= obstacle[2]:
                limit = max((p[2] for p in protected), default=left)
                right = min(right, max(obstacle[0], limit))
        if left >= right:
            continue
        left_profile.extend([[left, y0], [left, y1]])
        right_profile.extend([[right, y0], [right, y1]])
    polygon = left_profile + list(reversed(right_profile))
    if not polygon:
        return _rectangle_polygon(box)
    # Remove only duplicate / collinear vertices. No error-tolerant smoothing.
    compact = []
    for p in polygon:
        p = [rounded(v) for v in p]
        if not compact or p != compact[-1]:
            compact.append(p)
    changed = True
    while changed and len(compact) > 4:
        changed = False
        for i in range(len(compact)):
            a, b, c = compact[i-1], compact[i], compact[(i+1) % len(compact)]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                compact.pop(i)
                changed = True
                break
    return compact


def _owned_segments(records: Sequence[dict[str, Any]], geometry: dimension_ledger.PageGeometry) -> list[dict[str, Any]]:
    by_id = {s['id']: s for s in geometry.segments}
    arrows = {a['id']: a for a in geometry.arrows}
    owned = {}
    for record in records:
        evidence = record.get('geometry', {})
        ids = set(evidence.get('line_segment_ids', []) + evidence.get('extension_segment_ids', [])
                  + record.get('_view_extension_ids', []))
        for arrow_id in evidence.get('arrow_ids', []):
            ids.update(arrows.get(arrow_id, {}).get('segment_ids', []))
        color = record.get('_text_color') or 0
        rgb = np.array([(color >> shift) & 255 for shift in (16, 8, 0)])
        for segment_id in sorted(ids):
            segment = by_id.get(segment_id)
            if not segment:
                continue
            if np.ptp(rgb) >= 60 and (segment.get('color') is None or np.max(np.abs(np.array(segment['color'])*255-rgb)) > 60):
                continue
            owned[segment_id] = {k: segment[k] for k in ('id', 'p1', 'p2', 'width', 'color')}
    return list(owned.values())


def _body_protection_boxes(
    region: dict[str, Any], regions: list[dict[str, Any]], body_mask: np.ndarray | None,
    scale: float, page_rect: Sequence[float],
) -> list[list[float]]:
    """Protect the actual silhouette, not every blank corner of its bounding box."""
    box = region['core_bbox']
    if body_mask is None or region.get('detection_basis') == 'embedded_image':
        return [expand_bbox(box, 2, page_rect)]
    x0, y0 = max(0, int(box[0]*scale)), max(0, int(box[1]*scale))
    x1, y1 = min(body_mask.shape[1], math.ceil(box[2]*scale)), min(body_mask.shape[0], math.ceil(box[3]*scale))
    ys, xs = np.nonzero(body_mask[y0:y1, x0:x1])
    xs, ys = (xs+x0)/scale, (ys+y0)/scale
    def score(b):
        dx = np.maximum(np.maximum(b[0]-xs, xs-b[2]), 0)
        dy = np.maximum(np.maximum(b[1]-ys, ys-b[3]), 0)
        cx, cy = bbox_center(b)
        return dx*dx+dy*dy+0.015*((xs-cx)**2+(ys-cy)**2)
    own_score = score(box)
    keep = np.ones(len(xs), dtype=bool)
    for other in regions:
        if other is not region and bbox_intersection(box, other['core_bbox']) > 0:
            keep &= own_score <= score(other['core_bbox'])
    xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return [expand_bbox(box, 2, page_rect)]
    bands = np.floor(ys/4).astype(int)
    return [expand_bbox([float(xs[bands==band].min()), float(ys[bands==band].min()),
                         float(xs[bands==band].max()), float(ys[bands==band].max())], 2, page_rect)
            for band in np.unique(bands)]


def _context_arrow_segments(boxes: Sequence[Sequence[float]], geometry: dimension_ledger.PageGeometry) -> list[dict[str, Any]]:
    by_id={s['id']:s for s in geometry.segments}
    result={}
    for box in boxes:
        candidates=[]
        for arrow in geometry.arrows:
            distance=bbox_gap(box,arrow['bbox'])
            segments=[by_id[i] for i in arrow.get('segment_ids',[]) if i in by_id]
            if distance>24 or not segments: continue
            if not all(max(s.get('color') or s.get('fill') or (0,0,0))<0.25 for s in segments): continue
            candidates.append((distance,arrow['id'],segments))
        if candidates:
            for s in min(candidates,key=lambda item:(item[0],item[1]))[2]:
                result[s['id']]={k:s[k] for k in ('id','p1','p2','width','color')}
    return list(result.values())


def _compact_owned_polygon(
    region: dict[str, Any], shape: tuple[int, int], scale: float,
) -> list[list[float]]:
    """Compact envelope of owned bodies, labels and *complete* vector corridors.

    Unlike the first iteration, spans are built from protected geometry, not a
    pruned ink mask. Smoothing only expands spans; it never erases a thin line.
    """
    image = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    for box in [*region['body_protection_boxes'], *region['dimension_protection_boxes'], *region.get('label_boxes', []), *region.get('context_text_boxes', [])]:
        draw.rectangle(tuple(v*scale for v in box), fill=255)
    for segment in region['geometry_segments']:
        draw.line([tuple(v*scale for v in p) for p in (segment['p1'], segment['p2'])],
                  fill=255, width=max(3, math.ceil((segment['width']+2)*scale)))
    mask = _dilate(np.asarray(image)>0, max(2, math.ceil(4*scale)))
    ys,xs = np.nonzero(mask)
    if not len(xs):
        return _rectangle_polygon(region['provisional_bbox'])
    band = max(2, round(5*scale))
    start, stop = int(ys.min()//band), int(ys.max()//band)+1
    spans = {}
    for i in range(start, stop):
        _, bx = np.nonzero(mask[i*band:min((i+1)*band, shape[0])])
        if len(bx): spans[i] = (float(bx.min()), float(bx.max()+1))
    occupied = sorted(spans)
    for a,b in zip(occupied, occupied[1:]):
        for i in range(a+1,b):
            fraction=(i-a)/(b-a)
            spans[i] = tuple(spans[a][j]*(1-fraction)+spans[b][j]*fraction for j in (0,1))
    expanded = {i: (min(spans[j][0] for j in (i-1,i,i+1) if j in spans),
                    max(spans[j][1] for j in (i-1,i,i+1) if j in spans)) for i in spans}
    left,right = [],[]
    for i in sorted(expanded):
        x0,x1=expanded[i]; y0,y1=i*band,min((i+1)*band,shape[0])
        left.extend([[rounded(x0/scale),rounded(y0/scale)],[rounded(x0/scale),rounded(y1/scale)]])
        right.extend([[rounded(x1/scale),rounded(y0/scale)],[rounded(x1/scale),rounded(y1/scale)]])
    # Only remove redundant collinear vertices, not approximate a line inward.
    polygon=left+list(reversed(right))
    compact=[]
    for point in polygon:
        if compact and point==compact[-1]: continue
        while len(compact)>=2 and ((compact[-2][0]==compact[-1][0]==point[0]) or
                                   (compact[-2][1]==compact[-1][1]==point[1])):
            compact.pop()
        compact.append(point)
    return compact


def _clipped_dashed_structure(
    region: dict[str, Any], regions: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry, polygon: Sequence[Sequence[float]],
    shape: tuple[int, int], scale: float,
) -> list[dict[str, Any]]:
    """Recover a styled structural line cut only in its middle by an inward bay.

    Both endpoints must already be inside the crop and the complete segment
    must lie uniquely in this view's core. This does not grow outer extents or
    fill arbitrary whitespace notches where a neighboring view may belong.
    """
    mask = _rasterize_polygon(polygon, shape, scale)
    box = region['core_bbox']
    existing = {s['id'] for s in region['geometry_segments']}
    recovered = []
    for segment in geometry.segments:
        if segment['id'] in existing or segment['length'] < max(20, min(box[2]-box[0], box[3]-box[1])*0.12):
            continue
        pattern = re.search(r'\[([^]]*)\]', segment.get('dashes') or '')
        lengths = re.findall(r'\d+(?:\.\d+)?', pattern.group(1)) if pattern else []
        if len(lengths) < 2 or not any(float(value)>0 for value in lengths):
            continue
        p1,p2 = segment['p1'],segment['p2']
        if any(point_rect_distance(p,box)>0 for p in (p1,p2)):
            continue
        count = max(3, math.ceil(segment['length']*scale)+1)
        points = np.linspace(p1,p2,count)
        if any(np.any((points[:,0]>=other['core_bbox'][0]) & (points[:,0]<=other['core_bbox'][2])
                      & (points[:,1]>=other['core_bbox'][1]) & (points[:,1]<=other['core_bbox'][3]))
               for other in regions if other is not region):
            continue
        xs = np.clip(np.rint(points[:,0]*scale).astype(int),0,shape[1]-1)
        ys = np.clip(np.rint(points[:,1]*scale).astype(int),0,shape[0]-1)
        covered = mask[ys,xs]
        if covered[0] and covered[-1] and not np.all(covered):
            recovered.append({key:segment[key] for key in ('id','p1','p2','width','color')})
    return recovered


def _fit_collision_polygons(
    content_mask: np.ndarray,
    regions: list[dict[str, Any]],
    dimensions: Sequence[dict[str, Any]],
    geometry: dimension_ledger.PageGeometry,
    page_data: dict[str, Any],
    scale: float,
    excluded_boxes: Sequence[Sequence[float]] = (),
    body_mask: np.ndarray | None = None,
    page_tokens: Sequence[dict[str, Any]] = (),
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
    for region in regions:
        region['context_text_boxes'] = []
    for token in page_tokens:
        # Standalone cutting-plane letters belong to the nearby source view,
        # not to a neighboring crop whose rectangular extent happens to reach it.
        if not re.fullmatch(r'[A-ZА-Я]{1,3}', token.get('normalized_text','').strip()):
            continue
        box = token['bbox']
        if box[0]<24 or box[1]<24 or box[2]>page_rect[2]-24 or box[3]>page_rect[3]-24:
            continue
        if any(bbox_intersection(box, obstacle)>0 for obstacle in fixed_obstacles):
            continue
        choices = [(bbox_gap(box, region['core_bbox']), index) for index,region in enumerate(regions)]
        if choices and min(choices)[0] <= 12:
            regions[min(choices)[1]]['context_text_boxes'].append(box)
    polygon_masks: list[np.ndarray] = []
    for region_index, region in enumerate(regions):
        records = [dimensions_by_id[value] for value in region["dimension_ids"] if value in dimensions_by_id]
        provisional = region["provisional_bbox"]
        region["dimension_text_boxes"] = [list(record["bbox"]) for record in records]
        region["dimension_text_colors"] = [record.get("_text_color") for record in records]
        region["dimension_protection_boxes"] = _dimension_protection_boxes(records, page_rect)
        owned_segments = _owned_segments(records, geometry)
        context_segments=_context_arrow_segments(region.get('context_text_boxes',[]),geometry)
        known_ids={s['id'] for s in owned_segments}
        owned_segments.extend(s for s in context_segments if s['id'] not in known_ids)
        region["geometry_segments"] = owned_segments
        geometry_boxes = [
            expand_bbox([min(s['p1'][0], s['p2'][0]), min(s['p1'][1], s['p2'][1]),
                         max(s['p1'][0], s['p2'][0]), max(s['p1'][1], s['p2'][1])],
                        max(2.0, s['width'] + 1.0), page_rect)
            for s in owned_segments
        ]
        region["geometry_protection_boxes"] = geometry_boxes
        region['body_protection_boxes'] = _body_protection_boxes(region, regions, body_mask, scale, page_rect)
        foreign_text_obstacles = [
            expand_bbox(dimensions_by_id[dimension_id]["bbox"], 3.0, page_rect)
            for other_index, other in enumerate(regions) if other_index != region_index
            for dimension_id in other["dimension_ids"] if dimension_id in dimensions_by_id
            and bbox_intersection(provisional, dimensions_by_id[dimension_id]["bbox"]) > 0
        ]
        foreign_view_obstacles = [
            expand_bbox(box, 3, page_rect)
            for other in regions if other is not region
            for box in [other['core_bbox'], *other.get('label_boxes', [])]
            if bbox_intersection(provisional, box) > 0
        ]
        fixed_collisions = [
            obstacle for obstacle in fixed_obstacles
            if bbox_intersection(provisional, obstacle) > 0
            and not any(bbox_intersection(record["bbox"], obstacle) > 0 for record in records)
        ]
        excluded_collisions = [
            obstacle for obstacle in excluded_boxes
            if region.get("detection_basis") != "embedded_image"
            and bbox_intersection(provisional, obstacle) > 0
        ]
        polygon = _compact_owned_polygon(region, content_mask.shape, scale)
        recovered = _clipped_dashed_structure(region, regions, geometry, polygon, content_mask.shape, scale)
        region['recovered_dashed_structure_ids'] = [segment['id'] for segment in recovered]
        if recovered:
            region['geometry_segments'].extend(recovered)
            polygon = _compact_owned_polygon(region, content_mask.shape, scale)
        polygon_box = _polygon_bbox(polygon)
        region["polygon"] = polygon
        region["bbox"] = [rounded(v) for v in polygon_box]
        region["polygon_area"] = rounded(_polygon_area(polygon))
        region["bbox_area_reduction_pct"] = rounded(
            max(0.0, 100.0 * (1.0 - region["polygon_area"] / max(bbox_area(provisional), 1.0))), 2)
        region["fit_applied"] = polygon != _rectangle_polygon(provisional)
        region["fit_reasons"] = [
            label for label, values in (
                ("foreign_dimension_text", foreign_text_obstacles),
                ("neighbor_view", foreign_view_obstacles),
                ("sheet_furniture", fixed_collisions), ("excluded_3d_model", excluded_collisions)
            ) if values
        ]
        region["excluded_model_boxes"] = [list(box) for box in excluded_collisions]
        region["fixed_exclusion_boxes"] = [list(box) for box in fixed_collisions]
        region['foreign_label_boxes'] = [
            list(box) for other in regions if other is not region for box in other.get('label_boxes', [])
            if not any(bbox_intersection(box, own_box) > 0 for own_box in region['dimension_protection_boxes'])
        ]
        polygon_masks.append(_rasterize_polygon(polygon, content_mask.shape, scale))
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
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False, annots=False
    )
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    mask = np.min(pixels, axis=2) < 245
    # Colored dimensions / datum frames / roughness marks are not view bodies.
    # Enable this only when the ledger demonstrates a colored-annotation drawing;
    # monochrome drawings retain the existing geometry-based path.
    token_colors = {token['id']: token.get('color') or 0 for token in page_tokens}
    colors = [token_colors.get(record.get('root_token_id'), 0) for record in page_dimensions]
    chromatic = sum(
        max((color >> 16) & 255, (color >> 8) & 255, color & 255)
        - min((color >> 16) & 255, (color >> 8) & 255, color & 255) >= 60
        for color in colors if color is not None
    )
    neutral = (np.max(pixels, axis=2) - np.min(pixels, axis=2) <= 25) & (np.max(pixels, axis=2) < 220)
    color_separated = chromatic >= 3 and chromatic >= len(colors) * 0.6 and np.count_nonzero(neutral) >= 200
    if color_separated:
        mask = neutral

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
        color = token.get('color') or 0
        channels = [(color >> shift) & 255 for shift in (16, 8, 0)]
        if color_separated and max(channels) - min(channels) >= 60:
            # The colored ink is already absent. Erasing its whole (possibly
            # rotated) text box would sever black part outlines underneath it.
            continue
        _mask_rect(mask, token["bbox"], scale, padding=max(1.5, token.get("size", 8.0) * 0.12))
    _remove_page_frames(mask, geometry, page_rect, scale)
    if not color_separated:
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
    geometry: dimension_ledger.PageGeometry | None = None,
) -> None:
    maximum_distance = max(34.0, min(page_rect[2], page_rect[3]) * 0.055)
    page_area = bbox_area(page_rect)
    # Extension lines locate the part even when arrows are far outside its body.
    segments = [s for s in geometry.segments if s['length'] >= 12] if geometry else []
    starts = np.asarray([s['p1'] for s in segments], dtype=float).reshape(-1, 2)
    ends = np.asarray([s['p2'] for s in segments], dtype=float).reshape(-1, 2)
    vectors = ends-starts
    lengths_sq = np.sum(vectors*vectors, axis=1)
    by_id = {s['id']: s for s in geometry.segments if 'id' in s} if geometry else {}
    for record in dimensions:
        points = _dimension_points(record)
        extension_candidates: set[int] = set()
        record['_view_extension_ids'] = []
        text_color = record.get('_text_color') or 0
        text_rgb = np.array([(text_color >> shift) & 255 for shift in (16, 8, 0)])
        colored_text = int(np.ptp(text_rgb)) >= 60
        linked = [by_id[i] for i in record.get('geometry', {}).get('line_segment_ids', []) if i in by_id]
        if colored_text and linked and not any(
            s.get('color') is not None and np.max(np.abs(np.asarray(s['color'])*255-text_rgb)) <= 60
            for s in linked
        ):
            # A blue radius label can be accidentally linked to a black section
            # arrow beside it. That arrow must not move the label to another view.
            points = []
        groups = [[point] for point in points]
        if len(points) == 2 and segments and record.get('type') != 'angle':
            direction = np.asarray(points[1])-np.asarray(points[0])
            span = float(np.linalg.norm(direction))
            if span > 1:
                across = np.abs(vectors @ (direction/span)) / np.sqrt(lengths_sq) < 0.5
                if colored_text:
                    across &= np.asarray([
                        s.get('color') is not None and np.max(np.abs(np.asarray(s['color'])*255-text_rgb)) <= 60
                        for s in segments
                    ])
                for group, point in zip(groups, points):
                    t = np.clip(np.sum((point-starts)*vectors, axis=1)/lengths_sq, 0, 1)
                    distances = np.linalg.norm(starts+t[:, None]*vectors-point, axis=1)
                    for index in np.flatnonzero(across & (distances <= 2.5)):
                        group.extend([tuple(starts[index]), tuple(ends[index])])
                        extension_candidates.add(int(index))
        choices = []
        for index, core in enumerate(cores):
            box = core["core_bbox"]
            text_distance = bbox_gap(record["bbox"], box)
            if points:
                density = core.get("dilated_pixels", bbox_area(box)) / max(bbox_area(box), 1.0)
                if density < 0.10 and bbox_area(box) > page_area * 0.02:
                    continue
                distances = [min(point_rect_distance(point, box) for point in group) for group in groups]
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
        owner_box = cores[best_index]['core_bbox']
        # Outer dimension chains can be far outside the part silhouette. The
        # old fixed page-distance gate dropped them after clean core detection.
        allowance = maximum_distance
        allowance = max(allowance, min(
            min(page_rect[2], page_rect[3]) * 0.22,
            math.hypot(owner_box[2]-owner_box[0], owner_box[3]-owner_box[1]) * 0.35,
        ))
        if best_distance <= allowance:
            cores[best_index]["assigned_dimensions"].append(record)
            record['_view_extension_ids'] = [
                segments[index]['id'] for index in sorted(extension_candidates)
                if 'id' in segments[index]
                and min(point_rect_distance(segments[index][end], owner_box) for end in ('p1', 'p2')) <= 8
            ]


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


def _assign_view_labels(
    cores: list[dict[str, Any]], tokens: Sequence[dict[str, Any]], page_rect: Sequence[float]
) -> None:
    """Respect both above-view headings with scales and below-view captions."""
    for core in cores:
        core['label_tokens'] = []
    for token in tokens:
        text = token.get("normalized_text", "").strip()
        if not VIEW_LABEL_RE.fullmatch(text):
            continue
        x, y = bbox_center(token['bbox'])
        scale_tokens = [other for other in tokens
                        if re.fullmatch(r'\d+\s*:\s*\d+', other.get('normalized_text', '').strip())
                        and abs(bbox_center(other['bbox'])[0]-x) <= 20
                        and 0 < bbox_center(other['bbox'])[1]-y <= max(30, token['bbox'][3]-token['bbox'][1])]
        choices = []
        for index, core in enumerate(cores):
            box = core['core_bbox']
            if not box[0] <= x <= box[2]:
                continue
            # Do not attach headings to components made from lettering/noise.
            if bbox_area(box) < bbox_area(page_rect) * 0.0035:
                continue
            below = abs(y-box[1]) < abs(y-box[3])
            gap = min(abs(y-box[1]), abs(y-box[3]))
            if box[1] < y < box[3] and gap > max(30, (box[3]-box[1])*0.12):
                continue
            if gap > min(page_rect[2], page_rect[3]) * 0.45:
                continue
            centering = abs(x-bbox_center(box)[0]) / max(box[2]-box[0], 1.0)
            choices.append((bool(scale_tokens) and not below, gap + 20*centering, index))
        if not choices:
            continue
        owner = cores[min(choices)[2]]
        owner['label_tokens'].append(token)
        # Keep the scale immediately under a title with the same view.
        owner['label_tokens'].extend(scale_tokens)


def _keep_core(core: dict[str, Any], page_area: float, scale: float) -> bool:
    assigned = core['assigned_dimensions']
    box = core['core_bbox']
    area = bbox_area(box)
    density = core['dilated_pixels'] / max(area * scale * scale, 1.0)
    if not assigned and area < page_area * 0.0035:
        return False
    if not assigned and core.get('source_kind') == 'line_art' and density < 0.30:
        width, height = box[2]-box[0], box[3]-box[1]
        if not (area >= page_area * 0.08 and max(width, height) / max(min(width, height), 0.01) <= 2.5 and density >= 0.12):
            return False
    if len(assigned) <= 1 and area < page_area * 0.004:
        return False
    return not (assigned and area < page_area * 0.01
                and all(record.get('type') == 'geometric_tolerance' for record in assigned))


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
            _assign_dimensions(cores, dimensions, page_rect, geometry)
            cores = _suppress_embedded_duplicates(cores)
            cores = _merge_regions(cores)
            cores = [core for core in cores if _keep_core(core, page_area, scale)]
            # Small rejected fragments used to take their dimensions with them.
            # Reassign against retained bodies only, never silently discard IDs.
            for core in cores:
                core['assigned_dimensions'] = []
            _assign_dimensions(cores, dimensions, page_rect, geometry)
            _assign_view_labels(cores, tokens, page_rect)

            regions = []
            for core in cores:
                assigned = core["assigned_dimensions"]
                core_area = bbox_area(core["core_bbox"])
                source_kind = core.get("source_kind", "line_art")
                line_art_density = core["dilated_pixels"] / max(core_area * scale * scale, 1.0)
                sort_box = expand_bbox(
                    bbox_union(
                        [core["core_bbox"]] + [record["bbox"] for record in assigned]
                    ),
                    max(5.0, min(page_rect[2], page_rect[3]) * 0.008),
                    page_rect,
                )
                label_tokens = core.get('label_tokens', [])
                label_boxes = [token['bbox'] for token in label_tokens]
                evidence_boxes = [core["core_bbox"], *label_boxes] + _dimension_evidence_boxes(
                    assigned, geometry
                )
                provisional_box = expand_bbox(
                    bbox_union(evidence_boxes),
                    max(5.0, min(page_rect[2], page_rect[3]) * 0.008),
                    page_rect,
                )
                labels = [token['normalized_text'] for token in label_tokens
                          if VIEW_LABEL_RE.fullmatch(token['normalized_text'])]
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
                        "label_boxes": label_boxes,
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
                body_mask=mask,
                page_tokens=tokens,
            )
            assigned_ids = {dimension_id for region in regions for dimension_id in region['dimension_ids']}
            unassigned = [record for record in dimensions if record['id'] not in assigned_ids]
            output_pages.append(
                {
                    "page": page_number,
                    "rect": page_rect,
                    "view_count": len(regions),
                    "excluded_embedded_3d_models": len(all_image_cores) - len(image_cores),
                    "views": regions,
                    "unassigned_dimension_ids": sorted(record['id'] for record in unassigned),
                    "unassigned_dimensions": [
                        {'id': record['id'], 'bbox': record['bbox'], 'text': record['raw_text'],
                         'reason': 'no_retained_core_within_assignment_gate'} for record in unassigned
                    ],
                }
            )

    return {
        "schema_version": "0.5",
        "source": str(input_path.resolve()),
        "dimension_ledger_source": ledger.get("source"),
        "parameters": {
            "render_scale": scale,
            "dilation_pt": dilation_pt,
            "region_shape": "compact_owned_geometry_envelope",
            "embedded_3d_models": "excluded",
        },
        "summary": {
            "pages": len(output_pages),
            "views": sum(page["view_count"] for page in output_pages),
            "unassigned_dimensions": sum(len(page['unassigned_dimension_ids']) for page in output_pages),
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
                        dashes="[2 2] 0",
                        overlay=True,
                    )
                label = (
                    f"{view['id']}  {view['dimension_count']} dims  "
                    + ' / '.join(view.get('labels', []))
                )
                page.insert_text(
                    (rect.x0 + 2, max(page.rect.y0 + 9, rect.y0 - 3)),
                    label,
                    fontsize=8,
                    color=color,
                    overlay=True,
                )
            for record in page_data.get('unassigned_dimensions', []):
                rect = fitz.Rect(record['bbox'])
                page.draw_rect(rect, color=(1, 0.45, 0), width=1, dashes='[3 2] 0', overlay=True)
                page.insert_text((rect.x0, max(8, rect.y0-2)), 'UNASSIGNED '+record['id'],
                                 fontsize=6, color=(0.8, 0.3, 0), overlay=True)
        document.save(output_path, garbage=3, deflate=True)


def _owned_vector_ink_mask(
    view: dict[str, Any], pixels: np.ndarray, origin: tuple[int, int], scale: float,
) -> np.ndarray:
    """Actual source ink along owned dimension / extension / arrow segments.

    Match the segment color blended with white, including antialiasing. Do not
    restore an unrelated model's black pixels along a blue dimension corridor.
    """
    mask = np.zeros(pixels.shape[:2], dtype=bool)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for segment in view.get('geometry_segments', []):
        color = tuple(segment.get('color') or (0, 0, 0))
        groups.setdefault(color, []).append(segment)
    for color, segments in groups.items():
        image = Image.new('L', (pixels.shape[1], pixels.shape[0]), 0)
        draw = ImageDraw.Draw(image)
        for segment in segments:
            points = [(p[0]*scale-origin[0], p[1]*scale-origin[1]) for p in (segment['p1'], segment['p2'])]
            draw.line(points, fill=255, width=max(3, int(math.ceil((segment['width']+2)*scale))))
        corridor = np.asarray(image) > 0
        ys, xs = np.nonzero(corridor)
        if not len(xs):
            continue
        rgb = np.asarray(color)*255.0
        direction = 255.0-rgb
        denominator = float(direction @ direction)
        if denominator < 1:
            continue
        source = pixels[ys, xs, :3].astype(float)
        amount = ((255-source) @ direction)/denominator
        residual = np.max(np.abs(source-(255-amount[:, None]*direction)), axis=1)
        keep = (amount >= 0.025) & (amount <= 1.05) & (residual <= 32)
        mask[ys[keep], xs[keep]] = True
    return mask


def _view_claim_ink(view: dict[str, Any], pixels: np.ndarray, scale: float) -> np.ndarray:
    """Source ink owned by a view; shared pixels are explicitly not exclusive."""
    ink = np.min(pixels,axis=2)<248
    body_image=Image.new('L',(pixels.shape[1],pixels.shape[0]),0)
    draw=ImageDraw.Draw(body_image)
    for box in view.get('body_protection_boxes', []):
        draw.rectangle(tuple(v*scale for v in box),fill=255)
    body=np.asarray(body_image)>0
    colors=[v for v in view.get('dimension_text_colors',[]) if v is not None]
    chromatic=sum(max((c>>16)&255,(c>>8)&255,c&255)-min((c>>16)&255,(c>>8)&255,c&255)>=60 for c in colors)
    if len(colors)>=3 and chromatic>=len(colors)*0.6:
        body &= (np.max(pixels,axis=2)-np.min(pixels,axis=2)<=35)
    claim=body & ink
    text_image=Image.new('L',(pixels.shape[1],pixels.shape[0]),0)
    draw=ImageDraw.Draw(text_image)
    for box in [*view.get('dimension_text_boxes', []), *view.get('label_boxes', []), *view.get('context_text_boxes', [])]:
        draw.rectangle(tuple(v*scale for v in box),fill=255)
    claim |= (np.asarray(text_image)>0) & ink
    claim |= _owned_vector_ink_mask(view,pixels,(0,0),scale)
    return claim


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
            claims={view['id']: _view_claim_ink(view,page_pixels,render_scale) for view in page_data['views']}
            claim_counts=np.zeros(page_ink.shape,dtype=np.uint16)
            for claim in claims.values(): claim_counts += claim
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
                for model_box in view.get("excluded_model_boxes", []) + view.get('fixed_exclusion_boxes', []) + view.get('foreign_label_boxes', []):
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
                            dtype=np.int32,
                        )
                        dominant = int(np.argmax(target))
                        if target.max() >= 180 and target.max() - np.partition(target, -2)[-2] >= 80:
                            others = [index for index in range(3) if index != dominant]
                            color_match = (
                                source_pixels[:, :, dominant] >= 180
                            ) & (
                                source_pixels[:, :, dominant].astype(np.int32)
                                >= np.maximum(
                                    source_pixels[:, :, others[0]].astype(np.int32),
                                    source_pixels[:, :, others[1]].astype(np.int32),
                                )
                                + 18
                            )
                        elif target.max() <= 80:
                            color_match = np.max(source_pixels, axis=2) <= 150
                        else:
                            delta = source_pixels.astype(np.int32) - target
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
                # The last pass protects geometry as well as text. A whitespace
                # notch or a model's broad image rectangle must not erase lines.
                crop_pixels = page_pixels[crop_px[1]:crop_px[3], crop_px[0]:crop_px[2]]
                vector_ink = _owned_vector_ink_mask(view, crop_pixels, crop_px[:2], render_scale)
                restored = int(np.count_nonzero(vector_ink & (alpha_pixels == 0)))
                alpha_pixels[vector_ink] = 255
                own_claim=claims[view['id']]
                exclusive_foreign=(claim_counts>0) & ~own_claim
                foreign_crop=exclusive_foreign[crop_px[1]:crop_px[3],crop_px[0]:crop_px[2]]
                foreign_removed=int(np.count_nonzero(foreign_crop & (alpha_pixels>0)))
                alpha_pixels[foreign_crop]=0
                view['foreign_ink_validation']={
                    'removed_exclusive_foreign_ink_pixels':foreign_removed,
                    'remaining_exclusive_foreign_ink_pixels':int(np.count_nonzero(foreign_crop & (alpha_pixels>0))),
                    'scope':'neighbor bodies, linked dimension geometry, text and captions; shared/unknown ink remains for review',
                }
                view['geometry_crop_validation'] = {
                    'owned_vector_ink_pixels': int(np.count_nonzero(vector_ink)),
                    'restored_vector_ink_pixels': restored,
                    'scope': 'linked dimension lines, recovered extensions and arrowheads; not all view geometry',
                }
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
        write_review_pdf(input_path, output_dir / "view-review.pdf", result)
        write_polygon_crops(input_path, output_dir / "crops", result)
        (output_dir / "view-regions.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    if summary['unassigned_dimensions']:
        print(f"WARNING: {summary['unassigned_dimensions']} dimension records are unassigned; "
              "see orange boxes in view-review.pdf and unassigned_dimensions in the JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
