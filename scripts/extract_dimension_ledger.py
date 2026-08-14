#!/usr/bin/env python3
"""Extract a traceable engineering-dimension ledger from a vector PDF.

The tool is deliberately local and deterministic: it reads the PDF text layer,
groups split tolerance fragments in the text's local coordinate system, parses
common engineering-dimension forms, and uses PDF vector paths as supporting
evidence for dimension lines, arrowheads, extension lines, and leaders.

It does not OCR scanned drawings and never silently drops uncertain numeric
annotations. Ambiguous candidates are written to ``needs-review.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24 compatibility
    import fitz  # type: ignore[no-redef]


NUMBER_RE = r"(?:\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d+(?:[.,]\d+)?|[.,]\d+)"
SIGNED_TOL_RE = re.compile(rf"^[+\-±]\s*{NUMBER_RE}\s*[°'\"]?$")
PURE_NUMBER_RE = re.compile(rf"^{NUMBER_RE}$")
NUMERIC_FRAGMENT_RE = re.compile(rf"^({NUMBER_RE})\s*([°'\"]?)$")
QUANTITY_PREFIX_RE = re.compile(r"^\(?\s*\d+\s*[-×xX]\s*\)?$")
METADATA_KEYWORDS = (
    "SCALE", "DATE", "DRAWING", "DWG", "SHEET", "REV", "REVISION",
    "MATERIAL", "WEIGHT", "CHECKED", "APPROVED", "TITLE", "PART NO",
    "比例", "日期", "图号", "图纸", "页码", "材料", "重量", "审核", "批准",
    "МАСШТАБ", "ДАТА", "ЛИСТ", "МАТЕРИАЛ", "МАССА",
)
SYMBOL_ONLY_RE = re.compile(
    r"^(?:Ø|R|SR|SØ|M|C|□|⌴|↧|°|′|″|\"|'|\(|\)|\[|\]|THRU|DEPTH|EQ|TYP)\.?$",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"^(?:mm|cm|m|in|inch|inches|\"|″)$", re.IGNORECASE)
FIT_RE = re.compile(r"^(?:[A-Za-z]{1,2}\d{1,2}|[A-Za-z]\d/[A-Za-z]\d)$")
ROUGHNESS_TEXT_RE = re.compile(
    rf"^(?P<parameter>Ra|Rz|Rt|Rq|Rmax)\s*(?P<value>{NUMBER_RE})$",
    re.IGNORECASE,
)
TECHNICAL_NOTE_HEADING_RE = re.compile(
    r"(?:技术要求|铸造要求|TECHNICAL\s+REQUIREMENTS?|ТЕХНИЧЕСКИЕ\s+ТРЕБОВАНИЯ)",
    re.IGNORECASE,
)
COMPLETE_PREFIXED_DIMENSION_RE = re.compile(
    rf"^(?:SR|S脴|脴|R|M|C)\s*{NUMBER_RE}(?:\s*[掳'\"])?$",
    re.IGNORECASE,
)


def rounded(value: float, digits: int = 3) -> float:
    value = round(float(value), digits)
    return 0.0 if value == -0.0 else value


def point(value: Any) -> tuple[float, float]:
    return float(value.x), float(value.y)


def vec_sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def vec_add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def vec_scale(a: Sequence[float], scale: float) -> tuple[float, float]:
    return a[0] * scale, a[1] * scale


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def length(a: Sequence[float]) -> float:
    return math.hypot(a[0], a[1])


def unit(a: Sequence[float]) -> tuple[float, float]:
    size = length(a)
    return (a[0] / size, a[1] / size) if size else (1.0, 0.0)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return length(vec_sub(a, b))


def normalize_180(angle: float) -> float:
    return angle % 180.0


def angle_difference(a: float, b: float) -> float:
    diff = abs(normalize_180(a) - normalize_180(b))
    return min(diff, 180.0 - diff)


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def bbox_union(boxes: Iterable[Sequence[float]]) -> list[float]:
    values = list(boxes)
    return [
        rounded(min(box[0] for box in values)),
        rounded(min(box[1] for box in values)),
        rounded(max(box[2] for box in values)),
        rounded(max(box[3] for box in values)),
    ]


def point_rect_distance(p: Sequence[float], box: Sequence[float]) -> float:
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def point_segment_distance(
    p: Sequence[float], a: Sequence[float], b: Sequence[float]
) -> float:
    ab = vec_sub(b, a)
    denom = dot(ab, ab)
    if not denom:
        return distance(p, a)
    t = max(0.0, min(1.0, dot(vec_sub(p, a), ab) / denom))
    projection = vec_add(a, vec_scale(ab, t))
    return distance(p, projection)


def local_range(bbox: Sequence[float], axis: Sequence[float]) -> tuple[float, float]:
    corners = (
        (bbox[0], bbox[1]), (bbox[2], bbox[1]),
        (bbox[2], bbox[3]), (bbox[0], bbox[3]),
    )
    values = [dot(corner, axis) for corner in corners]
    return min(values), max(values)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    translations = str.maketrans(
        {
            "∅": "Ø", "⌀": "Ø", "Φ": "Ø", "φ": "Ø", "Ф": "Ø",
            "−": "-", "–": "-", "—": "-", "﹣": "-",
            "＋": "+", "×": "×", "º": "°", "˚": "°",
            "“": '"', "”": '"', "″": '"', "′": "'",
        }
    )
    text = text.translate(translations)
    text = re.sub(r"\+\s*/\s*-", "±", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(text: str) -> float | None:
    value = normalize_text(text).replace(",", ".")
    mixed = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", value)
    if mixed and int(mixed.group(3)):
        return int(mixed.group(1)) + int(mixed.group(2)) / int(mixed.group(3))
    fraction = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if fraction and int(fraction.group(2)):
        return int(fraction.group(1)) / int(fraction.group(2))
    try:
        return float(value)
    except ValueError:
        return None


def first_number(text: str) -> tuple[str, float] | None:
    match = re.search(NUMBER_RE, text)
    if not match:
        return None
    value = parse_number(match.group(0))
    return (match.group(0), value) if value is not None else None


def direction_angle(direction: Sequence[float]) -> float:
    # PDF page coordinates point down; return conventional counter-clockwise degrees.
    value = (-math.degrees(math.atan2(direction[1], direction[0]))) % 360.0
    for cardinal in (0.0, 90.0, 180.0, 270.0, 360.0):
        if abs(value - cardinal) < 0.05:
            return 0.0 if cardinal == 360.0 else cardinal
    return rounded(value, 2)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PageGeometry:
    segments: list[dict[str, Any]]
    arrows: list[dict[str, Any]]
    adjacency: dict[str, set[str]]
    symbols: list[dict[str, Any]]


def extract_text_tokens(page: Any, page_number: int) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    data = page.get_text("dict")
    sequence = 0
    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            direction = tuple(float(value) for value in line.get("dir", (1.0, 0.0)))
            if length(direction) < 0.5:
                direction = (1.0, 0.0)
            direction = unit(direction)
            for span_index, span in enumerate(spans):
                raw = span.get("text", "")
                text = raw.strip()
                if not text:
                    continue
                sequence += 1
                bbox = [rounded(value) for value in span["bbox"]]
                try:
                    recovered = fitz.recover_quad(direction, span)
                    quad = [
                        [rounded(recovered.ul.x), rounded(recovered.ul.y)],
                        [rounded(recovered.ur.x), rounded(recovered.ur.y)],
                        [rounded(recovered.lr.x), rounded(recovered.lr.y)],
                        [rounded(recovered.ll.x), rounded(recovered.ll.y)],
                    ]
                except Exception:
                    quad = [
                        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                    ]
                tokens.append(
                    {
                        "id": f"P{page_number}-T{sequence:04d}",
                        "page": page_number,
                        "block": block_index,
                        "line": line_index,
                        "span": span_index,
                        "text": text,
                        "normalized_text": normalize_text(text),
                        "line_text": line_text,
                        "bbox": bbox,
                        "quad": quad,
                        "origin": [rounded(value) for value in span.get("origin", bbox[:2])],
                        "font": span.get("font", ""),
                        "size": rounded(span.get("size", 0.0), 2),
                        "flags": span.get("flags", 0),
                        "color": span.get("color", 0),
                        "direction": [rounded(direction[0], 5), rounded(direction[1], 5)],
                        "rotation_deg": direction_angle(direction),
                    }
                )
    return tokens


def add_segment(
    segments: list[dict[str, Any]], page_number: int, path_index: int,
    p1: Sequence[float], p2: Sequence[float], drawing: dict[str, Any],
) -> None:
    segment_length = distance(p1, p2)
    if segment_length < 0.2:
        return
    segment_id = f"P{page_number}-S{len(segments) + 1:05d}"
    angle = normalize_180(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    segments.append(
        {
            "id": segment_id,
            "path_id": f"P{page_number}-V{path_index:04d}",
            "p1": [rounded(p1[0]), rounded(p1[1])],
            "p2": [rounded(p2[0]), rounded(p2[1])],
            "length": rounded(segment_length),
            "angle_deg": rounded(angle, 2),
            "width": rounded(drawing.get("width") or 0.0, 3),
            "color": drawing.get("color"),
            "fill": drawing.get("fill"),
            "dashes": drawing.get("dashes"),
        }
    )


def detect_vector_concentricity_symbol(
    drawing: dict[str, Any], page_number: int, path_index: int
) -> dict[str, Any] | None:
    """Recognize the two concentric outlined circles used for coaxiality.

    CAD exporters commonly flatten the glyph into two sequential closed
    polylines.  The small size, near-square bounds, two closed loops, common
    center, and nested diameters make this substantially safer than treating
    every pair of circles in the drawing as a GD&T characteristic.
    """
    rect = drawing.get("rect")
    line_items = [item for item in drawing.get("items", []) if item[0] == "l"]
    if rect is None or not 24 <= len(line_items) <= 240:
        return None
    width, height = float(rect.width), float(rect.height)
    if not 5.0 <= min(width, height) or max(width, height) > 28.0:
        return None
    if max(width, height) / min(width, height) > 1.25:
        return None

    lines = [(point(item[1]), point(item[2])) for item in line_items]
    loops: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
    current: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for line in lines:
        if current and distance(current[-1][1], line[0]) > 0.45:
            loops.append(current)
            current = []
        current.append(line)
    if current:
        loops.append(current)
    if len(loops) != 2 or any(len(loop) < 12 for loop in loops):
        return None
    if any(distance(loop[-1][1], loop[0][0]) > 0.45 for loop in loops):
        return None

    loop_boxes: list[list[float]] = []
    for loop in loops:
        vertices = [value for line in loop for value in line]
        loop_boxes.append(
            [
                min(value[0] for value in vertices),
                min(value[1] for value in vertices),
                max(value[0] for value in vertices),
                max(value[1] for value in vertices),
            ]
        )
    loop_boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    outer, inner = loop_boxes
    outer_center, inner_center = bbox_center(outer), bbox_center(inner)
    if distance(outer_center, inner_center) > min(width, height) * 0.08:
        return None
    outer_size = max(outer[2] - outer[0], outer[3] - outer[1])
    inner_size = max(inner[2] - inner[0], inner[3] - inner[1])
    if not 0.25 <= inner_size / outer_size <= 0.72:
        return None

    return {
        "id": f"P{page_number}-SYM{path_index:04d}",
        "path_id": f"P{page_number}-V{path_index:04d}",
        "kind": "coaxiality",
        "text": "\u25ce",
        "bbox": [rounded(rect.x0), rounded(rect.y0), rounded(rect.x1), rounded(rect.y1)],
        "center": [rounded(outer_center[0]), rounded(outer_center[1])],
        "rotation_deg": 0.0,
    }


def detect_vector_diameter_symbol(
    drawing: dict[str, Any], page_number: int, path_index: int
) -> dict[str, Any] | None:
    """Recognize an outlined diameter glyph exported as a small vector path.

    Some CAD PDF exporters preserve numbers as text but convert ``Ø`` into a
    polyline: one closed oval plus a diagonal stroke.  Requiring both features
    keeps ordinary circles, degree marks, arrowheads, and drawing geometry out.
    """
    rect = drawing.get("rect")
    line_items = [item for item in drawing.get("items", []) if item[0] == "l"]
    if rect is None or not 18 <= len(line_items) <= 100:
        return None
    lines = [(point(item[1]), point(item[2])) for item in line_items]
    vertices = [value for line in lines for value in line]
    glyph_box = [
        min(value[0] for value in vertices), min(value[1] for value in vertices),
        max(value[0] for value in vertices), max(value[1] for value in vertices),
    ]
    width, height = glyph_box[2] - glyph_box[0], glyph_box[3] - glyph_box[1]
    if not 3.0 <= min(width, height) or max(width, height) > 18.0:
        return None

    lengths = [distance(start, end) for start, end in lines]
    slash_index = max(range(len(lines)), key=lengths.__getitem__)
    slash_start, slash_end = lines[slash_index]
    slash_length = lengths[slash_index]
    # The slash becomes horizontal or vertical when the whole annotation is
    # rotated.  Its angle in page coordinates is therefore not a glyph test.
    if slash_length < max(width, height) * 0.95:
        return None
    center = bbox_center(glyph_box)
    if point_segment_distance(center, slash_start, slash_end) > min(width, height) * 0.28:
        return None

    oval = [line for index, line in enumerate(lines) if index != slash_index]
    connected = sum(
        distance(oval[index][1], oval[index + 1][0]) <= 0.35
        for index in range(len(oval) - 1)
    )
    closed = distance(oval[-1][1], oval[0][0]) <= 0.35
    if connected < len(oval) * 0.85 or not closed:
        return None

    rotation = 90.0 if width > height else 0.0
    return {
        "id": f"P{page_number}-SYM{path_index:04d}",
        "path_id": f"P{page_number}-V{path_index:04d}",
        "kind": "diameter",
        "text": "Ø",
        "bbox": [rounded(value) for value in glyph_box],
        "center": [rounded(center[0]), rounded(center[1])],
        "rotation_deg": rotation,
    }


def extract_vector_geometry(page: Any, page_number: int) -> PageGeometry:
    segments: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    path_vertices: dict[str, list[tuple[float, float]]] = {}
    path_drawings: dict[str, dict[str, Any]] = {}
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    for path_index, drawing in enumerate(drawings, start=1):
        path_id = f"P{page_number}-V{path_index:04d}"
        path_drawings[path_id] = drawing
        symbol = detect_vector_concentricity_symbol(drawing, page_number, path_index)
        if symbol is None:
            symbol = detect_vector_diameter_symbol(drawing, page_number, path_index)
        if symbol is not None:
            symbols.append(symbol)
            path_vertices[path_id] = []
            continue
        vertices: list[tuple[float, float]] = []
        for item in drawing.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = point(item[1]), point(item[2])
                add_segment(segments, page_number, path_index, p1, p2, drawing)
                vertices.extend((p1, p2))
            elif kind == "re":
                rect = item[1]
                corners = (
                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                    (rect.x1, rect.y1), (rect.x0, rect.y1),
                )
                for index in range(4):
                    add_segment(
                        segments, page_number, path_index,
                        corners[index], corners[(index + 1) % 4], drawing,
                    )
                vertices.extend(corners)
            elif kind == "qu":
                quad = item[1]
                corners = (point(quad.ul), point(quad.ur), point(quad.lr), point(quad.ll))
                for index in range(4):
                    add_segment(
                        segments, page_number, path_index,
                        corners[index], corners[(index + 1) % 4], drawing,
                    )
                vertices.extend(corners)
        path_vertices[path_id] = vertices

    arrows = detect_filled_arrows(path_vertices, path_drawings, page_number)
    arrows.extend(detect_open_arrows(segments, page_number))
    arrows = deduplicate_arrows(arrows)
    adjacency = build_segment_adjacency(segments)
    return PageGeometry(segments=segments, arrows=arrows, adjacency=adjacency, symbols=symbols)


def _frame_leader_evidence(
    frame: Sequence[float], frame_segment_ids: set[str], geometry: PageGeometry
) -> tuple[list[str], list[str]]:
    """Return vector segments and arrowheads connected to a control frame."""
    tolerance = 0.9

    def on_boundary(value: Sequence[float]) -> bool:
        x, y = value
        on_vertical = frame[1] - tolerance <= y <= frame[3] + tolerance and min(
            abs(x - frame[0]), abs(x - frame[2])
        ) <= tolerance
        on_horizontal = frame[0] - tolerance <= x <= frame[2] + tolerance and min(
            abs(y - frame[1]), abs(y - frame[3])
        ) <= tolerance
        return on_vertical or on_horizontal

    def outside(value: Sequence[float]) -> bool:
        return not (
            frame[0] - tolerance <= value[0] <= frame[2] + tolerance
            and frame[1] - tolerance <= value[1] <= frame[3] + tolerance
        )

    by_id = {segment["id"]: segment for segment in geometry.segments}
    seeds = [
        segment["id"]
        for segment in geometry.segments
        if segment["id"] not in frame_segment_ids
        and (
            (on_boundary(segment["p1"]) and outside(segment["p2"]))
            or (on_boundary(segment["p2"]) and outside(segment["p1"]))
        )
    ]
    visited = set(seeds)
    frontier = [(segment_id, 0) for segment_id in seeds]
    while frontier and len(visited) <= 24:
        segment_id, depth = frontier.pop(0)
        if depth >= 4:
            continue
        for neighbor in sorted(geometry.adjacency.get(segment_id, set())):
            if neighbor not in visited and neighbor not in frame_segment_ids:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
    endpoints = [
        endpoint
        for segment_id in visited
        for endpoint in (by_id[segment_id]["p1"], by_id[segment_id]["p2"])
    ]
    arrow_ids = nearby_arrow_ids(geometry.arrows, endpoints, 3.0) if endpoints else []
    return sorted(visited), arrow_ids


def detect_geometric_tolerance_frames(
    tokens: Sequence[dict[str, Any]], geometry: PageGeometry,
    default_unit: str | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Assemble strict vector feature-control frames into structured records.

    This first implementation intentionally supports the evidence combination
    seen in ZM-786: a coaxiality characteristic, diameter tolerance zone,
    numeric tolerance, and datum reference.  Requiring the complete closed
    frame plus all four semantic components prevents ordinary tables and bare
    numbers from being consumed.
    """
    horizontal = [
        segment for segment in geometry.segments
        if angle_difference(segment["angle_deg"], 0.0) <= 1.0
        and 18.0 <= segment["length"] <= 240.0
    ]
    vertical = [
        segment for segment in geometry.segments
        if angle_difference(segment["angle_deg"], 90.0) <= 1.0
        and 7.0 <= segment["length"] <= 40.0
    ]
    candidates: list[tuple[float, list[float], list[dict[str, Any]]]] = []
    for index, top in enumerate(horizontal):
        top_x = sorted((top["p1"][0], top["p2"][0]))
        top_y = (top["p1"][1] + top["p2"][1]) / 2.0
        for bottom in horizontal[index + 1:]:
            bottom_x = sorted((bottom["p1"][0], bottom["p2"][0]))
            bottom_y = (bottom["p1"][1] + bottom["p2"][1]) / 2.0
            height = abs(bottom_y - top_y)
            if not 8.0 <= height <= 32.0:
                continue
            if max(abs(top_x[0] - bottom_x[0]), abs(top_x[1] - bottom_x[1])) > 0.8:
                continue
            width = min(top_x[1], bottom_x[1]) - max(top_x[0], bottom_x[0])
            if width < height * 2.2:
                continue
            frame = [top_x[0], min(top_y, bottom_y), top_x[1], max(top_y, bottom_y)]
            frame_vertical = [
                segment for segment in vertical
                if frame[0] - 0.8 <= sum((segment["p1"][0], segment["p2"][0])) / 2.0 <= frame[2] + 0.8
                and abs(min(segment["p1"][1], segment["p2"][1]) - frame[1]) <= 0.8
                and abs(max(segment["p1"][1], segment["p2"][1]) - frame[3]) <= 0.8
            ]
            xs = sorted(
                {
                    rounded(sum((segment["p1"][0], segment["p2"][0])) / 2.0, 1)
                    for segment in frame_vertical
                }
            )
            if (
                len(xs) < 4
                or abs(xs[0] - frame[0]) > 1.0
                or abs(xs[-1] - frame[2]) > 1.0
            ):
                continue
            frame_segments = [top, bottom, *frame_vertical]
            candidates.append((width * height, frame, frame_segments))

    records: list[dict[str, Any]] = []
    consumed: set[str] = set()
    seen_frames: set[tuple[float, ...]] = set()
    for _, frame, frame_segments in sorted(candidates, reverse=True):
        signature = tuple(rounded(value, 1) for value in frame)
        if signature in seen_frames:
            continue
        seen_frames.add(signature)
        inside_tokens = [
            token for token in tokens
            if frame[0] - 0.5 <= bbox_center(token["bbox"])[0] <= frame[2] + 0.5
            and frame[1] - 0.5 <= bbox_center(token["bbox"])[1] <= frame[3] + 0.5
        ]
        inside_symbols = [
            symbol for symbol in geometry.symbols
            if frame[0] - 0.5 <= symbol["center"][0] <= frame[2] + 0.5
            and frame[1] - 0.5 <= symbol["center"][1] <= frame[3] + 0.5
        ]
        characteristic = next(
            (symbol for symbol in inside_symbols if symbol["kind"] == "coaxiality"), None
        )
        diameter = next(
            (symbol for symbol in inside_symbols if symbol["kind"] == "diameter"), None
        )
        datum_tokens = [
            token for token in inside_tokens
            if re.fullmatch(r"[A-Z](?:\d)?", token["normalized_text"])
        ]
        numeric_tokens = [
            token for token in inside_tokens
            if re.fullmatch(NUMBER_RE, token["normalized_text"])
        ]
        if not characteristic or not diameter or len(datum_tokens) != 1 or len(numeric_tokens) != 1:
            continue
        tolerance_token = numeric_tokens[0]
        tolerance_value = parse_number(tolerance_token["normalized_text"])
        if tolerance_value is None or tolerance_value < 0:
            continue
        datum = datum_tokens[0]["normalized_text"]
        frame_segment_ids = {segment["id"] for segment in frame_segments}
        leader_ids, arrow_ids = _frame_leader_evidence(frame, frame_segment_ids, geometry)
        all_line_ids = sorted(frame_segment_ids | set(leader_ids))
        raw_text = f"\u25ce | {datum} | \u2300{tolerance_token['normalized_text']}"
        records.append(
            {
                "id": "",
                "page": tolerance_token["page"],
                "raw_text": raw_text,
                "normalized_text": raw_text,
                "canonical_text": f"\u25ce | \u2300{tolerance_token['normalized_text']} | {datum}",
                "type": "geometric_tolerance",
                "nominal_text": None,
                "nominal": None,
                "unit": default_unit,
                "unit_source": "command_line" if default_unit else None,
                "tolerance_upper": None,
                "tolerance_lower": None,
                "tolerance_unit": None,
                "quantity": 1,
                "reference": False,
                "fit": None,
                "thread_pitch": None,
                "angle_degrees": None,
                "angle_minutes": None,
                "angle_seconds": None,
                "distribution_angle_deg": None,
                "surface_roughness_parameter": None,
                "geometric_characteristic": "coaxiality",
                "characteristic_symbol": "\u25ce",
                "geometric_tolerance": rounded(tolerance_value, 6),
                "geometric_tolerance_unit": default_unit,
                "tolerance_zone": "diameter",
                "datum_references": [datum],
                "controlled_feature": "leader_indicated_surface_axis",
                "parse_notes": [],
                "rotation_deg": 0.0,
                "direction": [1.0, 0.0],
                "bbox": [rounded(value) for value in frame],
                "font_size": max(token["size"] for token in inside_tokens),
                "root_token_id": tolerance_token["id"],
                "fragment_token_ids": [datum_tokens[0]["id"]],
                "vector_symbol_ids": [characteristic["id"], diameter["id"]],
                "fragments": [
                    {
                        "id": datum_tokens[0]["id"], "text": datum_tokens[0]["text"],
                        "role": "datum_reference", "bbox": datum_tokens[0]["bbox"],
                        "size": datum_tokens[0]["size"],
                    }
                ],
                "context_line_text": " ".join(token["line_text"] for token in inside_tokens),
                "assembly_basis": "closed_vector_feature_control_frame",
                "geometry": {
                    "relationship": "feature_control_frame",
                    "score": 100.0,
                    "score_margin": None,
                    "unique": True,
                    "line_segment_ids": all_line_ids,
                    "arrow_ids": arrow_ids,
                    "extension_segment_ids": [],
                    "frame_bbox": [rounded(value) for value in frame],
                    "leader_connected": bool(leader_ids),
                    "controlled_feature_kind": "surface_axis",
                    "confidence_basis": "closed_frame+coaxiality+diameter+tolerance+datum",
                },
                "status": "accepted",
                "review_reason": None,
            }
        )
        consumed.update(token["id"] for token in inside_tokens)
    return records, consumed


def unique_points(values: Iterable[Sequence[float]], tolerance: float = 0.5) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for value in values:
        candidate = (float(value[0]), float(value[1]))
        if not any(distance(candidate, previous) <= tolerance for previous in result):
            result.append(candidate)
    return result


def polygon_tip(vertices: Sequence[Sequence[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    center = (
        sum(value[0] for value in vertices) / len(vertices),
        sum(value[1] for value in vertices) / len(vertices),
    )
    best_vertex = tuple(vertices[0])
    best_angle = 181.0
    for index, current in enumerate(vertices):
        before = vec_sub(vertices[index - 1], current)
        after = vec_sub(vertices[(index + 1) % len(vertices)], current)
        cosine = max(-1.0, min(1.0, dot(unit(before), unit(after))))
        angle = math.degrees(math.acos(cosine))
        if angle < best_angle:
            best_angle = angle
            best_vertex = tuple(current)
    return best_vertex, unit(vec_sub(best_vertex, center))


def detect_filled_arrows(
    path_vertices: dict[str, list[tuple[float, float]]],
    path_drawings: dict[str, dict[str, Any]],
    page_number: int,
) -> list[dict[str, Any]]:
    arrows: list[dict[str, Any]] = []
    for path_id, raw_vertices in path_vertices.items():
        drawing = path_drawings[path_id]
        if drawing.get("fill") is None:
            continue
        # Some CAD exporters place both endpoint arrowheads in one filled PDF
        # drawing object.  Treating the whole object as one polygon yields six
        # vertices and used to discard both valid triangles.  Split line items
        # into connected components before applying the small-polygon filter.
        edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for item in drawing.get("items", []):
            kind = item[0]
            if kind == "l":
                edges.append((point(item[1]), point(item[2])))
            elif kind == "re":
                rect = item[1]
                corners = (
                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                    (rect.x1, rect.y1), (rect.x0, rect.y1),
                )
                edges.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))
            elif kind == "qu":
                quad = item[1]
                corners = (point(quad.ul), point(quad.ur), point(quad.lr), point(quad.ll))
                edges.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))

        components: list[list[tuple[float, float]]] = []
        for left, right in edges:
            touching = [
                index for index, vertices in enumerate(components)
                if any(distance(endpoint, vertex) <= 0.35 for endpoint in (left, right) for vertex in vertices)
            ]
            if not touching:
                components.append([left, right])
                continue
            target = touching[0]
            components[target].extend((left, right))
            for index in reversed(touching[1:]):
                components[target].extend(components.pop(index))

        component_vertices = [unique_points(vertices) for vertices in components]
        if not component_vertices:
            component_vertices = [unique_points(raw_vertices)]
        for vertices in component_vertices:
            if not 3 <= len(vertices) <= 5:
                continue
            xs = [value[0] for value in vertices]
            ys = [value[1] for value in vertices]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            if min(width, height) < 0.4 or max(width, height) > 24.0:
                continue
            tip, direction = polygon_tip(vertices)
            arrows.append(
                {
                    "id": f"P{page_number}-A{len(arrows) + 1:04d}",
                    "kind": "filled",
                    "tip": [rounded(tip[0]), rounded(tip[1])],
                    "direction": [rounded(direction[0], 4), rounded(direction[1], 4)],
                    "bbox": [rounded(min(xs)), rounded(min(ys)), rounded(max(xs)), rounded(max(ys))],
                    "segment_ids": [],
                }
            )
    return arrows


def build_endpoint_grid(
    segments: Sequence[dict[str, Any]], cell: float = 1.5
) -> dict[tuple[int, int], list[tuple[str, tuple[float, float], int]]]:
    grid: dict[tuple[int, int], list[tuple[str, tuple[float, float], int]]] = {}
    for segment in segments:
        for endpoint_index, endpoint in enumerate((segment["p1"], segment["p2"])):
            key = (round(endpoint[0] / cell), round(endpoint[1] / cell))
            grid.setdefault(key, []).append((segment["id"], tuple(endpoint), endpoint_index))
    return grid


def detect_open_arrows(
    segments: Sequence[dict[str, Any]], page_number: int
) -> list[dict[str, Any]]:
    short = [segment for segment in segments if 1.0 <= segment["length"] <= 18.0]
    grid = build_endpoint_grid(short)
    by_id = {segment["id"]: segment for segment in short}
    arrows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entries in grid.values():
        for left_index, left in enumerate(entries):
            for right in entries[left_index + 1:]:
                if left[0] == right[0]:
                    continue
                pair = tuple(sorted((left[0], right[0])))
                if pair in seen_pairs or distance(left[1], right[1]) > 1.2:
                    continue
                seen_pairs.add(pair)
                first, second = by_id[left[0]], by_id[right[0]]
                ratio = first["length"] / second["length"]
                if not 0.45 <= ratio <= 2.2:
                    continue
                tip = ((left[1][0] + right[1][0]) / 2, (left[1][1] + right[1][1]) / 2)
                far_first = first["p2"] if left[2] == 0 else first["p1"]
                far_second = second["p2"] if right[2] == 0 else second["p1"]
                arm_first, arm_second = unit(vec_sub(far_first, tip)), unit(vec_sub(far_second, tip))
                opening = math.degrees(math.acos(max(-1.0, min(1.0, dot(arm_first, arm_second)))))
                if not 15.0 <= opening <= 80.0:
                    continue
                backwards = unit(vec_add(arm_first, arm_second))
                direction = vec_scale(backwards, -1.0)
                arrows.append(
                    {
                        "id": f"P{page_number}-AO{len(arrows) + 1:04d}",
                        "kind": "open",
                        "tip": [rounded(tip[0]), rounded(tip[1])],
                        "direction": [rounded(direction[0], 4), rounded(direction[1], 4)],
                        "bbox": bbox_union(
                            ([tip[0], tip[1], tip[0], tip[1]],
                             [far_first[0], far_first[1], far_first[0], far_first[1]],
                             [far_second[0], far_second[1], far_second[0], far_second[1]])
                        ),
                        "segment_ids": [first["id"], second["id"]],
                    }
                )
    return arrows


def deduplicate_arrows(arrows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arrow in sorted(arrows, key=lambda value: value["kind"] != "filled"):
        duplicate = next(
            (
                existing for existing in result
                if distance(arrow["tip"], existing["tip"]) <= 1.5
                or (
                    existing["kind"] == "filled" and arrow["kind"] == "open"
                    and all(
                        abs(left - right) <= 1.5
                        for left, right in zip(existing["bbox"], arrow["bbox"])
                    )
                )
            ),
            None,
        )
        if duplicate is None:
            result.append(dict(arrow))
        elif arrow["segment_ids"]:
            duplicate["segment_ids"] = list(
                dict.fromkeys([*duplicate["segment_ids"], *arrow["segment_ids"]])
            )
    for index, arrow in enumerate(result, start=1):
        arrow["id"] = f"{arrow['id'].split('-A')[0]}-A{index:04d}"
    return result


def build_segment_adjacency(
    segments: Sequence[dict[str, Any]], tolerance: float = 1.4
) -> dict[str, set[str]]:
    grid = build_endpoint_grid(segments, cell=tolerance)
    adjacency = {segment["id"]: set() for segment in segments}
    offsets = (-1, 0, 1)
    for key, entries in grid.items():
        nearby: list[tuple[str, tuple[float, float], int]] = []
        for dx in offsets:
            for dy in offsets:
                nearby.extend(grid.get((key[0] + dx, key[1] + dy), []))
        for segment_id, endpoint, _ in entries:
            for other_id, other_endpoint, _ in nearby:
                if segment_id != other_id and distance(endpoint, other_endpoint) <= tolerance:
                    adjacency[segment_id].add(other_id)
    return adjacency


def is_root_candidate(token: dict[str, Any]) -> bool:
    text = token["normalized_text"]
    if not re.search(r"\d", text) or len(text) > 100:
        return False
    if (
        SIGNED_TOL_RE.fullmatch(text)
        or re.fullmatch(rf"[+\-±]\s*{NUMBER_RE}\s*[°'\"]?", text)
        or QUANTITY_PREFIX_RE.fullmatch(text)
    ):
        return False
    if re.fullmatch(r"[+\-±]", text):
        return False
    return first_number(text) is not None


def fragment_role(text: str) -> str | None:
    value = normalize_text(text)
    if re.fullmatch(r"[+\-±]", value) or SIGNED_TOL_RE.fullmatch(value):
        return "tolerance"
    if NUMERIC_FRAGMENT_RE.fullmatch(value):
        return "number"
    if QUANTITY_PREFIX_RE.fullmatch(value):
        return "quantity_prefix"
    if SYMBOL_ONLY_RE.fullmatch(value):
        return "symbol"
    if UNIT_RE.fullmatch(value):
        return "unit"
    if ROUGHNESS_TEXT_RE.fullmatch(value):
        return None
    # R20, SR10, M16, C2, etc. already carry their own dimension type and
    # nominal value.  FIT_RE would otherwise mistake R20 for a fit such as H7
    # and attach the complete neighboring dimension to another root (R10 R20).
    if COMPLETE_PREFIXED_DIMENSION_RE.fullmatch(value):
        return None
    if FIT_RE.fullmatch(value):
        return "fit"
    return None


def compatible_rotation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return angle_difference(left["rotation_deg"], right["rotation_deg"]) <= 10.0


def vector_symbol_fragments(
    root: dict[str, Any], vector_symbols: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    u = tuple(root["direction"])
    v = (-u[1], u[0])
    root_u = local_range(root["bbox"], u)
    root_v = local_range(root["bbox"], v)
    root_size = max(root["size"], 4.0)
    matches: list[tuple[float, dict[str, Any]]] = []
    for symbol in vector_symbols:
        # A vectorized diameter glyph has no reliable text rotation metadata.
        # Baseline position relative to the root is the stronger attachment
        # test and also supports arbitrary rotated annotations.
        if symbol["kind"] != "diameter":
            continue
        symbol_u = local_range(symbol["bbox"], u)
        symbol_v = local_range(symbol["bbox"], v)
        along_gap = max(root_u[0] - symbol_u[1], 0.0)
        perpendicular_gap = interval_gap(root_v, symbol_v)
        before_root = sum(symbol_u) / 2.0 < sum(root_u) / 2.0
        if before_root and along_gap <= root_size * 0.75 and perpendicular_gap <= root_size * 0.35:
            token = {
                "id": symbol["id"], "page": root["page"],
                "block": root["block"], "line": root["line"], "span": -1,
                "text": "Ø", "normalized_text": "Ø", "line_text": "Ø",
                "bbox": symbol["bbox"], "quad": [], "origin": symbol["center"],
                "font": "[vector-diameter-glyph]", "size": root["size"],
                "flags": 0, "color": 0, "direction": list(root["direction"]),
                "rotation_deg": root["rotation_deg"], "vector_symbol": True,
            }
            matches.append((along_gap + perpendicular_gap * 2.0, token))
    if not matches:
        return []
    _, nearest = min(matches, key=lambda value: (value[0], value[1]["id"]))
    return [{"token": nearest, "role": "symbol"}]


def collect_fragments(
    root: dict[str, Any], page_tokens: Sequence[dict[str, Any]],
    vector_symbols: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    u = tuple(root["direction"])
    v = (-u[1], u[0])
    root_u = local_range(root["bbox"], u)
    root_v = local_range(root["bbox"], v)
    root_center_u = sum(root_u) / 2
    root_center_v = sum(root_v) / 2
    root_size = max(root["size"], 4.0)
    candidates: list[tuple[float, dict[str, Any], str]] = []
    symbol_fragments = vector_symbol_fragments(root, vector_symbols)

    def stacked_number_partner(candidate: dict[str, Any]) -> bool:
        candidate_u = sum(local_range(candidate["bbox"], u)) / 2
        candidate_v = sum(local_range(candidate["bbox"], v)) / 2
        for other in page_tokens:
            if other["id"] in {root["id"], candidate["id"]}:
                continue
            if not NUMERIC_FRAGMENT_RE.fullmatch(other["normalized_text"]):
                continue
            if not compatible_rotation(root, other):
                continue
            other_u = sum(local_range(other["bbox"], u)) / 2
            other_v = sum(local_range(other["bbox"], v)) / 2
            if (
                abs(candidate_u - other_u) <= max(root_size * 0.65, 4.0)
                and root_size * 0.3 <= abs(candidate_v - other_v) <= root_size * 2.0
                and other_u >= root_u[1] - root_size * 0.5
            ):
                return True
        return False

    def signed_number_partner(candidate: dict[str, Any]) -> bool:
        candidate_u = sum(local_range(candidate["bbox"], u)) / 2
        candidate_v = sum(local_range(candidate["bbox"], v)) / 2
        for other in page_tokens:
            if other["id"] in {root["id"], candidate["id"]}:
                continue
            if not re.fullmatch(r"[+\-±]", other["normalized_text"]):
                continue
            if not compatible_rotation(root, other):
                continue
            other_u = sum(local_range(other["bbox"], u)) / 2
            other_v = sum(local_range(other["bbox"], v)) / 2
            if abs(candidate_u - other_u) <= root_size * 1.6 and abs(candidate_v - other_v) <= root_size * 0.6:
                return True
        return False

    def quote_anchored_fraction_part(candidate: dict[str, Any]) -> bool:
        if not re.fullmatch(r"\d", candidate["normalized_text"]):
            return False
        candidate_u = sum(local_range(candidate["bbox"], u)) / 2
        candidate_v = sum(local_range(candidate["bbox"], v)) / 2
        return stacked_number_partner(candidate) and any(
            other["normalized_text"] in {'"', "″"}
            and compatible_rotation(root, other)
            and 0 <= sum(local_range(other["bbox"], u)) / 2 - candidate_u <= root_size * 2.0
            and abs(sum(local_range(other["bbox"], v)) / 2 - candidate_v) <= root_size
            for other in page_tokens
            if other["id"] not in {root["id"], candidate["id"]}
        )

    for candidate in page_tokens:
        if candidate["id"] == root["id"] or not compatible_rotation(root, candidate):
            continue
        role = fragment_role(candidate["normalized_text"])
        if role is None:
            continue
        ratio = candidate["size"] / root_size if root_size else 1.0
        max_ratio = 1.35 if role == "symbol" and candidate["normalized_text"] in {'"', "′", "″", "'"} else 1.25
        if not 0.30 <= ratio <= max_ratio:
            continue
        candidate_u = local_range(candidate["bbox"], u)
        candidate_v = local_range(candidate["bbox"], v)
        center_u = sum(candidate_u) / 2
        center_v = sum(candidate_v) / 2
        along_gap = max(candidate_u[0] - root_u[1], root_u[0] - candidate_u[1], 0.0)
        perpendicular_gap = max(candidate_v[0] - root_v[1], root_v[0] - candidate_v[1], 0.0)
        if along_gap > max(root_size * 3.2, 24.0) or perpendicular_gap > max(root_size * 1.7, 13.0):
            continue
        same_line = candidate["block"] == root["block"] and candidate["line"] == root["line"]
        side = center_u - root_center_u
        vertical = center_v - root_center_v
        if role == "number":
            # Unsigned neighboring numbers are accepted only as smaller stacked
            # tolerances. This conservative rule avoids merging adjacent dimensions.
            if abs(vertical) > root_size * 1.5:
                continue
            # Equal-size numeric tokens are complete neighboring dimensions,
            # not stacked deviations. A real split deviation either uses a
            # smaller font or has an explicit nearby sign.
            if (
                candidate["size"] >= root["size"] * 0.92
                and not signed_number_partner(candidate)
                and not quote_anchored_fraction_part(candidate)
            ):
                continue
            right_or_aligned = center_u >= root_u[1] - root_size * 0.5
            stacked = right_or_aligned and (
                (
                    candidate["size"] <= root["size"] * 0.88
                    and abs(vertical) >= root_size * 0.18
                )
                or stacked_number_partner(candidate)
                or signed_number_partner(candidate)
            )
            if not stacked:
                continue
            if along_gap > root_size * 1.6:
                continue
        elif role == "tolerance":
            if abs(vertical) > root_size * 1.5:
                continue
        elif role in {"symbol", "quantity_prefix"}:
            loose_unit_mark = candidate["normalized_text"] in {'"', "'", "°"}
            if loose_unit_mark:
                if side < -root_size or abs(vertical) > root_size * 1.5:
                    continue
            elif not same_line or abs(vertical) > root_size * 0.65:
                if role != "quantity_prefix" or not symbol_fragments:
                    continue
                symbol = symbol_fragments[0]["token"]
                symbol_u = local_range(symbol["bbox"], u)
                symbol_v = local_range(symbol["bbox"], v)
                bridged = (
                    candidate_u[1] <= symbol_u[0] + root_size * 0.25
                    and interval_gap(candidate_u, symbol_u) <= root_size * 0.75
                    and interval_gap(candidate_v, symbol_v) <= root_size * 0.35
                )
                if not bridged:
                    continue
            if role == "quantity_prefix" and side > 0:
                continue
        elif role in {"unit", "fit"} and side < -root_size:
            continue
        score = along_gap + perpendicular_gap * 1.5 + abs(vertical) * 0.15
        candidates.append((score, candidate, role))

    # Keep only the nearest member for competing sign/number locations; a complete
    # annotation can still retain two stacked deviations.
    chosen: list[dict[str, Any]] = list(symbol_fragments)
    tolerance_signs: set[str] = set()
    symmetric_tolerance_selected = False
    role_counts: dict[str, int] = {"symbol": len(symbol_fragments)}
    for _, candidate, role in sorted(candidates, key=lambda item: item[0]):
        if any(candidate["id"] == value["token"]["id"] for value in chosen):
            continue
        if role == "tolerance":
            value = candidate["normalized_text"]
            if "±" in value:
                if symmetric_tolerance_selected or tolerance_signs:
                    continue
                symmetric_tolerance_selected = True
            else:
                sign = "+" if "+" in value else ("-" if "-" in value else value)
                if symmetric_tolerance_selected or sign in tolerance_signs:
                    continue
                tolerance_signs.add(sign)
        if role == "number" and role_counts.get(role, 0) >= 2:
            continue
        if role in {"symbol", "unit", "fit", "quantity_prefix"} and role_counts.get(role, 0) >= 2:
            continue
        value = dict(candidate)
        chosen.append({"token": value, "role": role})
        role_counts[role] = role_counts.get(role, 0) + 1
        if len(chosen) >= 8:
            break
    return chosen


def resolve_shared_fragment_owners(
    roots: Sequence[dict[str, Any]],
    fragments_by_root: dict[str, list[dict[str, Any]]],
) -> None:
    """Give every tolerance/number fragment one spatially nearest owner.

    Adjacent stacked dimensions can have overlapping search windows.  A small
    deviation such as ``+7`` must not be copied into both nominal dimensions.
    """
    roots_by_id = {root["id"]: root for root in roots}
    owners: dict[str, list[str]] = {}
    subordinate_roots: set[str] = set()
    for root_id, fragments in fragments_by_root.items():
        for fragment in fragments:
            if fragment["role"] in {"tolerance", "number"}:
                owners.setdefault(fragment["token"]["id"], []).append(root_id)
                child = roots_by_id.get(fragment["token"]["id"])
                parent = roots_by_id[root_id]
                if child and child["size"] <= parent["size"] * 0.9:
                    subordinate_roots.add(child["id"])

    def ownership_score(root: dict[str, Any], token: dict[str, Any]) -> float:
        u = tuple(root["direction"])
        v = (-u[1], u[0])
        root_u = local_range(root["bbox"], u)
        token_u = local_range(token["bbox"], u)
        root_v_center = sum(local_range(root["bbox"], v)) / 2.0
        token_v_center = sum(local_range(token["bbox"], v)) / 2.0
        along_gap = max(token_u[0] - root_u[1], root_u[0] - token_u[1], 0.0)
        before_penalty = max(root_u[0] - token_u[1], 0.0)
        return along_gap + abs(token_v_center - root_v_center) * 1.5 + before_penalty * 2.0

    for fragment_id, root_ids in owners.items():
        if len(root_ids) < 2:
            continue
        all_root_ids = root_ids
        canonical_root_ids = [root_id for root_id in root_ids if root_id not in subordinate_roots]
        if canonical_root_ids:
            root_ids = canonical_root_ids
        token = next(
            fragment["token"]
            for root_id in root_ids
            for fragment in fragments_by_root[root_id]
            if fragment["token"]["id"] == fragment_id
        )
        winner = min(
            root_ids,
            key=lambda root_id: (ownership_score(roots_by_id[root_id], token), root_id),
        )
        for root_id in all_root_ids:
            if root_id != winner:
                fragments_by_root[root_id] = [
                    fragment for fragment in fragments_by_root[root_id]
                    if fragment["token"]["id"] != fragment_id
                ]


def detect_imperial_fraction_clusters(
    page_tokens: Sequence[dict[str, Any]], vector_symbols: Sequence[dict[str, Any]] = (),
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    """Give each stacked imperial fraction one canonical numeric root.

    CAD text commonly stores ``5 1/4\"`` as four peer spans.  Generic nearest-
    fragment matching lets every digit claim the quote and creates duplicate
    dimensions.  A quote-anchored cluster establishes ownership before the
    ordinary candidate loop.
    """
    clusters: dict[str, list[dict[str, Any]]] = {}
    consumed: set[str] = set()
    quotes = [token for token in page_tokens if token["normalized_text"] in {'"', "″"}]
    common_denominators = {2, 4, 8, 16, 32, 64}

    for quote in sorted(quotes, key=lambda token: token["id"]):
        u = tuple(quote["direction"])
        v = (-u[1], u[0])
        quote_u = sum(local_range(quote["bbox"], u)) / 2
        quote_v = sum(local_range(quote["bbox"], v)) / 2
        size = max(quote["size"] / 1.25, 4.0)
        nearby: list[dict[str, Any]] = []
        for token in page_tokens:
            if token["id"] == quote["id"] or token["id"] in consumed:
                continue
            if not compatible_rotation(quote, token):
                continue
            text = token["normalized_text"]
            if not (PURE_NUMBER_RE.fullmatch(text) or re.fullmatch(r"\d+\s+\d+", text)):
                continue
            token_u = sum(local_range(token["bbox"], u)) / 2
            token_v = sum(local_range(token["bbox"], v)) / 2
            if (
                token_u < quote_u + size * 0.15
                and quote_u - token_u <= size * 3.2
                and abs(token_v - quote_v) <= size * 1.35
            ):
                nearby.append(token)

        pair_options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        pure = [token for token in nearby if PURE_NUMBER_RE.fullmatch(token["normalized_text"])]
        for index, first in enumerate(pure):
            for second in pure[index + 1:]:
                first_value = parse_number(first["normalized_text"])
                second_value = parse_number(second["normalized_text"])
                if first_value is None or second_value is None:
                    continue
                numerator, denominator = (
                    (first, second) if first_value < second_value else (second, first)
                )
                denominator_value = max(first_value, second_value)
                numerator_value = min(first_value, second_value)
                if int(denominator_value) not in common_denominators or numerator_value >= denominator_value:
                    continue
                first_u = sum(local_range(first["bbox"], u)) / 2
                second_u = sum(local_range(second["bbox"], u)) / 2
                first_v = sum(local_range(first["bbox"], v)) / 2
                second_v = sum(local_range(second["bbox"], v)) / 2
                if abs(first_u - second_u) > size * 0.40:
                    continue
                vertical = abs(first_v - second_v)
                if not size * 0.45 <= vertical <= size * 1.65:
                    continue
                pair_u = (first_u + second_u) / 2
                pair_v = (first_v + second_v) / 2
                score = abs(quote_u - pair_u) + abs(quote_v - pair_v) * 0.5
                pair_options.append((score, numerator, denominator))

        root: dict[str, Any] | None = None
        fraction_tokens: list[dict[str, Any]] = []
        if pair_options:
            _, numerator, denominator = min(
                pair_options, key=lambda item: (item[0], item[1]["id"], item[2]["id"])
            )
            pair_u = sum(local_range(numerator["bbox"], u) + local_range(denominator["bbox"], u)) / 4
            whole_options: list[tuple[float, dict[str, Any]]] = []
            for token in pure:
                if token["id"] in {numerator["id"], denominator["id"]}:
                    continue
                token_u = sum(local_range(token["bbox"], u)) / 2
                token_v = sum(local_range(token["bbox"], v)) / 2
                if (
                    token_u < pair_u - size * 0.25
                    and pair_u - token_u <= size * 2.7
                    and abs(token_v - quote_v) <= size * 0.48
                ):
                    whole_options.append((pair_u - token_u, token))
            root = min(whole_options, key=lambda item: (item[0], item[1]["id"]))[1] if whole_options else numerator
            fraction_tokens = [numerator, denominator]
        else:
            # A span like ``4 3`` contains the whole and numerator together.
            for combined in (token for token in nearby if re.fullmatch(r"\d+\s+\d+", token["normalized_text"])):
                _, numerator_text = combined["normalized_text"].split()
                numerator_value = int(numerator_text)
                combined_u = sum(local_range(combined["bbox"], u)) / 2
                denominator_options = []
                for denominator in pure:
                    value = parse_number(denominator["normalized_text"])
                    denominator_u = sum(local_range(denominator["bbox"], u)) / 2
                    if (
                        value is not None and int(value) in common_denominators
                        and numerator_value < value
                        and abs(denominator_u - combined_u) <= size * 1.2
                    ):
                        denominator_options.append((abs(quote_u - denominator_u), denominator))
                if denominator_options:
                    root = combined
                    fraction_tokens = [min(denominator_options, key=lambda item: item[0])[1]]
                    break
        if root is None:
            continue

        component_ids = {root["id"], quote["id"], *(token["id"] for token in fraction_tokens)}
        fragments: list[dict[str, Any]] = []
        for token in fraction_tokens:
            if token["id"] != root["id"]:
                fragments.append({"token": token, "role": "number"})
        fragments.append({"token": quote, "role": "symbol"})
        diameter_fragments = vector_symbol_fragments(root, vector_symbols)
        fragments.extend(diameter_fragments)

        root_u = sum(local_range(root["bbox"], u)) / 2
        root_v = sum(local_range(root["bbox"], v)) / 2
        for token in page_tokens:
            if token["id"] in component_ids or not compatible_rotation(root, token):
                continue
            text = token["normalized_text"]
            role = fragment_role(text)
            if role not in {"symbol", "quantity_prefix"} or text in {'"', "″", "′", "'", "°"}:
                continue
            token_u_range = local_range(token["bbox"], u)
            token_v_range = local_range(token["bbox"], v)
            token_u = sum(token_u_range) / 2
            token_v = sum(token_v_range) / 2
            close_to_root = (
                token_u < root_u
                and root_u - token_u <= size * 2.0
                and abs(token_v - root_v) <= size * 0.65
            )
            bridged_by_diameter = role == "quantity_prefix" and any(
                token_u < sum(local_range(fragment["token"]["bbox"], u)) / 2 < root_u
                and interval_gap(token_u_range, local_range(fragment["token"]["bbox"], u)) <= size * 0.75
                and interval_gap(token_v_range, local_range(fragment["token"]["bbox"], v)) <= size * 0.35
                for fragment in diameter_fragments
            )
            if close_to_root or bridged_by_diameter:
                fragments.append({"token": token, "role": role})

        unique_fragments: dict[str, dict[str, Any]] = {
            fragment["token"]["id"]: fragment for fragment in fragments
        }
        clusters[root["id"]] = list(unique_fragments.values())
        consumed.update(component_ids)

    # Merge a smaller quote-anchored fraction following ± into the preceding
    # imperial dimension, e.g. Ø3 5/8" + ± + 1/16".
    for marker in (token for token in page_tokens if token["normalized_text"] == "±"):
        u = tuple(marker["direction"])
        v = (-u[1], u[0])
        marker_u = sum(local_range(marker["bbox"], u)) / 2
        marker_v = sum(local_range(marker["bbox"], v)) / 2
        before: list[tuple[float, str]] = []
        after: list[tuple[float, str]] = []
        for root_id, fragments in clusters.items():
            root = next(token for token in page_tokens if token["id"] == root_id)
            if not compatible_rotation(marker, root):
                continue
            members = [root] + [fragment["token"] for fragment in fragments]
            box = bbox_union(member["bbox"] for member in members)
            cluster_u = local_range(box, u)
            cluster_v = sum(local_range(box, v)) / 2
            if abs(cluster_v - marker_v) > max(marker["size"] * 1.2, 8.0):
                continue
            if cluster_u[1] <= marker_u and marker_u - cluster_u[1] <= max(marker["size"] * 1.5, 10.0):
                before.append((marker_u - cluster_u[1], root_id))
            if cluster_u[0] >= marker_u and cluster_u[0] - marker_u <= max(marker["size"] * 1.5, 10.0):
                after.append((cluster_u[0] - marker_u, root_id))
        if not before or not after:
            continue
        owner_id = min(before, key=lambda item: (item[0], item[1]))[1]
        tolerance_id = min(after, key=lambda item: (item[0], item[1]))[1]
        if owner_id == tolerance_id:
            continue
        tolerance_root = next(token for token in page_tokens if token["id"] == tolerance_id)
        owner_root = next(token for token in page_tokens if token["id"] == owner_id)
        if tolerance_root["size"] > owner_root["size"] * 0.90:
            continue
        tolerance_fragments = clusters[tolerance_id]
        tolerance_text = compose_text(tolerance_root, tolerance_fragments)
        if not re.fullmatch(rf"{NUMBER_RE}\s*\"", tolerance_text):
            continue
        tolerance_members = [marker, tolerance_root] + [
            fragment["token"] for fragment in tolerance_fragments
        ]
        synthetic = {
            "id": f"{marker['id']}-IMPERIAL-TOL", "page": marker["page"],
            "block": marker["block"], "line": marker["line"], "span": marker["span"],
            "text": f"±{tolerance_text}", "normalized_text": f"±{tolerance_text}",
            "line_text": f"±{tolerance_text}",
            "bbox": bbox_union(member["bbox"] for member in tolerance_members),
            "quad": [], "origin": marker["origin"], "font": marker["font"],
            "size": marker["size"], "flags": marker["flags"], "color": marker["color"],
            "direction": marker["direction"], "rotation_deg": marker["rotation_deg"],
            "synthetic_imperial_tolerance": True,
            "source_token_ids": [member["id"] for member in tolerance_members],
        }
        clusters[owner_id].append({"token": synthetic, "role": "tolerance"})
        del clusters[tolerance_id]
    return clusters, consumed


def infer_fraction_text(root: dict[str, Any], fragments: Sequence[dict[str, Any]]) -> str | None:
    u = tuple(root["direction"])
    v = (-u[1], u[0])
    items = [root] + [fragment["token"] for fragment in fragments]
    if not any('"' in item["normalized_text"] for item in items):
        return None
    numeric_items: list[dict[str, Any]] = []
    for item in items:
        match = NUMERIC_FRAGMENT_RE.fullmatch(item["normalized_text"])
        if not match:
            continue
        numeric_items.append(
            {
                "token": item,
                "value_text": match.group(1),
                "value": parse_number(match.group(1)),
                "u": sum(local_range(item["bbox"], u)) / 2,
                "v": sum(local_range(item["bbox"], v)) / 2,
            }
        )
    root_text = root["normalized_text"].replace('"', "").strip()
    root_u_center = sum(local_range(root["bbox"], u)) / 2

    def prefix_text(before_u: float, excluded: set[str]) -> str:
        prefixes: list[tuple[float, str]] = []
        for item in items:
            text = item["normalized_text"]
            if item["id"] in excluded or text in {'"', "′", "″", "'"}:
                continue
            if not (QUANTITY_PREFIX_RE.fullmatch(text) or re.fullmatch(r"(?:SØ|Ø|SR|R)", text, re.I)):
                continue
            item_u = sum(local_range(item["bbox"], u)) / 2
            if item_u <= before_u + root["size"] * 0.25:
                prefixes.append((item_u, text))
        return "".join(text for _, text in sorted(prefixes))

    def suffix_text(excluded: set[str]) -> str:
        suffixes = []
        for fragment in fragments:
            token = fragment["token"]
            if token["id"] in excluded or fragment["role"] != "tolerance":
                continue
            token_u = sum(local_range(token["bbox"], u)) / 2
            suffixes.append((token_u, token["normalized_text"]))
        return " ".join(text for _, text in sorted(suffixes))

    # Some CAD exporters keep ``whole numerator`` in one span and place only
    # the denominator below it, e.g. ``4 3`` + stacked ``4`` + quote.
    embedded = re.fullmatch(r"(?:(SØ|Ø|SR|R)\s*)?(\d+)\s+(\d+)", root_text, re.I)
    if embedded:
        numerator_value = int(embedded.group(3))
        denominators = [
            item for item in numeric_items
            if item["token"]["id"] != root["id"]
            and item["value"] is not None
            and int(item["value"]) in {2, 4, 8, 16, 32, 64}
            and numerator_value < item["value"]
        ]
        if denominators:
            denominator = min(
                denominators,
                key=lambda item: (abs(item["u"] - root_u_center), item["token"]["id"]),
            )
            external = prefix_text(root_u_center, {root["id"], denominator["token"]["id"]})
            prefix = external + (embedded.group(1) or "")
            base_text = f"{prefix}{embedded.group(2)} {embedded.group(3)}/{denominator['value_text']}\""
            suffix = suffix_text({root["id"], denominator["token"]["id"]})
            return normalize_text(f"{base_text} {suffix}" if suffix else base_text)
    pair_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    common_denominators = {2, 4, 8, 16, 32, 64}
    for index, first in enumerate(numeric_items):
        for second in numeric_items[index + 1:]:
            if abs(first["u"] - second["u"]) > max(root["size"] * 0.8, 6.0):
                continue
            if abs(first["v"] - second["v"]) < max(root["size"] * 0.28, 2.0):
                continue
            numerator, denominator = sorted((first, second), key=lambda value: value["v"])
            if (
                numerator["value"] is None or denominator["value"] is None
                or int(denominator["value"]) not in common_denominators
                or numerator["value"] >= denominator["value"]
            ):
                continue
            pair_candidates.append((abs(first["u"] - second["u"]), numerator, denominator))
    if not pair_candidates:
        return None
    _, numerator, denominator = min(pair_candidates, key=lambda value: value[0])
    pair_ids = {numerator["token"]["id"], denominator["token"]["id"]}
    if root["id"] in pair_ids:
        whole_candidates = [
            item for item in numeric_items
            if item["token"]["id"] not in pair_ids and item["u"] < numerator["u"] - root["size"] * 0.25
        ]
        base = max(whole_candidates, key=lambda value: value["u"])["value_text"] if whole_candidates else ""
    else:
        base = root_text
    base_prefix = ""
    prefixed_base = re.fullmatch(r"(SØ|Ø|SR|R)\s*(\d+)", base, re.I)
    if prefixed_base:
        base_prefix, base = prefixed_base.group(1), prefixed_base.group(2)
    fraction_u = min(numerator["u"], denominator["u"])
    external_prefix = prefix_text(fraction_u, pair_ids | {root["id"]})
    fraction = f"{numerator['value_text']}/{denominator['value_text']}\""
    body = f"{base} {fraction}" if base else fraction
    result = f"{external_prefix}{base_prefix}{body}"
    suffix = suffix_text(pair_ids | {root["id"]})
    return normalize_text(f"{result} {suffix}" if suffix else result)


def compose_text(root: dict[str, Any], fragments: Sequence[dict[str, Any]]) -> str:
    fraction = infer_fraction_text(root, fragments)
    if fraction:
        return fraction
    u = tuple(root["direction"])
    items = [root] + [fragment["token"] for fragment in fragments]
    baseline = []
    stacked = []
    root_v = sum(local_range(root["bbox"], (-u[1], u[0]))) / 2
    for item in items:
        center_u = sum(local_range(item["bbox"], u)) / 2
        center_v = sum(local_range(item["bbox"], (-u[1], u[0]))) / 2
        record = (center_u, center_v, item["normalized_text"])
        if item["id"] == root["id"] or abs(center_v - root_v) <= max(root["size"] * 0.35, 3.0):
            baseline.append(record)
        else:
            stacked.append(record)
    baseline_text = " ".join(value[2] for value in sorted(baseline))
    baseline_text = re.sub(rf"^({NUMBER_RE})\s+(R|SR|SØ|Ø|M|C)$", r"\2\1", baseline_text, flags=re.I)
    baseline_text = re.sub(r"(\d+\s*[-×xX])\s+Ø\s+(?=\d)", r"\1Ø", baseline_text)
    baseline_text = re.sub(r"\b(S?Ø|SR|R|M|C)\s+(?=\d)", r"\1", baseline_text, flags=re.I)
    if not stacked:
        return normalize_text(baseline_text)
    stack_text = "/".join(value[2] for value in sorted(stacked, key=lambda value: value[1]))
    return normalize_text(f"{baseline_text} {stack_text}")


def parse_tolerances(
    root: dict[str, Any], fragments: Sequence[dict[str, Any]], combined_text: str
) -> tuple[float | None, float | None, list[str]]:
    notes: list[str] = []
    plus_minus = re.search(rf"±\s*({NUMBER_RE})", combined_text)
    if plus_minus:
        tolerance = parse_number(plus_minus.group(1))
        if tolerance is not None:
            return tolerance, -tolerance, notes

    signed: list[tuple[str, float, float]] = []
    unsigned: list[tuple[float, float, float]] = []
    sign_only: list[tuple[str, float, float]] = []
    u = tuple(root["direction"])
    v = (-u[1], u[0])
    for text_match in re.finditer(rf"([+\-])\s*({NUMBER_RE})", combined_text):
        value = parse_number(text_match.group(2))
        if value is not None:
            signed.append((text_match.group(1), value, 0.0))
    for fragment in fragments:
        if fragment["role"] not in {"tolerance", "number"}:
            continue
        token = fragment["token"]
        text = token["normalized_text"]
        local_u = sum(local_range(token["bbox"], u)) / 2
        local_v = sum(local_range(token["bbox"], v)) / 2
        match = re.fullmatch(rf"([+\-])\s*({NUMBER_RE})", text)
        if match:
            value = parse_number(match.group(2))
            item = (match.group(1), value, local_v)
            if value is not None and not any(existing[:2] == item[:2] for existing in signed):
                signed.append(item)
        elif re.fullmatch(r"[+\-]", text):
            sign_only.append((text, local_u, local_v))
        elif numeric_match := NUMERIC_FRAGMENT_RE.fullmatch(text):
            value = parse_number(numeric_match.group(1))
            if value is not None:
                unsigned.append((value, local_u, local_v))

    used_unsigned: set[int] = set()
    for sign, sign_u, sign_v in sign_only:
        choices = [
            (abs(value_u - sign_u) + abs(value_v - sign_v) * 2.0, index, value, value_v)
            for index, (value, value_u, value_v) in enumerate(unsigned)
            if index not in used_unsigned
            and abs(value_u - sign_u) <= max(root["size"] * 1.8, 12.0)
            and abs(value_v - sign_v) <= max(root["size"] * 0.85, 6.0)
        ]
        if choices:
            _, index, value, value_v = min(choices)
            if not any(existing[0] == sign for existing in signed):
                signed.append((sign, value, value_v))
                used_unsigned.add(index)

    upper = next((value for sign, value, _ in signed if sign == "+"), None)
    lower_value = next((value for sign, value, _ in signed if sign == "-"), None)
    lower = -lower_value if lower_value is not None else None
    if unsigned and (upper is not None or lower is not None):
        root_v = sum(local_range(root["bbox"], v)) / 2
        for index, (value, _, location) in enumerate(unsigned):
            if index in used_unsigned:
                continue
            if location < root_v and upper is None:
                upper = value
            elif location >= root_v and lower is None:
                lower = value
    fraction_is_complete = bool(re.search(r"\d+\s*/\s*\d+\s*\"", combined_text))
    if len(unsigned) >= 2 and upper is None and lower is None and not fraction_is_complete:
        notes.append("stacked_unsigned_values_may_be_limits")
    return upper, lower, notes


def parse_annotation(
    root: dict[str, Any], fragments: Sequence[dict[str, Any]], default_unit: str | None
) -> dict[str, Any]:
    combined = compose_text(root, fragments)
    working = combined
    quantity = 1
    quantity_match = re.match(
        r"^\s*\(?\s*(\d+)\s*[-×xX]\s*\)?\s*(?=(?:Ø|R|SR|SØ|M|C|□))",
        working, re.I,
    )
    if quantity_match:
        quantity = int(quantity_match.group(1))
        working = working[quantity_match.end():]
    reference = (
        (working.strip().startswith("(") and working.strip().endswith(")"))
        or working.rstrip().endswith("*")
    )
    upper = working.upper()
    roughness_match = ROUGHNESS_TEXT_RE.fullmatch(working.strip())
    if roughness_match:
        dimension_type = "surface_roughness"
    elif "SPACING" in upper:
        dimension_type = "spacing"
    elif re.search(
        r"\d+\s*(?:只|个)?\s*(?:槽|孔|吊耳|HOLES?|SLOTS?|LUGS?)(?:.*均布)?",
        upper,
        re.I,
    ):
        dimension_type = "feature_count"
    elif re.search(r"\bM\s*\d", upper):
        dimension_type = "thread"
    elif "SØ" in upper:
        dimension_type = "spherical_diameter"
    elif re.search(r"\bSR\s*\d", upper):
        dimension_type = "spherical_radius"
    elif "Ø" in upper:
        dimension_type = "diameter"
    elif re.search(r"(?:^|\s|\()R\s*\d", upper):
        dimension_type = "radius"
    elif "°" in upper:
        dimension_type = "angle"
    elif re.search(r"(?:^|\s)C\s*\d", upper) or ("×" in upper and "45°" in upper):
        dimension_type = "chamfer"
    elif "□" in upper:
        dimension_type = "square"
    else:
        dimension_type = "linear"

    nominal_match = first_number(working)
    nominal_text = nominal_match[0] if nominal_match else None
    nominal = nominal_match[1] if nominal_match else None
    angle_degrees = angle_minutes = angle_seconds = None
    distribution_angle = None
    roughness_parameter = roughness_match.group("parameter") if roughness_match else None
    if dimension_type == "feature_count":
        count_match = re.match(r"\s*(\d+)", working)
        if count_match:
            quantity = int(count_match.group(1))
    if dimension_type == "angle":
        angle_match = re.search(
            rf"({NUMBER_RE})\s*°(?:\s*({NUMBER_RE})\s*')?(?:\s*({NUMBER_RE})\s*\")?",
            working,
        )
        if angle_match:
            angle_degrees = parse_number(angle_match.group(1))
            angle_minutes = parse_number(angle_match.group(2)) if angle_match.group(2) else 0.0
            angle_seconds = parse_number(angle_match.group(3)) if angle_match.group(3) else 0.0
            if angle_degrees is not None:
                nominal = angle_degrees + (angle_minutes or 0.0) / 60.0 + (angle_seconds or 0.0) / 3600.0
                nominal_text = angle_match.group(0)
    elif dimension_type == "spacing":
        distribution_match = re.search(rf"@\s*({NUMBER_RE})\s*°", working)
        if distribution_match:
            distribution_angle = parse_number(distribution_match.group(1))
    explicit_unit = None
    if "\"" in working:
        explicit_unit = "in"
    else:
        unit_match = re.search(r"\b(mm|cm|in|inch|inches)\b", working, re.I)
        if unit_match:
            explicit_unit = unit_match.group(1).lower()
    if dimension_type == "angle":
        explicit_unit = "deg"
    elif dimension_type == "surface_roughness":
        explicit_unit = "µm"
    tolerance_upper, tolerance_lower, notes = parse_tolerances(root, fragments, combined)
    tolerance_unit = None
    tolerance_marker = re.search(rf"(?:±|[+\-])\s*{NUMBER_RE}\s*([°'\"]?)", combined)
    if tolerance_marker:
        marker_unit = tolerance_marker.group(1)
        if marker_unit == '"' and dimension_type != "angle":
            tolerance_unit = explicit_unit or default_unit
        else:
            tolerance_unit = {"°": "deg", "'": "arcmin", '"': "arcsec"}.get(
                marker_unit, explicit_unit or default_unit
            )
    elif tolerance_upper is not None or tolerance_lower is not None:
        tolerance_unit = explicit_unit or default_unit
    if dimension_type == "angle" and (tolerance_upper is not None or tolerance_lower is not None):
        deviation_marks = [
            match.group(1)
            for fragment in fragments
            if fragment["role"] in {"tolerance", "number"}
            if (match := re.search(r"([°'\"])[^°'\"]*$", fragment["token"]["normalized_text"]))
        ]
        if "'" in deviation_marks:
            tolerance_unit = "arcmin"
        elif '"' in deviation_marks:
            tolerance_unit = "arcsec"
        elif "°" in deviation_marks:
            tolerance_unit = "deg"
    if "�" in combined:
        notes.append("unmapped_glyph_in_text_layer")
    # ``working`` has the quantity prefix removed, so the dash in ``2-Ø40``
    # cannot be mistaken for an unparsed negative tolerance marker.
    if re.search(r"[+\-±]", working) and tolerance_upper is None and tolerance_lower is None:
        notes.append("unparsed_tolerance_marker")
    if combined.count("Ø") + len(re.findall(r"\bR\s*\d", combined, re.I)) > 1:
        notes.append("multiple_nominal_values_in_text")
    numeric_groups = re.findall(NUMBER_RE, working)
    has_tolerance_marker = bool(re.search(r"[+\-±]", working))
    has_fraction = bool(re.search(r"\d+\s*/\s*\d+", working))
    if (
        len(numeric_groups) > 1
        and dimension_type not in {"angle", "thread", "spacing", "feature_count"}
        and not has_tolerance_marker
        and not has_fraction
    ):
        notes.append("multiple_unqualified_numeric_values")
    pitch = None
    if dimension_type == "thread":
        pitch_match = re.search(rf"\bM\s*{NUMBER_RE}\s*[×xX]\s*({NUMBER_RE})", working, re.I)
        if pitch_match:
            pitch = parse_number(pitch_match.group(1))
    fit_match = None
    if dimension_type in {"linear", "diameter", "spherical_diameter"}:
        fit_match = re.search(r"\b([A-Za-z]{1,2}\d{1,2}(?:/[A-Za-z]{1,2}\d{1,2})?)\b", working)
    return {
        "raw_text": combined,
        "normalized_text": combined,
        "type": dimension_type,
        "nominal_text": nominal_text,
        "nominal": rounded(nominal, 6) if nominal is not None else None,
        "unit": explicit_unit or default_unit,
        "unit_source": "explicit" if explicit_unit else ("command_line" if default_unit else None),
        "tolerance_upper": rounded(tolerance_upper, 6) if tolerance_upper is not None else None,
        "tolerance_lower": rounded(tolerance_lower, 6) if tolerance_lower is not None else None,
        "tolerance_unit": tolerance_unit,
        "quantity": quantity,
        "reference": reference,
        "fit": fit_match.group(1) if fit_match else None,
        "thread_pitch": rounded(pitch, 6) if pitch is not None else None,
        "angle_degrees": rounded(angle_degrees, 6) if angle_degrees is not None else None,
        "angle_minutes": rounded(angle_minutes, 6) if angle_minutes is not None else None,
        "angle_seconds": rounded(angle_seconds, 6) if angle_seconds is not None else None,
        "distribution_angle_deg": rounded(distribution_angle, 6) if distribution_angle is not None else None,
        "surface_roughness_parameter": roughness_parameter,
        "parse_notes": notes,
    }


def segment_line_distance(segment: dict[str, Any], p: Sequence[float]) -> float:
    a, b = segment["p1"], segment["p2"]
    direction = unit(vec_sub(b, a))
    normal = (-direction[1], direction[0])
    return abs(dot(vec_sub(p, a), normal))


def interval_gap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(left[0] - right[1], right[0] - left[1], 0.0)


def collinear_group(
    base: dict[str, Any], nearby: Sequence[dict[str, Any]], max_gap: float,
    bridge_interval: Sequence[float] | None = None,
    ordinary_gap: float = 12.0,
) -> list[dict[str, Any]]:
    axis = unit(vec_sub(base["p2"], base["p1"]))
    origin = base["p1"]
    base_interval = sorted((0.0, dot(vec_sub(base["p2"], origin), axis)))
    group = [base]
    current_range = [base_interval[0], base_interval[1]]
    changed = True
    while changed:
        changed = False
        for segment in nearby:
            if segment in group or angle_difference(base["angle_deg"], segment["angle_deg"]) > 4.0:
                continue
            if max(segment_line_distance(base, segment["p1"]), segment_line_distance(base, segment["p2"])) > 1.8:
                continue
            interval = sorted(
                (dot(vec_sub(segment["p1"], origin), axis), dot(vec_sub(segment["p2"], origin), axis))
            )
            gap = interval_gap(interval, current_range)
            gap_interval = (
                [current_range[1], interval[0]]
                if interval[0] > current_range[1]
                else [interval[1], current_range[0]]
            )
            bridges_text = (
                bridge_interval is not None
                and interval_gap(gap_interval, bridge_interval) <= ordinary_gap * 0.35
            )
            if gap <= ordinary_gap or (gap <= max_gap and bridges_text):
                group.append(segment)
                current_range[0] = min(current_range[0], interval[0])
                current_range[1] = max(current_range[1], interval[1])
                changed = True
    return group


def group_endpoints(
    group: Sequence[dict[str, Any]], base: dict[str, Any]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    axis = unit(vec_sub(base["p2"], base["p1"]))
    origin = tuple(base["p1"])
    values: list[tuple[float, tuple[float, float]]] = []
    for segment in group:
        for endpoint in (segment["p1"], segment["p2"]):
            values.append((dot(vec_sub(endpoint, origin), axis), tuple(endpoint)))
    values.sort(key=lambda item: item[0])
    return values[0][1], values[-1][1], axis, (values[0][0], values[-1][0])


def nearby_arrow_ids(
    arrows: Sequence[dict[str, Any]], endpoints: Sequence[Sequence[float]], tolerance: float
) -> list[str]:
    """Find arrows attached to any endpoint by either their tip or their base.

    A filled CAD arrow is often drawn with its base touching the dimension-line
    endpoint while its geometric tip points outward.  Tip-only distance then
    reports a false gap roughly equal to the arrow length.  The arrow bounding
    box is a conservative proxy for that base contact; line-direction checks in
    the callers still prevent perpendicular arrows from being borrowed.
    """
    return [
        arrow["id"] for arrow in arrows
        if any(
            distance(arrow["tip"], endpoint) <= tolerance
            or (
                arrow.get("bbox")
                and point_rect_distance(endpoint, arrow["bbox"]) <= max(0.8, tolerance * 0.15)
            )
            for endpoint in endpoints
        )
    ]


def collinear_arrow_ids(
    arrows: Sequence[dict[str, Any]], group: Sequence[dict[str, Any]],
    line_angle: float, tolerance: float,
) -> list[str]:
    """Find arrow tips on a dimension line, including internal junctions.

    Short dimensions often put text outside the measured span and continue the
    line past an arrow.  In that layout the true arrow tips are not the outer
    endpoints of the merged collinear group.
    """
    result: list[str] = []
    for arrow in arrows:
        direction = arrow.get("direction", (1.0, 0.0))
        arrow_angle = normalize_180(math.degrees(math.atan2(direction[1], direction[0])))
        if angle_difference(line_angle, arrow_angle) > 25.0:
            continue
        tip_on_line = any(
            point_segment_distance(arrow["tip"], segment["p1"], segment["p2"]) <= tolerance
            for segment in group
        )
        base_touches_split_line = bool(arrow.get("bbox")) and any(
            point_rect_distance(endpoint, arrow["bbox"]) <= max(0.8, tolerance * 0.3)
            for segment in group
            for endpoint in (segment["p1"], segment["p2"])
        )
        if tip_on_line or base_touches_split_line:
            result.append(arrow["id"])
    return result


def extension_segments(
    endpoints: Sequence[Sequence[float]], line_angle: float,
    segments: Sequence[dict[str, Any]], minimum_length: float,
) -> list[str]:
    result: list[str] = []
    for endpoint in endpoints:
        for segment in segments:
            if segment["length"] < minimum_length:
                continue
            perpendicular = angle_difference(line_angle, segment["angle_deg"])
            if not 60.0 <= perpendicular <= 120.0:
                continue
            if min(distance(endpoint, segment["p1"]), distance(endpoint, segment["p2"])) <= 2.5:
                result.append(segment["id"])
    return list(dict.fromkeys(result))


def best_dimension_line(
    annotation: dict[str, Any], geometry: PageGeometry, page_rect: Sequence[float]
) -> dict[str, Any] | None:
    center = bbox_center(annotation["bbox"])
    size = max(annotation["font_size"], 4.0)
    diagonal = math.hypot(page_rect[2] - page_rect[0], page_rect[3] - page_rect[1])
    search_radius = max(size * 7.0, 30.0)
    nearby = [
        segment for segment in geometry.segments
        if 2.0 <= segment["length"] <= diagonal * 0.82
        and point_segment_distance(center, segment["p1"], segment["p2"]) <= search_radius
    ]
    arrow_outline_segment_ids = {
        segment_id
        for arrow in geometry.arrows
        for segment_id in arrow.get("segment_ids", [])
    }
    nearby.sort(key=lambda segment: point_segment_distance(center, segment["p1"], segment["p2"]))
    candidates: list[dict[str, Any]] = []
    for base in nearby[:28]:
        text_width = annotation["bbox"][2] - annotation["bbox"][0]
        base_axis = unit(vec_sub(base["p2"], base["p1"]))
        base_origin_projection = dot(base["p1"], base_axis)
        text_bridge_interval = [
            value - base_origin_projection
            for value in local_range(annotation["bbox"], base_axis)
        ]
        group = collinear_group(
            base,
            nearby,
            max_gap=max(size * 5.0, text_width + size * 4.0, 25.0),
            bridge_interval=text_bridge_interval,
            ordinary_gap=max(8.0, size * 1.35),
        )
        start, end, axis, group_interval = group_endpoints(group, base)
        normal = (-axis[1], axis[0])
        perpendicular_distance = abs(dot(vec_sub(center, start), normal))
        text_interval = local_range(annotation["bbox"], axis)
        origin_projection = dot(start, axis)
        projected_group = (origin_projection, origin_projection + distance(start, end))
        if projected_group[0] > projected_group[1]:
            projected_group = (projected_group[1], projected_group[0])
        text_inside = interval_gap(text_interval, projected_group) <= size * 1.5
        arrow_tolerance = max(3.0, size * 0.55)
        attached_arrow_ids = set(nearby_arrow_ids(geometry.arrows, (start, end), arrow_tolerance))
        arrow_ids = [
            arrow["id"]
            for arrow in geometry.arrows
            if arrow["id"] in attached_arrow_ids
            and angle_difference(
                base["angle_deg"],
                normalize_180(
                    math.degrees(
                        math.atan2(
                            arrow.get("direction", (1.0, 0.0))[1],
                            arrow.get("direction", (1.0, 0.0))[0],
                        )
                    )
                ),
            )
            <= 25.0
        ]
        arrow_ids.extend(
            collinear_arrow_ids(geometry.arrows, group, base["angle_deg"], arrow_tolerance * 0.55)
        )
        arrow_ids = list(dict.fromkeys(arrow_ids))
        arrows_by_id = {arrow["id"]: arrow for arrow in geometry.arrows}
        minimum_arrow_span = max(2.0, size * 0.18)
        arrow_ids = [
            arrow_id
            for arrow_id in arrow_ids
            if arrow_id in arrows_by_id
            and (
                not arrows_by_id[arrow_id].get("bbox")
                or max(
                    arrows_by_id[arrow_id]["bbox"][2] - arrows_by_id[arrow_id]["bbox"][0],
                    arrows_by_id[arrow_id]["bbox"][3] - arrows_by_id[arrow_id]["bbox"][1],
                )
                >= minimum_arrow_span
            )
        ]
        ext_ids = extension_segments((start, end), base["angle_deg"], nearby, size * 0.8)
        root_axis = tuple(annotation["direction"])
        root_angle = normalize_180(math.degrees(math.atan2(root_axis[1], root_axis[0])))
        orientation_diff = angle_difference(root_angle, base["angle_deg"])
        orientation_bonus = 12.0 if orientation_diff <= 15.0 else (4.0 if orientation_diff >= 75.0 else 0.0)

        projections = []
        for segment in group:
            projections.append(sorted((dot(segment["p1"], axis), dot(segment["p2"], axis))))
        left_of_text = any(value[1] <= text_interval[0] + size * 0.4 for value in projections)
        right_of_text = any(value[0] >= text_interval[1] - size * 0.4 for value in projections)
        score = max(0.0, 42.0 - perpendicular_distance * 3.0)
        score += 15.0 if text_inside else 0.0
        score += 22.0 if left_of_text and right_of_text else 0.0
        score += min(len(arrow_ids), 2) * 22.0
        score += min(len(ext_ids), 2) * 10.0
        score += orientation_bonus
        if annotation.get("type", "linear") == "linear" and 20.0 <= orientation_diff <= 60.0:
            # In aligned drawings, a plain linear value follows its dimension
            # line.  This prevents dense diagonal construction lines from
            # outranking the close, collinear line; arrows can still overcome
            # the penalty when the drawing deliberately uses rotated leaders.
            score -= 18.0
        relationship = "nearby_line"
        if len(arrow_ids) >= 2 or (left_of_text and right_of_text) or len(ext_ids) >= 2:
            relationship = "dimension_line"
        elif len(arrow_ids) == 1:
            relationship = "leader"
        candidates.append(
            {
                "relationship": relationship,
                "score": rounded(score, 2),
                "line_segment_ids": [segment["id"] for segment in group],
                "arrow_ids": arrow_ids,
                "extension_segment_ids": ext_ids,
                "line_endpoints": [[rounded(value) for value in start], [rounded(value) for value in end]],
                "line_angle_deg": rounded(base["angle_deg"], 2),
                "text_to_line_distance": rounded(perpendicular_distance, 2),
                "_arrow_outline_only": len(arrow_ids) >= 2 and all(
                    segment["id"] in arrow_outline_segment_ids
                    or any(
                        arrow_id in arrows_by_id
                        and arrows_by_id[arrow_id].get("bbox")
                        and point_rect_distance(segment["p1"], arrows_by_id[arrow_id]["bbox"]) <= 0.8
                        and point_rect_distance(segment["p2"], arrows_by_id[arrow_id]["bbox"]) <= 0.8
                        for arrow_id in arrow_ids
                    )
                    for segment in group
                ),
            }
        )
    if not candidates:
        return None
    distinct: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        signature = (
            candidate["relationship"],
            tuple(sorted(candidate["line_segment_ids"])),
            tuple(sorted(candidate["arrow_ids"])),
            tuple(sorted(candidate["extension_segment_ids"])),
        )
        previous = distinct.get(signature)
        if previous is None or candidate["score"] > previous["score"]:
            distinct[signature] = candidate
    candidates = list(distinct.values())
    candidates = [
        candidate for candidate in candidates
        if not candidate["_arrow_outline_only"]
        or not any(
            not alternative["_arrow_outline_only"]
            and set(candidate["arrow_ids"]) == set(alternative["arrow_ids"])
            for alternative in candidates
        )
    ]
    candidates.sort(key=lambda value: (-value["score"], tuple(sorted(value["line_segment_ids"]))))
    best = candidates[0]
    best["score_margin"] = rounded(best["score"] - (candidates[1]["score"] if len(candidates) > 1 else 0.0), 2)
    best["unique"] = best["score"] >= 62.0 and best["score_margin"] >= 7.0
    best.pop("_arrow_outline_only", None)
    return best


def find_leader_chain(annotation: dict[str, Any], geometry: PageGeometry) -> dict[str, Any] | None:
    box = annotation["bbox"]
    center = bbox_center(box)
    size = max(annotation["font_size"], 4.0)
    by_id = {segment["id"]: segment for segment in geometry.segments}
    starts = [
        segment for segment in geometry.segments
        if min(point_rect_distance(segment["p1"], box), point_rect_distance(segment["p2"], box)) <= size * 1.6
        and segment["length"] >= 2.0
    ]
    arrows_by_tip = geometry.arrows
    direct_candidates: list[dict[str, Any]] = []
    direct_starts = [
        segment
        for segment in starts
        if min(point_rect_distance(segment["p1"], box), point_rect_distance(segment["p2"], box))
        <= max(1.5, size * 0.1)
    ]
    for start in direct_starts:
        frontier = [(start["id"], [start["id"]])]
        seen = {start["id"]}
        while frontier:
            segment_id, path = frontier.pop(0)
            segment = by_id[segment_id]
            matching_arrows = [
                arrow
                for arrow in arrows_by_tip
                if any(
                    distance(arrow["tip"], endpoint) <= max(3.0, size * 0.35)
                    for endpoint in (segment["p1"], segment["p2"])
                )
                and angle_difference(
                    segment["angle_deg"],
                    normalize_180(
                        math.degrees(
                            math.atan2(
                                arrow.get("direction", (1.0, 0.0))[1],
                                arrow.get("direction", (1.0, 0.0))[0],
                            )
                        )
                    ),
                )
                <= 25.0
            ]
            if matching_arrows:
                arrow = min(matching_arrows, key=lambda value: distance(value["tip"], center))
                chain_length = math.fsum(by_id[value]["length"] for value in path)
                if chain_length <= size * 12.0:
                    continuation_ids = [
                        value["id"]
                        for value in geometry.segments
                        if value["id"] not in path
                        and value["path_id"] == start["path_id"]
                        and angle_difference(value["angle_deg"], start["angle_deg"]) <= 2.0
                        and min(
                            point_rect_distance(value["p1"], box),
                            point_rect_distance(value["p2"], box),
                        )
                        <= max(2.0, size * 0.25)
                    ]
                    direct_candidates.append(
                        {
                            "relationship": "leader",
                            "score": rounded(95.0 - chain_length / max(size, 1.0), 2),
                            "score_margin": None,
                            "unique": True,
                            "line_segment_ids": path + continuation_ids,
                            "arrow_ids": [arrow["id"]],
                            "extension_segment_ids": [],
                            "leader_end": [rounded(value) for value in arrow["tip"]],
                            "chain_length": rounded(chain_length, 2),
                            "direct_path": True,
                        }
                    )
                continue
            if len(path) >= 4:
                continue
            for neighbor in sorted(geometry.adjacency.get(segment_id, set())):
                if neighbor in seen or by_id[neighbor]["path_id"] != start["path_id"]:
                    continue
                seen.add(neighbor)
                frontier.append((neighbor, path + [neighbor]))

    if direct_candidates:
        return min(
            direct_candidates,
            key=lambda value: (value["chain_length"], len(value["line_segment_ids"])),
        )

    best: dict[str, Any] | None = None
    for start in starts[:20]:
        visited = {start["id"]}
        frontier = [(start["id"], 0)]
        while frontier and len(visited) <= 24:
            segment_id, depth = frontier.pop(0)
            if depth >= 4:
                continue
            for neighbor in sorted(geometry.adjacency.get(segment_id, set())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        ordered_visited = sorted(visited)
        endpoints = [
            endpoint for segment_id in ordered_visited
            for endpoint in (by_id[segment_id]["p1"], by_id[segment_id]["p2"])
        ]
        arrow_ids = nearby_arrow_ids(geometry.arrows, endpoints, max(3.0, size * 0.5))
        if not arrow_ids:
            continue
        farthest = max(endpoints, key=lambda endpoint: distance(endpoint, center))
        chain_length = math.fsum(by_id[segment_id]["length"] for segment_id in ordered_visited)
        score = 55.0 + min(distance(farthest, center) / max(size, 1.0), 25.0)
        score += 8.0 if len(arrow_ids) == 1 else -4.0 * max(0, len(arrow_ids) - 1)
        score -= 0.25 * max(0, len(visited) - 12)
        candidate = {
            "relationship": "leader",
            "score": rounded(score, 2),
            "score_margin": None,
            "unique": len(arrow_ids) == 1,
            "line_segment_ids": sorted(visited),
            "arrow_ids": arrow_ids,
            "extension_segment_ids": [],
            "leader_end": [rounded(value) for value in farthest],
            "chain_length": rounded(chain_length, 2),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def associate_geometry(
    annotation: dict[str, Any], geometry: PageGeometry, page_rect: Sequence[float]
) -> dict[str, Any]:
    dimension = best_dimension_line(annotation, geometry, page_rect)
    leader = find_leader_chain(annotation, geometry)
    if (
        leader
        and leader.get("direct_path")
        and annotation.get("type") in {"radius", "spherical_radius"}
    ):
        selected = leader
    elif leader and (dimension is None or dimension["relationship"] == "nearby_line" or leader["score"] > dimension["score"] + 12):
        selected = leader
    elif dimension:
        selected = dimension
    else:
        selected = {
            "relationship": "none", "score": 0.0, "score_margin": None,
            "unique": False, "line_segment_ids": [], "arrow_ids": [],
            "extension_segment_ids": [],
        }

    # A low score margin can merely mean several segments describe the same
    # crowded detail. Promote only physically interpretable evidence patterns;
    # bare nearby lines remain uncertain regardless of score.
    size = max(annotation["font_size"], 4.0)
    arrow_count = len(selected.get("arrow_ids", []))
    relationship = selected["relationship"]
    if relationship == "dimension_line" and arrow_count >= 2 and selected["score"] >= 70.0:
        selected["unique"] = True
        selected["confidence_basis"] = "two_or_more_arrowheads_on_dimension_line"
    elif relationship == "leader" and selected.get("direct_path") and arrow_count == 1:
        selected["unique"] = True
        selected["confidence_basis"] = "same_path_single_arrow_leader"
    elif (
        relationship in {"dimension_line", "leader"}
        and arrow_count == 1
        and selected["score"] >= 70.0
        and selected.get("text_to_line_distance") is not None
        and selected["text_to_line_distance"] <= size * 0.55
    ):
        selected["unique"] = True
        selected["confidence_basis"] = "single_arrowhead_with_close_text_attachment"
    elif (
        relationship == "leader"
        and arrow_count >= 2
        and selected["score"] >= 55.0
        and selected.get("chain_length", math.inf) <= size * 12.0
    ):
        selected["unique"] = True
        selected["confidence_basis"] = "text_connected_two_arrow_chain"
    elif selected.get("unique"):
        selected["confidence_basis"] = "score_margin_unique"
    else:
        selected["confidence_basis"] = "insufficient_unique_geometry"
    return selected


def surface_roughness_evidence(
    annotation: dict[str, Any], geometry: PageGeometry
) -> dict[str, Any] | None:
    """Recognize the two-stroke ISO surface-texture check symbol by geometry."""
    if not re.fullmatch(NUMBER_RE, annotation["normalized_text"].strip()):
        return None
    center = bbox_center(annotation["bbox"])
    size = max(annotation["font_size"], 4.0)
    nearby = [
        segment for segment in geometry.segments
        if 0.75 * size <= segment["length"] <= 3.6 * size
        and point_segment_distance(center, segment["p1"], segment["p2"]) <= 1.35 * size
    ]
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for index, first in enumerate(nearby):
        for second in nearby[index + 1:]:
            if first["path_id"] != second["path_id"]:
                continue
            shared = min(
                distance(left, right)
                for left in (first["p1"], first["p2"])
                for right in (second["p1"], second["p2"])
            )
            if shared > 0.8:
                continue
            short, long = sorted((first, second), key=lambda segment: segment["length"])
            ratio = long["length"] / max(short["length"], 0.1)
            angle = angle_difference(short["angle_deg"], long["angle_deg"])
            if not 1.65 <= ratio <= 2.35 or not 45.0 <= angle <= 75.0:
                continue
            score = (
                point_segment_distance(center, short["p1"], short["p2"])
                + point_segment_distance(center, long["p1"], long["p2"])
                + abs(ratio - 2.0) * size
                + abs(angle - 60.0) * 0.1
            )
            if best is None or score < best[0]:
                best = (score, short, long)
    if best is None:
        return None
    _, short, long = best
    return {
        "relationship": "surface_roughness_symbol",
        "score": rounded(max(70.0, 100.0 - best[0] * 1.5), 2),
        "score_margin": None,
        "unique": True,
        "line_segment_ids": [short["id"], long["id"]],
        "arrow_ids": [],
        "extension_segment_ids": [],
        "confidence_basis": "two_stroke_surface_texture_symbol",
    }


def detect_title_block_rect(
    geometry: PageGeometry, page_rect: Sequence[float]
) -> list[float] | None:
    width = page_rect[2] - page_rect[0]
    height = page_rect[3] - page_rect[1]
    horizontal: list[tuple[float, float, float, float]] = []
    for segment in geometry.segments:
        if angle_difference(segment["angle_deg"], 0.0) > 2.0:
            continue
        x0, x1 = sorted((segment["p1"][0], segment["p2"][0]))
        y = (segment["p1"][1] + segment["p2"][1]) / 2.0
        if (
            y >= page_rect[1] + height * 0.60
            and width * 0.14 <= x1 - x0 <= width * 0.80
            and x1 >= page_rect[0] + width * 0.72
            and x0 >= page_rect[0] + width * 0.35
        ):
            horizontal.append((x0, x1, y, x1 - x0))
    groups: list[list[tuple[float, float, float, float]]] = []
    for line in sorted(horizontal, key=lambda value: value[0]):
        group = next((values for values in groups if abs(values[0][0] - line[0]) <= 4.0), None)
        if group is None:
            groups.append([line])
        else:
            group.append(line)
    eligible = [group for group in groups if len(group) >= 3]
    if not eligible:
        return None
    group = max(eligible, key=lambda values: (len(values), sum(value[3] for value in values)))
    top = min(value[2] for value in group)
    bottom = max(value[2] for value in group)
    if bottom - top < height * 0.06:
        return None
    left = min(value[0] for value in group)
    right = max(value[1] for value in horizontal if value[2] >= top - 3.0)
    return [rounded(left - 2.0), rounded(top - 2.0), rounded(right + 2.0), rounded(page_rect[3])]


def detect_technical_note_rects(
    tokens: Sequence[dict[str, Any]], page_rect: Sequence[float],
    title_block_rect: Sequence[float] | None = None,
) -> list[list[float]]:
    """Find explicit technical-requirement blocks headed by a known label."""
    page_width = page_rect[2] - page_rect[0]
    result: list[list[float]] = []
    for heading in tokens:
        if not TECHNICAL_NOTE_HEADING_RE.search(heading["normalized_text"]):
            continue
        left = max(page_rect[0], heading["bbox"][0] - max(heading["size"], 4.0))
        right = page_rect[2]
        if (
            title_block_rect
            and title_block_rect[0] >= heading["bbox"][0] + page_width * 0.08
        ):
            right = title_block_rect[0]
        else:
            aligned = [
                token for token in tokens
                if token["bbox"][1] >= heading["bbox"][1]
                and abs(token["bbox"][0] - heading["bbox"][0]) <= page_width * 0.04
            ]
            if aligned:
                right = min(page_rect[2], max(token["bbox"][2] for token in aligned) + 4.0)
        result.append([
            rounded(left), rounded(heading["bbox"][1]),
            rounded(right), rounded(page_rect[3]),
        ])
    return result


def metadata_reason(
    annotation: dict[str, Any], page_rect: Sequence[float],
    title_block_rect: Sequence[float] | None = None,
    technical_note_rects: Sequence[Sequence[float]] = (),
) -> str | None:
    text = annotation["normalized_text"].strip()
    line_text = normalize_text(annotation["context_line_text"]).upper()
    center = bbox_center(annotation["bbox"])
    width, height = page_rect[2] - page_rect[0], page_rect[3] - page_rect[1]
    if any(
        rect[0] <= center[0] <= rect[2] and rect[1] <= center[1] <= rect[3]
        for rect in technical_note_rects
    ):
        return "inside_detected_technical_note_block"
    if title_block_rect and (
        title_block_rect[0] <= center[0] <= title_block_rect[2]
        and title_block_rect[1] <= center[1] <= title_block_rect[3]
    ):
        return "inside_detected_title_block"
    if re.fullmatch(r"\d+\s*:\s*\d+", text):
        return "scale_not_dimension"
    if re.fullmatch(r"[A-Z0-9-]*\s*\(\s*\d+\s*:\s*\d+\s*\)", text, re.IGNORECASE):
        return "scale_not_dimension"
    if re.search(r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", text):
        return "date_not_dimension"
    if re.fullmatch(r"[A-Za-zА-Яа-я]{1,8}-\d{3,}(?:[-./][\wА-Яа-я]+)*", text):
        return "probable_drawing_number"
    if any(keyword in line_text for keyword in METADATA_KEYWORDS):
        return "title_or_note_metadata"
    near_border = (
        center[0] <= page_rect[0] + width * 0.025
        or center[0] >= page_rect[2] - width * 0.025
        or center[1] <= page_rect[1] + height * 0.025
        or center[1] >= page_rect[3] - height * 0.025
    )
    if near_border and re.fullmatch(r"[A-Za-z]?\d{1,2}", text):
        return "drawing_border_coordinate"
    identifier_border = (
        center[0] <= page_rect[0] + width * 0.06
        or center[0] >= page_rect[2] - width * 0.06
        or center[1] <= page_rect[1] + height * 0.06
        or center[1] >= page_rect[3] - height * 0.06
    )
    if identifier_border and re.fullmatch(r"\d{5,}", text):
        return "drawing_border_identifier"
    if (
        annotation["geometry"]["relationship"] in {"none", "nearby_line"}
        and center[0] > page_rect[0] + width * 0.55
        and center[1] > page_rect[1] + height * 0.65
        and re.fullmatch(r"\d{5,}(?:[-./]\w+)*", text)
    ):
        return "probable_drawing_number"
    return None


def classify_status(
    annotation: dict[str, Any], page_rect: Sequence[float],
    title_block_rect: Sequence[float] | None = None,
    technical_note_rects: Sequence[Sequence[float]] = (),
) -> tuple[str, str | None]:
    metadata = metadata_reason(
        annotation, page_rect, title_block_rect, technical_note_rects
    )
    if metadata:
        return "metadata", metadata
    if re.match(r"^[+\-±]", annotation["raw_text"]):
        return "needs_review", "possible_orphan_tolerance_fragment"
    if annotation["nominal"] is None:
        return "needs_review", "nominal_value_not_parsed"
    if annotation["parse_notes"]:
        return "needs_review", ";".join(annotation["parse_notes"])
    geometry = annotation["geometry"]
    if annotation.get("assembly_basis") == "quote_anchored_imperial_fraction":
        arrow_count = len(geometry.get("arrow_ids", []))
        close_text = (
            geometry.get("text_to_line_distance") is not None
            and geometry["text_to_line_distance"] <= max(annotation["font_size"], 4.0) * 0.55
        )
        if (
            geometry["relationship"] == "dimension_line"
            and geometry["score"] >= 65.0
            and (arrow_count >= 1 or close_text)
        ) or (
            geometry["relationship"] == "leader"
            and geometry["score"] >= 55.0
            and arrow_count >= 2
        ):
            return "accepted", None
    strong_symbol = annotation["type"] != "linear"
    explicit_tolerance = annotation["tolerance_upper"] is not None or annotation["tolerance_lower"] is not None
    if geometry["relationship"] in {"dimension_line", "leader"} and geometry["unique"]:
        return "accepted", None
    if strong_symbol or explicit_tolerance:
        reason = None if geometry["relationship"] != "nearby_line" else "geometry_link_not_unique"
        return ("accepted", reason) if reason is None else ("needs_review", reason)
    if geometry["relationship"] == "none":
        return "needs_review", "numeric_text_without_vector_dimension_evidence"
    if not geometry["unique"]:
        return "needs_review", "geometry_link_not_unique"
    return "accepted", None


def deduplicate_annotations(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def token_set(record: dict[str, Any]) -> frozenset[str]:
        return frozenset([record["root_token_id"], *record["fragment_token_ids"]])

    def preference(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            len(record["parse_notes"]),
            0 if "/" in record["raw_text"] else 1,
            0 if record["type"] != "linear" else 1,
            -len(record["fragment_token_ids"]),
            -record["geometry"]["score"],
        )

    by_signature: dict[frozenset[str], dict[str, Any]] = {}
    for record in records:
        signature = token_set(record)
        current = by_signature.get(signature)
        if current is None or preference(record) < preference(current):
            by_signature[signature] = record
    unique = list(by_signature.values())
    suppressed: set[str] = set()
    for candidate in unique:
        for owner in unique:
            if candidate is owner or candidate["root_token_id"] not in owner["fragment_token_ids"]:
                continue
            owner_tokens = token_set(owner)
            candidate_tokens = token_set(candidate)
            smaller_fragment = candidate["font_size"] <= owner["font_size"] * 0.9
            fraction_owner = (
                "/" in owner["raw_text"] and '"' in owner["raw_text"]
                and candidate_tokens.issubset(owner_tokens)
            )
            orphan_tolerance = bool(re.match(r"^[+\-±]", candidate["raw_text"]))
            if smaller_fragment or fraction_owner or orphan_tolerance:
                suppressed.add(candidate["root_token_id"])
                break
    return [record for record in unique if record["root_token_id"] not in suppressed]


def mark_shared_fragment_conflicts(records: Sequence[dict[str, Any]]) -> None:
    owners: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] == "metadata":
            continue
        for fragment_id in record["fragment_token_ids"]:
            owners.setdefault(fragment_id, []).append(record)
    for fragment_id, conflicting in owners.items():
        if len(conflicting) < 2:
            continue
        conflict_ids = sorted(record["root_token_id"] for record in conflicting)
        reason = f"shared_fragment_conflict:{fragment_id}:" + ",".join(conflict_ids)
        for record in conflicting:
            record["status"] = "needs_review"
            existing = record.get("review_reason")
            record["review_reason"] = f"{existing};{reason}" if existing else reason


def fragment_evidence_ids(fragments: Sequence[dict[str, Any]]) -> list[str]:
    """Return physical source ids as well as any synthetic fragment id."""
    result: list[str] = []
    for fragment in fragments:
        token = fragment["token"]
        result.extend((token["id"], *token.get("source_token_ids", [])))
    return list(dict.fromkeys(result))


def residual_dimension_fragment_kind(token: dict[str, Any]) -> str | None:
    role = fragment_role(token["normalized_text"])
    if role in {"quantity_prefix", "tolerance", "fit", "unit"}:
        return role
    if role == "symbol" and token["normalized_text"].upper() not in {
        "(", ")", "[", "]", "THRU", "DEPTH", "EQ", "TYP",
        # A bare C/c is commonly a section or border label. A split chamfer
        # prefix is still attached by the primary spatial assembler.
        "C",
    }:
        return role
    return None


def reconcile_evidence_inventory(
    records: Sequence[dict[str, Any]], tokens: Sequence[dict[str, Any]],
    geometries: dict[int, PageGeometry], page_rects: dict[int, list[float]],
    page_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Account for every raw token and detected vector annotation symbol.

    The primary parser still performs all assembly.  This final pass only
    exposes suspicious leftovers and revokes green status when a leftover is
    spatially compatible with an accepted dimension.
    """
    contexts = {page["page"]: page for page in page_payloads}
    owners: dict[str, list[dict[str, Any]]] = {}
    claimed_symbols: set[str] = set()
    for record in records:
        for token_id in (record["root_token_id"], *record["fragment_token_ids"]):
            owners.setdefault(token_id, []).append(record)
        claimed_symbols.update(record.get("vector_symbol_ids", []))

    def in_region(center: Sequence[float], rect: Sequence[float] | None) -> bool:
        return bool(rect) and rect[0] <= center[0] <= rect[2] and rect[1] <= center[1] <= rect[3]

    def border_metadata(token: dict[str, Any]) -> bool:
        rect = page_rects[token["page"]]
        center = bbox_center(token["bbox"])
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        return (
            center[0] <= rect[0] + width * 0.025
            or center[0] >= rect[2] - width * 0.025
            or center[1] <= rect[1] + height * 0.025
            or center[1] >= rect[3] - height * 0.025
        )

    def nearby_record(token: dict[str, Any]) -> dict[str, Any] | None:
        size = max(token["size"], 4.0)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            if record["page"] != token["page"] or record["status"] == "metadata":
                continue
            if angle_difference(token["rotation_deg"], record["rotation_deg"]) > 10.0:
                continue
            u = tuple(record["direction"])
            v = (-u[1], u[0])
            along = interval_gap(local_range(token["bbox"], u), local_range(record["bbox"], u))
            perpendicular = interval_gap(local_range(token["bbox"], v), local_range(record["bbox"], v))
            if along <= size * 2.5 and perpendicular <= size * 0.8:
                candidates.append((along + perpendicular * 2.0, record))
        return min(candidates, key=lambda value: (value[0], value[1]["id"]))[1] if candidates else None

    def revoke_green(record: dict[str, Any], reason: str) -> None:
        if record["status"] == "accepted":
            record["status"] = "needs_review"
        existing = record.get("review_reason")
        if reason not in (existing or ""):
            record["review_reason"] = f"{existing};{reason}" if existing else reason

    dispositions: list[dict[str, Any]] = []
    unresolved_text: list[dict[str, Any]] = []
    for token in tokens:
        token_owners = owners.get(token["id"], [])
        if token_owners:
            dimension_owners = [record for record in token_owners if record["status"] != "metadata"]
            dispositions.append(
                {
                    "token_id": token["id"],
                    "disposition": "dimension_component" if dimension_owners else "metadata_component",
                    "owner_ids": [record["id"] for record in token_owners],
                }
            )
            continue
        context = contexts[token["page"]]
        center = bbox_center(token["bbox"])
        metadata_region = (
            in_region(center, context.get("detected_title_block"))
            or any(in_region(center, rect) for rect in context.get("detected_technical_note_blocks", []))
            or border_metadata(token)
        )
        kind = None if metadata_region else residual_dimension_fragment_kind(token)
        if kind:
            nearest = nearby_record(token)
            item = {
                "token_id": token["id"], "text": token["normalized_text"],
                "kind": kind, "page": token["page"], "bbox": token["bbox"],
                "nearby_dimension_id": nearest["id"] if nearest else None,
            }
            unresolved_text.append(item)
            dispositions.append({"token_id": token["id"], "disposition": "unresolved_dimension_fragment"})
            if nearest:
                revoke_green(nearest, f"unreconciled_dimension_fragment:{token['id']}")
        else:
            dispositions.append(
                {
                    "token_id": token["id"],
                    "disposition": "metadata_region" if metadata_region else "non_dimension_text",
                }
            )

    unresolved_symbols: list[dict[str, Any]] = []
    for page, geometry in geometries.items():
        for symbol in geometry.symbols:
            if symbol["id"] in claimed_symbols:
                continue
            nearby = [
                (point_rect_distance(symbol["center"], record["bbox"]), record)
                for record in records
                if record["page"] == page and record["status"] != "metadata"
                and point_rect_distance(symbol["center"], record["bbox"])
                <= max(record["font_size"] * 1.5, 12.0)
            ]
            nearest = min(nearby, key=lambda value: (value[0], value[1]["id"]))[1] if nearby else None
            item = {**symbol, "page": page, "nearby_dimension_id": nearest["id"] if nearest else None}
            unresolved_symbols.append(item)
            if nearest:
                revoke_green(nearest, f"unreconciled_vector_symbol:{symbol['id']}")

    counts = dict(Counter(item["disposition"] for item in dispositions))
    return {
        "passed": not unresolved_text and not unresolved_symbols,
        "text_token_dispositions": dispositions,
        "text_disposition_counts": counts,
        "unresolved_text_fragments": unresolved_text,
        "vector_symbols_total": sum(len(geometry.symbols) for geometry in geometries.values()),
        "vector_symbols_claimed": len(claimed_symbols),
        "unresolved_vector_symbols": unresolved_symbols,
    }


def analyze_pdf(input_path: Path, default_unit: str | None = None) -> dict[str, Any]:
    with fitz.open(input_path) as document:
        all_tokens: list[dict[str, Any]] = []
        page_payloads: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        internal_geometries: dict[int, PageGeometry] = {}
        page_rects: dict[int, list[float]] = {}
        for page_index, page in enumerate(document, start=1):
            page_rect = [rounded(page.rect.x0), rounded(page.rect.y0), rounded(page.rect.x1), rounded(page.rect.y1)]
            page_rects[page_index] = page_rect
            tokens = extract_text_tokens(page, page_index)
            geometry = extract_vector_geometry(page, page_index)
            title_block_rect = detect_title_block_rect(geometry, page_rect)
            technical_note_rects = detect_technical_note_rects(
                tokens, page_rect, title_block_rect
            )
            geometric_tolerances, geometric_tolerance_consumed = (
                detect_geometric_tolerance_frames(tokens, geometry, default_unit)
            )
            records.extend(geometric_tolerances)
            imperial_clusters, imperial_consumed = detect_imperial_fraction_clusters(
                tokens, geometry.symbols
            )
            generic_tokens = [
                token for token in tokens
                if token["id"] not in imperial_consumed
                and token["id"] not in geometric_tolerance_consumed
            ]
            internal_geometries[page_index] = geometry
            all_tokens.extend(tokens)
            page_payloads.append(
                {
                    "page": page_index,
                    "rect": page_rect,
                    "rotation": page.rotation,
                    "text_token_count": len(tokens),
                    "vector_segment_count": len(geometry.segments),
                    "arrow_candidate_count": len(geometry.arrows),
                    "vector_diameter_symbol_count": sum(
                        symbol["kind"] == "diameter" for symbol in geometry.symbols
                    ),
                    "vector_symbol_count": len(geometry.symbols),
                    "geometric_tolerance_frame_count": len(geometric_tolerances),
                    "detected_title_block": title_block_rect,
                    "detected_technical_note_blocks": technical_note_rects,
                }
            )
            roots = [token for token in tokens if is_root_candidate(token)]
            fragments_by_root: dict[str, list[dict[str, Any]]] = {}
            active_roots: list[dict[str, Any]] = []
            for root in roots:
                if root["id"] in geometric_tolerance_consumed:
                    continue
                if root["id"] in imperial_consumed and root["id"] not in imperial_clusters:
                    continue
                fragments = imperial_clusters.get(root["id"])
                if fragments is None:
                    fragments = collect_fragments(root, generic_tokens, geometry.symbols)
                active_roots.append(root)
                fragments_by_root[root["id"]] = fragments
            resolve_shared_fragment_owners(active_roots, fragments_by_root)

            for root in active_roots:
                fragments = fragments_by_root[root["id"]]
                parsed = parse_annotation(root, fragments, default_unit)
                annotation = {
                    "id": "",
                    "page": page_index,
                    **parsed,
                    "rotation_deg": root["rotation_deg"],
                    "direction": root["direction"],
                    "bbox": bbox_union([root["bbox"]] + [value["token"]["bbox"] for value in fragments]),
                    "font_size": root["size"],
                    "root_token_id": root["id"],
                    "fragment_token_ids": fragment_evidence_ids(fragments),
                    "vector_symbol_ids": [
                        value["token"]["id"] for value in fragments
                        if value["token"].get("vector_symbol")
                    ],
                    "fragments": [
                        {
                            "id": value["token"]["id"],
                            "text": value["token"]["text"],
                            "role": value["role"],
                            "bbox": value["token"]["bbox"],
                            "size": value["token"]["size"],
                        }
                        for value in fragments
                    ],
                    "context_line_text": root["line_text"],
                }
                if root["id"] in imperial_clusters:
                    annotation["assembly_basis"] = "quote_anchored_imperial_fraction"
                roughness = surface_roughness_evidence(annotation, geometry)
                if roughness:
                    annotation["type"] = "surface_roughness"
                    annotation["unit"] = "µm"
                    annotation["unit_source"] = "surface_texture_convention"
                    annotation["surface_roughness_parameter"] = "Ra"
                    annotation["geometry"] = roughness
                else:
                    annotation["geometry"] = associate_geometry(annotation, geometry, page_rect)
                annotation["status"], annotation["review_reason"] = classify_status(
                    annotation, page_rect, title_block_rect, technical_note_rects
                )
                records.append(annotation)

        if not all_tokens:
            raise ValueError("PDF has no extractable text layer; OCR is intentionally not included in this tool.")

        records = deduplicate_annotations(records)
        mark_shared_fragment_conflicts(records)
        records.sort(key=lambda value: (value["page"], value["bbox"][1], value["bbox"][0]))
        page_counters: dict[int, int] = {}
        for record in records:
            page_counters[record["page"]] = page_counters.get(record["page"], 0) + 1
            record["id"] = f"P{record['page']}-D{page_counters[record['page']]:04d}"
        reconciliation = reconcile_evidence_inventory(
            records, all_tokens, internal_geometries, page_rects, page_payloads
        )
        accepted = [record for record in records if record["status"] == "accepted"]
        review = [record for record in records if record["status"] == "needs_review"]
        metadata = [record for record in records if record["status"] == "metadata"]
        return {
            "schema_version": "1.1",
            "source": str(input_path.resolve()),
            "default_unit": default_unit,
            "summary": {
                "pages": len(page_payloads),
                "raw_text_tokens": len(all_tokens),
                "dimension_candidates": len(accepted) + len(review),
                "accepted": len(accepted),
                "needs_review": len(review),
                "green_rate_pct": rounded(
                    len(accepted) * 100.0 / max(len(accepted) + len(review), 1), 2
                ),
                "metadata_candidates": len(metadata),
                "reconciliation_passed": reconciliation["passed"],
                "unresolved_text_fragments": len(reconciliation["unresolved_text_fragments"]),
                "unresolved_vector_symbols": len(reconciliation["unresolved_vector_symbols"]),
                "recovered_vector_diameter_symbols": sum(
                    sum(
                        symbol["kind"] == "diameter"
                        for symbol in internal_geometries[page["page"]].symbols
                    )
                    for page in page_payloads
                ),
                "geometric_tolerance_frames": sum(
                    page["geometric_tolerance_frame_count"] for page in page_payloads
                ),
            },
            "pages": page_payloads,
            "dimensions": accepted + review,
            "metadata_candidates": metadata,
            "reconciliation": reconciliation,
            "raw_text_tokens": all_tokens,
            "_internal_geometries": internal_geometries,
            "_page_rects": page_rects,
        }


CSV_FIELDS = (
    "id", "page", "status", "review_reason", "raw_text", "type", "nominal_text",
    "nominal", "unit", "tolerance_upper", "tolerance_lower", "tolerance_unit",
    "quantity", "reference", "fit", "thread_pitch", "angle_degrees", "angle_minutes",
    "angle_seconds", "distribution_angle_deg", "rotation_deg", "bbox", "geometry_relationship",
    "geometry_score", "geometry_unique", "root_token_id", "fragment_token_ids",
    "geometric_characteristic", "characteristic_symbol", "geometric_tolerance",
    "geometric_tolerance_unit", "tolerance_zone", "datum_references", "controlled_feature",
)


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_internal") and key != "_page_rects"}


def write_csv(path: Path, dimensions: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in dimensions:
            row = dict(record)
            row["bbox"] = json.dumps(record["bbox"], ensure_ascii=False)
            row["geometry_relationship"] = record["geometry"]["relationship"]
            row["geometry_score"] = record["geometry"]["score"]
            row["geometry_unique"] = record["geometry"]["unique"]
            row["fragment_token_ids"] = ";".join(record["fragment_token_ids"])
            row["datum_references"] = ";".join(record.get("datum_references", []))
            writer.writerow(row)


def markdown_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# 结构化尺寸清单", "",
        f"- 来源：`{result['source']}`",
        f"- 自动接受：{summary['accepted']}",
        f"- 待人工核验：{summary['needs_review']}",
        f"- 核销复盘：{'通过' if summary.get('reconciliation_passed') else '存在未解释残片'}",
        f"- 未解释文字残片：{summary.get('unresolved_text_fragments', 0)}",
        f"- 未核销矢量符号：{summary.get('unresolved_vector_symbols', 0)}", "",
        "| ID | 页 | 状态 | 完整标注 | 类型 | 名义值 | 上偏差 | 下偏差 | 角度 | 几何关联 | 核验原因 |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for record in result["dimensions"]:
        lines.append(
            "| " + " | ".join(
                markdown_escape(value) for value in (
                    record["id"], record["page"], record["status"], record["raw_text"],
                    record["type"], record["nominal"], record["tolerance_upper"],
                    record["tolerance_lower"], record["rotation_deg"],
                    record["geometry"]["relationship"], record["review_reason"],
                )
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_pdf(input_path: Path, output_path: Path, result: dict[str, Any]) -> None:
    geometries: dict[int, PageGeometry] = result["_internal_geometries"]
    with fitz.open(input_path) as document:
        for record in result["dimensions"]:
            page = document[record["page"] - 1]
            color = (0.0, 0.65, 0.1) if record["status"] == "accepted" else (1.0, 0.45, 0.0)
            rect = fitz.Rect(record["bbox"])
            page.draw_rect(rect, color=color, width=0.8, overlay=True)
            label_y = max(page.rect.y0 + 7, rect.y0 - 2)
            page.insert_text(
                (rect.x0, label_y), record["id"], fontsize=5.5,
                color=color, overlay=True,
            )
            linked = set(record["geometry"].get("line_segment_ids", []))
            linked.update(record["geometry"].get("extension_segment_ids", []))
            by_id = {segment["id"]: segment for segment in geometries[record["page"]].segments}
            for segment_id in sorted(linked):
                segment = by_id.get(segment_id)
                if segment:
                    page.draw_line(segment["p1"], segment["p2"], color=(0.0, 0.55, 0.85), width=0.8, overlay=True)
        reconciliation = result.get("reconciliation", {})
        residuals = [
            *reconciliation.get("unresolved_text_fragments", []),
            *reconciliation.get("unresolved_vector_symbols", []),
        ]
        for item in residuals:
            page = document[item["page"] - 1]
            rect = fitz.Rect(item["bbox"])
            page.draw_rect(rect, color=(1.0, 0.2, 0.0), width=1.0, overlay=True)
            label = f"U-{item.get('token_id', item.get('id', '?'))}"
            page.insert_text(
                (rect.x0, max(page.rect.y0 + 7, rect.y0 - 2)), label,
                fontsize=5.5, color=(1.0, 0.2, 0.0), overlay=True,
            )
        document.save(output_path, garbage=3, deflate=True)


def write_outputs(
    input_path: Path, output_dir: Path, result: dict[str, Any], review_pdf: bool = True
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    public = public_result(result)
    json_dump(output_dir / "dimension-ledger.json", public)
    json_dump(output_dir / "raw-text.json", public["raw_text_tokens"])
    json_dump(output_dir / "reconciliation.json", public["reconciliation"])
    json_dump(
        output_dir / "needs-review.json",
        [record for record in public["dimensions"] if record["status"] == "needs_review"],
    )
    write_csv(output_dir / "dimensions.csv", public["dimensions"])
    write_markdown(output_dir / "dimensions.md", public)
    if review_pdf:
        write_review_pdf(input_path, output_dir / "review.pdf", result)


def create_self_test_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=420, height=300)
    # Horizontal dimension with a text gap, two extension lines, and open arrows.
    page.draw_line((50, 90), (165, 90), width=0.6)
    page.draw_line((255, 90), (370, 90), width=0.6)
    page.draw_line((50, 60), (50, 120), width=0.6)
    page.draw_line((370, 60), (370, 120), width=0.6)
    page.draw_line((50, 90), (58, 86), width=0.6)
    page.draw_line((50, 90), (58, 94), width=0.6)
    page.draw_line((370, 90), (362, 86), width=0.6)
    page.draw_line((370, 90), (362, 94), width=0.6)
    page.insert_text((180, 94), "25 +/-0.1", fontsize=12)
    # Rotated dimension.
    page.draw_line((105, 150), (105, 260), width=0.6)
    page.draw_line((105, 150), (101, 158), width=0.6)
    page.draw_line((105, 150), (109, 158), width=0.6)
    page.draw_line((105, 260), (101, 252), width=0.6)
    page.draw_line((105, 260), (109, 252), width=0.6)
    page.insert_text((95, 220), "40", fontsize=12, rotate=90)
    # Radius leader.
    page.insert_text((255, 205), "R12", fontsize=12)
    page.draw_line((250, 202), (220, 225), width=0.6)
    page.draw_line((220, 225), (226, 217), width=0.6)
    page.draw_line((220, 225), (229, 228), width=0.6)
    # Split angular tolerance text.
    page.insert_text((175, 145), "74°10'", fontsize=12)
    page.insert_text((218, 145), "+/-5'", fontsize=9)
    # Vector title block and metadata that must not enter the dimension ledger.
    for y in (250, 270, 290):
        page.draw_line((250, y), (410, y), width=0.6)
    for x in (250, 330, 410):
        page.draw_line((x, 250), (x, 295), width=0.6)
    page.insert_text((305, 265), "SCALE 1:2", fontsize=8)
    page.insert_text((350, 285), "987654", fontsize=8)
    document.save(path)
    document.close()


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="dimension-ledger-") as temp:
        root = Path(temp)
        pdf = root / "self-test.pdf"
        output = root / "output"
        create_self_test_pdf(pdf)
        result = analyze_pdf(pdf, default_unit="mm")
        write_outputs(pdf, output, result)
        dimensions = result["dimensions"]
        horizontal = next((record for record in dimensions if record["nominal"] == 25.0), None)
        radius = next((record for record in dimensions if record["type"] == "radius" and record["nominal"] == 12.0), None)
        rotated = next((record for record in dimensions if record["nominal"] == 40.0), None)
        angular = next((record for record in dimensions if record["type"] == "angle"), None)
        assert horizontal, "horizontal dimension missing"
        assert horizontal["tolerance_upper"] == 0.1 and horizontal["tolerance_lower"] == -0.1
        assert horizontal["geometry"]["relationship"] == "dimension_line"
        assert radius and radius["geometry"]["relationship"] in {"leader", "dimension_line"}
        assert rotated and abs(rotated["rotation_deg"] - 90.0) < 0.1
        assert angular and angular["tolerance_upper"] == 5.0 and angular["tolerance_unit"] == "arcmin"
        assert not any(record["raw_text"] == "1:2" for record in dimensions)
        assert result["pages"][0]["detected_title_block"] is not None
        assert (output / "review.pdf").exists()
        print("SELF-TEST OK")
        print(json.dumps(result["summary"], ensure_ascii=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a structured dimension ledger and vector-geometry evidence from a PDF."
    )
    parser.add_argument("input", nargs="?", type=Path, help="vector PDF with an extractable text layer")
    parser.add_argument("-o", "--output", type=Path, help="output directory")
    parser.add_argument("--unit", help="default unit when the annotation has no explicit unit")
    parser.add_argument("--no-review-pdf", action="store_true", help="skip the annotated review PDF")
    parser.add_argument("--self-test", action="store_true", help="run the built-in synthetic PDF check")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if not args.input:
        print("error: input PDF is required", file=sys.stderr)
        return 2
    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 2
    output_dir = (args.output or input_path.with_name(f"{input_path.stem}-dimension-ledger")).resolve()
    try:
        result = analyze_pdf(input_path, default_unit=args.unit)
        write_outputs(input_path, output_dir, result, review_pdf=not args.no_review_pdf)
    except (ValueError, fitz.FileDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = result["summary"]
    print(f"Output: {output_dir}")
    print(
        f"Dimensions: {summary['dimension_candidates']} "
        f"(accepted {summary['accepted']}, needs review {summary['needs_review']}, "
        f"green {summary['green_rate_pct']}%)"
    )
    print(
        f"Evidence: {summary['raw_text_tokens']} text tokens, "
        f"{sum(page['vector_segment_count'] for page in result['pages'])} vector segments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
