"""Regression checks for extract_dimension_ledger.py using generated PDFs only."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_dimension_ledger as ledger


def token(text: str, *, size: float = 12, bbox=(100, 100, 140, 116), rotation=0):
    radians = rotation * 3.141592653589793 / 180
    direction = [round(__import__("math").cos(radians), 6), round(-__import__("math").sin(radians), 6)]
    return {
        "id": "P1-T0001",
        "page": 1,
        "block": 0,
        "line": 0,
        "span": 0,
        "text": text,
        "normalized_text": ledger.normalize_text(text),
        "line_text": text,
        "bbox": list(bbox),
        "quad": [],
        "origin": list(bbox[:2]),
        "font": "Helvetica",
        "size": size,
        "flags": 0,
        "color": 0,
        "direction": direction,
        "rotation_deg": rotation,
    }


class ParserTests(unittest.TestCase):
    def parse(self, text: str):
        return ledger.parse_annotation(token(text), [], "mm")

    def test_symbol_normalization(self):
        self.assertEqual(ledger.normalize_text("Φ25 −0.1"), "Ø25 -0.1")
        self.assertEqual(ledger.normalize_text("25 +/- 0.2"), "25 ± 0.2")

    def test_common_dimension_types(self):
        cases = {
            "Ø25": ("diameter", 25.0),
            "R12": ("radius", 12.0),
            "M20×1.5": ("thread", 20.0),
            "4-Ø18": ("diameter", 18.0),
            "45°": ("angle", 45.0),
            "4.750 SPACING @ 45°": ("spacing", 4.75),
            "(5×)R25": ("radius", 25.0),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = self.parse(text)
                self.assertEqual((parsed["type"], parsed["nominal"]), expected)
        self.assertEqual(self.parse("4-Ø18")["quantity"], 4)
        self.assertEqual(self.parse("(5×)R25")["quantity"], 5)
        self.assertEqual(self.parse("M20×1.5")["thread_pitch"], 1.5)
        self.assertEqual(self.parse("4.750 SPACING @ 45°")["distribution_angle_deg"], 45.0)

    def test_tolerances_and_angle_minutes(self):
        symmetric = self.parse("25 ±0.1")
        self.assertEqual((symmetric["tolerance_upper"], symmetric["tolerance_lower"]), (0.1, -0.1))
        unilateral = self.parse("25 +0.2 -0.1")
        self.assertEqual((unilateral["tolerance_upper"], unilateral["tolerance_lower"]), (0.2, -0.1))
        angle = self.parse("74°10' ±5'")
        self.assertAlmostEqual(angle["nominal"], 74 + 10 / 60, places=6)
        self.assertEqual((angle["tolerance_upper"], angle["tolerance_unit"]), (5.0, "arcmin"))

    def test_fraction_and_reference(self):
        parsed = self.parse('(3 5/8")')
        self.assertEqual(parsed["nominal"], 3.625)
        self.assertTrue(parsed["reference"])

    def test_split_imperial_fraction_preserves_prefix_and_quote(self):
        root = token("2", bbox=(100, 100, 110, 116))
        numerator = token("1", bbox=(114, 93, 120, 108))
        numerator["id"] = "P1-T0002"
        denominator = token("2", bbox=(114, 108, 120, 123))
        denominator["id"] = "P1-T0003"
        quote = token('"', size=13.4, bbox=(124, 100, 130, 116))
        quote["id"] = "P1-T0004"
        diameter = {
            "id": "P1-SYM0001", "kind": "diameter", "text": "Ø",
            "bbox": [86, 101, 96, 115], "center": [91, 108], "rotation_deg": 0.0,
        }
        fragments = ledger.collect_fragments(root, [root, numerator, denominator, quote], [diameter])
        parsed = ledger.parse_annotation(root, fragments, "mm")
        self.assertEqual(parsed["raw_text"], 'Ø2 1/2"')
        self.assertEqual((parsed["type"], parsed["nominal"], parsed["unit"]), ("diameter", 2.5, "in"))
        clusters, consumed = ledger.detect_imperial_fraction_clusters(
            [root, numerator, denominator, quote], [diameter]
        )
        self.assertEqual(set(clusters), {root["id"]})
        self.assertEqual(consumed, {root["id"], numerator["id"], denominator["id"], quote["id"]})

    def test_radius_prefix_is_canonical_root_for_following_stacked_fraction(self):
        radius = token("R3", size=10.05, bbox=(100, 103, 112, 119))
        numerator = token("5", size=10.76, bbox=(114, 96, 120, 110))
        numerator["id"] = "P1-T0002"
        denominator = token("8", size=10.78, bbox=(114, 110, 120, 124))
        denominator["id"] = "P1-T0003"
        quote = token('"', size=13.47, bbox=(124, 103, 130, 119))
        quote["id"] = "P1-T0004"
        tokens = [radius, numerator, denominator, quote]
        clusters, consumed = ledger.detect_imperial_fraction_clusters(tokens)
        self.assertEqual(set(clusters), {radius["id"]})
        self.assertEqual(consumed, {token["id"] for token in tokens})
        parsed = ledger.parse_annotation(radius, clusters[radius["id"]], "mm")
        self.assertEqual(parsed["raw_text"], 'R3 5/8"')
        self.assertEqual((parsed["type"], parsed["nominal"], parsed["unit"]), ("radius", 3.625, "in"))

    def test_imperial_fraction_tolerance_unit_is_inches(self):
        parsed = self.parse('Ø3 5/8" ±1/16"')
        self.assertEqual((parsed["nominal"], parsed["tolerance_upper"], parsed["tolerance_lower"]), (3.625, 0.0625, -0.0625))
        self.assertEqual(parsed["tolerance_unit"], "in")

    def test_border_identifier_is_metadata(self):
        annotation = {
            "normalized_text": "780672",
            "context_line_text": "780672",
            "bbox": [60, 18, 100, 32],
            "geometry": {"relationship": "none"},
        }
        self.assertEqual(
            ledger.metadata_reason(annotation, [0, 0, 1200, 840]),
            "drawing_border_identifier",
        )

    def test_view_name_with_parenthesized_scale_is_metadata(self):
        annotation = {
            "normalized_text": "I(1:1)",
            "context_line_text": "I(1:1)",
            "bbox": [200, 370, 255, 392],
            "geometry": {"relationship": "nearby_line"},
        }
        self.assertEqual(
            ledger.metadata_reason(annotation, [0, 0, 1200, 840]),
            "scale_not_dimension",
        )

    def test_explicit_technical_requirement_block_is_metadata(self):
        heading = token("技术要求：", bbox=(100, 200, 160, 214))
        note = token("1、圆度允许偏差不超过0.25%。", bbox=(100, 220, 280, 234))
        note["id"] = "NOTE"
        rects = ledger.detect_technical_note_rects(
            [heading, note], [0, 0, 500, 400], [300, 250, 500, 400]
        )
        annotation = {
            "normalized_text": note["normalized_text"],
            "context_line_text": note["line_text"],
            "bbox": note["bbox"],
            "geometry": {"relationship": "none"},
        }
        self.assertEqual(
            ledger.metadata_reason(
                annotation, [0, 0, 500, 400], [300, 250, 500, 400], rects
            ),
            "inside_detected_technical_note_block",
        )

    def test_shared_fragment_forces_review(self):
        records = [
            {"root_token_id": "T1", "fragment_token_ids": ["F1"], "status": "accepted", "review_reason": None},
            {"root_token_id": "T2", "fragment_token_ids": ["F1"], "status": "accepted", "review_reason": None},
        ]
        ledger.mark_shared_fragment_conflicts(records)
        self.assertEqual([row["status"] for row in records], ["needs_review", "needs_review"])
        self.assertTrue(all("shared_fragment_conflict:F1" in row["review_reason"] for row in records))

    def test_vector_diameter_and_split_quantity_are_reassembled(self):
        root = token("40", bbox=(100, 100, 125, 116))
        quantity = token("2-", bbox=(65, 100, 79, 116))
        quantity["id"] = "P1-T0002"
        quantity["line"] = 1
        symbol = {
            "id": "P1-SYM0001", "kind": "diameter", "text": "Ø",
            "bbox": [82, 101, 95, 115], "center": [88.5, 108], "rotation_deg": 0.0,
        }
        fragments = ledger.collect_fragments(root, [root, quantity], [symbol])
        parsed = ledger.parse_annotation(root, fragments, "mm")
        self.assertEqual(parsed["raw_text"], "2-Ø40")
        self.assertEqual((parsed["type"], parsed["quantity"], parsed["nominal"]), ("diameter", 2, 40.0))
        self.assertEqual(parsed["parse_notes"], [])

    def test_rotated_vector_diameter_bridges_quantity_to_imperial_fraction(self):
        root = token("5", size=10.76, bbox=(637.927, 173.434, 654.245, 186.077), rotation=300)
        numerator = token("8", size=10.78, bbox=(626.964, 179.75, 643.288, 192.405), rotation=300)
        numerator["id"] = "P1-T0002"
        quote = token('"', size=13.47, bbox=(629.761, 183.887, 652.255, 196.53), rotation=300)
        quote["id"] = "P1-T0003"
        quantity = token("2-", size=10.76, bbox=(619.198, 154.883, 638.306, 172.359), rotation=300)
        quantity["id"] = "P1-T0004"
        symbol = {
            "id": "P1-SYM0001", "kind": "diameter", "text": "Ø",
            "bbox": [629.2, 170.29, 639.8, 177.19], "center": [634.5, 173.74],
            # Deliberately unreliable metadata: attachment must use baseline geometry.
            "rotation_deg": 90.0,
        }
        clusters, _ = ledger.detect_imperial_fraction_clusters(
            [root, numerator, quote, quantity], [symbol]
        )
        parsed = ledger.parse_annotation(root, clusters[root["id"]], "mm")
        self.assertEqual(parsed["raw_text"], '2-Ø5/8"')
        self.assertEqual((parsed["type"], parsed["quantity"], parsed["nominal"]), ("diameter", 2, 0.625))

    def test_rotated_vector_diameter_accepts_axis_aligned_slash(self):
        center = (635.4, 173.74)
        oval = []
        for index in range(48):
            first = 2 * math.pi * index / 48
            second = 2 * math.pi * (index + 1) / 48
            oval.append(
                (
                    "l",
                    fitz.Point(center[0] + 4.4 * math.cos(first), center[1] + 3.45 * math.sin(first)),
                    fitz.Point(center[0] + 4.4 * math.cos(second), center[1] + 3.45 * math.sin(second)),
                )
            )
        slash = ("l", fitz.Point(629.2, 174.49), fitz.Point(639.8, 172.99))
        drawing = {"rect": fitz.Rect(631.0, 170.29, 639.8, 177.19), "items": [*oval, slash]}
        symbol = ledger.detect_vector_diameter_symbol(drawing, 1, 7)
        self.assertIsNotNone(symbol)
        self.assertLessEqual(symbol["bbox"][0], 629.2)

    def test_unclaimed_quantity_revokes_green_status(self):
        root = token("40", bbox=(100, 100, 125, 116))
        quantity = token("2-", bbox=(75, 100, 95, 116))
        quantity["id"] = "QUANTITY"
        record = {
            "id": "P1-D0001", "page": 1, "status": "accepted", "review_reason": None,
            "root_token_id": root["id"], "fragment_token_ids": [], "vector_symbol_ids": [],
            "rotation_deg": 0.0, "direction": [1.0, 0.0], "bbox": root["bbox"],
            "font_size": 12.0,
        }
        result = ledger.reconcile_evidence_inventory(
            [record], [root, quantity], {1: ledger.PageGeometry([], [], {}, [])},
            {1: [0, 0, 500, 400]},
            [{"page": 1, "detected_title_block": None, "detected_technical_note_blocks": []}],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["unresolved_text_fragments"][0]["token_id"], "QUANTITY")
        self.assertEqual(record["status"], "needs_review")

    def test_bare_section_letter_c_is_not_a_high_risk_residual(self):
        self.assertIsNone(ledger.residual_dimension_fragment_kind(token("C")))
        self.assertIsNone(ledger.residual_dimension_fragment_kind(token("c")))

    def test_tolerance_fragments_do_not_jump_to_adjacent_dimension_line(self):
        root = token("2240", bbox=(100, 120, 130, 134))
        sign = token("-", size=9, bbox=(136, 88, 141, 98))
        sign["id"] = "P1-T0002"
        value = token("15", size=9, bbox=(141, 88, 152, 98))
        value["id"] = "P1-T0003"
        fragments = ledger.collect_fragments(root, [root, sign, value])
        self.assertEqual(fragments, [])

    def test_complete_radius_dimension_is_not_attached_as_fit_fragment(self):
        root = token("R10", bbox=(100, 100, 128, 116))
        root["id"] = "R10"
        neighbor = token("R20", bbox=(133, 100, 161, 116))
        neighbor["id"] = "R20"
        fragments = ledger.collect_fragments(root, [root, neighbor])
        self.assertEqual(fragments, [])
        self.assertIsNone(ledger.fragment_role("R20"))

    def test_real_fit_designations_remain_attachable_fragments(self):
        root = token("50", bbox=(100, 100, 122, 116))
        root["id"] = "NOMINAL"
        fit = token("H7", bbox=(126, 100, 142, 116))
        fit["id"] = "FIT"
        fragments = ledger.collect_fragments(root, [root, fit])
        self.assertEqual(
            [(item["token"]["id"], item["role"]) for item in fragments],
            [("FIT", "fit")],
        )

    def test_equal_size_neighboring_dimensions_are_not_number_fragments(self):
        root = token("553", bbox=(100, 100, 124, 116))
        root["id"] = "ROOT"
        neighbor = token("646", bbox=(130, 90, 154, 106))
        neighbor["id"] = "NEIGHBOR"
        third = token("678", bbox=(130, 110, 154, 126))
        third["id"] = "THIRD"
        fragments = ledger.collect_fragments(root, [root, neighbor, third])
        self.assertEqual(fragments, [])

    def test_textual_surface_roughness_is_independent_annotation(self):
        parsed = self.parse("Rz40")
        self.assertIsNone(ledger.fragment_role("Rz40"))
        self.assertEqual(parsed["type"], "surface_roughness")
        self.assertEqual(parsed["nominal"], 40.0)
        self.assertEqual(parsed["unit"], "µm")
        self.assertEqual(parsed["surface_roughness_parameter"], "Rz")
        self.assertIsNone(parsed["fit"])

    def test_distributed_lug_callout_is_feature_count(self):
        parsed = self.parse("8只吊耳圆周均布")
        self.assertEqual(parsed["type"], "feature_count")
        self.assertEqual(parsed["quantity"], 8)

    def test_closed_coaxiality_frame_is_one_structured_record(self):
        def segment(identifier, p1, p2):
            math = __import__("math")
            angle = ledger.normalize_180(
                math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            )
            return {
                "id": identifier, "path_id": "V1", "p1": list(p1), "p2": list(p2),
                "length": ledger.distance(p1, p2), "angle_deg": angle,
                "width": 0.5, "color": (1.0, 0.0, 1.0), "fill": None,
                "dashes": "[] 0",
            }

        segments = [
            segment("TOP", (100, 100), (170, 100)),
            segment("BOTTOM", (100, 120), (170, 120)),
            segment("LEFT", (100, 100), (100, 120)),
            segment("SEP1", (120, 100), (120, 120)),
            segment("SEP2", (140, 100), (140, 120)),
            segment("RIGHT", (170, 100), (170, 120)),
            segment("LAND", (85, 110), (100, 110)),
            segment("LEADER", (85, 110), (60, 135)),
        ]
        symbols = [
            {"id": "COAX", "kind": "coaxiality", "text": "◎", "bbox": [103, 103, 117, 117], "center": [110, 110], "rotation_deg": 0},
            {"id": "DIA", "kind": "diameter", "text": "⌀", "bbox": [143, 104, 151, 116], "center": [147, 110], "rotation_deg": 0},
        ]
        arrows = [
            {"id": "ARROW", "kind": "open", "tip": [60, 135], "direction": [-0.7, 0.7], "bbox": [60, 130, 67, 137], "segment_ids": []}
        ]
        datum = token("B", bbox=(125, 103, 134, 117))
        datum["id"] = "DATUM"
        tolerance = token("3", bbox=(155, 103, 164, 117))
        tolerance["id"] = "TOLERANCE"
        geometry = ledger.PageGeometry(
            segments, arrows, ledger.build_segment_adjacency(segments), symbols
        )

        records, consumed = ledger.detect_geometric_tolerance_frames(
            [datum, tolerance], geometry, "mm"
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(consumed, {"DATUM", "TOLERANCE"})
        self.assertEqual(record["type"], "geometric_tolerance")
        self.assertEqual(record["geometric_characteristic"], "coaxiality")
        self.assertEqual(record["geometric_tolerance"], 3.0)
        self.assertEqual(record["tolerance_zone"], "diameter")
        self.assertEqual(record["datum_references"], ["B"])
        self.assertEqual(record["geometry"]["relationship"], "feature_control_frame")
        self.assertEqual(record["geometry"]["arrow_ids"], ["ARROW"])

    def test_status_accepts_explainable_arrow_evidence_but_not_bare_nearby_line(self):
        base = {
            "normalized_text": "25", "context_line_text": "25", "nominal": 25.0,
            "raw_text": "25", "parse_notes": [], "type": "linear",
            "tolerance_upper": None, "tolerance_lower": None, "bbox": [100, 100, 130, 114],
        }
        strong = dict(base, geometry={
            "relationship": "dimension_line", "score": 80.0, "unique": True,
            "arrow_ids": ["A1", "A2"],
        })
        weak = dict(base, geometry={
            "relationship": "nearby_line", "score": 80.0, "unique": False,
            "arrow_ids": [],
        })
        self.assertEqual(ledger.classify_status(strong, [0, 0, 500, 400])[0], "accepted")
        self.assertEqual(ledger.classify_status(weak, [0, 0, 500, 400])[0], "needs_review")

    def test_owned_imperial_fraction_uses_arrow_evidence(self):
        annotation = {
            "normalized_text": '4 3/4"', "context_line_text": "4", "nominal": 4.75,
            "raw_text": '4 3/4"', "parse_notes": [], "type": "linear",
            "unit": "in", "tolerance_upper": None, "tolerance_lower": None,
            "bbox": [100, 100, 140, 116], "font_size": 12,
            "assembly_basis": "quote_anchored_imperial_fraction",
            "geometry": {
                "relationship": "dimension_line", "score": 70.0, "unique": False,
                "arrow_ids": ["A1"], "text_to_line_distance": 10.0,
            },
        }
        self.assertEqual(ledger.classify_status(annotation, [0, 0, 500, 400])[0], "accepted")

    def test_internal_arrow_tips_on_extended_dimension_line(self):
        group = [{"p1": [50, 100], "p2": [150, 100]}]
        arrows = [
            {"id": "A1", "tip": [85, 100], "direction": [1, 0]},
            {"id": "A2", "tip": [105, 100], "direction": [-1, 0]},
            {"id": "A3", "tip": [95, 110], "direction": [0, 1]},
        ]
        self.assertEqual(
            ledger.collinear_arrow_ids(arrows, group, 0.0, 1.0),
            ["A1", "A2"],
        )

    def test_internal_filled_arrow_base_touching_split_line_is_collinear(self):
        group = [
            {"p1": [50, 100], "p2": [85, 100]},
            {"p1": [90, 100], "p2": [150, 100]},
        ]
        arrows = [
            {
                "id": "A1", "tip": [85, 100], "direction": [-1, 0],
                "bbox": [85, 97.5, 90, 102.5],
            }
        ]
        self.assertEqual(
            ledger.collinear_arrow_ids(arrows, group, 0.0, 2.0),
            ["A1"],
        )

    def test_filled_arrow_base_contact_counts_as_endpoint_attachment(self):
        arrows = [
            {
                "id": "LEFT", "kind": "filled", "tip": [44.6, 100],
                "direction": [-1, 0], "bbox": [44.6, 97.5, 50, 102.5],
            },
            {
                "id": "RIGHT", "kind": "filled", "tip": [155.4, 100],
                "direction": [1, 0], "bbox": [150, 97.5, 155.4, 102.5],
            },
        ]
        self.assertEqual(
            ledger.nearby_arrow_ids(arrows, ([50, 100], [150, 100]), 3.0),
            ["LEFT", "RIGHT"],
        )

    def test_arrow_base_attachment_still_requires_line_direction(self):
        segments = [
            {"id": "VERTICAL", "p1": [100, 40], "p2": [100, 160], "length": 120.0, "angle_deg": 90.0},
        ]
        arrows = [
            {"id": "TOP", "tip": [94.6, 40], "direction": [-1, 0], "bbox": [94.6, 37.5, 100, 42.5], "segment_ids": []},
            {"id": "BOTTOM", "tip": [105.4, 160], "direction": [1, 0], "bbox": [100, 157.5, 105.4, 162.5], "segment_ids": []},
        ]
        geometry = ledger.PageGeometry(segments, arrows, {}, [])
        annotation = {
            "type": "linear", "bbox": [112, 90, 150, 106],
            "font_size": 12.0, "direction": [0.0, 1.0],
        }
        result = ledger.best_dimension_line(annotation, geometry, [0, 0, 200, 200])
        self.assertEqual(result["arrow_ids"], [])

    def test_arrow_outline_is_not_selected_as_dimension_line(self):
        segments = [
            {"id": "BASE", "p1": [50, 100], "p2": [150, 100], "length": 100.0, "angle_deg": 0.0},
            {"id": "ARM1", "p1": [85, 100], "p2": [94, 98.6], "length": 9.11, "angle_deg": 171.16},
            {"id": "ARM2", "p1": [105, 100], "p2": [114, 98.6], "length": 9.11, "angle_deg": 171.16},
        ]
        arrows = [
            {"id": "A1", "tip": [85, 100], "direction": [1, 0], "segment_ids": ["ARM1"]},
            {"id": "A2", "tip": [105, 100], "direction": [-1, 0], "segment_ids": ["ARM2"]},
        ]
        geometry = ledger.PageGeometry(segments, arrows, {}, [])
        annotation = {
            "bbox": [112, 82, 124, 102],
            "font_size": 12.0,
            "direction": [1.0, 0.0],
        }
        result = ledger.best_dimension_line(annotation, geometry, [0, 0, 200, 200])
        self.assertEqual(result["line_segment_ids"], ["BASE"])
        self.assertEqual(result["arrow_ids"], ["A1", "A2"])

    def test_perpendicular_endpoint_arrows_are_not_borrowed_by_dimension_line(self):
        segments = [
            {"id": "VERTICAL", "p1": [100, 40], "p2": [100, 160], "length": 120.0, "angle_deg": 90.0},
        ]
        arrows = [
            {"id": "LEFT", "tip": [100, 40], "direction": [1, 0], "bbox": [100, 36, 108, 44], "segment_ids": []},
            {"id": "RIGHT", "tip": [100, 160], "direction": [-1, 0], "bbox": [92, 156, 100, 164], "segment_ids": []},
        ]
        geometry = ledger.PageGeometry(segments, arrows, {}, [])
        annotation = {
            "bbox": [112, 90, 150, 106],
            "font_size": 12.0,
            "direction": [1.0, 0.0],
        }
        result = ledger.best_dimension_line(annotation, geometry, [0, 0, 200, 200])
        self.assertEqual(result["arrow_ids"], [])
        self.assertNotEqual(result["relationship"], "dimension_line")

    def test_collinear_group_does_not_bridge_large_non_text_gap(self):
        base = {
            "id": "BASE", "p1": [100, 300], "p2": [100, 340],
            "length": 40.0, "angle_deg": 90.0,
        }
        foreign = {
            "id": "FOREIGN", "p1": [100, 400], "p2": [100, 430],
            "length": 30.0, "angle_deg": 90.0,
        }
        group = ledger.collinear_group(
            base,
            [base, foreign],
            max_gap=100.0,
            bridge_interval=[10.0, 30.0],
            ordinary_gap=16.0,
        )
        self.assertEqual([segment["id"] for segment in group], ["BASE"])

    def test_radius_uses_text_attached_same_path_single_arrow_leader(self):
        def segment(identifier, p1, p2):
            angle = ledger.normalize_180(
                __import__("math").degrees(__import__("math").atan2(p2[1] - p1[1], p2[0] - p1[0]))
            )
            return {
                "id": identifier, "path_id": "LEADER_PATH", "p1": list(p1), "p2": list(p2),
                "length": ledger.distance(p1, p2), "angle_deg": angle,
            }

        segments = [
            segment("LAND", (100, 116), (80, 116)),
            segment("LEADER", (80, 116), (60, 136)),
        ]
        arrows = [
            {
                "id": "TIP", "tip": [60, 136], "direction": [-0.7071, 0.7071],
                "bbox": [60, 130, 67, 137], "segment_ids": [],
            }
        ]
        geometry = ledger.PageGeometry(
            segments, arrows, ledger.build_segment_adjacency(segments), []
        )
        annotation = {
            "type": "radius", "bbox": [100, 100, 130, 116],
            "font_size": 12.0, "direction": [1.0, 0.0],
        }
        result = ledger.associate_geometry(annotation, geometry, [0, 0, 300, 220])
        self.assertEqual(result["relationship"], "leader")
        self.assertEqual(result["arrow_ids"], ["TIP"])
        self.assertEqual(result["line_segment_ids"], ["LAND", "LEADER"])
        self.assertEqual(result["confidence_basis"], "same_path_single_arrow_leader")

    def test_radius_keeps_same_path_continuation_split_by_text(self):
        def segment(identifier, p1, p2):
            angle = ledger.normalize_180(
                __import__("math").degrees(__import__("math").atan2(p2[1] - p1[1], p2[0] - p1[0]))
            )
            return {
                "id": identifier, "path_id": "LEADER_PATH", "p1": list(p1), "p2": list(p2),
                "length": ledger.distance(p1, p2), "angle_deg": angle,
            }

        segments = [
            segment("LAND", (100, 116), (80, 116)),
            segment("LEADER", (80, 116), (60, 136)),
            segment("CONTINUATION", (130, 116), (170, 116)),
        ]
        arrows = [
            {
                "id": "TIP", "tip": [60, 136], "direction": [-0.7071, 0.7071],
                "bbox": [60, 130, 67, 137], "segment_ids": [],
            }
        ]
        geometry = ledger.PageGeometry(
            segments, arrows, ledger.build_segment_adjacency(segments), []
        )
        annotation = {
            "type": "radius", "bbox": [100, 100, 130, 116],
            "font_size": 12.0, "direction": [1.0, 0.0],
        }
        result = ledger.associate_geometry(annotation, geometry, [0, 0, 300, 220])
        self.assertEqual(
            result["line_segment_ids"],
            ["LAND", "LEADER", "CONTINUATION"],
        )

    def test_open_base_corners_do_not_duplicate_filled_triangle(self):
        bbox = [10, 10, 18, 14]
        arrows = [
            {"id": "P1-A0001", "kind": "filled", "tip": [10, 12], "bbox": bbox, "segment_ids": []},
            {"id": "P1-AO0001", "kind": "open", "tip": [18, 10], "bbox": bbox, "segment_ids": ["S1", "S2"]},
            {"id": "P1-AO0002", "kind": "open", "tip": [18, 14], "bbox": bbox, "segment_ids": ["S2", "S3"]},
        ]
        result = ledger.deduplicate_arrows(arrows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["kind"], "filled")

    def test_two_filled_arrowheads_in_one_pdf_path_are_split(self):
        point = ledger.fitz.Point
        drawings = {
            "P1-V0001": {
                "fill": (0.0, 0.0, 1.0),
                "items": [
                    ("l", point(50, 40), point(47.5, 48)),
                    ("l", point(47.5, 48), point(52.5, 48)),
                    ("l", point(50, 160), point(52.5, 152)),
                    ("l", point(52.5, 152), point(47.5, 152)),
                ],
            }
        }
        vertices = {
            "P1-V0001": [
                (50, 40), (47.5, 48), (47.5, 48), (52.5, 48),
                (50, 160), (52.5, 152), (52.5, 152), (47.5, 152),
            ]
        }
        arrows = ledger.detect_filled_arrows(vertices, drawings, 1)
        self.assertEqual(len(arrows), 2)
        self.assertEqual([arrow["tip"] for arrow in arrows], [[50.0, 40.0], [50.0, 160.0]])

    def test_shared_stacked_tolerance_goes_to_nearest_nominal(self):
        upper = token("437.6", bbox=(100, 80, 130, 96))
        upper["id"] = "UPPER"
        lower = token("356", bbox=(100, 104, 124, 120))
        lower["id"] = "LOWER"
        plus = token("+", size=8, bbox=(124, 98, 129, 110))
        plus["id"] = "PLUS"
        fragments = {
            "UPPER": [{"token": plus, "role": "tolerance"}],
            "LOWER": [{"token": plus, "role": "tolerance"}],
        }
        ledger.resolve_shared_fragment_owners([upper, lower], fragments)
        self.assertEqual(fragments["UPPER"], [])
        self.assertEqual([item["token"]["id"] for item in fragments["LOWER"]], ["PLUS"])

    def test_tolerance_sign_is_not_owned_by_subordinate_number_root(self):
        nominal = token("356", bbox=(100, 104, 124, 120))
        nominal["id"] = "NOMINAL"
        deviation = token("7", size=8, bbox=(129, 98, 134, 110))
        deviation["id"] = "DEVIATION"
        plus = token("+", size=8, bbox=(124, 98, 129, 110))
        plus["id"] = "PLUS"
        fragments = {
            "NOMINAL": [
                {"token": plus, "role": "tolerance"},
                {"token": deviation, "role": "number"},
            ],
            "DEVIATION": [{"token": plus, "role": "tolerance"}],
        }
        ledger.resolve_shared_fragment_owners([nominal, deviation], fragments)
        self.assertEqual(
            [item["token"]["id"] for item in fragments["NOMINAL"]],
            ["PLUS", "DEVIATION"],
        )
        self.assertEqual(fragments["DEVIATION"], [])

    def test_two_stroke_surface_texture_symbol(self):
        annotation = {
            "normalized_text": "6.3",
            "bbox": [100, 80, 120, 94],
            "font_size": 10.0,
        }
        segments = [
            {"id": "SHORT", "path_id": "V1", "p1": [116, 96], "p2": [126, 96], "length": 10.0, "angle_deg": 0.0},
            {"id": "LONG", "path_id": "V1", "p1": [116, 96], "p2": [126, 113], "length": 19.72, "angle_deg": 59.53},
            {"id": "NOISE", "path_id": "V2", "p1": [90, 90], "p2": [130, 90], "length": 40.0, "angle_deg": 0.0},
        ]
        geometry = ledger.PageGeometry(segments, [], {}, [])
        evidence = ledger.surface_roughness_evidence(annotation, geometry)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["relationship"], "surface_roughness_symbol")
        self.assertEqual(evidence["line_segment_ids"], ["SHORT", "LONG"])

    def test_plain_number_without_symbol_is_not_surface_roughness(self):
        annotation = {
            "normalized_text": "6.3",
            "bbox": [100, 80, 120, 94],
            "font_size": 10.0,
        }
        geometry = ledger.PageGeometry([], [], {}, [])
        self.assertIsNone(ledger.surface_roughness_evidence(annotation, geometry))

    def test_stacked_angular_zero_lower_deviation(self):
        root = token("30°", size=12.7, bbox=(472.2, 355.1, 506.2, 375.1), rotation=350.78)
        plus = token("+", size=10.5, bbox=(501.1, 352.9, 513.2, 365.7), rotation=350.78)
        upper = token("0.25°", size=9.6, bbox=(508.0, 354.0, 536.3, 369.4), rotation=350.78)
        lower = token("0°", size=9.5, bbox=(506.4, 363.7, 520.9, 376.9), rotation=350.78)
        plus["id"], upper["id"], lower["id"] = "PLUS", "UPPER", "LOWER"
        fragments = ledger.collect_fragments(root, [root, plus, upper, lower])
        self.assertEqual(
            {item["token"]["id"] for item in fragments},
            {"PLUS", "UPPER", "LOWER"},
        )
        parsed = ledger.parse_annotation(root, fragments, "mm")
        self.assertEqual(parsed["tolerance_upper"], 0.25)
        self.assertEqual(parsed["tolerance_lower"], 0.0)
        self.assertEqual(parsed["tolerance_unit"], "deg")
        self.assertEqual(parsed["parse_notes"], [])

    def test_angular_deviation_unit_comes_from_tolerance_fragment(self):
        root = token("30°", size=12.7, bbox=(472.2, 355.1, 506.2, 375.1), rotation=350.78)
        upper = token("+0.25'", size=9.6, bbox=(501.1, 354.0, 536.3, 369.4), rotation=350.78)
        lower = token("0°", size=9.5, bbox=(506.4, 363.7, 520.9, 376.9), rotation=350.78)
        fragments = [
            {"token": upper, "role": "tolerance"},
            {"token": lower, "role": "number"},
        ]
        parsed = ledger.parse_annotation(root, fragments, "mm")
        self.assertEqual(parsed["tolerance_upper"], 0.25)
        self.assertEqual(parsed["tolerance_lower"], 0.0)
        self.assertEqual(parsed["tolerance_unit"], "arcmin")


class PdfPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dimension-ledger-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_existing_synthetic_pipeline(self):
        pdf = self.root / "self-test.pdf"
        ledger.create_self_test_pdf(pdf)
        result = ledger.analyze_pdf(pdf, "mm")
        by_nominal = {row["nominal"]: row for row in result["dimensions"]}
        self.assertEqual(by_nominal[25.0]["geometry"]["relationship"], "dimension_line")
        self.assertEqual(by_nominal[40.0]["rotation_deg"], 90.0)
        self.assertEqual(by_nominal[12.0]["type"], "radius")
        self.assertIsNotNone(result["pages"][0]["detected_title_block"])
        self.assertFalse(any(row["nominal"] == 987654 for row in result["dimensions"]))

    def test_multi_page_and_output_consistency(self):
        pdf = self.root / "multi.pdf"
        document = fitz.open()
        for page_number in range(2):
            page = document.new_page(width=420, height=300)
            y = 80 + page_number * 20
            page.draw_line((50, y), (155, y), width=0.6)
            page.draw_line((245, y), (370, y), width=0.6)
            page.draw_line((50, y), (58, y - 4), width=0.6)
            page.draw_line((50, y), (58, y + 4), width=0.6)
            page.draw_line((370, y), (362, y - 4), width=0.6)
            page.draw_line((370, y), (362, y + 4), width=0.6)
            page.insert_text((180, y + 4), str(100 + page_number), fontsize=12)
        document.save(pdf)
        document.close()

        result = ledger.analyze_pdf(pdf, "mm")
        output = self.root / "result"
        ledger.write_outputs(pdf, output, result)
        self.assertEqual(result["summary"]["pages"], 2)
        self.assertEqual({row["page"] for row in result["dimensions"]}, {1, 2})
        public = json.loads((output / "dimension-ledger.json").read_text(encoding="utf-8"))
        review = json.loads((output / "needs-review.json").read_text(encoding="utf-8"))
        self.assertEqual(public["summary"]["needs_review"], len(review))
        self.assertTrue((output / "review.pdf").is_file())

    def test_no_text_layer_is_rejected(self):
        pdf = self.root / "empty.pdf"
        document = fitz.open()
        page = document.new_page()
        page.draw_line((20, 20), (100, 100))
        document.save(pdf)
        document.close()
        with self.assertRaisesRegex(ValueError, "no extractable text layer"):
            ledger.analyze_pdf(pdf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
