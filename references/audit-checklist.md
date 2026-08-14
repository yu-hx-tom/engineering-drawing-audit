# Engineering Drawing Audit Checklist

## Pass 1: Document Inventory

- Confirm filenames and which drawing is authoritative.
- Record page count, page orientation, scale, language, and scan/vector status.
- Extract vector text and create overview renders only as needed.
- Inspect scans at native resolution first.
- Render or crop at 300 DPI only when needed; use 600 DPI for ambiguous local regions.
- Inventory principal views, sections, enlarged details, note blocks, and title blocks.

## Pass 2: Feature Map

Use one row per physical feature:

| ID | Customer location | Redrawn location | Physical feature | Correspondence evidence |
|---|---|---|---|---|
| F01 | Detail B | Main section, left side | Hook-root transition | Leader endpoints, R20 neighbor, section orientation |

Do not use page proximity as correspondence evidence.

## Pass 3: Dimension Comparison

| Feature ID | Customer | Redrawn | Symbol/tolerance | Status |
|---|---:|---:|---|---|
| F01 | 15 degrees | 15 degrees | angle | Match |

Check nominal values, tolerance placement, diameter/radius symbols, quantities, pitch, datums,
surface texture, and reference status.

Before entering a value, record:

- both extension-line endpoints or the leader endpoint;
- the physical feature name;
- the dimension-line shape: straight for a probable linear dimension, or arc-shaped between two
  rays for a probable angular dimension;
- dimension type: linear, angle, radius, diameter, or reference;
- every qualifier, including `*`, parentheses, tolerances, quantity, and units.

Treat the dimension-line shape as strong structural evidence. Confirm an angular reading from the
arc dimension line, its center or vertex, the two bounding rays, and a degree or minute symbol when
visible. Do not confuse a radius, diameter, arc-length callout, or a straight leader pointing to a
curved surface with an angular dimension.

Do not map dimensions by numeral or page proximity. The same value may control unrelated features.

For an unclear, conflicting, or implausible tolerance:

- preserve raw text separately from the parsed value, including comma versus decimal point, sign,
  upper/lower placement, and decimal places;
- parse the decimal convention from the drawing language or title block and normalize units only in
  a calculation copy;
- test every visually plausible decimal shift, including common 10x or 100x errors;
- check absolute magnitude against nominal size, neighboring gaps or thicknesses, dimensional
  chains, and cited standards; handle zero deviations without ratio tests;
- use implausibility to trigger reinspection, never to overwrite a legible source;
- keep source and redrawn values separate, for example `-1,5` versus `-15`.

## Pass 4: Handwriting Resolution

For every unclear character:

1. Write every evidence-supported candidate; do not invent a second candidate.
2. Compare the character with other examples on the same drawing.
3. Trace leaders and extension lines.
4. Check dimensional-chain and geometric plausibility.
5. Compare the corresponding redrawn feature.
6. Retain uncertainty when evidence conflicts.

Examples of common traps:

- `3` read as `5`;
- `15 degrees` read as `12 degrees`;
- an angle from one feature paired with a nearby angle from another feature;
- a radius symbol read as a diameter symbol;
- minute tolerance read as a degree value;
- a reference dimension reported as a manufacturing dimension.
- a trailing `*` read as a degree mark;
- a width or height read as an angle without an angular arc or vertex;
- rotated `13` read as `2` or `3` because the leading `1` overlaps a dimension line;
- one redrawn dimension matched to two different customer features.

A clear arc dimension line spanning two rays plus a readable degree or minute symbol is sufficient
to identify an angle. Seek extra evidence only when those elements are unclear or missing. If the
extension lines are parallel and terminate on two surfaces, prefer a linear-dimension candidate
even when a raised asterisk resembles a degree symbol.

## Pass 5: Technical-Note Mapping

Run this pass only for a complete audit or when the user requests technical requirements,
translation, materials, or manufacturing notes. Skip it for a dimensions-only request.

Map by meaning:

| Customer note | Meaning | Redrawn equivalent | Status |
|---|---|---|---|
| 7 | Unspecified casting radii | Note or general symbol | Present/Missing/Uncertain |

Check title-block fields and drawing symbols before calling a note missing.

## Final Output

### Concise Mode

Order the report as:

1. Confirmed errors.
2. Confirmed omissions.
3. Material needs-manual-confirmation items.
4. Scan and interpretation limitations.

Do not expand correctly matched dimensions unless they are needed as evidence.

### Detailed Mode

Use one subsection per customer view. Name views by the drawing label when available, otherwise by
an unambiguous descriptive name such as `Main section`, `Top view`, or `Enlarged hook detail`.

Start each subsection with a view-mapping summary:

| Customer view | Customer annotations | Redrawn corresponding view(s) | Mapping note |
|---|---:|---|---|
| Enlarged hook detail | 6 | Main view, left hook area | Detail merged into main view |

Then include one row for every customer dimension annotation:

| ID | Customer annotation | Physical feature controlled | Redrawn view/location | Redrawn annotation | Status | Problem evidence / note |
|---|---|---|---|---|---|---|
| V4-D01 | R20 | Hook-root internal fillet | Main view, left hook | R20 | Match | |
| V4-D02 | 15 degrees | Hook flank angle | Main view, left hook | 15 degrees | Equivalent expression | |
| V4-D03 | 8 | Hook tip thickness | Not found | None | Needs manual confirmation | Source handwriting unclear; no equivalent located |

For `Match` and clear `Equivalent expression` rows, leave the evidence field empty. Add evidence
only for errors, omissions, and `Needs manual confirmation` items.

Apply these rules:

1. Use the customer view order as the report structure, regardless of redrawn view count.
2. Give each customer annotation a stable ID such as `V2-D07`.
3. Include nominal value, symbol, tolerance, quantity, reference status, and units as one
   annotation; do not silently drop qualifiers.
4. Describe the physical feature, not merely the page position.
5. Allow many customer views to map to one redrawn view and one customer view to map across
   multiple redrawn views.
6. Use `Match`, `Confirmed error`, `Confirmed omission`, `Equivalent expression`, or
   `Needs manual confirmation` as status.
7. Write `Not found` only after searching all redrawn views, notes, title-block fields, and
   equivalent drafting conventions.
8. Record duplicated or redrawing-only annotations in a separate subsection after all customer
   views; do not force them into an unrelated customer row.
9. Reconcile totals at the end:

| Reconciliation | Count |
|---|---:|
| Customer annotations inventoried | N |
| Matched directly | N |
| Equivalent expressions | N |
| Confirmed errors | N |
| Confirmed omissions | N |
| Needs manual confirmation | N |

The status counts must sum to the customer annotation total. Then list redrawing-only or duplicated
annotations separately because they are outside that total.

Conclude with:

1. Confirmed errors and omissions.
2. Needs-manual-confirmation items.
3. Redrawing-only or duplicated annotations.
4. View consolidation assessment.
5. Scan and interpretation limitations.

Do not annotate the source PDF until the finding list is finalized.
