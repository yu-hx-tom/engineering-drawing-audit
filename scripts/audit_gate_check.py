"""
audit_gate_check.py - Pre-Report Residual Security Gate & Targeted Diff Engine
==============================================================================
Implements Approach 1 (Visual Primary Audit + Script Safety Gate & Targeted Residuals):
1. Duplicate Nominal Inspection (flags collision risks like repeated 30, 15, 25).
2. Consumptive Ledger Clearance (takes visual matched pairs, clears from total pools).
3. Exact Residual Derivation:
   - Customer Residuals: C_all - C_matched (Targeted for Confirmed Omission inspection)
   - Redraw Residuals: R_all - R_matched (Targeted for Redraw-Only / Equivalent inspection)
4. Targeted 600 DPI Crop Generation for all residuals for instant secondary verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding='utf-8')


def sanitize_filename(text: str) -> str:
    """Sanitize text for use in file names."""
    clean = re.sub(r'[^\w\.-]', '_', text)
    return clean[:30]


def generate_crop(
    pdf_path: str | Path,
    output_png: str | Path,
    page_no: int,
    bbox: List[float],
    padding: float = 40.0,
    dpi: int = 600,
) -> Optional[str]:
    """Generate high-DPI crop around bbox [x0, y0, x1, y1] with padding."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        if not (1 <= page_no <= doc.page_count):
            doc.close()
            return None

        page = doc.load_page(page_no - 1)
        rect = page.rect

        x0 = max(0.0, bbox[0] - padding)
        y0 = max(0.0, bbox[1] - padding)
        x1 = min(rect.width, bbox[2] + padding)
        y1 = min(rect.height, bbox[3] + padding)

        clip = pymupdf.Rect(x0, y0, x1, y1)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        pix.save(str(output_png))
        doc.close()
        return str(output_png)
    except Exception as e:
        return None


def run_gate_check(
    customer_ledger_path: str | Path,
    redraw_ledger_path: str | Path,
    matched_pairs_path: Optional[str | Path] = None,
    customer_pdf_path: Optional[str | Path] = None,
    redraw_pdf_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    with open(customer_ledger_path, 'r', encoding='utf-8-sig') as f:
        cust_data = json.load(f)
    with open(redraw_ledger_path, 'r', encoding='utf-8-sig') as f:
        red_data = json.load(f)

    cust_dims = cust_data.get('dimensions', [])
    red_dims = red_data.get('dimensions', [])

    cust_map = {d['id']: d for d in cust_dims}
    red_map = {d['id']: d for d in red_dims}

    # 1. Inspect Duplicate Nominals on Customer Drawing
    nom_counter_cust = Counter()
    for d in cust_dims:
        nom = d.get('nominal')
        if nom is not None and d.get('type') not in ['roughness', 'thread']:
            nom_counter_cust[nom] += 1
    duplicates_cust = {k: v for k, v in nom_counter_cust.items() if v > 1}

    # 2. Inspect Duplicate Nominals on Redraw Drawing
    nom_counter_red = Counter()
    for d in red_dims:
        nom = d.get('nominal')
        if nom is not None and d.get('type') not in ['roughness', 'thread']:
            nom_counter_red[nom] += 1

    # 3. Collision Risks
    collision_risks = []
    for val, c_count in duplicates_cust.items():
        r_count = nom_counter_red.get(val, 0)
        if c_count != r_count:
            collision_risks.append({
                'nominal': val,
                'customer_count': c_count,
                'redraw_count': r_count,
                'risk': f"同名数值碰撞高危: 客户图出现 {c_count} 处，重绘图出现 {r_count} 处 (差额 {c_count - r_count})"
            })

    # 4. Resolve Matched Pairs (from Agent Phase 3 file or Fallback Reconciler)
    matched_pairs: List[Dict[str, Any]] = []
    consumed_cust_ids: Set[str] = set()
    consumed_red_ids: Set[str] = set()

    if matched_pairs_path and os.path.exists(matched_pairs_path):
        with open(matched_pairs_path, 'r', encoding='utf-8-sig') as f:
            pairs_data = json.load(f)
        raw_pairs = pairs_data.get('matched_pairs', pairs_data if isinstance(pairs_data, list) else [])
        for p in raw_pairs:
            c_id = p.get('customer_id') or (p.get('customer') or {}).get('id')
            r_id = p.get('redraw_id') or (p.get('redraw') or {}).get('id')
            if c_id and c_id in cust_map:
                consumed_cust_ids.add(c_id)
            if r_id and r_id in red_map:
                consumed_red_ids.add(r_id)
            matched_pairs.append(p)
    else:
        # Automated fallback baseline using reconcile_ledgers
        from reconcile_ledgers import reconcile
        rec_res = reconcile(customer_ledger_path, redraw_ledger_path, assume_complete=True)
        for p in rec_res.get('matched_pairs', []):
            c_id = (p.get('customer') or {}).get('id')
            r_id = (p.get('redraw') or {}).get('id')
            if c_id:
                consumed_cust_ids.add(c_id)
            if r_id:
                consumed_red_ids.add(r_id)
            matched_pairs.append(p)
        for p in rec_res.get('equivalent_pairs', []):
            c_id = (p.get('customer') or {}).get('id')
            r_id = (p.get('redraw') or {}).get('id')
            if c_id:
                consumed_cust_ids.add(c_id)
            if r_id:
                consumed_red_ids.add(r_id)
            matched_pairs.append(p)

    # 5. Exact Set Difference for Residuals
    unmatched_cust = [d for d in cust_dims if d['id'] not in consumed_cust_ids]
    unmatched_red = [d for d in red_dims if d['id'] not in consumed_red_ids]

    # 6. Generate Targeted 600 DPI Crops for Residuals if PDFs provided
    crops_dir = None
    if output_dir:
        crops_dir = os.path.join(output_dir, "crops", "residuals")

    cust_residual_items = []
    for d in unmatched_cust:
        crop_path = None
        if customer_pdf_path and crops_dir and d.get('bbox'):
            crop_name = f"crop_cust_{d['id']}_{sanitize_filename(d['raw_text'])}.png"
            crop_target = os.path.join(crops_dir, crop_name)
            crop_path = generate_crop(customer_pdf_path, crop_target, d.get('page', 1), d['bbox'])

        cust_residual_items.append({
            'id': d['id'],
            'raw_text': d['raw_text'],
            'nominal': d.get('nominal'),
            'type': d.get('type'),
            'page': d.get('page', 1),
            'bbox': d.get('bbox'),
            'crop_path': crop_path,
        })

    red_residual_items = []
    for d in unmatched_red:
        crop_path = None
        if redraw_pdf_path and crops_dir and d.get('bbox'):
            crop_name = f"crop_red_{d['id']}_{sanitize_filename(d['raw_text'])}.png"
            crop_target = os.path.join(crops_dir, crop_name)
            crop_path = generate_crop(redraw_pdf_path, crop_target, d.get('page', 1), d['bbox'])

        red_residual_items.append({
            'id': d['id'],
            'raw_text': d['raw_text'],
            'nominal': d.get('nominal'),
            'type': d.get('type'),
            'page': d.get('page', 1),
            'bbox': d.get('bbox'),
            'crop_path': crop_path,
        })

    # Strict Gate Status Check
    # Gate passes only if 0 customer residuals and 0 collision risks
    gate_passed = (len(cust_residual_items) == 0 and len(collision_risks) == 0)

    summary = {
        'gate_passed': gate_passed,
        'counts': {
            'customer_total': len(cust_dims),
            'customer_matched': len(consumed_cust_ids),
            'customer_residual_count': len(cust_residual_items),
            'redraw_total': len(red_dims),
            'redraw_matched': len(consumed_red_ids),
            'redraw_residual_count': len(red_residual_items),
        },
        'collision_risks': collision_risks,
        'customer_residuals': cust_residual_items,
        'redraw_residuals': red_residual_items,
        'matched_pairs_count': len(matched_pairs),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "audit-gate-result.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit Gate Check & Targeted Residual Diff Engine")
    parser.add_argument("customer_ledger", help="Path to customer dimension-ledger.json")
    parser.add_argument("redraw_ledger", help="Path to redraw dimension-ledger.json")
    parser.add_argument("--matched-pairs", "-m", help="Path to Phase 3 visual_matched_pairs.json (optional)")
    parser.add_argument("--customer-pdf", help="Path to customer drawing PDF for auto-cropping")
    parser.add_argument("--redraw-pdf", help="Path to redraw drawing PDF for auto-cropping")
    parser.add_argument("-o", "--output-dir", help="Output directory for results and crops")

    args = parser.parse_args()

    res = run_gate_check(
        args.customer_ledger,
        args.redraw_ledger,
        matched_pairs_path=args.matched_pairs,
        customer_pdf_path=args.customer_pdf,
        redraw_pdf_path=args.redraw_pdf,
        output_dir=args.output_dir,
    )

    c = res['counts']
    print("=" * 65)
    print("阶段 4：安检门残差与靶向定向复核清单 (Targeted Residuals)")
    print("=" * 65)
    print(f"Gate 状态        : {'✅ PASS (全部对平，无残差)' if res['gate_passed'] else '🚨 INTERCEPTED (强制拦截，存在待定向复核项)'}")
    print(f"客户原图核销统计 : 总数 {c['customer_total']} | 已核销 {c['customer_matched']} | 待二次复核残差: 🚨 {c['customer_residual_count']} 项")
    print(f"重绘工程图统计   : 总数 {c['redraw_total']} | 已核销 {c['redraw_matched']} | 待二次复核残差: 🚨 {c['redraw_residual_count']} 项")
    print(f"同名碰撞风险组数 : {len(res['collision_risks'])} 组高危同名数值")
    print("-" * 65)

    if res['collision_risks']:
        print("\n[⚠️ 高危同名数值碰撞列表 (必须在视觉上解耦不同的物理特征)]:")
        for r in res['collision_risks']:
            print(f"  * 数值 {r['nominal']}: 客户图 {r['customer_count']} 处 vs 重绘图 {r['redraw_count']} 处 -> {r['risk']}")

    if res['customer_residuals']:
        print(f"\n[🚨 客户原图未核销残差清单 (重点定向确认是否属于重绘漏注 Confirmed Omission)]:")
        for idx, item in enumerate(res['customer_residuals'], 1):
            crop_info = f" | 切图: {item['crop_path']}" if item.get('crop_path') else ""
            print(f"  {idx:2d}. {item['id']:8s} | {item['raw_text']:22s} | type={item['type']:10s} | bbox={[round(x,1) for x in (item['bbox'] or [])]}{crop_info}")

    if res['redraw_residuals']:
        print(f"\n[🚨 重绘工程图未核销残差清单 (重点定向确认是否属于重绘新增或等价表达)]:")
        for idx, item in enumerate(res['redraw_residuals'], 1):
            crop_info = f" | 切图: {item['crop_path']}" if item.get('crop_path') else ""
            print(f"  {idx:2d}. {item['id']:8s} | {item['raw_text']:22s} | type={item['type']:10s} | bbox={[round(x,1) for x in (item['bbox'] or [])]}{crop_info}")

    print("=" * 65)


if __name__ == '__main__':
    main()
