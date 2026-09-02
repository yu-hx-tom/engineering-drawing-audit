"""
reconcile_ledgers.py - Dual-Ledger Deterministic 1:1 Reconciliation Engine
==========================================================================
Performs mathematically rigorous, consumptive 1:1 reconciliation between
Customer Dimension Ledger and Redraw Dimension Ledger to eliminate Nominal
Collisions and ensure zero-leakage omission/difference detection.
"""

import json
import math
import os
import re
import sys
from typing import Dict, List, Any, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

TITLE_BLOCK_PATTERNS = [
    r"ISO\s*\d+", r"SS-ISO", r"SS-EN", r"SMS\s*\d+", r">4000mm", r"<4000mm",
    r"\(SA\s*\d+", r"Font\s+Helvetika", r"acceptance\s+level", r"CH\d+",
    r"^\d{4}-\d{2}-\d{2}$", r"^\d{3,}$" # Document numbers in title block
]

def is_title_block_noise(raw: str) -> bool:
    """Filter out standard title block ISO notes, drawing numbers, dates."""
    for p in TITLE_BLOCK_PATTERNS:
        if re.search(p, raw, re.IGNORECASE):
            # But preserve real dimensions like 1552, 1032, etc. if purely numeric and not an ISO note
            if any(term in raw for term in ["ISO", "SS-", "SMS", "SA 11", "Helvetika", "acceptance", "level"]):
                return True
    return False

def parse_degree_minutes(text: str) -> Optional[float]:
    """Parse degree-minute-second strings like 33°1' or 32°58' into decimal degrees."""
    m = re.search(r"(\d+)\s*°\s*(?:(\d+)\s*['\u2032\u0027])?\s*(?:(\d+(?:\.\d+)?)\s*[\"\u2033])?", text)
    if m:
        deg = float(m.group(1))
        mins = float(m.group(2)) if m.group(2) else 0.0
        secs = float(m.group(3)) if m.group(3) else 0.0
        return deg + mins / 60.0 + secs / 3600.0
    return None

def normalize_dim(dim: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dimension attributes for signature comparison."""
    d_type = dim.get('type', 'linear')
    nom = dim.get('nominal')
    raw = dim.get('raw_text', '').strip()
    
    # Check if raw has diameter symbol
    has_dia = 'Ø' in raw or '\u2300' in raw or '\xd8' in raw or 'DIA' in raw.upper()
    if has_dia and d_type == 'linear':
        d_type = 'diameter'
        
    has_radius = raw.startswith('R') or raw.startswith('r') or ' R' in raw
    if has_radius and d_type == 'linear':
        d_type = 'radius'
        
    has_angle = '°' in raw or 'deg' in raw.lower()
    if has_angle and d_type == 'linear':
        d_type = 'angle'
        
    # Handle angle decimal vs degree-minutes
    deg_val = parse_degree_minutes(raw)
    
    # Tolerances
    tol_up = dim.get('tolerance_upper')
    tol_low = dim.get('tolerance_lower')
    
    # Reference status
    is_ref = dim.get('is_ref', False) or '(' in raw or ')' in raw or 'REF' in raw.upper()
    
    is_noise = is_title_block_noise(raw)
    
    return {
        'id': dim.get('id', ''),
        'raw_text': raw,
        'type': d_type,
        'nominal': nom,
        'deg_val': deg_val,
        'tolerance_upper': tol_up,
        'tolerance_lower': tol_low,
        'is_ref': is_ref,
        'is_noise': is_noise,
        'bbox': dim.get('bbox', []),
        'page': dim.get('page', 1),
        'confidence': dim.get('confidence', 1.0)
    }

def reconcile(cust_file: str, red_file: str) -> Dict[str, Any]:
    with open(cust_file, 'r', encoding='utf-8') as f:
        cust_json = json.load(f)
    with open(red_file, 'r', encoding='utf-8') as f:
        red_json = json.load(f)
        
    cust_dims = [normalize_dim(d) for d in cust_json.get('dimensions', [])]
    red_dims = [normalize_dim(d) for d in red_json.get('dimensions', [])]
    
    matched_pairs = []
    equivalent_pairs = []
    difference_pairs = []
    
    # Consumptive pools
    unmatched_cust = list(range(len(cust_dims)))
    unmatched_red = list(range(len(red_dims)))
    
    consumed_c = set()
    consumed_r = set()
    
    # -------------------------------------------------------------
    # PASS 1: Exact 1:1 Match (Type, Nominal, Tolerances, Is_Ref)
    # -------------------------------------------------------------
    for ci in unmatched_cust:
        c = cust_dims[ci]
        c_nom = c['nominal']
        c_type = c['type']
        
        if c_nom is None or c['is_noise']:
            continue
            
        best_rj = None
        for rj in unmatched_red:
            if rj in consumed_r:
                continue
            r = red_dims[rj]
            r_nom = r['nominal']
            r_type = r['type']
            
            # Type compatibility (allow linear ↔ diameter/radius if nominal matches exactly)
            type_compat = (c_type == r_type) or (c_type in ['linear', 'diameter'] and r_type in ['linear', 'diameter'])
            if not type_compat:
                continue
                
            # Nominal must match within 0.001
            if r_nom is not None and abs(c_nom - r_nom) < 0.001:
                # Check tolerances
                c_up, c_low = c['tolerance_upper'], c['tolerance_lower']
                r_up, r_low = r['tolerance_upper'], r['tolerance_lower']
                
                tol_match = (c_up == r_up) and (c_low == r_low)
                ref_match = (c['is_ref'] == r['is_ref'])
                
                if tol_match and ref_match and (c_type == r_type):
                    best_rj = rj
                    break
                elif tol_match:
                    if best_rj is None:
                        best_rj = rj
                        
        if best_rj is not None:
            consumed_c.add(ci)
            consumed_r.add(best_rj)
            r = red_dims[best_rj]
            if c['is_ref'] != r['is_ref']:
                difference_pairs.append({
                    'customer': c,
                    'redraw': r,
                    'status': 'Difference to note',
                    'reason': f"参考尺寸属性变更: 客户={'REF' if c['is_ref'] else '标准'} vs 重绘={'REF' if r['is_ref'] else '标准'}"
                })
            elif c['type'] != r['type']:
                matched_pairs.append({
                    'customer': c,
                    'redraw': r,
                    'status': 'Match',
                    'reason': f"符号补充: 客户={c['raw_text']} ↔ 重绘={r['raw_text']}"
                })
            else:
                matched_pairs.append({
                    'customer': c,
                    'redraw': r,
                    'status': 'Match',
                    'reason': '完全一致'
                })

    # -------------------------------------------------------------
    # PASS 2: Equivalent Expression (Degree-Minutes conversion, Arrays)
    # -------------------------------------------------------------
    for ci in unmatched_cust:
        if ci in consumed_c:
            continue
        c = cust_dims[ci]
        if c['is_noise']:
            continue
        
        # Check Angle Conversion (e.g. 33.014° vs 33°1')
        if c['type'] == 'angle' or c['deg_val'] is not None or '°' in c['raw_text']:
            c_angle = c['deg_val'] if c['deg_val'] is not None else c['nominal']
            if c_angle is not None:
                best_rj = None
                for rj in unmatched_red:
                    if rj in consumed_r:
                        continue
                    r = red_dims[rj]
                    if r['type'] == 'angle' or r['deg_val'] is not None or '°' in r['raw_text']:
                        r_angle = r['deg_val'] if r['deg_val'] is not None else r['nominal']
                        if r_angle is not None and abs(c_angle - r_angle) < 0.05: # within 3 arcminutes
                            best_rj = rj
                            break
                if best_rj is not None:
                    consumed_c.add(ci)
                    consumed_r.add(best_rj)
                    equivalent_pairs.append({
                        'customer': c,
                        'redraw': red_dims[best_rj],
                        'status': 'Equivalent expression',
                        'reason': f"角度制式转换: {c['raw_text']} ↔ {red_dims[best_rj]['raw_text']}"
                    })
                    continue

    # -------------------------------------------------------------
    # PASS 3: Close Nominal / Tolerance Discrepancy (Difference to note / Confirmed error)
    # -------------------------------------------------------------
    for ci in unmatched_cust:
        if ci in consumed_c:
            continue
        c = cust_dims[ci]
        c_nom = c['nominal']
        c_type = c['type']
        
        if c_nom is None or c['is_noise']:
            continue
            
        best_rj = None
        best_diff = 999999.0
        for rj in unmatched_red:
            if rj in consumed_r:
                continue
            r = red_dims[rj]
            r_nom = r['nominal']
            r_type = r['type']
            
            # Type compatibility
            type_compat = (c_type == r_type) or (c_type in ['linear', 'diameter'] and r_type in ['linear', 'diameter'])
            if not type_compat:
                continue
                
            if r_nom is not None:
                diff = abs(c_nom - r_nom)
                rel_diff = diff / max(abs(c_nom), 1.0)
                
                # Check if nominally close (< 1% relative or < 5mm absolute on large parts)
                if (diff < 5.0 and rel_diff < 0.02) or (c_nom > 500 and diff < 10.0):
                    if diff < best_diff:
                        best_diff = diff
                        best_rj = rj
                        
        if best_rj is not None:
            consumed_c.add(ci)
            consumed_r.add(best_rj)
            r = red_dims[best_rj]
            diff_val = r['nominal'] - c['nominal']
            difference_pairs.append({
                'customer': c,
                'redraw': r,
                'status': 'Difference to note' if abs(diff_val) <= 1.0 else 'Confirmed error',
                'reason': f"名义值存在差值: 客户={c['nominal']} vs 重绘={r['nominal']} (差 {diff_val:+.2f} mm)"
            })

    # -------------------------------------------------------------
    # PASS 4: Zero-Leakage Residuals (Confirmed Omissions & Redraw Added)
    # -------------------------------------------------------------
    omissions = [cust_dims[ci] for ci in unmatched_cust if ci not in consumed_c and not cust_dims[ci]['is_noise']]
    redraw_added = [red_dims[rj] for rj in unmatched_red if rj not in consumed_r and not red_dims[rj]['is_noise']]
    
    summary = {
        'total_customer': len([d for d in cust_dims if not d['is_noise']]),
        'total_redraw': len([d for d in red_dims if not d['is_noise']]),
        'matched_count': len(matched_pairs),
        'equivalent_count': len(equivalent_pairs),
        'difference_count': len(difference_pairs),
        'omission_count': len(omissions),
        'redraw_added_count': len(redraw_added),
        'matched_pairs': matched_pairs,
        'equivalent_pairs': equivalent_pairs,
        'difference_pairs': difference_pairs,
        'omissions': omissions,
        'redraw_added': redraw_added
    }
    
    return summary

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: py reconcile_ledgers.py <customer_ledger.json> <redraw_ledger.json> [-o <output_dir>]")
        sys.exit(1)
        
    cust_p = sys.argv[1]
    red_p = sys.argv[2]
    out_dir = None
    if len(sys.argv) >= 5 and sys.argv[3] == '-o':
        out_dir = sys.argv[4]
        
    res = reconcile(cust_p, red_p)
    
    print(f"=======================================================")
    print(f"RECONCILIATION SUMMARY")
    print(f"=======================================================")
    print(f"Customer Total Dims : {res['total_customer']}")
    print(f"Redraw Total Dims   : {res['total_redraw']}")
    print(f"-------------------------------------------------------")
    print(f"Match (完全一致)    : {res['matched_count']}")
    print(f"Equivalent (等价)   : {res['equivalent_count']}")
    print(f"Difference (差异)   : {res['difference_count']}")
    print(f"Omission (明确漏注) : {res['omission_count']}")
    print(f"Redraw Added (新增) : {res['redraw_added_count']}")
    print(f"-------------------------------------------------------")
    
    if res['omissions']:
        print(f"\n[!!! CONFIRMED OMISSIONS ({len(res['omissions'])} items) !!!]")
        for d in res['omissions']:
            print(f"  - {d['id']:8s} | {d['raw_text']:25s} | type={d['type']:10s} | bbox={[round(x,1) for x in d['bbox']]}")
            
    if res['difference_pairs']:
        print(f"\n[!!! DIFFERENCES / ERRORS ({len(res['difference_pairs'])} items) !!!]")
        for d in res['difference_pairs']:
            print(f"  - {d['customer']['raw_text']} ↔ {d['redraw']['raw_text']} | {d['reason']}")
            
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'reconciliation-summary.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nSaved structured summary to {os.path.join(out_dir, 'reconciliation-summary.json')}")
