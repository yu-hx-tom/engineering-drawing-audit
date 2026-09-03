"""
audit_gate_check.py - Pre-Report Residual Security Gate
======================================================
Implements Approach 1 (Visual Primary Audit + Script Safety Gate).
Before the agent writes the final audit report, this gate verifies:
1. Duplicate Nominal Inspection (flags numbers like 30, 15, 25 appearing multiple times).
2. Consumptive 1:1 Matching Check (ensures 1 redraw dim cannot satisfy 2 customer dims).
3. Residual Leakage Interception (flags any customer dimension left unverified).
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def run_gate_check(customer_ledger_path: str, redraw_ledger_path: str, output_dir: str = None) -> dict:
    with open(customer_ledger_path, 'r', encoding='utf-8-sig') as f:
        cust_data = json.load(f)
    with open(redraw_ledger_path, 'r', encoding='utf-8-sig') as f:
        red_data = json.load(f)

    cust_dims = cust_data.get('dimensions', [])
    red_dims = red_data.get('dimensions', [])

    # 1. Duplicate Nominal Inspection on Customer Drawing
    nom_counter_cust = Counter()
    for d in cust_dims:
        nom = d.get('nominal')
        if nom is not None and d.get('type') not in ['roughness', 'thread']:
            nom_counter_cust[nom] += 1

    duplicates_cust = {k: v for k, v in nom_counter_cust.items() if v > 1}

    # 2. Duplicate Nominal Inspection on Redraw Drawing
    nom_counter_red = Counter()
    for d in red_dims:
        nom = d.get('nominal')
        if nom is not None and d.get('type') not in ['roughness', 'thread']:
            nom_counter_red[nom] += 1

    # 3. Collision Risk Warning
    collision_risks = []
    for val, c_count in duplicates_cust.items():
        r_count = nom_counter_red.get(val, 0)
        if c_count != r_count:
            collision_risks.append({
                'nominal': val,
                'customer_count': c_count,
                'redraw_count': r_count,
                'risk': f"同名数值碰撞高危: 客户图出现 {c_count} 次，重绘图出现 {r_count} 次 (差额 {c_count - r_count})"
            })

    # 4. Strict 1:1 Consumptive Pool Reconciliation
    from reconcile_ledgers import reconcile
    reconcile_res = reconcile(customer_ledger_path, redraw_ledger_path)

    gate_passed = len(reconcile_res['omissions']) == 0 and len(collision_risks) == 0

    result = {
        'gate_passed': gate_passed,
        'customer_total': len(cust_dims),
        'redraw_total': len(red_dims),
        'duplicate_nominals_in_customer': duplicates_cust,
        'collision_risks': collision_risks,
        'unaccounted_omissions': [
            {
                'id': d['id'],
                'raw_text': d['raw_text'],
                'nominal': d.get('nominal'),
                'type': d.get('type'),
                'bbox': d.get('bbox')
            }
            for d in reconcile_res['omissions']
        ],
        'differences_to_verify': [
            {
                'customer': d['customer']['raw_text'],
                'redraw': d['redraw']['raw_text'],
                'reason': d['reason']
            }
            for d in reconcile_res['difference_pairs']
        ]
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "audit-gate-result.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: py audit_gate_check.py <customer_ledger.json> <redraw_ledger.json> [-o <out_dir>]")
        sys.exit(1)

    c_p = sys.argv[1]
    r_p = sys.argv[2]
    o_d = sys.argv[4] if len(sys.argv) >= 5 and sys.argv[3] == '-o' else None

    res = run_gate_check(c_p, r_p, o_d)
    print("=" * 60)
    print("AUDIT GATE CHECK RESULT (思路一: 报告前安检门)")
    print("=" * 60)
    print(f"Gate Status      : {'PASS (放行)' if res['gate_passed'] else 'INTERCEPTED (强制拦截，存在未核销项或同名碰撞)'}")
    print(f"Customer Dims    : {res['customer_total']}")
    print(f"Redraw Dims      : {res['redraw_total']}")
    print(f"Duplicate Numbers: {len(res['duplicate_nominals_in_customer'])} groups in customer drawing")
    print(f"Collision Risks  : {len(res['collision_risks'])} high-risk collisions detected")
    print(f"Omissions Left   : {len(res['unaccounted_omissions'])} items must be confirmed in drawing")
    print("-" * 60)
    if res['collision_risks']:
        print("\n[🚨 高危同名数值碰撞清单 (必须在视觉上解耦不同的物理特征)]:")
        for risk in res['collision_risks']:
            print(f"  * 数值 {risk['nominal']}: 客户图 {risk['customer_count']} 处 vs 重绘图 {risk['redraw_count']} 处 -> {risk['risk']}")

    if res['unaccounted_omissions']:
        print("\n[🚨 安检门拦截的未核销尺寸 (禁止直接判定为通过，必须二次看图核销)]:")
        for o in res['unaccounted_omissions'][:10]:
            print(f"  * {o['id']:8s} | {o['raw_text']:20s} | type={o['type']}")
        if len(res['unaccounted_omissions']) > 10:
            print(f"  ... 另有 {len(res['unaccounted_omissions']) - 10} 项")
