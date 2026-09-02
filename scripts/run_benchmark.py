import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

from reconcile_ledgers import reconcile

base_dir = r"d:\claude-code-space\antigravitywork\图纸审核"
projects = [
    "780691-定锥-452.5920",
    "780684-定锥-452.0842",
    "780683-定锥-452.0832",
    "780682-动锥-452.0820",
    "780685-动锥-452.4900"
]

print("==========================================================================")
print("BENCHMARK RECONCILIATION ACROSS ALL 5 AUDIT PROJECTS")
print("==========================================================================")

for p in projects:
    p_path = os.path.join(base_dir, p)
    cust_ledger = os.path.join(p_path, "ledger_customer", "dimension-ledger.json")
    red_ledger = os.path.join(p_path, "ledger_redraw", "dimension-ledger.json")
    
    if not os.path.exists(cust_ledger) or not os.path.exists(red_ledger):
        print(f"[-] Skipping {p}: Ledger files missing.")
        continue
        
    res = reconcile(cust_ledger, red_ledger)
    
    print(f"\n==========================================================================")
    print(f"PROJECT: {p}")
    print(f"Customer Dims: {res['total_customer']} | Redraw Dims: {res['total_redraw']}")
    print(f"Match: {res['matched_count']} | Equivalent: {res['equivalent_count']} | Difference: {res['difference_count']} | Omission: {res['omission_count']} | Redraw Added: {res['redraw_added_count']}")
    print(f"--------------------------------------------------------------------------")
    
    if res['omissions']:
        print(f"  [OMISSIONS ({len(res['omissions'])})]:")
        for d in res['omissions']:
            # Filter out non-dimension title block text noise
            print(f"    * {d['id']:8s} | {d['raw_text']:30s} | type={d['type']:10s}")
            
    if res['difference_pairs']:
        print(f"  [DIFFERENCES ({len(res['difference_pairs'])})]:")
        for d in res['difference_pairs']:
            print(f"    * {d['customer']['raw_text']:25s} ↔ {d['redraw']['raw_text']:25s} | {d['reason']}")

