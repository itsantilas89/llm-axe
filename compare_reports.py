#!/usr/bin/env python3
import json

# Load both reports
with open('output/evaluation/qa_consistency_report_live.json', encoding='utf-8') as f:
    old = json.load(f)
with open('output/evaluation/qa_consistency_report_fuzzy.json', encoding='utf-8') as f:
    new = json.load(f)

# Compare a few items
for i, (o, n) in enumerate(zip(old['results'][:3], new['results'][:3])):
    prog = o.get("programme_name", "N/A")
    print(f'\n=== Item {i+1}: {prog} ===')
    print(f'OLD: {o["consistency_score"]:.1%} ({o["consistency_consistent"]}/{o["consistency_applicable"]} applicable)')
    print(f'NEW: {n["consistency_score"]:.1%} ({n["consistency_consistent"]}/{n["consistency_applicable"]} applicable)')
    
    # Show question detail changes
    if o.get('details') and n.get('details'):
        for j in range(min(2, len(o['details']), len(n['details']))):
            od = o['details'][j] if j < len(o['details']) else {}
            nd = n['details'][j] if j < len(n['details']) else {}
            q_id = od.get('question_id', f'q{j+1}')
            print(f'\n  {q_id}:')
            print(f'    OLD: consistent={od.get("is_consistent")}, reason={od.get("consistency_reason", "N/A")[:70]}')
            print(f'    NEW: consistent={nd.get("is_consistent")}, reason={nd.get("consistency_reason", "N/A")[:70]}')
