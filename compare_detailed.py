import json
old = json.load(open('output/evaluation/qa_consistency_report_live.json', encoding='utf-8'))
new = json.load(open('output/evaluation/qa_consistency_report_fuzzy.json', encoding='utf-8'))

old_items = [r for r in old.get('results', []) if r.get('status') == 'OK']
new_items = [r for r in new.get('results', []) if r.get('status') == 'OK']

print(f'OLD: {len(old_items)} items, AVG {old.get("summary", {}).get("avg_consistency", 0):.1%}')
print(f'NEW: {len(new_items)} items, AVG {new.get("summary", {}).get("avg_consistency", 0):.1%}')

# Show scores
print('\nOLD scores:', [f"{r['consistency_score']:.0%}" for r in old_items[:6]])
print('NEW scores:', [f"{r['consistency_score']:.0%}" for r in new_items[:6]])

# Find changes
for old_r, new_r in zip(old_items, new_items):
    if old_r['consistency_score'] != new_r['consistency_score']:
        print(f"\n{new_r['programme_name']}: {old_r['consistency_score']:.0%} → {new_r['consistency_score']:.0%}")
