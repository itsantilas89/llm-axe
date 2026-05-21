import json

old = json.load(open('output/evaluation/qa_consistency_report_live.json', encoding='utf-8'))
new = json.load(open('output/evaluation/qa_consistency_report_fuzzy.json', encoding='utf-8'))

# Count consistency items
old_applicable = sum(r.get('consistency_applicable', 0) for r in old['results'] if r.get('status') == 'OK')
new_applicable = sum(r.get('consistency_applicable', 0) for r in new['results'] if r.get('status') == 'OK')

old_consistent = sum(r.get('consistency_consistent', 0) for r in old['results'] if r.get('status') == 'OK')
new_consistent = sum(r.get('consistency_consistent', 0) for r in new['results'] if r.get('status') == 'OK')

print(f"OLD: {old_consistent}/{old_applicable} = {old_consistent/old_applicable:.1%}")
print(f"NEW: {new_consistent}/{new_applicable} = {new_consistent/new_applicable:.1%}")

# Show the breakdown
print(f"\nOLD applicable items per program:")
for r in old['results']:
    if r.get('status') == 'OK':
        print(f"  {r.get('programme_name', 'N/A')}: {r['consistency_consistent']}/{r['consistency_applicable']}")

print(f"\nNEW applicable items per program:")
for r in new['results']:
    if r.get('status') == 'OK':
        print(f"  {r.get('programme_name', 'N/A')}: {r['consistency_consistent']}/{r['consistency_applicable']}")
