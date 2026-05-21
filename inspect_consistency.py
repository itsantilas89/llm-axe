import json

# Get improved report
improved = json.load(open('output/evaluation/qa_consistency_report_improved.json', encoding='utf-8'))

# Look at one detailed result
result = improved['results'][0]  # First program
print(f"Program: {result['programme_name']}")
print(f"URL: {result['url']}")
print(f"Consistency: {result['consistency_score']:.0%} ({result['consistency_consistent']}/{result['consistency_applicable']})")
print()

# Show details of first 3 questions
for i, detail in enumerate(result['details'][:3]):
    print(f"Q{i+1}: {detail['question'][:70]}...")
    print(f"  Applicable: {detail.get('consistency_applicable', False)}")
    print(f"  Consistent: {detail.get('is_consistent', False)}")
    print(f"  Reason: {detail.get('reason', 'N/A')[:80]}")
    print(f"  Fields checked: {detail['fields_checked']}")
    if detail.get('extracted_values'):
        vals = detail['extracted_values']
        print(f"  Values: {vals[:2]}...")
    print()
