import json
from pathlib import Path

# List actual Q&A questions from stored responses
qa_files = list(Path('output/va4_product_discoverer').glob('*_llm_raw_*.txt'))
questions = {}

for f in qa_files[:3]:
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'qa_responses' in data:
            for q, a in data['qa_responses'].items():
                if q not in questions:
                    questions[q] = 0
                questions[q] += 1
    except:
        pass

# Print unique questions
print(f"Found {len(questions)} unique questions:\n")
for i, q in enumerate(sorted(questions.keys()), 1):
    print(f"{i}. {q[:100]}...")
    if len(q) > 100:
        print(f"   (full: {q})\n")
