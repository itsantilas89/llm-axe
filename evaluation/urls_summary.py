#!/usr/bin/env python
"""Generate a summary of all URLs with their classification and validation status."""

import json
from pathlib import Path
from collections import defaultdict

# Load validation report
validation_report = json.loads(
    Path("output/evaluation/batch_validation_report_detailed.json").read_text(encoding="utf-8")
)

# Build coverage map
coverage_map = {}
for result in validation_report["results"]:
    class_file = Path(result["classification_file"]).name
    coverage = result["summary"]["coverage"]
    coverage_map[class_file] = coverage

# Process classification files
print("=" * 120)
print(f"{'STATUS':<15} | {'CATEGORY':<30} | {'COVERAGE':<10} | {'URL':<60}")
print("=" * 120)

files = sorted(Path("output/va4_product_discoverer").glob("*_classification.json"))
status_counts = defaultdict(int)

for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        url = data.get("url", "N/A")
        classification = data.get("classification", {})
        is_relevant = classification.get("is_relevant", False)
        category = classification.get("primary_category", "unknown")
        confidence = classification.get("confidence", 0)
        
        # Get coverage from validation report
        coverage = coverage_map.get(f.name, "N/A")
        coverage_str = f"{coverage:.0%}" if isinstance(coverage, float) else "SKIP"
        
        if is_relevant:
            status = "✓ RELEVANT"
            status_counts["relevant"] += 1
        else:
            status = "✗ NOT"
            status_counts["not_relevant"] += 1
        
        url_short = url[:58] if len(url) > 58 else url
        print(f"{status:<15} | {category:<30} | {coverage_str:<10} | {url_short:<60}")
        
    except Exception as e:
        print(f"ERROR processing {f.name}: {e}")

print("=" * 120)
print(f"\nSummary:")
print(f"  Relevant:     {status_counts['relevant']}")
print(f"  Not relevant: {status_counts['not_relevant']}")
print(f"  Total files:  {len(files)}")
print(f"\nValidation Coverage: 0% (NO values from extracted_data found in source HTML)")
print("\nThis indicates either:")
print("  1. VA4 mis-classified content (e.g., nutrition blogs as energy programs)")
print("  2. Extracted data doesn't match source HTML")
print("  3. VA3 extraction failed or created hallucinated content")
