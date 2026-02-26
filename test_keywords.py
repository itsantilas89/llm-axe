#!/usr/bin/env python3
"""Test keyword detection for energy programs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from llm_axe.va4_product_discoverer import _has_energy_keywords

# Test with the actual program data
data = {
    'programme_name': 'Αναβαθμίζω το Σπίτι μου',
    'description': 'Δάνειο για ενεργειακή αναβάθμιση'
}

print("Testing keyword detection...")
print(f"Programme: {data['programme_name']}")
print(f"Description: {data['description']}")
print()

result = _has_energy_keywords(data)
print(f"\n{'✓' if result else '✗'} Keyword check result: {result}")
