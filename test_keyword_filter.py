#!/usr/bin/env python3
"""Quick test of the dual keyword filter."""

# Direct import without network
import os
import sys
os.chdir('C:\\Users\\stavr\\Desktop\\HMMY\\Thesis\\llm-axe')
sys.path.insert(0, 'C:\\Users\\stavr\\Desktop\\HMMY\\Thesis\\llm-axe')

# Import and test
from llm_axe.va4_product_discoverer import _has_energy_keywords

# Suppress debug output temporarily
import io
from contextlib import redirect_stdout

print("=" * 70)
print("KEYWORD FILTER TEST")
print("=" * 70)

tests = [
    ("Real financing program (should PASS)", 
     "Αναβαθμίζω το σπίτι μου - Δάνειο για ενεργειακή αναβάθμιση κατοικιών"),
    
    ("Blog about nutrition/energy (should FAIL)", 
     "16 τροφές που ενισχύουν την ενέργεια του οργανισμού - Διατροφή και ενέργεια"),
    
    ("Blog with generic 'program' word (should FAIL)",
     "Το πρόγραμμα εξοικονόμησης ενέργειας στη ζωή σας χωρίς δάνειο"),
    
    ("Actual green loan program (should PASS)",
     "Πράσινο στεγαστικό δάνειο με χρηματοδότηση για ενεργειακή αναβάθμιση"),
]

for name, text in tests:
    print(f"\n{name}")
    print(f"Text: {text[:60]}...")
    
    # Capture output
    f = io.StringIO()
    with redirect_stdout(f):
        result = _has_energy_keywords(text)
    
    output = f.getvalue()
    status = "✓ PASS (accepted)" if result else "✗ FAIL (rejected)"
    print(f"Result: {status}")
    if output:
        lines = output.strip().split('\n')
        for line in lines:
            if '[DEBUG]' in line or '[WARN]' in line:
                print(f"  {line}")

