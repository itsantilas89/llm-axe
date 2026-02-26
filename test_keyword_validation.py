#!/usr/bin/env python3
"""Test keyword validation without LLM calls"""

from llm_axe.va4_product_discoverer import _has_energy_keywords

# Test 1: Microsoft app data (should be rejected)
microsoft_data = {
    "programme_name": "Lenovo Vantage",
    "description": "Free download and install on Windows",
    "programme_objective": "",
    "eligible_interventions": []
}

# Test 2: Energy program (should pass)
energy_data = {
    "programme_name": "Εξοικονομώ 2024",
    "description": "Πρόγραμμα ενεργειακής αναβάθμισης με επιδότηση για θερμομόνωση",
    "eligible_interventions": ["Θερμομόνωση", "Αλλαγή κουφωμάτων"]
}

# Test 3: Green housing loan (should pass)
green_housing_data = {
    "programme_name": "Αναβαθμίζω το σπίτι μου",
    "description": "Στεγαστικό δάνειο με υποχρεωτική ενεργειακή αναβάθμιση",
    "energy_performance_targets": "Ελάχιστη κλάση Β+"
}

print("\n" + "="*60)
print("KEYWORD VALIDATION TEST")
print("="*60)

print("\n[TEST 1] Microsoft App (should reject):")
result1 = _has_energy_keywords(microsoft_data)
print(f"  Result: {result1} (expected: False)")
print(f"  ✓ PASS" if not result1 else f"  ✗ FAIL")

print("\n[TEST 2] Energy Program (should accept):")
result2 = _has_energy_keywords(energy_data)
print(f"  Result: {result2} (expected: True)")
print(f"  ✓ PASS" if result2 else f"  ✗ FAIL")

print("\n[TEST 3] Green Housing Loan (should accept):")
result3 = _has_energy_keywords(green_housing_data)
print(f"  Result: {result3} (expected: True)")
print(f"  ✓ PASS" if result3 else f"  ✗ FAIL")

print("\n" + "="*60)
all_pass = (not result1) and result2 and result3
print(f"\nOVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
print("="*60 + "\n")
