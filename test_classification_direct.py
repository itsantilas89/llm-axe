#!/usr/bin/env python3
"""Test classification directly with mock data"""

from llm_axe.va4_product_discoverer import classify_product
from llm_axe.models import OllamaChat
import json

# Initialize LLM
llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")

# Test 1: Microsoft app (completely irrelevant)
print("\n" + "="*80)
print("TEST 1: Microsoft App (should be REJECTED immediately)")
print("="*80)
microsoft_data = {
    "programme_name": "Lenovo Vantage",
    "description": "Free download and install on Windows OS. Manage your PC settings.",
    "programme_objective": "PC management software",
    "eligible_interventions": []
}

result1 = classify_product(llm, microsoft_data)
print("\nRESULT:")
print(json.dumps(result1, ensure_ascii=False, indent=2))
print(f"\n✓ CORRECT" if not result1["is_relevant"] else f"✗ WRONG - Should be rejected!")

# Test 2: Energy upgrade program (should be accepted)
print("\n" + "="*80)
print("TEST 2: Energy Upgrade Program (should be ACCEPTED)")
print("="*80)
energy_data = {
    "programme_name": "Εξοικονομώ 2024",
    "description": "Πρόγραμμα επιδότησης για ενεργειακή αναβάθμιση κατοικιών",
    "programme_objective": "Μείωση ενεργειακής κατανάλωσης",
    "eligible_interventions": [
        "Θερμομόνωση εξωτερικών τοίχων",
        "Αλλαγή κουφωμάτων",
        "Εγκατάσταση φωτοβολταϊκών"
    ],
    "energy_performance_targets": "Αναβάθμιση κατά 2 ενεργειακές κλάσεις"
}

result2 = classify_product(llm, energy_data)
print("\nRESULT:")
print(json.dumps(result2, ensure_ascii=False, indent=2))
print(f"\n✓ CORRECT" if result2["is_relevant"] else f"✗ WRONG - Should be accepted!")

# Test 3: Green housing loan (should be accepted)
print("\n" + "="*80)
print("TEST 3: Green Housing Loan (should be ACCEPTED)")
print("="*80)
green_housing_data = {
    "programme_name": "Αναβαθμίζω το σπίτι μου - Eurobank",
    "description": "Στεγαστικό δάνειο με υποχρεωτική ενεργειακή αναβάθμιση",
    "programme_objective": "Χρηματοδότηση αγοράς ή κατασκευής με ενεργειακές παρεμβάσεις",
    "property_requirements": ["Ελάχιστη ενεργειακή κλάση Β+"],
    "eligible_interventions": [
        "Θερμομόνωση",
        "Φωτοβολταϊκά",
        "Ηλιακοί συλλέκτες"
    ]
}

result3 = classify_product(llm, green_housing_data)
print("\nRESULT:")
print(json.dumps(result3, ensure_ascii=False, indent=2))
print(f"\n✓ CORRECT" if result3["is_relevant"] and result3["primary_category"] == "green_housing_loan" else f"✗ WRONG - Should be green_housing_loan!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Test 1 (Microsoft): {'✓ PASS' if not result1['is_relevant'] else '✗ FAIL'}")
print(f"Test 2 (Energy Program): {'✓ PASS' if result2['is_relevant'] else '✗ FAIL'}")
print(f"Test 3 (Green Housing): {'✓ PASS' if result3['is_relevant'] else '✗ FAIL'}")
print("="*80 + "\n")
