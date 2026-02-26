#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for VA4 Product Discoverer

This script tests the basic functionality of VA4 without requiring actual URLs.
It uses mock data to verify the classification logic.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_axe.va4_product_discoverer import (
    classify_product, 
    CATEGORIES_OF_INTEREST,
    CATEGORIES_NOT_OF_INTEREST,
    ALL_CATEGORIES
)
from llm_axe.models import OllamaChat

def test_categories_defined():
    """Test that all category definitions are properly set."""
    print("\n" + "="*70)
    print("TEST 1: Έλεγχος ορισμού κατηγοριών")
    print("="*70)
    
    print("\nΚατηγορίες ενδιαφέροντος:")
    for key, desc in CATEGORIES_OF_INTEREST.items():
        print(f"  ✓ {key}: {desc}")
    
    print("\nΚατηγορίες χωρίς ενδιαφέρον:")
    for key, desc in CATEGORIES_NOT_OF_INTEREST.items():
        print(f"  ✗ {key}: {desc}")
    
    print(f"\nΣύνολο κατηγοριών: {len(ALL_CATEGORIES)}")
    print("✓ TEST PASSED: Όλες οι κατηγορίες ορίστηκαν σωστά")
    return True

def test_mock_classification():
    """Test classification with mock data."""
    print("\n" + "="*70)
    print("TEST 2: Κατηγοριοποίηση με mock δεδομένα")
    print("="*70)
    
    # Mock extracted data for a green loan program
    mock_data_green_loan = {
        "programme_name": "Πρόγραμμα Εξοικονομώ 2025",
        "description": "Πρόγραμμα επιδότησης για ενεργειακή αναβάθμιση κατοικιών με εγκατάσταση φωτοβολταϊκών και θερμομόνωση",
        "eligible_parties": ["Φυσικά πρόσωπα", "Ιδιοκτήτες κατοικιών"],
        "eligible_interventions": [
            "Φωτοβολταϊκά συστήματα",
            "Θερμομόνωση",
            "Ανάλλαγή κουφωμάτων"
        ],
        "funding_type": "Επιδότηση + Δάνειο",
        "maximum_funding_amount": "60.000€"
    }
    
    # Mock data for electric vehicle (not of interest)
    mock_data_ev = {
        "programme_name": "Επιδότηση Ηλεκτρικών Οχημάτων 2026",
        "description": "Πρόγραμμα επιδότησης για αγορά ηλεκτρικών αυτοκινήτων",
        "eligible_parties": ["Φυσικά πρόσωπα"],
        "eligible_interventions": [
            "Αγορά ηλεκτρικού αυτοκινήτου",
            "Εγκατάσταση φορτιστή"
        ],
        "maximum_funding_amount": "8.000€"
    }
    
    try:
        llm = OllamaChat(model="deepseek-r1:latest")
        
        # Test 1: Green loan (should be relevant)
        print("\n[Test 2.1] Κατηγοριοποίηση: Πρόγραμμα Εξοικονομώ")
        classification1 = classify_product(llm, mock_data_green_loan)
        
        print(f"  Κατηγορία: {classification1.get('primary_category')}")
        print(f"  Ενδιαφέρον: {'✓ ΝΑΙ' if classification1.get('is_relevant') else '✗ ΟΧΙ'}")
        print(f"  Βεβαιότητα: {classification1.get('confidence', 0.0):.0%}")
        
        if classification1.get('is_relevant') and classification1.get('primary_category') in CATEGORIES_OF_INTEREST:
            print("  ✓ PASS: Σωστή κατηγοριοποίηση (relevant)")
        else:
            print("  ✗ FAIL: Λάθος κατηγοριοποίηση")
            return False
        
        # Test 2: Electric vehicle (should NOT be relevant)
        print("\n[Test 2.2] Κατηγοριοποίηση: Ηλεκτρικά Οχήματα")
        classification2 = classify_product(llm, mock_data_ev)
        
        print(f"  Κατηγορία: {classification2.get('primary_category')}")
        print(f"  Ενδιαφέρον: {'✓ ΝΑΙ' if classification2.get('is_relevant') else '✗ ΟΧΙ'}")
        print(f"  Βεβαιότητα: {classification2.get('confidence', 0.0):.0%}")
        
        if not classification2.get('is_relevant'):
            print("  ✓ PASS: Σωστή κατηγοριοποίηση (not relevant)")
        else:
            print("  ✗ FAIL: Λάθος κατηγοριοποίηση (θα έπρεπε να μην είναι relevant)")
            return False
        
        print("\n✓ TEST PASSED: Η κατηγοριοποίηση λειτουργεί σωστά")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False

def test_output_structure():
    """Test that output directories and files can be created."""
    print("\n" + "="*70)
    print("TEST 3: Έλεγχος δημιουργίας output")
    print("="*70)
    
    try:
        from llm_axe.va4_product_discoverer import ensure_outputs_dir, save_classification_result
        
        # Test output directory creation
        out_dir = ensure_outputs_dir()
        print(f"\n  Output directory: {out_dir}")
        
        if os.path.exists(out_dir):
            print("  ✓ Output directory exists")
        else:
            print("  ✗ Failed to create output directory")
            return False
        
        # Test saving classification result
        mock_url = "https://example.com/test-program"
        mock_extracted = {"programme_name": "Test Program"}
        mock_classification = {
            "is_relevant": True,
            "primary_category": "energy_upgrade",
            "confidence": 0.9
        }
        
        saved_path = save_classification_result(mock_url, mock_extracted, mock_classification)
        print(f"  Test file saved to: {saved_path}")
        
        if os.path.exists(saved_path):
            print("  ✓ Classification file created successfully")
            
            # Verify JSON is valid
            with open(saved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('classification') and data.get('extracted_data'):
                print("  ✓ JSON structure is valid")
            else:
                print("  ✗ JSON structure is invalid")
                return False
            
            # Clean up test file
            os.remove(saved_path)
            print("  ✓ Test file cleaned up")
        else:
            print("  ✗ Failed to create classification file")
            return False
        
        print("\n✓ TEST PASSED: Output system works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False

def test_import_va3():
    """Test that VA3 components can be imported."""
    print("\n" + "="*70)
    print("TEST 4: Έλεγχος ενσωμάτωσης VA3")
    print("="*70)
    
    try:
        from llm_axe.va3_scraper_to_template import (
            scrape_page, 
            extract_json, 
            TEMPLATE_DEFAULT
        )
        
        print("\n  ✓ scrape_page imported")
        print("  ✓ extract_json imported")
        print("  ✓ TEMPLATE_DEFAULT imported")
        
        # Check template structure
        template = TEMPLATE_DEFAULT[0]
        required_fields = [
            'programme_name', 
            'description', 
            'eligible_parties',
            'eligible_interventions'
        ]
        
        for field in required_fields:
            if field in template:
                print(f"  ✓ Template field '{field}' exists")
            else:
                print(f"  ✗ Template field '{field}' missing")
                return False
        
        print("\n✓ TEST PASSED: VA3 integration successful")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("VA4 PRODUCT DISCOVERER - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Ορισμός Κατηγοριών", test_categories_defined),
        ("Mock Κατηγοριοποίηση", test_mock_classification),
        ("Output System", test_output_structure),
        ("VA3 Integration", test_import_va3),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nΣύνολο: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Όλα τα tests πέρασαν επιτυχώς!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) απέτυχαν")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
