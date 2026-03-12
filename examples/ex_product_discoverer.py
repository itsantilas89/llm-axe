#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of VA4 Product Discoverer

This script demonstrates how to use va4_product_discoverer to:
1. Scrape and extract data from a URL
2. Classify the program/product into relevant categories
3. Start an interactive Q&A session if the program is of interest
"""

import sys
import os

# Add parent directory to path so we can import llm_axe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_axe.va4_product_discoverer import process_url, log
from llm_axe.models import OllamaChat

def example_single_url():
    """Example: Process a single URL"""
    print("\n" + "="*70)
    print("ΠΑΡΑΔΕΙΓΜΑ: Ανάλυση ενός URL")
    print("="*70 + "\n")
    
    # Initialize LLM
    llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
    
    # Example URLs to try (replace with real ones)
    example_urls = [
        # Green loan program
        "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/eksoikonomo-2025",
        # Housing loan
        "https://www.nbg.gr/el/retail/loans/home-loans",
    ]
    
    # Let user choose or input custom URL
    print("Επιλέξτε URL ή εισάγετε δικό σας:")
    for i, url in enumerate(example_urls, 1):
        print(f"{i}. {url}")
    print(f"{len(example_urls) + 1}. Εισαγωγή custom URL")
    
    try:
        choice = input("\nΕπιλογή> ").strip()
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(example_urls):
                url = example_urls[idx - 1]
            elif idx == len(example_urls) + 1:
                url = input("Εισάγετε URL> ").strip()
            else:
                print("Μη έγκυρη επιλογή")
                return
        else:
            url = choice
        
        if not url:
            print("Δεν δόθηκε URL")
            return
        
        # Process the URL
        extracted_data, classification = process_url(url, llm, enable_qa=True)
        
        # Print summary
        print("\n" + "="*70)
        print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
        print("="*70)
        print(f"Πρόγραμμα: {extracted_data.get('programme_name', 'N/A')}")
        print(f"Κατηγορία: {classification.get('primary_category', 'N/A')}")
        print(f"Ενδιαφέρων: {'ΝΑΙ' if classification.get('is_relevant') else 'ΟΧΙ'}")
        print(f"Βεβαιότητα: {classification.get('confidence', 0.0):.0%}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Ακύρωση από χρήστη")
    except Exception as e:
        print(f"\n[ERROR] Σφάλμα: {e}")

def example_batch_processing():
    """Example: Process multiple URLs and compare"""
    print("\n" + "="*70)
    print("ΠΑΡΑΔΕΙΓΜΑ: Ομαδική επεξεργασία URLs")
    print("="*70 + "\n")
    
    # Initialize LLM
    llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
    
    # Example URLs covering different categories
    urls_to_test = [
        # Add real URLs here
        "https://example.com/green-loan",
        "https://example.com/electric-vehicle-subsidy",
        "https://example.com/home-renovation-program",
    ]
    
    results = []
    
    for i, url in enumerate(urls_to_test, 1):
        print(f"\n[{i}/{len(urls_to_test)}] Επεξεργασία: {url}")
        try:
            extracted_data, classification = process_url(
                url, llm, enable_qa=False  # Disable Q&A for batch processing
            )
            results.append({
                "url": url,
                "programme_name": extracted_data.get("programme_name", "N/A"),
                "category": classification.get("primary_category", "unknown"),
                "relevant": classification.get("is_relevant", False),
                "confidence": classification.get("confidence", 0.0)
            })
        except Exception as e:
            print(f"[ERROR] Αποτυχία για {url}: {e}")
            results.append({
                "url": url,
                "programme_name": "ERROR",
                "category": "error",
                "relevant": False,
                "confidence": 0.0
            })
    
    # Print summary table
    print("\n" + "="*70)
    print("ΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("="*70)
    print(f"{'Πρόγραμμα':<30} {'Κατηγορία':<20} {'Ενδιαφέρον':<12} {'Βεβαιότητα'}")
    print("-"*70)
    for r in results:
        relevant_symbol = "✓" if r["relevant"] else "✗"
        print(f"{r['programme_name'][:30]:<30} {r['category']:<20} {relevant_symbol:<12} {r['confidence']:.0%}")
    print("="*70 + "\n")
    
    # Count relevant vs not relevant
    relevant_count = sum(1 for r in results if r["relevant"])
    print(f"Σύνολο: {len(results)} URLs")
    print(f"Ενδιαφέροντα: {relevant_count}")
    print(f"Μη ενδιαφέροντα: {len(results) - relevant_count}\n")

def example_interactive_mode():
    """Example: Start interactive mode"""
    print("\n" + "="*70)
    print("ΠΑΡΑΔΕΙΓΜΑ: Διαδραστικό Μοδέ")
    print("="*70 + "\n")
    
    # Initialize LLM
    llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
    
    print("Εισάγετε URLs για ανάλυση. Γράψτε 'exit' για έξοδο.\n")
    
    while True:
        try:
            url = input("🔗 URL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Έξοδος")
            break
        
        if not url:
            continue
        
        if url.lower() in {"exit", "quit", "q"}:
            print("[INFO] Έξοδος")
            break
        
        try:
            process_url(url, llm, enable_qa=True)
        except Exception as e:
            print(f"\n[ERROR] Σφάλμα: {e}\n")

def main():
    """Main function to run examples"""
    
    print("\n" + "="*70)
    print("VA4 PRODUCT DISCOVERER - ΠΑΡΑΔΕΙΓΜΑΤΑ ΧΡΗΣΗΣ")
    print("="*70)
    print("\nΕπιλέξτε παράδειγμα:")
    print("1. Ανάλυση ενός URL με Q&A")
    print("2. Ομαδική επεξεργασία πολλών URLs")
    print("3. Διαδραστικό μοδέ")
    print("4. Έξοδος")
    
    try:
        choice = input("\nΕπιλογή> ").strip()
        
        if choice == "1":
            example_single_url()
        elif choice == "2":
            example_batch_processing()
        elif choice == "3":
            example_interactive_mode()
        elif choice == "4":
            print("Έξοδος")
        else:
            print("Μη έγκυρη επιλογή")
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Ακύρωση από χρήστη")
    except Exception as e:
        print(f"\n[ERROR] Σφάλμα: {e}")

if __name__ == "__main__":
    main()
