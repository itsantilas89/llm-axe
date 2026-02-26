#!/usr/bin/env python
"""
Complete Example: Green Loan Product Discovery Pipeline
This script demonstrates the complete workflow of discovering and extracting
energy efficiency-related loan products from Greek banks using trusted sources.
"""

import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_axe.models import OllamaChat
from llm_axe.va4_product_discoverer import (
    load_trusted_sources,
    discover_and_extract_products,
    ensure_outputs_dir
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Main pipeline for green loan product discovery."""
    
    print_section("GREEN LOAN PRODUCT DISCOVERY PIPELINE")
    print("This script will discover and extract energy efficiency-related")
    print("loan products from Greek banks' websites.")
    
    # Step 1: Load and display trusted sources
    print_section("STEP 1: Loading Trusted Sources")
    
    trusted_sources = load_trusted_sources()
    
    if not trusted_sources:
        print("[ERROR] No trusted sources found in trusted_sources.json")
        print("Please ensure the file exists and contains valid JSON.")
        return False
    
    print(f"[✓] Loaded {len(trusted_sources)} source categories")
    
    if "greek_banks_green_loans_stegastika" in trusted_sources:
        urls = trusted_sources["greek_banks_green_loans_stegastika"]
        print(f"[✓] Found {len(urls)} Greek bank URLs for green loans")
        print("\nTrusted Banks:")
        banks = [
            ("Eurobank", urls[0]),
            ("Crediabank", urls[1]),
            ("Piraeus Bank", urls[2]),
            ("National Bank of Greece", urls[3]),
            ("Alpha Bank", urls[4]),
        ]
        for i, (name, url) in enumerate(banks, 1):
            print(f"  {i}. {name}")
            print(f"     → {url}")
    else:
        print("[ERROR] 'greek_banks_green_loans_stegastika' category not found")
        return False
    
    # Step 2: Confirm before proceeding
    print_section("STEP 2: Confirmation")
    
    confirm = input("\nProceed with product discovery? This may take several minutes. [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("[INFO] Cancelled by user")
        return False
    
    # Step 3: Initialize LLM
    print_section("STEP 3: Initializing LLM")
    
    print("Initializing OllamaChat with deepseek-r1:latest...")
    try:
        llm = OllamaChat(model="deepseek-r1:latest")
        print("[✓] LLM initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False
    
    # Step 4: Run discovery for each bank
    print_section("STEP 4: Discovering Products")
    
    all_products = []
    output_dir = ensure_outputs_dir()
    
    for i, (bank_name, bank_url) in enumerate(banks, 1):
        print(f"\n[{i}/{len(banks)}] Processing {bank_name}...")
        
        try:
            products = discover_and_extract_products(
                bank_name=bank_name,
                bank_url=bank_url,
                llm=llm
            )
            
            if products:
                print(f"    → Found {len(products)} relevant products")
                all_products.extend(products)
            else:
                print(f"    → No relevant products found")
                
        except Exception as e:
            print(f"    [ERROR] Failed to process {bank_name}: {e}")
            continue
    
    # Step 5: Generate summary report
    print_section("STEP 5: Summary Report")
    
    print(f"[✓] Discovery Complete!")
    print(f"    Total products discovered: {len(all_products)}")
    print(f"    Output directory: {output_dir}")
    
    if all_products:
        # Group by bank
        by_bank = {}
        for product in all_products:
            bank = product.get("bank_name", "Unknown")
            if bank not in by_bank:
                by_bank[bank] = []
            by_bank[bank].append(product)
        
        print("\nProducts by Bank:")
        for bank, prods in sorted(by_bank.items()):
            print(f"  • {bank}: {len(prods)} product(s)")
            for prod in prods:
                name = prod.get("programme_name", "Unknown")
                rate = prod.get("interest_rate", "N/A")
                print(f"    - {name} (Rate: {rate})")
    
    # Step 6: Save comprehensive summary
    print_section("STEP 6: Saving Results")
    
    summary_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "discovery_complete": True,
        "total_banks_processed": len(banks),
        "total_products_discovered": len(all_products),
        "products": all_products,
    }
    
    summary_path = os.path.join(output_dir, f"discovery_summary_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Summary saved to: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary: {e}")
    
    # Step 7: Next steps
    print_section("NEXT STEPS")
    
    print("\n1. Review the extracted products:")
    print(f"   → Check: {output_dir}")
    
    print("\n2. For manual verification or deeper analysis:")
    print("   → Use VA3 (va3_scraper_to_template.py) with specific product URLs")
    
    print("\n3. Integrate products into your analysis:")
    print("   → Load summary JSON and process programmatically")
    
    print("\n4. Update trusted sources for more banks:")
    print("   → Edit llm_axe/trusted_sources.json")
    
    print("\n[✓] Discovery pipeline complete!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
