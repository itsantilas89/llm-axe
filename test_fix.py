#!/usr/bin/env python3
"""
Quick test of the fixed program name fallback logic.
"""
import sys
from llm_axe.va4_product_discoverer import process_url
from llm_axe.models import OllamaChat

# Initialize LLM
llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")

# Test URL
url = "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/anabathmizo-to-spiti-mou"

# Run process
try:
    extracted_data, classification, _experiment_id = process_url(url, llm, enable_qa=False)
    print("\n✓ Process completed successfully!")
except Exception as e:
    print(f"\n✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
