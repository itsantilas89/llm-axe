#!/usr/bin/env python3
"""Quick test to debug URL classification"""

from llm_axe.va4_product_discoverer import process_url
from llm_axe.models import OllamaChat
import json

# Initialize LLM
llm = OllamaChat(model="deepseek-r1:latest")

# Test URL
url = "https://apps.microsoft.com/detail/9wzdncrfj4mv?hl=en-US&gl=GR"

print(f"\n{'='*80}")
print(f"Testing URL: {url}")
print(f"{'='*80}\n")

# Process URL
result = process_url(url, llm)

# Print result
print("\n" + "="*80)
print("RESULT:")
print("="*80)
print(json.dumps(result, ensure_ascii=False, indent=2))
