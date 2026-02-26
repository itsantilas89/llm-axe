#!/usr/bin/env python3
"""
Quick test of the simplified logging system.

Run this to create a test log entry.
"""

from llm_axe.simple_logger import log_experiment

# Test creating an experiment log
experiment_id = log_experiment(
    model="llama3.2",
    temperature=0.5,
    hyperparameters={"max_retries": 2, "keyword_filter": True},
    prompt="Example prompt for testing",
    sources=["https://example.com"],
    terminal_output="Test output: Classification successful"
)

print(f"✓ Test experiment created: {experiment_id}")
print(f"✓ Log file: logs/{experiment_id}.json")
print("\nView it with:")
print(f"  cat logs/{experiment_id}.json | python -m json.tool")
