# Experiment Logging System (Simple)

## Overview
Every VA4 classification is automatically logged to `logs/` directory for tracking experiments and reproducibility.

## What Gets Logged

For each classification, a JSON file is created at: `logs/{experiment_id}.json`

**Example**: `logs/exp_20260226_123456_a1b2c3d4.json`

### Log Structure
```json
{
  "experiment_id": "exp_20260226_123456_a1b2c3d4",
  "timestamp": "2026-02-26T12:34:56.000000",
  "model": "llama3.2",
  "temperature": 0.1,
  "hyperparameters": {
    "max_retries": 2,
    "keyword_filter": true
  },
  "prompt": "User prompt preview (first 300 chars)",
  "sources": [
    "https://website.com/program"
  ],
  "terminal_output": "Category: energy_upgrade | Confidence: 92% | Relevant: true"
}
```

## Naming Convention

- **experiment_id** format: `exp_{YYYYMMDD}_{HHMMSS}_{short_uuid}`
- **File**: `logs/{experiment_id}.json`

Use this same experiment_id to name the extracted template JSON for correlation:
- Classification: `logs/exp_20260226_123456_a1b2c3d4.json`
- Template: `output/va4_product_discoverer/exp_20260226_123456_a1b2c3d4_template.json`

## How to Use

### View Logs
```bash
ls logs/
```

### Read a Specific Log
```bash
cat logs/exp_20260226_123456_a1b2c3d4.json | python -m json.tool
```

### Process URL with Logging
```bash
python -m llm_axe.va4_product_discoverer "https://example.com/program"
```

The console output will show:
```
[LOG] Experiment: exp_20260226_123456_a1b2c3d4
```

## Fields Explained

| Field | Description |
|-------|-------------|
| `experiment_id` | Unique identifier, matches template file names |
| `timestamp` | ISO 8601 UTC time |
| `model` | LLM model used (e.g., llama3.2) |
| `temperature` | Temperature parameter for LLM |
| `hyperparameters` | Other tuned parameters (max_retries, etc.) |
| `prompt` | Preview of the prompt sent to LLM |
| `sources` | List of source URLs processed |
| `terminal_output` | Summary of classification result |

## Analyzing Experiments

### Find All Energy Programs
```bash
grep -l '"Relevant: true"' logs/*.json
```

### Find Failed Classifications
```bash
grep -l '"ERROR:' logs/*.json
```

### Count by Temperature
```bash
grep '"temperature"' logs/*.json | sort | uniq -c
```

### Extract Average Confidence
```python
import json, os, statistics
confs = []
for f in os.listdir("logs"):
    if f.endswith(".json"):
        with open(f"logs/{f}") as fp:
            data = json.load(fp)
            if "Confidence:" in data["terminal_output"]:
                conf_str = data["terminal_output"].split("Confidence: ")[1].split("%")[0]
                confs.append(float(conf_str))
print(f"Average confidence: {statistics.mean(confs):.0%}")
```

## Auto-Logging

Logging happens automatically in `classify_product()`. To disable for a specific call:

```python
classification, exp_id = classify_product(llm, extracted_data, log_it=False)
```

## Output Directory

```
logs/
├── exp_20260226_123456_a1b2c3d4.json
├── exp_20260226_123457_b1c2d3e4.json
└── exp_20260226_123458_c1d2e3f4.json
```

---

**Simple, lean, and auditable. Every experiment is recorded for reproducibility.**
