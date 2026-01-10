# Verification & Quality Assurance Guide

## Τι Είναι το Verification Layer;

Το **Verification Layer** είναι ένα σύνολο ελέγχων που εκτελούνται **μετά** την εξαγωγή δεδομένων, για να βεβαιωθούμε ότι:

✅ Τα εξαγόμενα δεδομένα είναι ακριβή (όχι hallucinated)
✅ Δεν έχασαν κάποιο σχετικό product
✅ Όλα τα απαιτούμενα πεδία συμπληρώθηκαν
✅ Τα δεδομένα είναι λογικά (π.χ., min ≤ max)

---

## Πώς Λειτουργεί;

### 1️⃣ Hallucination Detection

**Τι κάνει:**
- Συγκρίνει τα extracted fields με το source text
- Ρωτάει το LLM: "Είναι αυτά τα στοιχεία όντως στο webpage;"

**Παράδειγμα:**
```json
{
  "programme_name": "Green Loan Plus",
  "interest_rate": "2.5%",
  "max_amount": "€100,000"
}
```

LLM checks: "Υπάρχει πραγματικά 2.5% στοιχείο;
- ✅ "Ναι, λέει 2.5% APR" → verified
- ❌ "Όχι, λέει 3.2%" → HALLUCINATION

### 2️⃣ Required Fields Check

**Τι κάνει:**
- Ελέγχει αν τα απαιτούμενα πεδία υπάρχουν και δεν είναι κενά

**Required fields:**
- `programme_name`
- `description`  
- `interest_rate`

**Confidence penalty:** -0.2 per missing field

### 3️⃣ Data Reasonableness Check

**Τι κάνει:**
- Ελέγχει λογικές σχέσεις (π.χ., min ≤ max)
- Ανιχνεύει λογικά λάθη

**Παράδειγμα:**
```
minimum_funding_amount: €100,000
maximum_funding_amount: €50,000  ❌ ERROR!
```

**Confidence penalty:** -0.3

### 4️⃣ Completeness Check

**Τι κάνει:**
- Ρωτάει το LLM: "Υπάρχουν άλλα products που δεν ανακαλύψαμε;"
- Ανιχνεύει potentially missed products

**Output:**
```
Potentially missed products: [
  "Green Loan Standard",
  "Home Energy Upgrade Fund"
]
```

---

## Confidence Scoring

Κάθε product λαμβάνει ένα **confidence score** (0.0 - 1.0):

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Excellent | ✅ Accept as-is |
| 0.7-0.89 | Good | ✅ Accept with minor review |
| 0.5-0.69 | Fair | ⚠️ Review manually |
| 0.3-0.49 | Poor | ❌ Reject or re-extract |
| <0.3 | Very Poor | ❌ Reject |

**Penalties:**
- Missing required field: -0.2
- Hallucinated field: -0.3
- Uncertain field: -0.1
- Unreasonable data (min > max): -0.3

---

## Verification Output Structure

Κάθε product χρησιμοποιεί αυτή τη δομή:

```json
{
  "programme_name": "Green Loan",
  "description": "...",
  "verification": {
    "is_valid": true,
    "confidence_score": 0.85,
    "hallucination_flags": [],
    "missing_fields": [],
    "warnings": [],
    "verification_details": {
      "verified": ["interest_rate", "loan_duration"],
      "hallucinated": [],
      "uncertain": ["eligible_interventions"]
    }
  }
}
```

---

## Πώς να Εκτελέσεις την Verification

### Βήμα 1: Εκτέλεση Discovery
```bash
python llm_axe/va4_product_discoverer.py
```

### Βήμα 2: Εξέταση του Summary Report
```bash
# Ο summary report περιέχει:
# - total_products_found
# - verified_products
# - hallucinations_detected
# - average_confidence_score

cat output/va4_product_discoverer/20250108T120000Z_discovery_summary.json
```

### Βήμα 3: Εξέταση Verification Report
```bash
python examples/ex_verify_discovery_results.py
```

---

## Ανάγνωση του Verification Report

### 📊 Summary Statistics
```
Total products discovered: 15
Verified products: 12
Hallucinations detected: 2
Average confidence score: 82%
```

### ⚠️ Products with Hallucinations
```
1. Eurobank Green Loan (Eurobank)
   ❌ interest_rate: "2.5%" not found in source
   ❌ max_amount: "€150,000" not explicitly stated
```

### 🔍 Low Confidence Products (< 60%)
```
1. NBG Sustainable Loan (NBG)
   Confidence: 52%
   URL: https://www.nbg.gr/...
   → Needs manual review
```

### 📋 Potentially Missed Products
```
Eurobank: 3 potentially missed products
  • Premium Green Loan
  • Standard Housing Upgrade Loan
  • Energy Efficiency Fund
```

---

## Manual Review Process

Αν ένα product έχει low confidence, θα πρέπει να:

1. **Πήγαινε στο source URL**
   ```
   https://www.eurobank.gr/...
   ```

2. **Σύγκρινε με τα extracted δεδομένα**
   - Είναι το interest rate ακριβές;
   - Είναι τα eligible interventions σωστά;

3. **Ενημέρωσε το JSON ή σημείωσε τα λάθη**
   ```json
   {
     "field_name": "correct_value",
     "manual_verified": true,
     "notes": "Verified against source page"
   }
   ```

---

## Troubleshooting

### Υψηλός αριθμός hallucinations;

**Πιθανά αίτια:**
- LLM temperature πολύ υψηλή
- Poor source text quality
- Ambiguous language in webpage

**Λύσεις:**
1. ↓ LLM temperature (πχ. 0.05 αντί 0.1)
2. Adjust extraction prompts για λιγότερο hallucination
3. Manual verification

### Χαμηλός average confidence;

**Πιθανά αίτια:**
- Πολλά missing fields
- Δύσκολο webpage structure
- Complex product descriptions

**Λύσεις:**
1. Improve page scraping (καλύτερο HTML parsing)
2. Use longer context in extraction
3. Multiple passes/extraction attempts

### Potentially missed products;

**Τι σημαίνει:**
- Το LLM εντόπισε ότι υπάρχουν άλλα products στη σελίδα
- Τα product links μπορεί να μην ήταν ανιχνεύσιμα από BeautifulSoup

**Λύσεις:**
1. Manual inspection της σελίδας
2. Add URLs manually στο trusted_sources.json
3. Use Selenium για JavaScript-rendered content

---

## Integration με Agents

Ο verification layer μπορεί να ενσωματωθεί σε agents:

```python
from llm_axe.agents import Agent, AgentType
from llm_axe.va4_product_discoverer import discover_and_extract_products

llm = OllamaChat(model="llama3.2:latest")
agent = Agent(llm, agent_type=AgentType.VALIDATOR)

# Discover products
products = discover_and_extract_products("Eurobank", "https://...", llm)

# Validate before passing to user
for product in products:
    confidence = product.get("verification", {}).get("confidence_score", 0)
    if confidence >= 0.7:
        # High quality - can present to user
        agent.ask(f"Summarize this product: {product}")
    else:
        # Low quality - needs review
        print(f"⚠️ Low confidence product, needs review: {product['programme_name']}")
```

---

## Best Practices

### ✅ DO:
- Always check the verification report
- Review products with confidence < 0.7
- Manually verify at least 10% of results
- Document any manual corrections
- Re-run discovery if many hallucinations detected

### ❌ DON'T:
- Trust all products blindly without checking confidence
- Use products with hallucination flags directly
- Ignore the "potentially missed products" warnings
- Assume high confidence = no manual review needed

---

## Performance Metrics

Κρατάμε track των metrics:

```json
{
  "discovery_date": "2025-01-08",
  "total_products": 15,
  "verified_percentage": 80,
  "hallucination_rate": 13.3,
  "average_confidence": 0.82,
  "average_missing_fields": 0.5,
  "average_extraction_time": "45s per product"
}
```

---

## Επόμενα Βήματα

Μετά την verification:

1. **Αποθήκευση Verified Products**
   ```bash
   # Μόνο products με confidence >= 0.7
   cp output/va4_product_discoverer/*extracted.json verified_products/
   ```

2. **Manual Review of Low-Confidence Products**
   - Create spreadsheet with low-confidence products
   - Manually verify and correct
   - Update JSON files

3. **Database Import**
   - Import verified products σε database
   - Create product comparison reports
   - Generate recommendations based on products

4. **Continuous Monitoring**
   - Re-run discovery weekly
   - Track confidence trends
   - Alert on new hallucinations

---

## Σχετικά Αρχεία

- `llm_axe/va4_product_discoverer.py` - Main discovery + verification module
- `examples/ex_complete_discovery_pipeline.py` - End-to-end pipeline
- `examples/ex_verify_discovery_results.py` - Verification report analyzer
- `output/va4_product_discoverer/` - Discovery results with verification metadata
