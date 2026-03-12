# VA4 Product Discoverer

## Περιγραφή

Το **VA4 Product Discoverer** είναι ένα εξελιγμένο σύστημα για αυτόματη ανακάλυψη, κατηγοριοποίηση και ανάλυση χρηματοδοτικών προγραμμάτων και προϊόντων. Συνδυάζει το VA3 scraper με ευφυή κατηγοριοποίηση και διαδραστικό σύστημα ερωτήσεων-απαντήσεων.

## Ροή Εργασίας

```
URL Input
    ↓
┌─────────────────────┐
│  VA3: Scraping &    │
│  Data Extraction    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  LLM Classification │
│  (Category Analysis)│
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Is Relevant?        │
│ ✓ YES → Q&A Mode    │
│ ✗ NO  → Skip        │
└─────────────────────┘
```

## Κατηγορίες Ενδιαφέροντος

Το σύστημα αναγνωρίζει τις εξής κατηγορίες που **μας ενδιαφέρουν**:

| Κατηγορία | Περιγραφή | Παραδείγματα |
|-----------|-----------|--------------|
| `housing_loan` | Στεγαστικά δάνεια | Δάνεια για αγορά ή κατασκευή κατοικίας |
| `energy_upgrade` | Ενεργειακή αναβάθμιση | Πρόγραμμα "Εξοικονομώ", ενεργειακές πιστοποιήσεις |
| `home_renovation` | Ανακαινίσεις σπιτιών | Δάνεια για επισκευές, ανακαίνιση κατοικίας |
| `home_renewables` | ΑΠΕ σε σπίτι | Φωτοβολταϊκά, αντλίες θερμότητας για κατοικίες |

### Κατηγορίες που ΔΕΝ μας ενδιαφέρουν:

- `electric_vehicles` - Ηλεκτρικά οχήματα
- `property_purchase` - Αγορές ακινήτων (χωρίς δανειοδότηση)
- `commercial_renewables` - ΑΠΕ για εμπορική/βιομηχανική χρήση
- `other` - Άλλα άσχετα θέματα

## Εγκατάσταση & Προαπαιτούμενα

```bash
# Εγκατάσταση dependencies
pip install -r requirements.txt

# Βεβαιωθείτε ότι το Ollama τρέχει
ollama serve

# Κατεβάστε το μοντέλο (αν δεν το έχετε)
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Χρήση

### 1. Βασική Χρήση (CLI)

```bash
# Ανάλυση ενός URL με interactive Q&A
python -m llm_axe.va4_product_discoverer "https://example.com/green-loan"

# Ανάλυση χωρίς Q&A (μόνο classification)
python -m llm_axe.va4_product_discoverer --no-qa "https://example.com/program"

# Χρήση διαφορετικού μοντέλου
python -m llm_axe.va4_product_discoverer --model llama3.1:8b-instruct-q4_K_M "https://..."

# Διαδραστικό μοδέ (χωρίς παραμέτρους)
python -m llm_axe.va4_product_discoverer
```

### 2. Χρήση από Python Script

```python
from llm_axe.va4_product_discoverer import process_url
from llm_axe.models import OllamaChat

# Initialize LLM
llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")

# Process URL
url = "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/eksoikonomo-2025"
extracted_data, classification = process_url(url, llm, enable_qa=True)

# Έλεγχος αποτελεσμάτων
if classification['is_relevant']:
    print(f"✓ Μας ενδιαφέρει!")
    print(f"Κατηγορία: {classification['primary_category']}")
    print(f"Βεβαιότητα: {classification['confidence']:.0%}")
else:
    print(f"✗ Δεν μας ενδιαφέρει")
```

### 3. Batch Processing

```python
from llm_axe.va4_product_discoverer import process_url
from llm_axe.models import OllamaChat

llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")

urls = [
    "https://example.com/url1",
    "https://example.com/url2",
    "https://example.com/url3",
]

results = []
for url in urls:
    try:
        extracted, classification = process_url(url, llm, enable_qa=False)
        results.append({
            'url': url,
            'relevant': classification['is_relevant'],
            'category': classification['primary_category'],
            'confidence': classification['confidence']
        })
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Φιλτράρισμα μόνο των relevant
relevant_urls = [r for r in results if r['relevant']]
print(f"Found {len(relevant_urls)} relevant programs out of {len(urls)}")
```

## Output Files

Το VA4 αποθηκεύει τα αποτελέσματα στον φάκελο `output/va4_product_discoverer/`:

```
output/va4_product_discoverer/
├── 20260213T120000Z_example_com_abc123_classification.json
├── 20260213T120130Z_eurobank_gr_def456_classification.json
└── ...
```

### Δομή Classification JSON:

```json
{
  "timestamp": "20260213T120000Z",
  "url": "https://example.com/program",
  "classification": {
    "is_relevant": true,
    "primary_category": "energy_upgrade",
    "secondary_categories": ["home_renovation"],
    "confidence": 0.95,
    "reasoning": "Το πρόγραμμα αφορά ενεργειακή αναβάθμιση κατοικιών...",
    "key_features": [
      "Επιδότηση έως 70%",
      "Ενεργειακή αναβάθμιση",
      "Φωτοβολταϊκά & μόνωση"
    ]
  },
  "extracted_data": {
    "programme_name": "Πρόγραμμα Εξοικονομώ 2025",
    "description": "...",
    "eligible_parties": [...],
    ...
  }
}
```

## Διαδραστικό Q&A Σύστημα

Όταν ένα πρόγραμμα κατηγοριοποιηθεί ως **relevant**, το σύστημα ενεργοποιεί αυτόματα τη λειτουργία Q&A:

```
================================================
ΔΙΑΔΡΑΣΤΙΚΟ ΣΥΣΤΗΜΑ ΕΡΩΤΗΣΕΩΝ
================================================

URL: https://example.com/program
Πρόγραμμα: Εξοικονομώ 2025
Κατηγορία: Ενεργειακή αναβάθμιση (Energy upgrade/efficiency)
Βεβαιότητα: 95%

Κύρια Χαρακτηριστικά:
  • Επιδότηση έως 70%
  • Κατοικίες έως 200τμ
  • Χρηματοδότηση ΑΠΕ

Μπορείς να κάνεις ερωτήσεις για το πρόγραμμα.
Γράψε 'exit' ή 'quit' για έξοδο.

❓ Ερώτηση> Ποιοι είναι οι δικαιούχοι;

💡 Απάντηση:
Σύμφωνα με το πρόγραμμα, δικαιούχοι είναι:
1. Φυσικά πρόσωπα με ετήσιο εισόδημα έως 25.000€
2. Ιδιοκτήτες κατοικιών μέχρι 200τμ
3. Δικαιούχοι που δεν έχουν λάβει επιδότηση τα τελευταία 5 έτη

❓ Ερώτηση> Τι επεμβάσεις καλύπτει;

💡 Απάντηση:
Το πρόγραμμα καλύπτει τις εξής επεμβάσεις:
...
```

### Παραδείγματα Ερωτήσεων:

- "Ποιο είναι το ανώτατο ποσό επιδότησης;"
- "Πότε λήγει η προθεσμία υποβολής;"
- "Ποιες επεμβάσεις είναι επιλέξιμες;"
- "Υπάρχει συγχρηματοδότηση;"
- "Πώς υποβάλλω αίτηση;"

## Προηγμένες Λειτουργίες

### Custom Categories

Μπορείτε να προσαρμόσετε τις κατηγορίες επεξεργάζοντας το αρχείο:

```python
# στο va4_product_discoverer.py
CATEGORIES_OF_INTEREST = {
    "housing_loan": "Στεγαστικό δάνειο",
    "energy_upgrade": "Ενεργειακή αναβάθμιση",
    # Προσθέστε δικές σας κατηγορίες...
    "solar_panels": "Φωτοβολταϊκά συστήματα",
}
```

### Integration με VA3

Το VA4 χρησιμοποιεί το VA3 για scraping:

```python
from llm_axe.va3_scraper_to_template import scrape_page, extract_json
from llm_axe.va4_product_discoverer import classify_product

# 1. Scrape με VA3
text = scrape_page(url)
extracted_data = extract_json(llm, text, template, url)

# 2. Classify με VA4
classification = classify_product(llm, extracted_data)
```

## Troubleshooting

### Το LLM δεν κατηγοριοποιεί σωστά

- Δοκιμάστε διαφορετικό μοντέλο: `--model llama3.1:8b-instruct-q4_K_M`
- Ελέγξτε τα εξαγόμενα δεδομένα από το VA3
- Αυξήστε το temperature (πειραματικά)

### Το scraping αποτυγχάνει

- Χρησιμοποιήστε το VA3 απευθείας για debugging
- Ελέγξτε αν το URL είναι προσβάσιμο
- Δοκιμάστε με `--no-qa` flag

### Classification JSON δεν αποθηκεύεται

- Ελέγξτε δικαιώματα στον φάκελο `output/`
- Βεβαιωθείτε ότι ο φάκελος υπάρχει

## Παραδείγματα Χρήσης

Δείτε το αρχείο [examples/ex_product_discoverer.py](../examples/ex_product_discoverer.py) για πλήρη παραδείγματα.

```bash
# Τρέξτε τα παραδείγματα
python examples/ex_product_discoverer.py
```

## API Reference

### `process_url(url, llm, enable_qa=True)`

Κύρια συνάρτηση επεξεργασίας URL.

**Παράμετροι:**
- `url` (str): URL ή file path προς ανάλυση
- `llm` (OllamaChat): Initialized LLM instance
- `enable_qa` (bool): Ενεργοποίηση Q&A mode (default: True)

**Returns:**
- Tuple[dict, dict]: (extracted_data, classification)

### `classify_product(llm, extracted_data)`

Κατηγοριοποίηση προγράμματος με βάση τα εξαγόμενα δεδομένα.

**Παράμετροι:**
- `llm` (OllamaChat): LLM instance
- `extracted_data` (dict): Δεδομένα από VA3

**Returns:**
- dict: Classification result με τα πεδία is_relevant, primary_category, confidence, κλπ.

### `interactive_qa(llm, extracted_data, classification, url)`

Διαδραστικό σύστημα ερωτήσεων-απαντήσεων.

**Παράμετροι:**
- `llm` (OllamaChat): LLM instance
- `extracted_data` (dict): Εξαγόμενα δεδομένα
- `classification` (dict): Αποτέλεσμα κατηγοριοποίησης
- `url` (str): Source URL

## License

Βλέπε [LICENSE](../LICENSE) file.

---

**Δημιουργήθηκε**: Φεβρουάριος 2026  
**Συγγραφέας**: llm-axe project  
**Έκδοση**: 1.0.0
