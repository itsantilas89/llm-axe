# Ενσωμάτωση Trusted Links & Product Discovery

## Σύνοψη

Ενσωματώθηκε αυτοματοποιημένη διαδικασία ανακάλυψης και εξαγωγής πληροφοριών για πράσινα δάνεια (energy efficiency loans) από ελληνικές τράπεζες.

### Τι Προστέθηκε:

1. **Ενημερωμένο `trusted_sources.json`** - Νέα κατηγορία με 5 ελληνικές τράπεζες
2. **VA4 Product Discoverer** (`va4_product_discoverer.py`) - Νέο module για αυτοματοποιημένη ανακάλυψη
3. **Παραδείγματα κώδικα** - Δύο νέα example scripts
4. **Τεκμηρίωση** - Αναλυτικό README για Product Discoverer

---

## Τι Περιέχεται;

### 1. Trusted Sources (Έμπιστες Πηγές)

**Αρχείο:** `llm_axe/trusted_sources.json`

```json
{
  "greek_banks_green_loans_stegastika": [
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina",
    "https://www.crediabank.com/idiotes/daneia/stegastika/",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia",
    "https://www.nbg.gr/el/idiwtes/daneia/stegastika-daneia",
    "https://www.alpha.gr/el/idiotes/daneia"
  ]
}
```

**Πλεονεκτήματα:**
- ✅ Δεν χρειάζεται να αναζητάς τα links στο internet
- ✅ Προηγμένες έμπιστες πηγές (επίσημες τραπεζικές ιστοσελίδες)
- ✅ Structured navigation εντός του domain κάθε τράπεζας

---

### 2. VA4 Product Discoverer

**Αρχείο:** `llm_axe/va4_product_discoverer.py`

Ένα νέο module που:
1. Φορτώνει τις trusted URLs από `trusted_sources.json`
2. Περιηγείται στις ιστοσελίδες των τραπεζών
3. Ανακαλύπτει product links
4. Αξιολογεί κάθε product με LLM (είναι σχετικό με energy efficiency;)
5. Εξάγει δομημένα δεδομένα (JSON) για τα σχετικά products

**Χαρακτηριστικά:**
- 🤖 LLM-based relevance evaluation (όχι απλή keyword matching)
- 🔍 Intelligent link filtering (παράλειψη pagination, search, login)
- ⏱️ Rate limiting (σεβασμός στους servers)
- 📊 Structured output (JSON template)
- ✅ Error handling (συνέχιση σε περίπτωση αποτυχίας)

---

### 3. Χρήση - Βασικό Παράδειγμα

**Απλή χρήση μέσω Python:**

```python
from llm_axe.models import OllamaChat
from llm_axe.va4_product_discoverer import discover_and_extract_products

# Αρχικοποίηση LLM
llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")

# Ανακάλυψη products από Eurobank
products = discover_and_extract_products(
    bank_name="Eurobank",
    bank_url="https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina",
    llm=llm
)

# Εμφάνιση αποτελεσμάτων
for product in products:
    print(f"📌 {product.get('programme_name')}")
    print(f"   Επιτόκιο: {product.get('interest_rate')}")
    print(f"   Περιγραφή: {product.get('description')[:100]}...")
```

**Εκτέλεση μέσω Command Line:**

```bash
python llm_axe/va4_product_discoverer.py
```

---

### 4. Παραδείγματα Κώδικα

Δύο νέα example scripts παρέχονται:

#### a. `examples/ex_product_discoverer.py`
Απλό παράδειγμα ανακάλυψης products από μία τράπεζα.

```bash
python examples/ex_product_discoverer.py
```

#### b. `examples/ex_complete_discovery_pipeline.py`
Πλήρης pipeline με όλες τις τράπεζες, user confirmation, summary reports.

```bash
python examples/ex_complete_discovery_pipeline.py
```

---

### 5. Δομή Εξαγόμενων Δεδομένων

Κάθε discovered product περιέχει:

```json
{
  "programme_name": "Πράσινο Δάνειο Eurobank",
  "description": "Δάνειο για ενεργειακή αναβάθμιση κατοικίας",
  "interest_rate": "2.5% - 3.5%",
  "eligible_interventions": [
    "Εγκατάσταση φωτοβολταϊκών",
    "Μόνωση", 
    "Αντικατάσταση παραθύρων"
  ],
  "funding_type": "Δάνειο",
  "bank_name": "Eurobank",
  "source_url": "https://www.eurobank.gr/...",
  "discovery_timestamp": "2025-01-08T12:34:56..."
}
```

---

### 6. Workflow - Πώς Λειτουργεί;

```
┌─────────────────────────────────────────────────────────┐
│  1. Φορτώνει trusted_sources.json                       │
│     ↓ 5 ελληνικές τράπεζες                             │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  2. Για κάθε τράπεζα:                                  │
│     a) Scrape τη σελίδα με τα πράσινα δάνεια         │
│     b) Εξαγωγή όλων των product links                │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  3. Για κάθε product link:                             │
│     a) Scrape το product page                         │
│     b) LLM: "Είναι σχετικό με energy efficiency;"     │
│     c) Αν ΝΑΙ → εξαγωγή δομημένων δεδομένων         │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  4. Αποθήκευση αποτελεσμάτων:                          │
│     output/va4_product_discoverer/                     │
│     ├── 20250108T123456Z_eurobank_prasina.json        │
│     ├── 20250108T123500Z_crediabank_housing.json      │
│     └── 20250108T123600Z_discovery_summary.json       │
└─────────────────────────────────────────────────────────┘
```

---

### 7. Εγκατάσταση & Εκτέλεση

**Προαπαιτούμενα:**
- Ollama τρέχει σε background (`ollama serve`)
- Python environment με llm-axe εγκατεστημένο

**Βήμα 1: Ενημέρωση trusted_sources.json**
✅ Ήδη κάνα (νέα κατηγορία προστέθηκε)

**Βήμα 2: Δοκιμή Discovery**
```bash
# Απλό παράδειγμα
python examples/ex_product_discoverer.py

# ή Πλήρες pipeline
python examples/ex_complete_discovery_pipeline.py
```

**Βήμα 3: Ανάκτηση Αποτελεσμάτων**
```bash
# Τα αποτελέσματα αποθηκεύονται στο:
ls output/va4_product_discoverer/
```

---

### 8. Δυνατότητες & Περιορισμοί

#### ✅ Τι Μπορείς Να Κάνεις:
- Αυτοματοποιημένη ανακάλυψη products από 5 ελληνικές τράπεζες
- LLM-powered relevance evaluation (όχι keyword matching)
- Δομημένη εξαγωγή δεδομένων (JSON)
- Rate limiting και error handling
- Πλήρη logging για debugging

#### ⚠️ Περιορισμοί:
- Ανταποκρίνεται στις πρώτες 10 σελίδες ανά τράπεζα (παραμετροποιήσιμο)
- Δεν κάνει recursive discovery (δεν ακολουθεί links μέσα σε products)
- Δεν κάνει caching (re-scrapes παρόλο που έχει κάνει κοινό ήδη)
- BeautifulSoup (δεν αντιμετωπίζει JavaScript-rendered content)

---

### 9. Επεκτάσεις & Μελλοντικές Βελτιώσεις

```python
# Προτεινόμενα future enhancements:

# 1. Προσθήκη νέων τραπεζών
"greek_banks_green_loans_stegastika": [
    "...",
    "https://newbank.gr/loans/green"  # Νέα τράπεζα
]

# 2. Recursive discovery (ακολούθηση links εντός products)
def discover_nested_products(url, depth=2):
    pass

# 3. Incremental processing (μόνο νέα products)
def discover_new_products_only(bank_url, last_check):
    pass

# 4. Caching
import pickle
cache = pickle.load(open("product_cache.pkl", "rb"))

# 5. Scheduling (run every day)
from schedule import every
every().day.at("02:00").do(discover_and_extract_products)
```

---

### 10. Integration με Agents

Ο Product Discoverer μπορεί να ενσωματωθεί σε agents:

```python
from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat
from llm_axe.va4_product_discoverer import discover_and_extract_products

llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
agent = OnlineAgent(llm)

# Ανακάλυψη products
products = discover_and_extract_products("Eurobank", "https://...", llm)

# Χρήση στο agent context
prompt = f"""
Βάσει αυτών των πράσινων δανείων: {products}
Ποια είναι τα κοινά χαρακτηριστικά; 
Ποιο είναι το καλύτερο για ένα σπίτι με 100τμ που θέλει μόνωση;
"""
response = agent.ask(prompt)
```

---

### 11. Βοήθεια & Debugging

**Προβλήματα & Λύσεις:**

| Πρόβλημα | Λύση |
|---------|------|
| "No trusted sources found" | Επαλήθευση `llm_axe/trusted_sources.json` |
| "Failed to scrape" | Έλεγχος internet connection, έλεγχος URL |
| "LLM query failed" | Έλεγχος ότι Ollama τρέχει (`ollama serve`) |
| Πολύ αργό | ↓ αριθμό links ανά σελίδα ή παράλληλα requests |
| Λάθος relevance | Tune LLM temperature ή system prompt |

**Logging:**
```python
# Όλες οι ενέργειες καταγράφονται στο stderr
# Πλήρη logs για debugging στο output directory
```

---

### 12. Παρακολούθηση Εξέλιξης

Κατά τη διάρκεια discovery θα δεις:

```
[INFO] Loading trusted sources...
[INFO] Found 1 source categories
[INFO] Processing 5 bank URLs
[DEBUG] Fetching https://www.eurobank.gr/...
[DEBUG] Scraped 5432 chars from https://...
[INFO] Found 42 links on the page
[1] Checking link: https://www.eurobank.gr/product1
    → Skipped (generic/pagination link)
[2] Checking link: https://www.eurobank.gr/green-loan-home
    → RELEVANT to energy efficiency!
    → Saved to: output/va4_product_discoverer/20250108T123456Z_eurobank_green_loan.json
```

---

## Σύνδεσμοι & Πληροφορίες

- 📖 **Πλήρη τεκμηρίωση:** `llm_axe/VA4_PRODUCT_DISCOVERER.md`
- 💻 **Source code:** `llm_axe/va4_product_discoverer.py`
- 📋 **Trusted sources:** `llm_axe/trusted_sources.json`
- 🎯 **Examples:** 
  - `examples/ex_product_discoverer.py`
  - `examples/ex_complete_discovery_pipeline.py`

---

## Τελικές Παρατηρήσεις

✅ **Ολοκληρώθηκε η ενσωμάτωση των trusted links**
- 5 ελληνικές τράπεζες με πράσινα δάνεια
- Αυτοματοποιημένη ανακάλυψη και αξιολόγηση products
- LLM-based relevance checking (όχι απλά keywords)
- Δομημένα αποτελέσματα σε JSON format

🚀 **Έτοιμο για χρήση:**
```bash
python examples/ex_complete_discovery_pipeline.py
```

💡 **Επόμενα βήματα:**
1. Δοκίμασε το discovery pipeline
2. Ανάκτησε τα αποτελέσματα από `output/va4_product_discoverer/`
3. Ενσωμάτωσε τα δεδομένα στη δική σου analysis
4. Προσθήκη νέων τραπεζών στο `trusted_sources.json` όπως χρειάζεται
