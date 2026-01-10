# GreenLoanAgent - User-Driven Product Recommendation

## Τι Είναι;

Το **GreenLoanAgent** είναι ένας ευφυής agent που:

✅ Δέχεται ερωτήσεις χρήστη σε φυσική γλώσσα  
✅ Αξιολογεί αν είναι σχετικές με green loans  
✅ Ανακαλύπτει προϊόντα από trusted bank sources  
✅ Εξάγει nested links από τα προϊόντα  
✅ Φιλτράρει αποτελέσματα βάσει της ερώτησης  
✅ Παρέχει προσωπικευμένες συστάσεις  
✅ Απαντάει σε follow-up ερωτήσεις  

---

## Αρχιτεκτονική

```
User Input
    ↓
Evaluate Relevance (LLM)
    ↓
├─ NOT relevant? → Return "Not related"
└─ Relevant? ↓
    Discover Products (VA4)
    ├─ Trusted bank URLs
    ├─ Extract product data
    └─ Extract nested links from products
    ↓
Filter by User Query (LLM)
    ├─ Rank products
    ├─ Score relevance
    └─ Top 10 matches
    ↓
Generate Recommendations (LLM)
    ├─ Personalized advice
    ├─ Product comparison
    └─ Next steps
    ↓
Return Response to User
```

---

## Πώς Λειτουργεί;

### 1️⃣ User Input
```
"Θέλω να εγκαταστήσω φωτοβολταϊκά στο σπίτι μου"
```

### 2️⃣ Relevance Check
```
LLM: Είναι αυτό σχετικό με πράσινα δάνεια;
Result: YES (confidence: 95%)
Topics: [solar, renewable energy]
Interventions: [photovoltaic installation]
```

### 3️⃣ Product Discovery
```
Eurobank Green Loans page
  ↓ Extract products:
  ├─ Πράσινο Δάνειο (General)
  ├─ Ηλιακή Ενέργεια (Solar-specific)
  └─ Ενεργειακή Αναβάθμιση (Energy Efficiency)
  
Crediabank Green Loans page
  ↓ Extract products + nested links
  
... (5 banks total)
```

### 4️⃣ Nested Link Extraction
```
For each product page:
  - Scan all internal links
  - Filter by domain (same bank only)
  - Skip generic links (pagination, search, etc.)
  - Extract product-related links
  → Find sub-products, variations, etc.
```

### 5️⃣ Filtering
```
LLM ranks products by relevance to user query:

Product: Ηλιακή Ενέργεια (Eurobank)
  Relevance: 95% ✅
  Match: "Solar installation" explicitly covered
  
Product: Δάνειο Αγοράς Αυτοκινήτου (Alpha)
  Relevance: 0% ❌
  Excluded: Not related to energy
```

### 6️⃣ Recommendation
```
LLM generates personalized response:

"Βάσει της ερώτησής σας, σας συνιστώ:

1. Ηλιακή Ενέργεια Eurobank (Relevance: 95%)
   Επιτόκιο: 2.5% | Διάρκεια: 15 χρόνια
   Καλύπτει: Εγκατάσταση φωτοβολταϊκών
   
2. Πράσινο Δάνειο Piraeus (Relevance: 85%)
   Επιτόκιο: 3.0% | Διάρκεια: 20 χρόνια
   Καλύπτει: Όλες οι ανανεώσιμες πηγές
   
Προτείνω τη λήψη επαφής με τις τράπεζες..."
```

---

## Χρήση

### Παράδειγμα 1: Standalone Script
```python
from llm_axe.models import OllamaChat
from llm_axe.green_loan_agent import GreenLoanAgent

llm = OllamaChat(model="llama3.2:latest")
agent = GreenLoanAgent(llm)

query = "Θέλω φωτοβολταϊκά για το σπίτι μου"
recommendation = agent.recommend(query)
print(recommendation)
```

### Παράδειγμα 2: Interactive Chat
```bash
python examples/ex_green_loan_chat.py
```

Chat Commands:
```
👤 Εσείς: Θέλω φωτοβολταϊκά
🤖 Agent: [Discoveries and recommendations]

👤 Εσείς: Ποια είναι τα επιτόκια;
🤖 Agent: [Follow-up answer]

👤 Εσείς: products
[Displays all discovered products]

👤 Εσείς: exit
[Closes chat]
```

### Παράδειγμα 3: Batch Processing
```python
queries = [
    "Φωτοβολταϊκά",
    "Μόνωση και παράθυρα",
    "Αντλία θερμότητας"
]

for query in queries:
    agent = GreenLoanAgent(llm)
    recommendation = agent.recommend(query)
    save_to_file(query, recommendation)
```

---

## Key Features

### 🔍 Smart Filtering
- LLM-based relevance ranking
- Multi-factor scoring (interventions, terms, bank reputation)
- Top 10 most relevant products returned

### 🔗 Nested Link Discovery
- Extracts product links from within bank websites
- Finds sub-products and variations
- Respects domain boundaries (same bank only)
- Avoids crawler traps (pagination, search, login)

### 💬 Conversational Interface
- Supports follow-up questions
- Maintains product context
- Adapts responses based on user needs

### 🎯 Verification-Aware
- Uses verification confidence scores
- Highlights high-confidence vs. low-confidence products
- Includes warnings for uncertain data

### 🌍 Multilingual
- Responds in Greek
- Understands Greek user queries
- Handles mixed Greek/English content

---

## API Reference

### Main Methods

#### `recommend(user_query: str) -> str`
Generate recommendations for user query.

**Args:**
- `user_query`: User's question in natural language

**Returns:**
- Formatted recommendation string in Greek

**Example:**
```python
recommendation = agent.recommend("Θέλω δάνειο για φωτοβολταϊκά")
print(recommendation)
```

#### `ask_followup(followup_query: str) -> str`
Answer follow-up questions about recommended products.

**Args:**
- `followup_query`: Follow-up question

**Returns:**
- Response based on discovered products

**Example:**
```python
followup = "Ποια είναι τα προσόντα;"
response = agent.ask_followup(followup)
```

#### `evaluate_user_prompt(user_query: str) -> Dict`
Analyze if query is relevant to green loans.

**Returns:**
- `is_relevant`: boolean
- `confidence`: 0-1 score
- `topics`: list of identified topics
- `interventions`: list of energy interventions

#### `discover_relevant_products(user_query: str) -> List[Dict]`
Discover and filter products relevant to query.

**Returns:**
- List of relevant products with relevance scores

#### `_extract_nested_links_from_product(product_url: str, product_text: str) -> List[Dict]`
Extract additional product links from within a product page.

**Returns:**
- List of nested product links

---

## Configuration

### Initialize with Custom Settings
```python
agent = GreenLoanAgent(
    llm=llm,
    temperature=0.5,  # Lower = more focused, Higher = more creative
    stream=True       # Stream responses token-by-token
)
```

### Adjust Filtering Thresholds
Modify in `_filter_products_by_relevance()`:
```python
# Filter: only keep products with relevance > 0.5
filtered = [p for p, score in sorted_products if score > 0.5]
```

---

## Examples

### Example 1: Solar Installation
```
Query: "Θέλω να εγκαταστήσω φωτοβολταϊκά"

Discovery:
  ✅ Eurobank Solar Loan (95% relevance)
  ✅ Crediabank Green Upgrade (85% relevance)
  ✅ NBG Renewable Energy (80% relevance)
  ❌ Alpha General Loan (15% relevance - filtered out)

Recommendation:
  "Με βάση την ερώτησή σας, τα φωτοβολταϊκά καλύπτονται..."
```

### Example 2: Building Insulation
```
Query: "Θέλουμε να μονώσουμε το σπίτι και να αντικαταστήσουμε παράθυρα"

Discovery:
  ✅ Piraeus Energy Efficiency (92% relevance)
  ✅ Eurobank Green Building (88% relevance)
  ✅ NBG Insulation Program (85% relevance)

Recommendation:
  "Η μόνωση και τα παράθυρα είναι κύρια αντικείμενα..."
```

### Example 3: Non-Related Query
```
Query: "Θέλω δάνειο για αυτοκίνητο"

Result:
  ❌ Not relevant to green loans
  
Response:
  "Η ερώτησή σας δεν σχετίζεται με πράσινα δάνεια
   ή ενεργειακές αναβαθμίσεις. Σας προτείνω
   να επικοινωνήσετε με τη τράπεζα για γενικά δάνεια."
```

---

## Workflow - Complete Example

```bash
# Step 1: Start interactive chat
python examples/ex_green_loan_chat.py

# Output:
# 🌱 ΕΛΛΗΝΙΚΟΣ ΠΡΑΚΤΟΡΕΥΤΗΣ ΠΡΑΣΙΝΩΝ ΔΑΝΕΙΩΝ
# 
# 👤 Εσείς: Θέλω φωτοβολταϊκά
# 🤖 Agent: Ανακαλύπτω σχετικά προϊόντα...
# 
# [Discovery process runs]
# 
# 🤖 Agent: Βάσει της ερώτησής σας, σας συνιστώ...
# [Recommendations]
# 
# 👤 Εσείς: products
# [Lists 10 most relevant products with details]
# 
# 👤 Εσείς: Ποιο έχει το καλύτερο επιτόκιο;
# 🤖 Agent: Το καλύτερο επιτόκιο έχει...
# 
# 👤 Εσείς: exit
# 👋 Αντίο!
```

---

## Performance

Typical timing:
- Relevance evaluation: ~2-3 seconds
- Product discovery (5 banks): ~60-90 seconds
- Nested link extraction: ~20-30 seconds
- Filtering & ranking: ~5-10 seconds
- Recommendation generation: ~3-5 seconds

**Total:** ~100-135 seconds per user query

---

## Limitations & Future Work

### Current Limitations
- Requires Ollama with llama3.2:latest
- Single-threaded discovery (sequential bank processing)
- Limited to first 15 products per query
- BeautifulSoup can't handle JavaScript-rendered content

### Future Enhancements
- Parallel product discovery for multiple banks
- Caching of discovered products
- Scheduled background discovery
- User preference learning
- Product comparison matrices
- Eligibility checklist generator
- Application form automation

---

## Integration Points

### With VA3 (Manual Scraper)
```python
# If user wants deeper info about a product:
product_url = filtered_products[0]['source_url']
# Run VA3: python llm_axe/va3_scraper_to_template.py <url>
```

### With VA4 (Automated Discovery)
```python
# GreenLoanAgent uses VA4 internally:
from llm_axe.va4_product_discoverer import discover_and_extract_products
products = discover_and_extract_products(bank_name, bank_url, llm)
```

### With Agents Framework
```python
from llm_axe.agents import OnlineAgent
from llm_axe.green_loan_agent import GreenLoanAgent

# Can extend or wrap with other agents
agent = GreenLoanAgent(llm)
# ... use with other agent pipelines
```

---

## See Also

- [VA4 Product Discoverer](./VA4_PRODUCT_DISCOVERER.md)
- [Verification Guide](./VERIFICATION_QUALITY_GUIDE_GR.md)
- [Integration Guide](./INTEGRATION_GUIDE_GR.md)
