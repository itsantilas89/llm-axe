# Funding Programs Web App

This webapp is a UI layer over the existing `llm_axe` project pipeline. It must
not maintain separate scraping, extraction, classification, merging, or QA
prompt logic.

Minimal FastAPI + plain HTML prototype for two actors:

- **Admin**: reviews/edits generated JSON and can run VA4 processing for an explicit URL or the latest relevant URLs from VA4 classification outputs.
- **User**: selects an available program, reads the JSON-backed details, and asks questions about it.

## Storage

The runtime program list prefers the latest merged evaluation classifications:

```text
output/evaluation/qa_full_runs/*/classifications/*_classification.json
```

Those files are treated as program-level records. Older generated extraction outputs are then loaded only as extra candidates when they are not already covered by the merged `source_urls`:

```text
output/va3_scraper_to_template/*_extracted.json
```

JSONs with a relevant classification are admin candidates; users only see records
whose `review_status` is `approved`.

## Run

From the repository root:

```bash
pip install -r requirements.txt
uvicorn webapp.backend.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## LLM Models

The UI follows the same VA4 model split as the project:

- `llama3.2:latest` as the fast model for admin pre-screening and extraction.
- `llama3.1:8b-instruct-q4_K_M` as the classification model.
- `llama3.1:8b-instruct-q4_K_M` as the QA model.

Extraction/classification runs through `llm_axe.va4_product_discoverer.process_url`.
QA runs through `llm_axe.va4_product_discoverer.ask_program_question`.
Merged dataset refresh runs through `llm_axe.merged_snapshot.refresh_merged_snapshot`.
The admin editor can review, approve/reject, and correct generated JSON records,
but programs enter the system through the existing URL-processing pipeline.

```powershell
$env:OLLAMA_HOST="http://localhost:11434"
$env:OLLAMA_QA_MODEL="llama3.1:8b-instruct-q4_K_M"
uvicorn webapp.backend.main:app --reload --port 8000
```

For admin extraction, pass the model from the Admin UI field. Use a larger/slower model there, because that crawl is expected to run rarely.
Admin extraction/classification does not set a default Ollama request timeout. To cap calls intentionally, set `OLLAMA_EXTRACT_TIMEOUT_SECONDS` or `OLLAMA_CLASSIFICATION_TIMEOUT_SECONDS`.

## Endpoints

- `GET /health`
- `GET /programs` (public approved programs)
- `GET /programs/{program_id}` (public approved program)
- `POST /chat/programs/{program_id}`
- `GET /admin/programs` (all candidate JSON records)
- `POST /admin/crawl`
- `GET /admin/jobs/{job_id}`
- `GET /admin/programs/{program_id}/raw`
- `PUT /admin/programs/{program_id}/raw`
