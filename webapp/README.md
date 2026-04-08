# Web App Prototype

This folder contains a minimal web app prototype split into:

- `backend/`: FastAPI API serving funding programs from existing JSON files.
- `frontend/`: Single-page plain HTML/CSS/JS UI.

## Run Backend

From repository root:

```bash
pip install -r requirements.txt
uvicorn webapp.backend.main:app --reload --port 8000
```

Open:

- `http://127.0.0.1:8000/` (single-page UI)
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/programs`
- `http://127.0.0.1:8000/docs`

## Current Endpoints

- `GET /health`
- `GET /programs`
- `GET /programs/{program_id}`
- `POST /chat/programs/{program_id}`
- `GET /admin/programs/{program_id}/raw`
- `PUT /admin/programs/{program_id}/raw`
