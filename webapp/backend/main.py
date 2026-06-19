from __future__ import annotations

import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_encoding()

from .admin_pipeline import run_admin_crawl
from .llm import answer_program_question
from .schemas import (
    AdminCrawlRequest,
    AdminCrawlResponse,
    AdminJobStatus,
    ChatRequest,
    ChatResponse,
    RawProgramResponse,
    RawProgramUpdate,
)
from .store import ProgramStore

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "webapp" / "frontend"
store = ProgramStore(ROOT_DIR)
jobs: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Funding Programs API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/health")
def health() -> dict[str, Any]:
    store.load()
    return {
        "status": "ok",
        "program_count": len(store.list_programs(public_only=True)),
        "candidate_count": len(store.records),
    }


@app.get("/programs")
def list_programs() -> list[dict[str, Any]]:
    store.load()
    return [item.model_dump() for item in store.list_programs(public_only=True)]


@app.get("/admin/programs")
def list_admin_programs() -> list[dict[str, Any]]:
    store.load()
    return [item.model_dump() for item in store.list_programs(public_only=False)]


@app.get("/programs/{program_id}")
def get_program(program_id: str) -> dict[str, Any]:
    store.load()
    details = store.get_program_details(program_id, public_only=True)
    if details is None:
        raise HTTPException(status_code=404, detail="Program not found")
    return details.model_dump()


@app.post("/chat/programs/{program_id}", response_model=ChatResponse)
def chat_about_program(program_id: str, body: ChatRequest) -> ChatResponse:
    store.load()
    details = store.get_program_details(program_id, public_only=True)
    if details is None:
        raise HTTPException(status_code=404, detail="Program not found")

    reply, used_llm, model = answer_program_question(details.raw, body.message, body.qa_model)
    return ChatResponse(
        reply=reply,
        used_llm=used_llm,
        model=model,
        suggested_questions=[
            "Ποιοι είναι δικαιούχοι;",
            "Τι ποσό χρηματοδότησης καλύπτει;",
            "Ποια είναι η προθεσμία;",
            "Ποιες παρεμβάσεις καλύπτονται;",
        ],
    )


@app.get("/admin/programs/{program_id}/raw", response_model=RawProgramResponse)
def get_raw_program(program_id: str) -> RawProgramResponse:
    store.load()
    record = store.get_raw(program_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Program not found")
    return RawProgramResponse(
        id=record.id,
        source_file=str(record.file_path.relative_to(ROOT_DIR)),
        data=record.raw,
    )


@app.put("/admin/programs/{program_id}/raw", response_model=RawProgramResponse)
def update_raw_program(program_id: str, body: RawProgramUpdate) -> RawProgramResponse:
    store.load()
    record = store.update_raw(program_id, body.data)
    if record is None:
        raise HTTPException(status_code=404, detail="Program not found or could not update file")

    store.load()
    updated = store.get_raw(record.id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Program disappeared after update")

    return RawProgramResponse(
        id=updated.id,
        source_file=str(updated.file_path.relative_to(ROOT_DIR)),
        data=updated.raw,
    )


@app.post("/admin/crawl", response_model=AdminCrawlResponse)
def start_admin_crawl(body: AdminCrawlRequest, background_tasks: BackgroundTasks) -> AdminCrawlResponse:
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "started_at": _now(),
        "finished_at": None,
        "message": "Σε αναμονή",
        "logs": [],
        "discovered": [],
        "saved_program_ids": [],
        "merged_snapshot": None,
        "errors": [],
    }
    background_tasks.add_task(run_admin_crawl, store, body, jobs[job_id])
    return AdminCrawlResponse(job_id=job_id, status="queued", message="Το admin crawl ξεκίνησε")


@app.get("/admin/jobs/{job_id}", response_model=AdminJobStatus)
def get_admin_job(job_id: str) -> AdminJobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return AdminJobStatus(**job)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
