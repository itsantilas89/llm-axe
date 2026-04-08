from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse, RawProgramResponse, RawProgramUpdate
from .store import ProgramStore

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "webapp" / "frontend"
store = ProgramStore(ROOT_DIR)

app = FastAPI(title="Funding Programs API", version="0.1.0")

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
def health() -> dict:
    return {"status": "ok", "program_count": len(store.records)}


@app.get("/programs")
def list_programs() -> list[dict]:
    return [item.model_dump() for item in store.list_programs()]


@app.get("/programs/{program_id}")
def get_program(program_id: str) -> dict:
    details = store.get_program_details(program_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Program not found")
    return details.model_dump()


@app.post("/chat/programs/{program_id}", response_model=ChatResponse)
def chat_about_program(program_id: str, body: ChatRequest) -> ChatResponse:
    details = store.get_program_details(program_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Program not found")

    q = body.message.lower().strip()
    if "deadline" in q or "προθεσμ" in q:
        reply = f"Η προθεσμία που έχουμε αποθηκευμένη είναι: {details.deadline}."
    elif "eligible" in q or "eligib" in q or "δικαι" in q:
        reply = f"Με βάση το πρόγραμμα, τα κριτήρια επιλεξιμότητας είναι: {details.eligibility}"
    elif "document" in q or "δικαιολογ" in q:
        reply = "Δεν έχουμε ξεχωριστό πεδίο δικαιολογητικών σε αυτό το JSON. Προτείνεται έλεγχος στον επίσημο σύνδεσμο του προγράμματος."
    else:
        reply = (
            f"Βάσει των αποθηκευμένων στοιχείων για το '{details.title}', "
            f"μπορώ να βοηθήσω σε επιλεξιμότητα, χρηματοδότηση και προθεσμίες."
        )

    return ChatResponse(
        reply=reply,
        suggested_questions=[
            "Am I eligible?",
            "What documents do I need?",
            "When is the deadline?",
        ],
    )


@app.get("/admin/programs/{program_id}/raw", response_model=RawProgramResponse)
def get_raw_program(program_id: str) -> RawProgramResponse:
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
    record = store.update_raw(program_id, body.data)
    if record is None:
        raise HTTPException(status_code=404, detail="Program not found or could not update file")

    # Refresh all list/detail mappings after write.
    store.load()
    updated = store.get_raw(program_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Program disappeared after update")

    return RawProgramResponse(
        id=updated.id,
        source_file=str(updated.file_path.relative_to(ROOT_DIR)),
        data=updated.raw,
    )
