from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProgramListItem(BaseModel):
    id: str
    title: str
    provider: str
    deadline: str
    status: str
    source_url: str


class ProgramDetails(BaseModel):
    id: str
    title: str
    provider: str
    description: str
    eligibility: str
    funding: str
    deadline: str
    link: str
    status: str
    raw: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    qa_model: str | None = Field(default=None, max_length=120)


class ChatResponse(BaseModel):
    reply: str
    suggested_questions: list[str]
    used_llm: bool
    model: str | None = None


class RawProgramResponse(BaseModel):
    id: str
    source_file: str
    data: dict[str, Any]


class RawProgramUpdate(BaseModel):
    data: dict[str, Any]


class AdminCrawlRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)
    include_known_links: bool = False
    model_fast: str | None = Field(default=None, max_length=120)
    model_classification: str | None = Field(default=None, max_length=120)


class AdminCrawlResponse(BaseModel):
    job_id: str
    status: str
    message: str


class AdminJobStatus(BaseModel):
    id: str
    status: str
    started_at: str
    finished_at: str | None = None
    message: str
    logs: list[str]
    discovered: list[dict[str, Any]]
    saved_program_ids: list[str]
    merged_snapshot: dict[str, Any] | None = None
    errors: list[str]
