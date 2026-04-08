from __future__ import annotations

from pydantic import BaseModel, Field


class ProgramListItem(BaseModel):
    id: str
    title: str
    provider: str
    deadline: str


class ProgramDetails(BaseModel):
    id: str
    title: str
    provider: str
    description: str
    eligibility: str
    funding: str
    deadline: str
    link: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    suggested_questions: list[str]


class RawProgramResponse(BaseModel):
    id: str
    source_file: str
    data: dict


class RawProgramUpdate(BaseModel):
    data: dict
