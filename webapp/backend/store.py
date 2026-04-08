from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ProgramDetails, ProgramListItem


@dataclass
class ProgramRecord:
    id: str
    file_path: Path
    index: int | None
    raw: dict[str, Any]


class ProgramStore:
    def __init__(self, root: Path):
        self.root = root
        self.records: dict[str, ProgramRecord] = {}
        self.load()

    def load(self) -> None:
        self.records.clear()
        # We only read curated output files that are likely to contain structured programs.
        candidates = [
            *self.root.glob("output/ideal-*.json"),
            *self.root.glob("output/va2_online_to_template/out*.json"),
            *self.root.glob("outputs/va2_response_template_outputs/*.json"),
        ]

        for file_path in sorted(set(candidates)):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(payload, list):
                for idx, item in enumerate(payload):
                    if isinstance(item, dict) and self._is_program(item):
                        program_id = self._make_id(file_path, idx)
                        self.records[program_id] = ProgramRecord(
                            id=program_id,
                            file_path=file_path,
                            index=idx,
                            raw=item,
                        )
            elif isinstance(payload, dict) and self._is_program(payload):
                program_id = self._make_id(file_path, None)
                self.records[program_id] = ProgramRecord(
                    id=program_id,
                    file_path=file_path,
                    index=None,
                    raw=payload,
                )

    def list_programs(self) -> list[ProgramListItem]:
        items = [self._to_list_item(record) for record in self.records.values()]
        return sorted(items, key=lambda x: x.title.lower())

    def get_program_details(self, program_id: str) -> ProgramDetails | None:
        record = self.records.get(program_id)
        if record is None:
            return None
        return self._to_program_details(record)

    def get_raw(self, program_id: str) -> ProgramRecord | None:
        return self.records.get(program_id)

    def update_raw(self, program_id: str, data: dict[str, Any]) -> ProgramRecord | None:
        record = self.records.get(program_id)
        if record is None:
            return None

        try:
            payload = json.loads(record.file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if record.index is None:
            payload = data
        else:
            if not isinstance(payload, list) or record.index >= len(payload):
                return None
            payload[record.index] = data

        record.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record.raw = data
        return record

    @staticmethod
    def _is_program(item: dict[str, Any]) -> bool:
        keys = {"programme_name", "description", "programme_objective", "eligible_parties", "managing_body"}
        return any(k in item for k in keys)

    @staticmethod
    def _make_id(file_path: Path, index: int | None) -> str:
        raw = f"{file_path.as_posix()}::{index if index is not None else 'root'}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
        return digest

    @staticmethod
    def _scalar_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                text = ProgramStore._scalar_text(item)
                if text:
                    parts.append(text)
            return "; ".join(parts)
        if isinstance(value, dict):
            parts: list[str] = []
            for _, field_value in value.items():
                text = ProgramStore._scalar_text(field_value)
                if text:
                    parts.append(text)
            return "; ".join(parts)
        return ""

    def _build_eligibility(self, raw: dict[str, Any]) -> str:
        segments: list[str] = []
        eligible_parties = self._scalar_text(raw.get("eligible_parties"))
        criteria = self._scalar_text(raw.get("eligibility_criteria"))
        property_reqs = self._scalar_text(raw.get("property_requirements"))

        if eligible_parties:
            segments.append(f"Eligible parties: {eligible_parties}")
        if criteria:
            segments.append(f"Criteria: {criteria}")
        if property_reqs:
            segments.append(f"Property requirements: {property_reqs}")
        return " | ".join(segments)

    def _build_funding(self, raw: dict[str, Any]) -> str:
        fields = [
            ("Min", raw.get("minimum_funding_amount")),
            ("Max", raw.get("maximum_funding_amount")),
            ("Coverage", raw.get("funding_coverage")),
            ("Interest", raw.get("interest_rate")),
            ("Budget", raw.get("total_budget")),
        ]
        parts: list[str] = []
        for label, value in fields:
            text = self._scalar_text(value)
            if text:
                parts.append(f"{label}: {text}")
        return " | ".join(parts)

    def _build_link(self, raw: dict[str, Any]) -> str:
        source = raw.get("source")
        if isinstance(source, list) and source:
            link = self._scalar_text(source[0])
            if link:
                return link
        if isinstance(source, str) and source.strip():
            return source.strip()

        contact_info = raw.get("contact_info")
        if isinstance(contact_info, list) and contact_info:
            for item in contact_info:
                text = self._scalar_text(item)
                if "http" in text:
                    return text

        announcement_date = self._scalar_text(raw.get("announcement_date"))
        if announcement_date.startswith("http"):
            return announcement_date

        return ""

    def _to_list_item(self, record: ProgramRecord) -> ProgramListItem:
        raw = record.raw
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title")) or "Untitled"
        provider = self._scalar_text(
            raw.get("loan_provider", {}).get("primary_provider_name") if isinstance(raw.get("loan_provider"), dict) else ""
        )
        if not provider:
            provider = self._scalar_text(raw.get("managing_body")) or "Unknown provider"

        deadline = self._scalar_text(raw.get("application_end_date") or raw.get("completion_deadline")) or "N/A"
        return ProgramListItem(id=record.id, title=title, provider=provider, deadline=deadline)

    def _to_program_details(self, record: ProgramRecord) -> ProgramDetails:
        raw = record.raw
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title")) or "Untitled"

        provider = self._scalar_text(
            raw.get("loan_provider", {}).get("primary_provider_name") if isinstance(raw.get("loan_provider"), dict) else ""
        )
        if not provider:
            provider = self._scalar_text(raw.get("managing_body")) or "Unknown provider"

        description = self._scalar_text(raw.get("description") or raw.get("programme_objective")) or "No description available."
        eligibility = self._build_eligibility(raw) or "No eligibility information available."
        funding = self._build_funding(raw) or "No funding information available."
        deadline = self._scalar_text(raw.get("application_end_date") or raw.get("completion_deadline")) or "N/A"
        link = self._build_link(raw)

        return ProgramDetails(
            id=record.id,
            title=title,
            provider=provider,
            description=description,
            eligibility=eligibility,
            funding=funding,
            deadline=deadline,
            link=link,
        )
