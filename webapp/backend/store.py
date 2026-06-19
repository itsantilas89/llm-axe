from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_axe.va3_scraper_to_template import TEMPLATE_DEFAULT
from llm_axe.va4_product_discoverer import list_latest_relevant_classification_urls

from .schemas import ProgramDetails, ProgramListItem


PROJECT_PROGRAM_PATTERNS = [
    "output/va3_scraper_to_template/*_extracted.json",
]
PROJECT_CLASSIFICATION_PATTERN = "output/va4_product_discoverer/*_classification.json"
EVALUATION_MERGED_CLASSIFICATION_DIR_PATTERN = "output/evaluation/qa_full_runs/*/classifications"
PREFERRED_MERGED_RUN_PREFIXES = (
    "webapp_merged_",
    "cli_merged_",
    "cli_batch_merged_",
    "merged_",
)


PROGRAM_TEMPLATE: dict[str, Any] = {
    **deepcopy(TEMPLATE_DEFAULT[0]),
    "id": "",
    "source_url": "",
    "source_urls": [],
    "review_status": "needs_review",
    "admin_notes": "",
    "scraped_at": "",
    "updated_at": "",
}


@dataclass
class ProgramRecord:
    id: str
    file_path: Path
    index: int | None
    raw: dict[str, Any]
    wrapper_key: str | None = None
    review_status_explicit: bool = False


class ProgramStore:
    def __init__(self, root: Path):
        self.root = root
        self.generated_programs_dir = root / "output" / "va3_scraper_to_template"
        self.records: dict[str, ProgramRecord] = {}
        self.title_index: dict[str, str] = {}
        self.source_index: dict[str, str] = {}
        self.classification_index: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        self.records.clear()
        self.title_index.clear()
        self.source_index.clear()
        self.classification_index = self._load_classification_index()

        merged_dir = self._latest_merged_classification_dir()
        if merged_dir is not None:
            self._load_merged_classification_dir(merged_dir)

        for pattern in PROJECT_PROGRAM_PATTERNS:
            for file_path in sorted(self.root.glob(pattern)):
                self._load_program_file(
                    file_path,
                    dedupe_title=True,
                    replace_existing=merged_dir is None,
                )

    def list_programs(self, public_only: bool = False) -> list[ProgramListItem]:
        records = self.records.values()
        if public_only:
            records = [record for record in records if self._is_public_record(record)]
        items = [self._to_list_item(record) for record in records]
        return sorted(items, key=lambda x: x.title.lower())

    def get_program_details(self, program_id: str, public_only: bool = False) -> ProgramDetails | None:
        record = self.records.get(program_id)
        if record is None:
            return None
        if public_only and not self._is_public_record(record):
            return None
        return self._to_program_details(record)

    def is_public_program(self, program_id: str) -> bool:
        record = self.records.get(program_id)
        return record is not None and self._is_public_record(record)

    def get_raw(self, program_id: str) -> ProgramRecord | None:
        return self.records.get(program_id)

    def list_known_source_urls(self) -> list[str]:
        return list_latest_relevant_classification_urls(
            str(self.root / "output" / "va4_product_discoverer")
        )

    def update_raw(self, program_id: str, data: dict[str, Any]) -> ProgramRecord | None:
        record = self.records.get(program_id)
        if record is None:
            return None

        clean = self.normalize_program(data)
        clean["id"] = record.id
        clean["updated_at"] = self._now()

        try:
            payload = json.loads(record.file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if record.wrapper_key:
            if not isinstance(payload, dict):
                return None
            wrapper_payload = dict(clean)
            if record.wrapper_key == "extracted_data":
                for key in ("classification", "field_sources", "merged", "merged_from_count", "source_files"):
                    wrapper_payload.pop(key, None)
            payload[record.wrapper_key] = wrapper_payload
            source_url = self._scalar_text(clean.get("source_url"))
            if source_url:
                payload["url"] = source_url
            payload["id"] = clean["id"]
            source_urls = clean.get("source_urls")
            if isinstance(source_urls, list):
                payload["source_urls"] = source_urls
            classification = clean.get("classification")
            if isinstance(classification, dict):
                payload["classification"] = classification
        elif record.index is None:
            payload = clean
        else:
            if not isinstance(payload, list) or record.index >= len(payload):
                return None
            payload[record.index] = clean

        self._write_json(record.file_path, payload)
        record.raw = clean
        return record

    def normalize_program(self, data: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        clean = {**deepcopy(PROGRAM_TEMPLATE), **data}

        if not clean.get("programme_name"):
            clean["programme_name"] = clean.get("program_name") or clean.get("title") or "Untitled program"

        source_url = self._scalar_text(clean.get("source_url"))
        if not source_url:
            source = clean.get("source")
            if isinstance(source, str):
                source_url = source
            elif isinstance(source, list) and source:
                source_url = self._scalar_text(source[0])
            clean["source_url"] = source_url

        source_urls = clean.get("source_urls")
        if not isinstance(source_urls, list):
            source_urls = []
        if source_url and source_url not in source_urls:
            source_urls.insert(0, source_url)
        clean["source_urls"] = source_urls

        clean["review_status"] = self._scalar_text(clean.get("review_status")) or "needs_review"
        clean["updated_at"] = self._scalar_text(clean.get("updated_at")) or now
        return clean

    def _load_program_file(self, file_path: Path, dedupe_title: bool, replace_existing: bool = True) -> None:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return

        require_title = dedupe_title
        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                raw = self._prepare_project_program(file_path, item) if dedupe_title else item
                if not self._is_known_irrelevant(raw) and self._is_program(raw, require_title=require_title):
                    self._add_record(file_path, idx, raw, dedupe_title=dedupe_title, replace_existing=replace_existing)
        elif isinstance(payload, dict):
            raw = self._prepare_project_program(file_path, payload) if dedupe_title else payload
            if not self._is_known_irrelevant(raw) and self._is_program(raw, require_title=require_title):
                self._add_record(file_path, None, raw, dedupe_title=dedupe_title, replace_existing=replace_existing)

    def _latest_merged_classification_dir(self) -> Path | None:
        candidates: list[Path] = []
        for classification_dir in self.root.glob(EVALUATION_MERGED_CLASSIFICATION_DIR_PATTERN):
            if not classification_dir.is_dir():
                continue
            if any(classification_dir.glob("*_classification.json")):
                if not self._is_complete_merged_run_dir(classification_dir):
                    continue
                candidates.append(classification_dir)
        if not candidates:
            return None
        preferred = [
            path for path in candidates
            if path.parent.name.startswith(PREFERRED_MERGED_RUN_PREFIXES)
        ]
        return max(preferred or candidates, key=self._merged_classification_dir_timestamp)

    def _is_complete_merged_run_dir(self, classification_dir: Path) -> bool:
        manifest_path = classification_dir.parent / "target_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(manifest, dict):
            return False

        selection = manifest.get("selection")
        if not isinstance(selection, dict):
            return False
        if selection.get("merge_by_program") is not True:
            return False
        if int(selection.get("max_programs") or 0) != 0:
            return False

        target_count = int(manifest.get("target_count") or 0)
        file_count = len(list(classification_dir.glob("*_classification.json")))
        return target_count > 0 and target_count == file_count

    def _merged_classification_dir_timestamp(self, classification_dir: Path) -> datetime:
        manifest_path = classification_dir.parent / "target_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            parsed = self._parse_timestamp(self._scalar_text(manifest.get("created_at")))
            if parsed is not None:
                return parsed

        try:
            return datetime.fromtimestamp(classification_dir.stat().st_mtime, timezone.utc)
        except OSError:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _load_merged_classification_dir(self, classification_dir: Path) -> None:
        for file_path in sorted(classification_dir.glob("*_classification.json")):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            raw, review_status_explicit = self._prepare_merged_classification_program(file_path, payload)
            if not self._is_known_irrelevant(raw) and self._is_program(raw, require_title=True):
                self._add_record(
                    file_path,
                    None,
                    raw,
                    wrapper_key="extracted_data",
                    dedupe_title=False,
                    replace_existing=True,
                    review_status_explicit=review_status_explicit,
                )

    def _prepare_merged_classification_program(self, file_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        extracted = payload.get("extracted_data")
        clean = dict(extracted) if isinstance(extracted, dict) else {}
        review_status_explicit = bool(
            self._scalar_text(clean.get("review_status")) or self._scalar_text(payload.get("review_status"))
        )
        if not self._scalar_text(clean.get("review_status")) and self._scalar_text(payload.get("review_status")):
            clean["review_status"] = self._scalar_text(payload.get("review_status"))

        source_url = self._scalar_text(payload.get("url") or clean.get("source_url"))
        source_urls: list[str] = []
        for values in (payload.get("source_urls"), clean.get("source_urls")):
            if not isinstance(values, list):
                continue
            for value in values:
                url = self._scalar_text(value)
                if url and url not in source_urls:
                    source_urls.append(url)
        if source_url and source_url not in source_urls:
            source_urls.insert(0, source_url)
        if not source_url and source_urls:
            source_url = source_urls[0]
        if source_url:
            clean["source_url"] = source_url
        clean["source_urls"] = source_urls

        classification = payload.get("classification")
        if isinstance(classification, dict):
            clean["classification"] = dict(classification)
        for field in ("field_sources", "merged", "merged_from_count", "source_files"):
            if field in payload:
                clean[field] = payload[field]

        timestamp = self._scalar_text(payload.get("timestamp"))
        if timestamp and not self._scalar_text(clean.get("scraped_at")):
            clean["scraped_at"] = timestamp
        file_time = self._file_timestamp(file_path)
        if file_time and not self._scalar_text(clean.get("updated_at")):
            clean["updated_at"] = file_time.isoformat()
        return self.normalize_program(clean), review_status_explicit

    def _prepare_project_program(self, file_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
        clean = dict(raw)
        source_url = self._scalar_text(clean.get("source_url")) or self._infer_source_url(file_path)
        if source_url:
            clean["source_url"] = source_url
            source_urls = clean.get("source_urls")
            if not isinstance(source_urls, list):
                source_urls = []
            if source_url not in source_urls:
                source_urls.insert(0, source_url)
            clean["source_urls"] = source_urls

        if not self._title_key(clean):
            inferred_title = self._title_from_url(source_url)
            if inferred_title:
                clean["programme_name"] = inferred_title

        classification = self._find_classification(clean, source_url)
        if classification:
            clean["classification"] = classification

        file_time = self._file_timestamp(file_path)
        if file_time and not self._scalar_text(clean.get("scraped_at")):
            clean["scraped_at"] = file_time.isoformat()
        return clean

    def _load_classification_index(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for file_path in sorted(self.root.glob(PROJECT_CLASSIFICATION_PATTERN)):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("classification"), dict):
                continue

            extracted = payload.get("extracted_data")
            if not isinstance(extracted, dict):
                extracted = {}

            classification = dict(payload["classification"])
            timestamp = self._record_timestamp(file_path, payload)
            entry = {
                "classification": classification,
                "timestamp": timestamp,
                "file_path": file_path,
            }

            source_url = self._scalar_text(payload.get("url") or extracted.get("source_url"))
            for key in self._classification_keys(extracted, source_url):
                current = out.get(key)
                if current is None or timestamp > current["timestamp"]:
                    out[key] = entry
        return out

    def _find_classification(self, raw: dict[str, Any], source_url: str) -> dict[str, Any] | None:
        for key in self._classification_keys(raw, source_url):
            entry = self.classification_index.get(key)
            if entry is not None:
                return dict(entry["classification"])
        return None

    def _classification_keys(self, raw: dict[str, Any], source_url: str) -> list[str]:
        keys: list[str] = []
        url = self._scalar_text(source_url)
        if url:
            keys.append(f"url:{url.rstrip('/').casefold()}")
        title = self._title_key(raw)
        if title:
            keys.append(f"title:{title}")
        return keys

    @staticmethod
    def _is_known_irrelevant(raw: dict[str, Any]) -> bool:
        classification = raw.get("classification")
        return isinstance(classification, dict) and classification.get("is_relevant") is False

    def _is_public_record(self, record: ProgramRecord) -> bool:
        status = self._scalar_text(record.raw.get("review_status")).casefold()
        return status == "approved"

    def _add_record(
        self,
        file_path: Path,
        index: int | None,
        raw: dict[str, Any],
        wrapper_key: str | None = None,
        dedupe_title: bool = False,
        replace_existing: bool = True,
        review_status_explicit: bool = False,
    ) -> None:
        program_id = self._safe_id(self._scalar_text(raw.get("id")) or self._derive_program_id(raw, file_path, index))
        if not self._safe_id(self._scalar_text(raw.get("id"))):
            raw["id"] = program_id
        title_key = self._title_key(raw)
        source_keys = self._source_keys(raw)
        if dedupe_title:
            existing_ids = []
            if title_key:
                existing_ids.append(self.title_index.get(title_key))
            for source_key in source_keys:
                existing_ids.append(self.source_index.get(source_key))

            if not replace_existing and any(item for item in existing_ids):
                for existing_id in {item for item in existing_ids if item}:
                    self._overlay_approved_status(existing_id, raw)
                return
            for existing_id in {item for item in existing_ids if item}:
                existing = self.records.get(existing_id)
                if existing is not None and not self._is_newer_record(file_path, raw, existing):
                    return

            for existing_id in {item for item in existing_ids if item}:
                self._remove_record(existing_id)
        self.records[program_id] = ProgramRecord(
            id=program_id,
            file_path=file_path,
            index=index,
            raw=raw,
            wrapper_key=wrapper_key,
            review_status_explicit=review_status_explicit,
        )
        if title_key:
            self.title_index[title_key] = program_id
        for source_key in source_keys:
            self.source_index[source_key] = program_id

    def _remove_record(self, record_id: str) -> None:
        self.records.pop(record_id, None)
        for key, value in list(self.title_index.items()):
            if value == record_id:
                self.title_index.pop(key, None)
        for key, value in list(self.source_index.items()):
            if value == record_id:
                self.source_index.pop(key, None)

    def _overlay_approved_status(self, record_id: str, raw: dict[str, Any]) -> None:
        status = self._scalar_text(raw.get("review_status")).casefold()
        if status != "approved":
            return
        existing = self.records.get(record_id)
        if existing is None:
            return
        if existing.review_status_explicit:
            return
        current = self._scalar_text(existing.raw.get("review_status")).casefold()
        if current in ("", "needs_review"):
            existing.raw["review_status"] = "approved"

    def _is_program(self, item: dict[str, Any], require_title: bool = False) -> bool:
        title = self._scalar_text(item.get("programme_name") or item.get("program_name") or item.get("title"))
        if title:
            return True
        if require_title:
            return False

        keys = {
            "programme_name",
            "program_name",
            "title",
            "description",
            "programme_objective",
            "eligible_parties",
            "managing_body",
            "source_url",
        }
        return any(self._scalar_text(item.get(k)) for k in keys)

    def _title_key(self, raw: dict[str, Any]) -> str:
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title"))
        return re.sub(r"\s+", " ", title).strip().casefold()

    def _source_key(self, raw: dict[str, Any]) -> str:
        source_url = self._scalar_text(raw.get("source_url"))
        return self._source_key_for_url(source_url)

    def _source_keys(self, raw: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        primary = self._source_key(raw)
        if primary:
            keys.append(primary)
        source_urls = raw.get("source_urls")
        if isinstance(source_urls, list):
            for value in source_urls:
                key = self._source_key_for_url(self._scalar_text(value))
                if key and key not in keys:
                    keys.append(key)
        return keys

    @staticmethod
    def _source_key_for_url(source_url: str) -> str:
        return source_url.rstrip("/").casefold() if source_url else ""

    def _is_newer_record(self, file_path: Path, raw: dict[str, Any], existing: ProgramRecord) -> bool:
        return self._record_timestamp(file_path, raw) > self._record_timestamp(existing.file_path, existing.raw)

    def _record_timestamp(self, file_path: Path, raw: dict[str, Any]) -> datetime:
        for field in ("updated_at", "scraped_at", "timestamp"):
            parsed = self._parse_timestamp(self._scalar_text(raw.get(field)))
            if parsed is not None:
                return parsed

        parsed = self._file_timestamp(file_path)
        if parsed is not None:
            return parsed

        try:
            return datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
        except OSError:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _file_timestamp(self, file_path: Path) -> datetime | None:
        match = re.match(r"^(\d{8}T\d{6}Z)_", file_path.name)
        if not match:
            return None
        return self._parse_timestamp(match.group(1))

    def _parse_timestamp(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            if re.match(r"^\d{8}T\d{6}Z$", value):
                return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    def _infer_source_url(self, file_path: Path) -> str:
        match = re.match(r"^\d{8}T\d{6}Z_(.+?)_[0-9a-f]{8}_extracted\.json$", file_path.name)
        if not match:
            return ""

        parts = match.group(1).split("_")
        if not parts:
            return ""

        host = parts[0]
        path = "/".join(part for part in parts[1:] if part)
        return f"https://{host}/{path}" if path else f"https://{host}"

    def _title_from_url(self, source_url: str) -> str:
        if not source_url:
            return ""
        slug = source_url.rstrip("/").split("/")[-1]
        slug = re.sub(r"[-_]+", " ", slug).strip()
        if not slug:
            return ""
        return " ".join(word[:1].upper() + word[1:] for word in slug.split())

    def _derive_program_id(self, raw: dict[str, Any], file_path: Path | None = None, index: int | None = None) -> str:
        source = self._scalar_text(raw.get("source_url")) or self._scalar_text(raw.get("source"))
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title"))
        file_key = f"{file_path.as_posix()}::{index}" if file_path else ""
        base = source or title or file_key or self._now()
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        title_slug = self._safe_id(title)
        return f"{title_slug[:60]}-{digest}" if title_slug else f"program-{digest}"

    @staticmethod
    def _safe_id(value: str) -> str:
        value = (value or "").strip().lower()
        value = re.sub(r"[^a-z0-9_-]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")
        return value[:90]

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
            for field_value in value.values():
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
            segments.append(f"Δικαιούχοι: {eligible_parties}")
        if criteria:
            segments.append(f"Κριτήρια: {criteria}")
        if property_reqs:
            segments.append(f"Ακίνητο: {property_reqs}")
        return "\n".join(segments)

    def _build_funding(self, raw: dict[str, Any]) -> str:
        fields = [
            ("Ελάχιστο", raw.get("minimum_funding_amount")),
            ("Μέγιστο", raw.get("maximum_funding_amount")),
            ("Κάλυψη", raw.get("funding_coverage")),
            ("Επιτόκιο", raw.get("interest_rate")),
            ("Προϋπολογισμός", raw.get("total_budget")),
            ("Τύπος", raw.get("funding_type")),
        ]
        parts: list[str] = []
        for label, value in fields:
            text = self._scalar_text(value)
            if text:
                parts.append(f"{label}: {text}")
        return "\n".join(parts)

    def _build_link(self, raw: dict[str, Any]) -> str:
        link = self._scalar_text(raw.get("source_url"))
        if link:
            return link

        source_urls = raw.get("source_urls")
        if isinstance(source_urls, list) and source_urls:
            link = self._scalar_text(source_urls[0])
            if link:
                return link

        contact_info = raw.get("contact_info")
        if isinstance(contact_info, list):
            for item in contact_info:
                text = self._scalar_text(item)
                if text.startswith("http"):
                    return text
        return ""

    def _to_list_item(self, record: ProgramRecord) -> ProgramListItem:
        raw = record.raw
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title")) or "Untitled"
        provider = self._provider(raw)
        deadline = self._scalar_text(raw.get("application_end_date") or raw.get("completion_deadline")) or "Δεν έχει οριστεί"
        status = self._scalar_text(raw.get("review_status")) or "needs_review"
        return ProgramListItem(
            id=record.id,
            title=title,
            provider=provider,
            deadline=deadline,
            status=status,
            source_url=self._build_link(raw),
        )

    def _to_program_details(self, record: ProgramRecord) -> ProgramDetails:
        raw = record.raw
        title = self._scalar_text(raw.get("programme_name") or raw.get("program_name") or raw.get("title")) or "Untitled"
        description = self._scalar_text(raw.get("description") or raw.get("programme_objective")) or "Δεν υπάρχει περιγραφή."
        eligibility = self._build_eligibility(raw) or "Δεν υπάρχουν στοιχεία επιλεξιμότητας."
        funding = self._build_funding(raw) or "Δεν υπάρχουν στοιχεία χρηματοδότησης."
        deadline = self._scalar_text(raw.get("application_end_date") or raw.get("completion_deadline")) or "Δεν έχει οριστεί"

        return ProgramDetails(
            id=record.id,
            title=title,
            provider=self._provider(raw),
            description=description,
            eligibility=eligibility,
            funding=funding,
            deadline=deadline,
            link=self._build_link(raw),
            status=self._scalar_text(raw.get("review_status")) or "needs_review",
            raw=raw,
        )

    def _provider(self, raw: dict[str, Any]) -> str:
        loan_provider = raw.get("loan_provider")
        if isinstance(loan_provider, dict):
            provider = self._scalar_text(loan_provider.get("primary_provider_name"))
            if provider:
                return provider
        return self._scalar_text(raw.get("managing_body")) or "Άγνωστος φορέας"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
