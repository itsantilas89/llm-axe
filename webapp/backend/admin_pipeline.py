from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from llm_axe.va4_product_discoverer import DEFAULT_MODEL_FAST, DEFAULT_MODEL_SMART, process_url
from llm_axe.merged_snapshot import refresh_merged_snapshot

from .llm import _env_float, make_ollama
from .schemas import AdminCrawlRequest
from .store import ProgramStore


ROOT_DIR = Path(__file__).resolve().parents[2]


def run_admin_crawl(store: ProgramStore, request: AdminCrawlRequest, job: dict[str, Any]) -> None:
    job["status"] = "running"
    job["message"] = "Εκτελείται VA4 processing"

    def log(message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        job["logs"].append(f"{ts} - {message}")

    try:
        explicit_urls = _clean_urls(request.urls)
        known_urls = store.list_known_source_urls() if request.include_known_links else []
        urls = _clean_urls([*explicit_urls, *known_urls])
        if not urls:
            job["message"] = "Δεν υπάρχουν URLs για VA4 processing"
            job["status"] = "completed"
            return

        explicit_url_keys = {url.rstrip("/").casefold() for url in explicit_urls}
        job["discovered"] = [
            {
                "url": url,
                "title": url,
                "score": 1,
                "reason": "explicit_url" if url.rstrip("/").casefold() in explicit_url_keys else "project_v4_relevant_url",
            }
            for url in urls
        ]

        model_fast = (request.model_fast or DEFAULT_MODEL_FAST).strip()
        model_classification = (request.model_classification or DEFAULT_MODEL_SMART).strip()
        log(f"LLM fast/extraction: {model_fast}")
        log(f"LLM classification: {model_classification}")
        llm_fast = make_ollama(model_fast, timeout_seconds=_env_float("OLLAMA_EXTRACT_TIMEOUT_SECONDS", None))
        llm_classification = make_ollama(model_classification, timeout_seconds=_env_float("OLLAMA_CLASSIFICATION_TIMEOUT_SECONDS", None))
        processed_successfully = False

        for index, url in enumerate(urls, start=1):
            job["message"] = f"VA4 processing {index}/{len(urls)}"
            log(f"VA4 process_url: {url}")
            try:
                extracted_data, classification, experiment_id = process_url(
                    url,
                    llm_fast,
                    llm_classification,
                    enable_qa=False,
                )
                processed_successfully = True
                if experiment_id:
                    log(f"Experiment: {experiment_id}")
                if not classification.get("is_relevant", False):
                    log(f"Παράλειψη μη σχετικού URL: {classification.get('reasoning', '')}")
                    continue

                store.load()
                saved_id = _find_loaded_program_id(store, url, extracted_data)
                if saved_id:
                    job["saved_program_ids"].append(saved_id)
                    log(f"Αποθηκεύτηκε generated JSON: {saved_id}")
                else:
                    log("Το VA4 ολοκληρώθηκε, αλλά δεν βρέθηκε loaded generated JSON.")
            except Exception as exc:
                error = f"Αποτυχία VA4 processing για {url}: {exc}"
                job["errors"].append(error)
                log(error)

        if processed_successfully:
            snapshot = refresh_merged_snapshot(ROOT_DIR, prefix="webapp_merged", log=log, timeout_seconds=600)
            job["merged_snapshot"] = snapshot
            if snapshot.get("status") == "ok":
                log(f"Merged snapshot έτοιμο: {snapshot.get('run_id')}")
            else:
                job["errors"].append(f"Merged snapshot refresh failed: {snapshot.get('message', '')}")

        store.load()
        if isinstance(job.get("merged_snapshot"), dict) and job["merged_snapshot"].get("status") == "ok":
            job["message"] = (
                f"Ολοκληρώθηκε: {len(job['saved_program_ids'])} JSON αρχεία, "
                f"merged snapshot {job['merged_snapshot'].get('run_id')}"
            )
        else:
            job["message"] = f"Ολοκληρώθηκε: {len(job['saved_program_ids'])} JSON αρχεία"
        job["status"] = "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["message"] = str(exc)
        job["errors"].append(str(exc))
    finally:
        job["finished_at"] = _now()


def _clean_urls(urls: list[str]) -> list[str]:
    clean_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        clean = url.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        clean_urls.append(clean)
    return clean_urls


def _find_loaded_program_id(store: ProgramStore, url: str, extracted_data: dict[str, Any]) -> str:
    target_url = (url or "").rstrip("/").casefold()
    target_title = ProgramStore._scalar_text(
        extracted_data.get("programme_name") or extracted_data.get("program_name") or extracted_data.get("title")
    ).casefold()

    for record in store.records.values():
        record_urls = [ProgramStore._scalar_text(record.raw.get("source_url"))]
        source_urls = record.raw.get("source_urls")
        if isinstance(source_urls, list):
            record_urls.extend(ProgramStore._scalar_text(item) for item in source_urls)
        for record_url in record_urls:
            if target_url and record_url.rstrip("/").casefold() == target_url:
                return record.id

        record_title = ProgramStore._scalar_text(
            record.raw.get("programme_name") or record.raw.get("program_name") or record.raw.get("title")
        ).casefold()
        if target_title and record_title == target_title:
            return record.id
    return ""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
