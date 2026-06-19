from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_axe.models import OllamaChat
from llm_axe.qa_prompting import (
    build_batch_qa_prompt,
    build_single_qa_prompt,
    deterministic_qa_out_of_scope_answer as canonical_out_of_scope_answer,
)
from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash
from llm_axe.va4_product_discoverer import _fill_missing_core_fields_from_source
from evaluation.batch_html_validator import find_matching_scraped


DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_timestamp(path: Path, payload: dict[str, Any]) -> datetime:
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        try:
            if re.match(r"^\d{8}T\d{6}Z$", timestamp):
                return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass

    match = re.match(r"^(\d{8}T\d{6}Z)_", path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def safe_record_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:90]


def stable_program_id(programme_name: str, url: str) -> str:
    base = str(url or programme_name or now_stamp())
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    title_slug = safe_record_id(programme_name)
    return f"{title_slug[:60]}-{digest}" if title_slug else f"program-{digest}"


def safe_console(text: Any) -> None:
    value = str(text)
    try:
        print(value, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            pruned = prune_empty(item)
            if pruned in ("", None, [], {}):
                continue
            cleaned[key] = pruned
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            pruned = prune_empty(item)
            if pruned in ("", None, [], {}):
                continue
            cleaned_list.append(pruned)
        return cleaned_list
    if isinstance(value, str):
        return value.strip()
    return value


def has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_value(item) for item in value)
    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())
    return value not in (None, "")


def normalized_program_key(programme_name: str, url: str) -> str:
    text = normalize_text(f"{programme_name} {url}")
    text = text.replace("ς", "σ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    if (
        ("εξοικονομ" in text or "exoikonom" in text or "eksoikonom" in text)
        and ("ανακαινιζ" in text or "anakain" in text or "anakoiniz" in text)
        and ("νεου" in text or "neous" in text)
    ):
        return "εξοικονομω ανακαινιζω για νεους"
    if "exoikonomo 2025" in text or "eksoikonomo 2025" in text or "εξοικονομω 2025" in text:
        return "εξοικονομω 2025"
    if "anavathmizo to spiti" in text or "αναβαθμιζω το σπιτι" in text:
        return "αναβαθμιζω το σπιτι μου"
    if "fotovoltaika ste stege" in text or "photoboltaika ste stege" in text or "φωτοβολταικα στη στεγη" in text:
        bank_markers = ("bank", "alpha.gr", "eurobank.gr", "nbg.gr", "piraeusbank.gr", "crediabank", "epirusbank")
        if any(marker in text for marker in bank_markers) and "gov.gr" not in text:
            domain = re.sub(r"^https?://", "", url or "").split("/", 1)[0]
            return f"φωτοβολταικα στη στεγη τραπεζικη χρηματοδοτηση {normalize_text(domain)}"
        return "φωτοβολταικα στη στεγη"
    return normalize_text(programme_name) or normalize_text(url)


def source_priority(url: str) -> int:
    normalized = normalize_text(url)
    official_markers = (
        "gov.gr",
        "ypen.gov.gr",
        "greece20.gov.gr",
        "exoikonomo2025.gov.gr",
        "stegasi.gov.gr",
        "hdb.gr",
        "energy-invest.gov.gr",
    )
    if any(marker in normalized for marker in official_markers):
        return 3
    bank_markers = ("bank", "alpha.gr", "eurobank.gr", "nbg.gr", "piraeusbank.gr", "crediabank.com")
    if any(marker in normalized for marker in bank_markers):
        return 2
    return 1


def date_source_priority(url: str) -> int:
    normalized = normalize_text(url)
    priority = source_priority(url)
    if "ypen.gov.gr" in normalized:
        priority += 4
    if "exoikonomo2025.gov.gr" in normalized or "exoikonomoneon.gov.gr" in normalized:
        priority += 3
    return priority


def value_score(field: str, value: Any, url: str) -> tuple[int, int, int]:
    text_value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    normalized_value = normalize_text(text_value).replace("ς", "σ")
    if field in {"description", "programme_objective"}:
        quality = source_priority(url)
        if any(
            marker in normalized_value
            for marker in (
                "piraeus app",
                "e-banking",
                "e banking",
                "ψηφιακο βοηθο",
                "created with sketchtool",
                "el english",
                "μεταβαση στο κυριο περιεχομενο",
                "χαμηλη οραση",
                "ανοιξτε το μενου",
            )
        ):
            quality -= 12
        if any(
            marker in normalized_value
            for marker in (
                "μπορειτε να λαβετε επιδοτηση",
                "στοχος του προγραμματος",
                "με το green",
                "αποκτηστε πρασινο",
                "αναβαθμιστε ενεργειακα",
                "παρεμβασεις που εξοικονομουν ενεργεια",
            )
        ):
            quality += 8
        if 50 <= len(text_value or "") <= 900:
            quality += 4
        if len(text_value or "") > 1200:
            quality -= 5
        return (quality, -abs(len(text_value or "") - 500), source_priority(url))
    if field == "programme_name":
        quality = source_priority(url)
        if 4 <= len(text_value or "") <= 80:
            quality += 6
        if len(text_value or "") > 120 or "μεταβαση στο περιεχομενο" in normalized_value:
            quality -= 10
        return (quality, -len(text_value or ""), source_priority(url))

    base = source_priority(url)
    if field == "funding_coverage":
        if "100%" in normalized_value:
            base += 8
        if re.search(r"\b(?:80|90)\s*%", normalized_value):
            base += 6
        if any(marker in normalized_value for marker in ("επιδοτ", "τοκ", "επιτοκ", "ατοκο")):
            base += 4
        if any(marker in normalized_value for marker in ("πληρης επιδοτηση", "ατοκο δανειο")):
            base += 5
        if "50%" in normalized_value and "ταμει" in normalized_value:
            base -= 4
        return (base, -len(text_value or ""), source_priority(url))
    elif field == "interest_rate":
        if re.search(r"\d+(?:[.,]\d+)?\s*%", text_value or ""):
            base += 6
        if normalized_value.strip() in {"0%", "0 %"}:
            base += 8
        if any(marker in normalized_value for marker in ("0%", "ατοκο", "ατοκοσ", "100% επιδοτ", "πληρης επιδοτηση")):
            base += 5
        if "euribor" in normalized_value:
            base += 3
        if "50%" in normalized_value and "100%" not in normalized_value and "ατοκ" not in normalized_value:
            base -= 3
        if "0,12%" in normalized_value and "εισφορ" in normalized_value:
            base -= 5
        if not re.search(r"\d|euribor|ατοκ|ατοκο|επιδοτ", normalized_value):
            base -= 4
        return (base, -len(text_value or ""), source_priority(url))
    elif field in {"loan_duration", "duration"}:
        if re.search(r"\d", text_value or ""):
            base += 3
        if re.search(r"\d+\s*(?:-|έως|εως|εωσ|μέχρι|μεχρι)\s*\d+", normalized_value):
            base += 2
        if ("anavathmizo" in normalize_text(url) or "anabathmizo" in normalize_text(url)) and (
            "3 εωσ 7" in normalized_value
            or "3 εως 7" in normalized_value
            or "3-7" in normalized_value
            or "3 - 7" in normalized_value
            or "3 ετη" in normalized_value and "7" in normalized_value
        ):
            base += 6
        if ("anavathmizo" in normalize_text(url) or "anabathmizo" in normalize_text(url)) and "30" in normalized_value:
            base -= 6
        return (base, -len(text_value or ""), source_priority(url))
    elif field in {"completion_delay_consequences", "post_completion_obligations"}:
        base += source_priority(url)
        if any(marker in normalized_value for marker in ("υποχρε", "πρεπει", "απαραιτη", "χρειαζεται")):
            base += 3
        if any(marker in normalized_value for marker in ("ανακαλ", "απενταξ", "επιστροφ", "υπερημερι", "επιπτωσ", "κυρωσ")):
            base += 4
        if any(marker in normalized_value for marker in ("ολοκληρω", "εκταμιευ", "προθεσμ", "εργασ", "εργου")):
            base += 2
        return (base, -len(text_value or ""), source_priority(url))
    if field in {
        "total_budget",
        "funding_sources",
        "managing_body",
        "announcement_date",
        "application_start_date",
        "application_end_date",
        "completion_deadline",
        "completion_delay_consequences",
        "post_completion_obligations",
        "eligibility_criteria",
        "eligible_parties",
    }:
        base += source_priority(url)
    length = min(len(text_value or ""), 4000)
    return (base, length, len(str(url or "")))


def valid_total_budget_value(value: Any) -> bool:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return False
    normalized = normalize_text(raw).replace("ς", "σ")
    has_currency_or_unit = bool(
        re.search(r"(€|ευρω|εκατ|εκατομμυρ|δισ|δις|million|billion|bn)", normalized, flags=re.IGNORECASE)
    )
    if not has_currency_or_unit:
        return False
    if re.search(r"\d", raw) is None:
        return False
    return True


def list_value_score(field: str, value: Any, target: dict[str, Any]) -> tuple[int, int, int, float]:
    values = value if isinstance(value, list) else [value]
    normalized_items = [
        normalize_text(json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item).replace("ς", "σ")
        for item in values
        if has_value(item)
    ]
    unique_items = list(dict.fromkeys(item for item in normalized_items if item))
    useful_count = len(unique_items)
    text_blob = " ".join(unique_items)
    quality = source_priority(target["url"])

    if field == "eligible_interventions":
        quality += sum(
            1
            for marker in (
                "κουφ",
                "θερμομον",
                "θερμαν",
                "ψυξ",
                "ζεστ",
                "απε",
                "φωτοβολται",
                "αντλι",
                "εξοικονομηση",
                "ενεργεια",
            )
            if marker in text_blob
        )
        if any(marker in text_blob for marker in ("ενωσιακη πολιτικη", "νοικοκυριων", "επιχορηγηση απο", "καυστηρα")):
            quality -= 5
    elif field == "property_requirements":
        quality += sum(
            2
            for marker in (
                "νομιμ",
                "κατεδαφιστ",
                "κυρια κατοικ",
                "πιστοποιητικ",
                "ενεργειακη αποδοση",
                "κατηγορια χαμηλοτερη",
            )
            if marker in text_blob
        )
        if useful_count > 8:
            quality -= 4
    elif field == "energy_performance_targets":
        quality += sum(
            2
            for marker in (
                "30%",
                "ενεργειακ",
                "κατηγορι",
                "πρωτογεν",
                "εξοικονομηση",
                "αναβαθμιση",
            )
            if marker in text_blob
        )
        if any(
            marker in text_blob
            for marker in (
                "αμοιβ",
                "τεχνικου συμβουλου",
                "ηλεκτρονικη ταυτοτητα",
                "κοστους των δυο ενεργειακων επιθεωρησεων",
            )
        ):
            quality -= 8
        if useful_count > 5:
            quality -= 2
    else:
        quality += min(useful_count, 6)

    timestamp = target.get("timestamp")
    timestamp_score = timestamp.timestamp() if hasattr(timestamp, "timestamp") else 0.0
    return (quality, useful_count, -len(text_blob), timestamp_score)


def merge_values(field: str, values: list[tuple[Any, dict[str, Any]]]) -> tuple[Any, Any]:
    non_empty = [(value, target) for value, target in values if has_value(value)]
    if field == "total_budget":
        non_empty = [(value, target) for value, target in non_empty if valid_total_budget_value(value)]
    if not non_empty:
        return "", None

    first_value = non_empty[0][0]
    if isinstance(first_value, list):
        if field in {
            "eligible_interventions",
            "property_requirements",
            "eligible_parties",
            "eligibility_criteria",
            "energy_performance_targets",
        }:
            chosen_value, chosen_target = max(
                non_empty,
                key=lambda item: list_value_score(field, item[0], item[1]),
            )
            return chosen_value, chosen_target["url"]

        merged = []
        source_urls = []
        seen = set()
        for value, target in non_empty:
            if not isinstance(value, list):
                value = [value]
            for item in value:
                key = normalize_text(json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item)
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                source_urls.append(target["url"])
        return merged, list(dict.fromkeys(source_urls))

    if isinstance(first_value, dict):
        merged_dict = {}
        field_sources = {}
        keys = set()
        for value, _target in non_empty:
            if isinstance(value, dict):
                keys.update(value.keys())
        for key in keys:
            child_values = [(value.get(key), target) for value, target in non_empty if isinstance(value, dict)]
            merged_value, merged_source = merge_values(f"{field}.{key}", child_values)
            if has_value(merged_value):
                merged_dict[key] = merged_value
                field_sources[key] = merged_source
        return merged_dict, field_sources

    if field in {"announcement_date", "application_start_date", "application_end_date", "completion_deadline"}:
        chosen_value, chosen_target = max(
            non_empty,
            key=lambda item: (
                date_source_priority(item[1]["url"]),
                item[1]["timestamp"].timestamp() if hasattr(item[1].get("timestamp"), "timestamp") else 0.0,
                value_score(field, item[0], item[1]["url"]),
            ),
        )
        return chosen_value, chosen_target["url"]

    chosen_value, chosen_target = max(
        non_empty,
        key=lambda item: value_score(field, item[0], item[1]["url"]),
    )
    return chosen_value, chosen_target["url"]


def repair_payload_from_source(payload: dict[str, Any], classification_path: Path, scraped_dir: Path) -> dict[str, Any]:
    repaired = copy.deepcopy(payload)
    scraped_file = find_matching_scraped(classification_path, scraped_dir)
    text = ""
    if scraped_file and scraped_file.exists():
        text = scraped_file.read_text(encoding="utf-8", errors="ignore")
    extracted = repaired.get("extracted_data") or {}
    if isinstance(extracted, dict) and text:
        repaired["extracted_data"] = _fill_missing_core_fields_from_source(
            extracted,
            text,
            str(repaired.get("url") or extracted.get("source_url") or ""),
        )
    repaired["repair_metadata"] = {
        "source_scraped_file": str(scraped_file) if scraped_file else "",
        "repair": "source_backed_core_field_repair",
    }
    return repaired


def log(run_dir: Path, text: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {text}"
    safe_console(line)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_questions(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"Questions file must be a JSON list of strings: {path}")
    return [item.strip() for item in raw if item.strip()]


def deterministic_out_of_scope_answer(question: str) -> str:
    return canonical_out_of_scope_answer(question)


def load_latest_relevant_sources(
    classification_dir: Path,
    scraped_dir: Path,
) -> list[dict[str, Any]]:
    latest_by_url: dict[str, tuple[datetime, Path, dict[str, Any]]] = {}
    for path in sorted(classification_dir.glob("*_classification.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        url = str(payload.get("url") or payload.get("extracted_data", {}).get("source_url") or "").strip()
        if not url:
            continue
        key = url.rstrip("/").casefold()
        timestamp = parse_timestamp(path, payload)
        current = latest_by_url.get(key)
        if current is None or timestamp > current[0]:
            latest_by_url[key] = (timestamp, path, payload)

    relevant_sources = [
        {
            "timestamp": timestamp,
            "path": path,
            "payload": repair_payload_from_source(payload, path, scraped_dir),
            "url": payload.get("url", ""),
            "programme_name": payload.get("extracted_data", {}).get("programme_name", "") or path.stem,
        }
        for timestamp, path, payload in latest_by_url.values()
        if payload.get("classification", {}).get("is_relevant") is True
    ]
    relevant_sources.sort(key=lambda item: (normalized_program_key(item["programme_name"], item["url"]), item["url"]))
    return relevant_sources


def merge_program_target(group: list[dict[str, Any]]) -> dict[str, Any]:
    group = sorted(group, key=lambda item: (source_priority(item["url"]), item["timestamp"]), reverse=True)
    template_keys = set()
    for item in group:
        extracted = item["payload"].get("extracted_data") or {}
        if isinstance(extracted, dict):
            template_keys.update(extracted.keys())

    merged_extracted: dict[str, Any] = {}
    field_sources: dict[str, Any] = {}
    for field in sorted(template_keys):
        values = [
            ((item["payload"].get("extracted_data") or {}).get(field), item)
            for item in group
        ]
        merged_value, merged_source = merge_values(field, values)
        if has_value(merged_value):
            merged_extracted[field] = merged_value
            field_sources[field] = merged_source

    source_urls = list(dict.fromkeys(item["url"] for item in group if item.get("url")))
    source_files = [str(item["path"]) for item in group]
    merged_extracted["source_urls"] = source_urls
    if source_urls and not merged_extracted.get("source_url"):
        merged_extracted["source_url"] = source_urls[0]

    primary = group[0]
    programme_name = str(merged_extracted.get("programme_name") or primary["programme_name"] or "")
    program_id = safe_record_id(merged_extracted.get("id")) or stable_program_id(programme_name, primary["url"])
    merged_extracted["id"] = program_id
    merged_payload = {
        "id": program_id,
        "timestamp": now_stamp(),
        "url": primary["url"],
        "source_urls": source_urls,
        "source_files": source_files,
        "merged_from_count": len(group),
        "merged": True,
        "field_sources": field_sources,
        "extracted_data": merged_extracted,
        "classification": {
            "is_relevant": True,
            "primary_category": primary["payload"].get("classification", {}).get("primary_category", "energy_upgrade"),
            "secondary_categories": sorted(
                {
                    category
                    for item in group
                    for category in (item["payload"].get("classification", {}).get("secondary_categories") or [])
                    if isinstance(category, str)
                }
            ),
            "confidence": max(
                float(item["payload"].get("classification", {}).get("confidence", 0) or 0)
                for item in group
            ),
            "reasoning": f"Merged source-backed JSON from {len(group)} relevant URL(s).",
            "key_features": sorted(
                {
                    feature
                    for item in group
                    for feature in (item["payload"].get("classification", {}).get("key_features") or [])
                    if isinstance(feature, str)
                }
            ),
        },
    }
    return {
        "timestamp": primary["timestamp"],
        "path": primary["path"],
        "payload": merged_payload,
        "url": primary["url"],
        "programme_name": programme_name,
        "source_targets": group,
    }


def build_program_targets(
    relevant_sources: list[dict[str, Any]],
    dedupe_by_title: bool,
    max_programs: int,
) -> list[dict[str, Any]]:
    if dedupe_by_title:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in relevant_sources:
            key = normalized_program_key(item["programme_name"], item["url"])
            grouped.setdefault(key, []).append(item)
        relevant = [merge_program_target(group) for group in grouped.values()]
    else:
        relevant = []
        for item in relevant_sources:
            single = merge_program_target([item])
            relevant.append(single)

    relevant.sort(key=lambda item: normalize_text(item["programme_name"]))
    if max_programs > 0:
        relevant = relevant[:max_programs]
    return relevant


def write_source_classifications(targets: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_by_path: dict[str, dict[str, Any]] = {}
    for target in targets:
        for source_target in target.get("source_targets", [target]):
            source = Path(source_target["path"])
            dest = out_dir / source.name
            dest.write_text(json.dumps(source_target["payload"], ensure_ascii=False, indent=2), encoding="utf-8")
            copied_by_path[str(source)] = {**source_target, "copied_path": dest}
    return list(copied_by_path.values())


def write_program_classifications(targets: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for index, target in enumerate(targets, 1):
        url = target["url"] or target["programme_name"] or f"program_{index}"
        safe_name = f"{index:03d}_{_make_safe_name(url)}_{_short_hash(url)}_classification.json"
        dest = out_dir / safe_name
        dest.write_text(json.dumps(target["payload"], ensure_ascii=False, indent=2), encoding="utf-8")
        materialized.append({**target, "copied_path": dest})
    return materialized


def build_prompt(program_data: dict[str, Any], question: str) -> list[dict[str, str]]:
    return build_single_qa_prompt(program_data, question)


def build_batch_prompt(program_data: dict[str, Any], question_items: list[dict[str, str]]) -> list[dict[str, str]]:
    return build_batch_qa_prompt(program_data, question_items)


def parse_jsonish(text: str) -> Any:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    candidates = []
    object_start = raw.find("{")
    object_end = raw.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(raw[object_start : object_end + 1])
    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(raw[array_start : array_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Could not parse model output as JSON")


def normalize_batch_answers(parsed: Any, question_items: list[dict[str, str]]) -> dict[str, str]:
    if isinstance(parsed, dict):
        raw_answers = parsed.get("answers", parsed.get("responses", []))
    else:
        raw_answers = parsed

    if not isinstance(raw_answers, list):
        return {}

    expected_ids = [item["question_id"] for item in question_items]
    by_id: dict[str, str] = {}
    positional: list[str] = []

    for item in raw_answers:
        if isinstance(item, dict):
            answer = str(item.get("answer", item.get("response", "")) or "").strip()
            qid = str(item.get("question_id", item.get("id", "")) or "").strip()
            if qid and answer:
                by_id[qid] = answer
            elif answer:
                positional.append(answer)
        elif isinstance(item, str) and item.strip():
            positional.append(item.strip())

    for index, answer in enumerate(positional):
        if index < len(expected_ids) and expected_ids[index] not in by_id:
            by_id[expected_ids[index]] = answer

    return {qid: by_id[qid] for qid in expected_ids if by_id.get(qid)}


def ask_llm(
    llm: OllamaChat,
    program_data: dict[str, Any],
    question: str,
    temperature: float,
    num_ctx: int,
    num_predict: int | None,
) -> str:
    options = {"num_ctx": num_ctx}
    if num_predict is not None and num_predict > 0:
        options["num_predict"] = num_predict
    try:
        return llm.ask(
            build_prompt(program_data, question),
            format="",
            temperature=temperature,
            **options,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def ask_llm_batch(
    llm: OllamaChat,
    program_data: dict[str, Any],
    question_items: list[dict[str, str]],
    temperature: float,
    num_ctx: int,
    num_predict: int | None,
) -> tuple[dict[str, str], str]:
    options = {"num_ctx": num_ctx}
    if num_predict is not None and num_predict > 0:
        options["num_predict"] = num_predict
    try:
        raw = llm.ask(
            build_batch_prompt(program_data, question_items),
            format="json",
            temperature=temperature,
            **options,
        ).strip()
        return normalize_batch_answers(parse_jsonish(raw), question_items), raw
    except Exception as exc:
        return {}, f"ERROR: {exc}"


def qa_payload_path(output_dir: Path, url: str, mode: str) -> Path:
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    return output_dir / f"{safe_name}_{mode}_qa_responses.json"


def load_existing_payload(path: Path, url: str, programme_name: str, classification_file: Path, mode: str) -> dict[str, Any]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("responses"), list):
                return payload
        except Exception:
            pass
    return {
        "timestamp": now_stamp(),
        "url": url,
        "programme_name": programme_name,
        "classification_file": classification_file.name,
        "qa_count": 0,
        "source": "qa_evaluation_suite",
        "mode": mode,
        "responses": [],
    }


def response_key(question_id: str, repeat_index: int) -> str:
    return f"{question_id}::r{repeat_index}"


def run_qa_mode(
    *,
    run_dir: Path,
    targets: list[dict[str, Any]],
    questions: list[str],
    output_dir: Path,
    llm: OllamaChat,
    mode: str,
    repeat_each: int,
    temperature: float,
    num_ctx: int,
    num_predict: int | None,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, batch_size)
    total_answers = len(targets) * len(questions) * repeat_each
    completed_answers = 0
    started = time.monotonic()

    for program_index, target in enumerate(targets, 1):
        payload = target["payload"]
        extracted = payload.get("extracted_data", {}) or {}
        url = str(payload.get("url") or extracted.get("source_url") or "")
        programme_name = str(extracted.get("programme_name") or target["programme_name"] or "")
        classification_file = Path(target["copied_path"])
        out_path = qa_payload_path(output_dir, url, mode)
        qa_payload = load_existing_payload(out_path, url, programme_name, classification_file, mode)
        done_keys = {
            response_key(str(item.get("question_id", "")), int(item.get("repeat_index", 1) or 1))
            for item in qa_payload.get("responses", [])
        }

        log(run_dir, f"[{mode}] Program {program_index}/{len(targets)}: {programme_name} | {url}")

        def record_answer(qid: str, question: str, answer: str, repeat_index: int, answer_mode: str = "llm") -> None:
            nonlocal completed_answers
            qa_payload["responses"].append(
                {
                    "question_id": qid,
                    "question": question,
                    "answer": answer,
                    "repeat_index": repeat_index,
                    "repeat_count": repeat_each,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "qa_evaluation_suite",
                    "answer_mode": answer_mode,
                }
            )
            qa_payload["qa_count"] = len(qa_payload["responses"])
            qa_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            done_keys.add(response_key(qid, repeat_index))

            completed_answers += 1
            elapsed = time.monotonic() - started
            rate = completed_answers / elapsed if elapsed else 0.0
            remaining = (total_answers - completed_answers) / rate if rate else 0.0
            log(
                run_dir,
                f"[{mode}] Answer {completed_answers}/{total_answers} "
                f"({programme_name}, {qid}) ETA {remaining/60:.1f}m",
            )

        for repeat_index in range(1, repeat_each + 1):
            pending: list[dict[str, str]] = []
            for question_index, question in enumerate(questions, 1):
                base_question_id = f"q{question_index}"
                qid = base_question_id if repeat_each == 1 else f"{base_question_id}_r{repeat_index}"
                key = response_key(qid, repeat_index)
                if key in done_keys:
                    completed_answers += 1
                    continue
                pending.append({"question_id": qid, "question": question})

            llm_pending = []
            for item in pending:
                deterministic_answer = deterministic_out_of_scope_answer(item["question"])
                if deterministic_answer:
                    record_answer(
                        item["question_id"],
                        item["question"],
                        deterministic_answer,
                        repeat_index,
                        answer_mode="deterministic_out_of_scope_guard",
                    )
                else:
                    llm_pending.append(item)
            if len(llm_pending) != len(pending):
                out_path.write_text(json.dumps(qa_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            pending = llm_pending

            if not pending:
                continue

            for batch_start in range(0, len(pending), batch_size):
                chunk = pending[batch_start : batch_start + batch_size]
                batch_answers, raw_batch = ask_llm_batch(
                    llm,
                    extracted,
                    chunk,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
                if len(batch_answers) != len(chunk):
                    missing = [item for item in chunk if item["question_id"] not in batch_answers]
                    log(
                        run_dir,
                        f"[{mode}] Batch returned {len(batch_answers)}/{len(chunk)} answers "
                        f"({programme_name}, repeat {repeat_index}); retrying {len(missing)} missing individually",
                    )
                    for item in missing:
                        batch_answers[item["question_id"]] = ask_llm(
                            llm,
                            extracted,
                            item["question"],
                            temperature=temperature,
                            num_ctx=num_ctx,
                            num_predict=num_predict,
                        )

                for item in chunk:
                    qid = item["question_id"]
                    question = item["question"]
                    answer = batch_answers.get(qid, "").strip() or f"ERROR: missing answer from batch. Raw batch: {raw_batch[:1000]}"
                    record_answer(qid, question, answer, repeat_index)
                out_path.write_text(json.dumps(qa_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_qa_files(source_dir: Path, output_path: Path) -> dict[str, Any]:
    files = sorted(source_dir.glob("*_qa_responses.json"))
    payloads = []
    answer_count = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payloads.append({"file": str(path), **payload})
        answer_count += len(payload.get("responses", []) or [])
    aggregate = {
        "timestamp": now_stamp(),
        "source_dir": str(source_dir),
        "qa_files": len(payloads),
        "answer_count": answer_count,
        "programs": payloads,
    }
    output_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def run_command(run_dir: Path, args: list[str]) -> int:
    log(run_dir, "RUN " + " ".join(args))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log_path = run_dir / "commands.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(args) + "\n")
        handle.write(completed.stdout)
        if completed.stderr:
            handle.write("\n[stderr]\n" + completed.stderr)
        handle.write(f"\n[exit {completed.returncode}]\n")
    if completed.stdout.strip():
        log(run_dir, completed.stdout.strip().splitlines()[-1])
    if completed.returncode != 0 and completed.stderr.strip():
        log(run_dir, "WARN " + completed.stderr.strip().splitlines()[-1])
    return completed.returncode


def build_html_report(run_dir: Path, manifest: dict[str, Any]) -> Path:
    def load_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    all15 = load_json(run_dir / "qa_all15_answers.json")
    repeat = load_json(run_dir / "qa_repeat_q1_11_x5_answers.json")
    consistency = load_json(run_dir / "qa_consistency_report.json")
    semantic = load_json(run_dir / "qa_semantic_report.json")
    repeatability = load_json(run_dir / "qa_repeatability_report.json")
    html_validation = load_json(run_dir / "html_json_report.json")
    completeness = load_json(run_dir / "json_completeness_report.json")
    visual_summary = load_json(run_dir / "plots" / "visual_summary.json")

    sem_metrics = semantic.get("semantic_validation", {}).get("global_metrics", {})
    rep_summary = repeatability.get("summary", {})
    html_cov = html_validation.get("summary", {}).get("overall_coverage", {})
    completeness_summary = completeness.get("summary", {})
    json_expected_coverage = float(completeness_summary.get("avg_expected_coverage", 0.0) or 0.0)
    consistency_results = [item for item in consistency.get("results", []) if item.get("status") == "OK"]
    avg_consistency = mean([float(item.get("consistency_score", 0.0) or 0.0) for item in consistency_results]) if consistency_results else 0.0

    rows = []
    for target in manifest.get("targets", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(target.get('programme_name', '')))}</td>"
            f"<td><a href=\"{html.escape(str(target.get('url', '')))}\">source</a></td>"
            f"<td>{html.escape(Path(str(target.get('copied_path', ''))).name)}</td>"
            "</tr>"
        )

    charts = []
    for chart in visual_summary.get("charts", []):
        chart_path = Path("plots") / chart
        charts.append(
            f"<figure><img src=\"{html.escape(chart_path.as_posix())}\" alt=\"{html.escape(chart)}\">"
            f"<figcaption>{html.escape(chart)}</figcaption></figure>"
        )

    payload_links = [
        ("Manifest", "target_manifest.json"),
        ("All 15 Q&A JSON", "qa_all15_answers.json"),
        ("Repeat Q1-Q11 x5 JSON", "qa_repeat_q1_11_x5_answers.json"),
        ("HTML vs JSON report", "html_json_report.json"),
        ("JSON completeness report", "json_completeness_report.json"),
        ("Q&A consistency report", "qa_consistency_report.json"),
        ("Semantic report", "qa_semantic_report.json"),
        ("Repeatability report", "qa_repeatability_report.json"),
        ("Visual summary", "plots/visual_summary.json"),
    ]

    html_text = f"""<!doctype html>
<html lang="el">
<head>
  <meta charset="utf-8">
  <title>QA Evaluation Report {html.escape(manifest.get('run_id', ''))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 6px; padding: 12px; background: #f9fbfd; }}
    .kpi {{ font-size: 26px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    figure {{ margin: 18px 0; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; border-radius: 4px; }}
    code, pre {{ background: #f2f4f7; padding: 2px 4px; border-radius: 4px; }}
    .links a {{ display: inline-block; margin: 0 12px 8px 0; }}
  </style>
</head>
<body>
  <h1>QA Evaluation Report</h1>
  <p>Run: <code>{html.escape(str(manifest.get('run_id', '')))}</code></p>
  <div class="grid">
    <div class="card"><div>Programs</div><div class="kpi">{manifest.get('target_count', 0)}</div></div>
    <div class="card"><div>All-15 answers</div><div class="kpi">{all15.get('answer_count', 0)}</div></div>
    <div class="card"><div>Repeat answers</div><div class="kpi">{repeat.get('answer_count', 0)}</div></div>
    <div class="card"><div>Answer-data consistency</div><div class="kpi">{avg_consistency:.1%}</div></div>
    <div class="card"><div>Semantic Token-F1</div><div class="kpi">{float(sem_metrics.get('token_f1', 0.0) or 0.0):.1%}</div></div>
    <div class="card"><div>Repeatability Token-F1</div><div class="kpi">{float(rep_summary.get('avg_token_f1', 0.0) or 0.0):.1%}</div></div>
    <div class="card"><div>HTML evidence found</div><div class="kpi">{(html_cov.get('found', 0) / html_cov.get('total_checked', 1) if html_cov.get('total_checked') else 0):.1%}</div></div>
    <div class="card"><div>JSON expected coverage</div><div class="kpi">{json_expected_coverage:.1%}</div></div>
  </div>

  <h2>JSON Artifacts</h2>
  <p class="links">{''.join(f'<a href="{href}">{html.escape(label)}</a>' for label, href in payload_links)}</p>

  <h2>Graphs</h2>
  {''.join(charts) if charts else '<p>No charts generated.</p>'}

  <h2>Target Programs</h2>
  <table>
    <thead><tr><th>Program</th><th>URL</th><th>Classification file</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>Embedded Summary JSON</h2>
  <pre>{html.escape(json.dumps({
      'manifest': manifest,
      'semantic_global_metrics': sem_metrics,
      'repeatability_summary': rep_summary,
      'html_validation_summary': html_validation.get('summary', {}),
      'json_completeness_summary': completeness_summary,
  }, ensure_ascii=False, indent=2))}</pre>
</body>
</html>
"""
    report_path = run_dir / "evaluation_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full QA evaluation suite for relevant classification files.")
    parser.add_argument("--classification-dir", default="output/va4_product_discoverer")
    parser.add_argument("--scraped-dir", default="output/va3_scraper_to_template")
    parser.add_argument("--questions-file", default="evaluation/questions.json")
    parser.add_argument("--output-root", default="output/evaluation/qa_full_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--repeat-batch-size", type=int, default=1)
    parser.add_argument("--max-programs", type=int, default=0)
    parser.add_argument("--no-dedupe-title", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or now_stamp()
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(ROOT / args.questions_file)
    if len(questions) < 15:
        raise ValueError("Expected at least 15 questions in questions.json")
    all15_questions = questions[:15]
    repeat_questions = questions[:11]

    relevant_sources = load_latest_relevant_sources(
        classification_dir=ROOT / args.classification_dir,
        scraped_dir=ROOT / args.scraped_dir,
    )
    targets = build_program_targets(
        relevant_sources,
        dedupe_by_title=not args.no_dedupe_title,
        max_programs=args.max_programs,
    )
    source_classifications = write_source_classifications(targets, run_dir / "source_classifications")
    copied_targets = write_program_classifications(targets, run_dir / "classifications")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification_dir": args.classification_dir,
        "selection": {
            "latest_by_url": True,
            "is_relevant": True,
            "merge_by_program": not args.no_dedupe_title,
            "program_key_rules": [
                "exact_normalized_title_or_url",
                "exoikonomo_2025_aliases",
                "exoikonomo_anakainizo_gia_neous_aliases",
                "anavathmizo_to_spiti_mou_aliases",
                "photovoltaika_sti_stegi_aliases",
            ],
            "max_programs": args.max_programs,
        },
        "question_counts": {
            "all15": len(all15_questions),
            "repeat_questions": len(repeat_questions),
            "repeat_count": args.repeat_count,
        },
        "llm": {
            "model": args.llm_model,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "generation_mode": "full_pruned_json_question_batches",
            "all15_batch_size": args.batch_size,
            "repeat_batch_size": args.repeat_batch_size,
            "prompt_context": "full_pruned_json_empty_fields_removed_no_string_truncation",
        },
        "source_url_count": len(source_classifications),
        "target_count": len(copied_targets),
        "targets": [
            {
                "programme_name": item["programme_name"],
                "url": item["url"],
                "source_path": str(item["path"]),
                "copied_path": str(item["copied_path"]),
                "timestamp": item["timestamp"].isoformat(),
                "merged_from_count": item["payload"].get("merged_from_count", 1),
                "source_urls": item["payload"].get("source_urls", [item["url"]]),
            }
            for item in copied_targets
        ],
    }
    (run_dir / "target_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(run_dir, f"Targets selected: {len(copied_targets)}")

    if not copied_targets:
        log(run_dir, "No relevant targets found.")
        return 1

    if not args.skip_qa:
        llm = OllamaChat(model=args.llm_model)
        run_qa_mode(
            run_dir=run_dir,
            targets=copied_targets,
            questions=all15_questions,
            output_dir=run_dir / "qa_all15",
            llm=llm,
            mode="all15",
            repeat_each=1,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            batch_size=args.batch_size,
        )
        run_qa_mode(
            run_dir=run_dir,
            targets=copied_targets,
            questions=repeat_questions,
            output_dir=run_dir / "qa_repeat_q1_11_x5",
            llm=llm,
            mode="repeat_q1_11_x5",
            repeat_each=args.repeat_count,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            batch_size=args.repeat_batch_size,
        )

    aggregate_qa_files(run_dir / "qa_all15", run_dir / "qa_all15_answers.json")
    aggregate_qa_files(run_dir / "qa_repeat_q1_11_x5", run_dir / "qa_repeat_q1_11_x5_answers.json")

    python = sys.executable
    run_command(
        run_dir,
        [
            python,
            "evaluation/json_completeness_validator.py",
            "--classification-dir",
            str(run_dir / "classifications"),
            "--output",
            str(run_dir / "json_completeness_report.json"),
        ],
    )
    run_command(
        run_dir,
        [
            python,
            "evaluation/batch_html_validator.py",
            "--classification-dir",
            str(run_dir / "source_classifications"),
            "--scraped-dir",
            str(ROOT / args.scraped_dir),
            "--output",
            str(run_dir / "html_json_report.json"),
            "--summary-only",
            "--exclude-fields",
            "classification",
        ],
    )

    if args.skip_qa:
        report_path = build_html_report(run_dir, manifest)
        log(run_dir, f"HTML report: {report_path}")
        log(run_dir, "DONE")
        return 0

    run_command(
        run_dir,
        [
            python,
            "evaluation/qa_consistency_validator.py",
            "--classification-dir",
            str(run_dir / "classifications"),
            "--qa-responses-dir",
            str(run_dir / "qa_all15"),
            "--output",
            str(run_dir / "qa_consistency_report.json"),
            "--summary-only",
        ],
    )
    semantic_cmd = [
        python,
        "evaluation/qa_semantic_validator.py",
        "--qa-report",
        str(run_dir / "qa_consistency_report.json"),
        "--output",
        str(run_dir / "qa_semantic_report.json"),
        "--summary-only",
    ]
    if not args.skip_bertscore:
        semantic_cmd.append("--enable-bertscore")
    if not args.skip_embeddings:
        semantic_cmd.append("--enable-embeddings")
        semantic_cmd.extend(["--embedding-model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"])
    run_command(run_dir, semantic_cmd)
    run_command(
        run_dir,
        [
            python,
            "evaluation/qa_repeatability_validator.py",
            "--qa-responses-dir",
            str(run_dir / "qa_repeat_q1_11_x5"),
            "--output",
            str(run_dir / "qa_repeatability_report.json"),
            "--min-repeats",
            str(args.repeat_count),
            "--stable-threshold",
            "0.80",
            "--summary-only",
        ],
    )
    run_command(
        run_dir,
        [
            python,
            "evaluation/metrics_visualizer.py",
            "--qa-consistency-report",
            str(run_dir / "qa_consistency_report.json"),
            "--qa-semantic-report",
            str(run_dir / "qa_semantic_report.json"),
            "--qa-repeatability-report",
            str(run_dir / "qa_repeatability_report.json"),
            "--html-report",
            str(run_dir / "html_json_report.json"),
            "--output-dir",
            str(run_dir / "plots"),
            "--top-n",
            "25",
        ],
    )

    report_path = build_html_report(run_dir, manifest)
    log(run_dir, f"HTML report: {report_path}")
    log(run_dir, "DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
