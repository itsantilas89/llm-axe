from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash, save_result
from llm_axe import va4_product_discoverer as va4


SUMMARY_INPUTS = [
    ROOT / ".codex_logs" / "greece_completeness_va4_batch_20260616T225519Z_summary.json",
    ROOT / ".codex_logs" / "greece_completeness_va4_supplemental_20260617T022017Z_summary.json",
    ROOT / ".codex_logs" / "repair_scoped_va4_outputs_summary.json",
]

SCRIPT_INPUTS = [
    ROOT / ".codex_logs" / "run_va4_greece_completeness_batch.py",
    ROOT / ".codex_logs" / "run_va4_greece_completeness_supplemental.py",
]

EXTRA_URLS = [
    "https://oikogeneia.gov.gr/programs/anavathmizo-to-spiti-mou/",
    "https://www.nbg.gr/el/idiwtes/daneia/stegastika-daneia/daneia-akinitwn/estia-prasini",
    "https://www.epirusbank.com/blog/deltia-tupou-1/183-neo-crematodotiko-proion-gia-dikaioucous-tou-programmatos-lpotoboltaika-ste-steger",
]

VA3_OUT = ROOT / "output" / "va3_scraper_to_template"
VA4_OUT = ROOT / "output" / "va4_product_discoverer"
SUMMARY_PATH = ROOT / ".codex_logs" / "repair_greece_completeness_va4_outputs_summary.json"


def _read_urls_from_summary(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        rows = payload.get("results") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    urls = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("url"), str):
            urls.append(row["url"])
    return urls


def _read_urls_from_script(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "URLS" in names:
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return []
                return [item for item in value if isinstance(item, str)]
    return []


def _unique_urls() -> list[str]:
    urls: list[str] = []
    for path in SUMMARY_INPUTS:
        urls.extend(_read_urls_from_summary(path))
    for path in SCRIPT_INPUTS:
        urls.extend(_read_urls_from_script(path))
    urls.extend(EXTRA_URLS)

    seen = set()
    unique = []
    for url in urls:
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        unique.append(url)
    return unique


def _keys(url: str) -> list[str]:
    variants = [url, url.rstrip("/")]
    if not url.endswith("/"):
        variants.append(url + "/")
    seen: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.append(item)
    return [f"{_make_safe_name(item)}_{_short_hash(item)}" for item in seen]


def _latest_file(root: Path, url: str, suffix: str) -> Path | None:
    files: list[Path] = []
    for key in _keys(url):
        files.extend(root.glob(f"*_{key}_{suffix}"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _read_json(path: Path | None):
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _previous_classification(url: str) -> dict:
    path = _latest_file(VA4_OUT, url, "classification.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    classification = payload.get("classification")
    return classification if isinstance(classification, dict) else {}


def _extract_row(payload) -> dict:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def _reject(reason: str, feature: str, confidence: float = 0.96) -> dict:
    return {
        "is_relevant": False,
        "primary_category": "other",
        "secondary_categories": [],
        "confidence": confidence,
        "reasoning": reason,
        "key_features": [feature],
    }


def _classification_for(url: str, extracted: dict, text: str) -> dict:
    deterministic = va4._deterministic_classification_if_obvious(extracted, url)
    if deterministic is not None:
        return deterministic

    previous = _previous_classification(url)
    if previous:
        guarded = va4._apply_deterministic_classification_guard(previous, extracted, url)
        if guarded:
            return guarded

    scope_ok, scope_reason, scope = va4._is_home_green_finance_candidate(
        " ".join([url, text[:5000]]),
        extracted,
        url,
    )
    if not scope_ok:
        return _reject(f"Deterministic scope reject: {scope_reason}", "deterministic_scope_reject")

    category = "energy_upgrade" if scope.get("has_home_appliance") else "green_housing_loan"
    return {
        "is_relevant": True,
        "primary_category": category,
        "secondary_categories": ["home_renewables", "energy_upgrade"],
        "confidence": 0.84,
        "reasoning": f"Deterministic scope pass: {scope_reason}",
        "key_features": ["deterministic_home_green_finance_scope"],
    }


def _write_latest_extracted(path: Path, original_payload, repaired: dict) -> None:
    payload = [repaired] if isinstance(original_payload, list) else repaired
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_latest_classification(path: Path, repaired: dict, classification: dict) -> None:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return
    payload["extracted_data"] = repaired
    payload["classification"] = classification
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(url: str, status: str, scraped_path: Path | None, extracted_path: Path | None, classification_path: Path | None, repaired: dict, classification: dict, notes: list[str]) -> dict:
    return {
        "url": url,
        "status": status,
        "programme_name": repaired.get("programme_name", ""),
        "is_relevant": classification.get("is_relevant"),
        "primary_category": classification.get("primary_category"),
        "minimum_funding_amount": repaired.get("minimum_funding_amount", ""),
        "maximum_funding_amount": repaired.get("maximum_funding_amount", ""),
        "funding_type": repaired.get("funding_type", ""),
        "eligible_interventions_count": len(repaired.get("eligible_interventions") or []),
        "scraped_chars": len(scraped_path.read_text(encoding="utf-8", errors="replace")) if scraped_path else 0,
        "scraped_path": str(scraped_path) if scraped_path else "",
        "extracted_path": str(extracted_path) if extracted_path else "",
        "classification_path": str(classification_path) if classification_path else "",
        "notes": notes,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rows = []
    for url in _unique_urls():
        notes: list[str] = []
        scraped_path = _latest_file(VA3_OUT, url, "scraped.txt")
        extracted_path = _latest_file(VA3_OUT, url, "extracted.json")
        classification_path = _latest_file(VA4_OUT, url, "classification.json")

        text = scraped_path.read_text(encoding="utf-8", errors="replace") if scraped_path else ""
        original_payload = _read_json(extracted_path)
        extracted = _extract_row(original_payload)

        if not extracted:
            classification = _previous_classification(url)
            if not classification:
                scope_ok, scope_reason, _scope = va4._is_home_green_finance_candidate(
                    " ".join([url, text[:5000]]),
                    {"source_url": url, "source_urls": [url]},
                    url,
                )
                feature = "missing_extracted_json_scope_pass" if scope_ok else "missing_extracted_json_scope_reject"
                classification = _reject(f"No extracted JSON found; scoped source check: {scope_reason}", feature)
            rows.append(
                _summarize(
                    url,
                    "missing_extracted_json",
                    scraped_path,
                    extracted_path,
                    classification_path,
                    {"source_url": url, "source_urls": [url]},
                    classification,
                    notes,
                )
            )
            continue

        before = json.dumps(extracted, ensure_ascii=False, sort_keys=True)
        repaired = va4._fill_missing_core_fields_from_source(extracted, text, url)
        classification = _classification_for(url, repaired, text)
        after = json.dumps(repaired, ensure_ascii=False, sort_keys=True)

        if before != after:
            notes.append("repaired_extracted_data")
        if extracted_path:
            _write_latest_extracted(extracted_path, original_payload, repaired)
        if classification_path:
            _write_latest_classification(classification_path, repaired, classification)

        rows.append(
            _summarize(
                url,
                "ok",
                scraped_path,
                extracted_path,
                classification_path,
                repaired,
                classification,
                notes,
            )
        )

    SUMMARY_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nSummary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
