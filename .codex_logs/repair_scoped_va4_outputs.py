from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash, save_result
from llm_axe import va4_product_discoverer as va4


URLS = [
    "https://exoikonomo2025.gov.gr/",
    "https://exoikonomo2025.gov.gr/odegos",
    "https://stegasi.gov.gr/programs/anavathmizo-to-spiti-mou/",
    "https://stegasi.gov.gr/programs/exoikonomo-2025/",
    "https://greece20.gov.gr/home-loans/",
    "https://www.gov.gr/ipiresies/periousia-kai-phorologia/diakheirise-akinetes-periousias/exoikonomo-2025",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/anabathmizo-to-spiti-mou",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/eksoikonomo-2025",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/prasino-daneio-spitiou-h-autokinhtou",
    "https://www.alpha.gr/el/idiotes/daneia/prasina-daneia",
    "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/programma-eksoikonomo-2025",
    "https://www.crediabank.com/idiotes/daneia/eco-katanalotika/eco-lyseis-energeiaki-anavathmisi-katoikias/",
    "https://www.crediabank.com/idiotes/daneia/stegastika/stegastiko-daneio-eco-home/",
    "https://www.crediabank.com/idiotes/daneia/stegastika/programma-anavathmizo-to-spiti-mou/",
    "https://exoikonomoepixeiro.energy-invest.gov.gr/",
    "https://hdb.gr/en/green-co-financing-loans/",
    "https://hlektra.gov.gr/home",
    "https://www.gov.gr/ipiresies/polites-kai-kathemerinoteta/periballon-kai-poioteta-zoes/photoboltaika-ste-stege",
]


VA3_OUT = ROOT / "output" / "va3_scraper_to_template"
VA4_OUT = ROOT / "output" / "va4_product_discoverer"
SUMMARY_PATH = ROOT / ".codex_logs" / "repair_scoped_va4_outputs_summary.json"


def _keys(url: str) -> list[str]:
    variants = [url, url.rstrip("/")]
    if not url.endswith("/"):
        variants.append(url + "/")
    seen = []
    for item in variants:
        if item and item not in seen:
            seen.append(item)
    return [f"{_make_safe_name(item)}_{_short_hash(item)}" for item in seen]


def _latest_file(root: Path, url: str, suffix: str) -> Path | None:
    files: list[Path] = []
    for key in _keys(url):
        files.extend(root.glob(f"*_{key}_{suffix}"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _latest_classification(url: str) -> dict:
    path = _latest_file(VA4_OUT, url, "classification.json")
    if not path:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    classification = payload.get("classification")
    return classification if isinstance(classification, dict) else {}


def _reject(url: str, reason: str, feature: str, confidence: float = 0.95) -> dict:
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

    previous = _latest_classification(url)
    if previous:
        guarded = va4._apply_deterministic_classification_guard(previous, extracted, url)
        if guarded:
            return guarded

    scope_ok, scope_reason, _scope = va4._is_home_green_finance_candidate(url, extracted, text)
    if not scope_ok:
        return _reject(url, f"Deterministic scope reject: {scope_reason}", "deterministic_scope_reject")
    return {
        "is_relevant": True,
        "primary_category": "green_housing_loan",
        "secondary_categories": ["energy_upgrade"],
        "confidence": 0.84,
        "reasoning": f"Deterministic scope pass: {scope_reason}",
        "key_features": ["deterministic_home_green_finance_scope"],
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    summary: list[dict] = []
    for url in URLS:
        scraped_path = _latest_file(VA3_OUT, url, "scraped.txt")
        extracted_path = _latest_file(VA3_OUT, url, "extracted.json")
        text = scraped_path.read_text(encoding="utf-8") if scraped_path else ""

        if not extracted_path:
            scope_ok, scope_reason, _scope = va4._is_home_green_finance_candidate(url, text)
            classification = _reject(
                url,
                (
                    "No extracted JSON was produced in the latest batch; "
                    f"scoped source check: {scope_reason}"
                ),
                "no_extracted_json_from_batch" if scope_ok else "prefilter_reject",
                0.9 if scope_ok else 0.96,
            )
            classification_path = va4.save_classification_result(
                url,
                {"source_url": url, "source_urls": [url]},
                classification,
            )
            summary.append(
                {
                    "url": url,
                    "status": "rejected_no_extracted_json",
                    "classification": classification,
                    "classification_path": classification_path,
                    "scraped_chars": len(text),
                }
            )
            continue

        raw = json.loads(extracted_path.read_text(encoding="utf-8"))
        extracted = raw[0] if isinstance(raw, list) and raw else raw
        if not isinstance(extracted, dict):
            extracted = {}

        repaired = va4._fill_missing_core_fields_from_source(extracted, text, url)
        extracted_out = save_result([repaired], url)
        classification = _classification_for(url, repaired, text)
        classification_out = va4.save_classification_result(url, repaired, classification)
        summary.append(
            {
                "url": url,
                "status": "repaired",
                "programme_name": repaired.get("programme_name", ""),
                "is_relevant": classification.get("is_relevant"),
                "primary_category": classification.get("primary_category"),
                "extracted_path": extracted_out,
                "classification_path": classification_out,
                "scraped_chars": len(text),
            }
        )

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
