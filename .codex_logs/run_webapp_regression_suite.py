from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash


BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = ROOT / ".codex_logs"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = OUT_DIR / f"webapp_regression_suite_{RUN_ID}.log"
SUMMARY_PATH = OUT_DIR / f"webapp_regression_suite_{RUN_ID}_summary.json"


CASES = [
    # Negative controls: not residential/home green finance.
    {
        "label": "Alpha EV-only loan",
        "expected": False,
        "url": "https://www.alpha.gr/el/idiotes/daneia/katanalotika-daneia/alpha-prasines-luseis-daneio-ilektriko-autokinito",
    },
    {
        "label": "Piraeus EV-only loan",
        "expected": False,
        "url": "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/daneia/proswpika/katanalotiko-daneio-ev-loan-gia-agora-ilektrikou-autokinitou",
    },
    {
        "label": "Kinoumai Ilektrika III",
        "expected": False,
        "url": "https://kinoumeilektrika3.gov.gr/",
    },
    {
        "label": "gov.gr Kinoumai Ilektrika III",
        "expected": False,
        "url": "https://www.gov.gr/ipiresies/polites-kai-kathemerinoteta/periballon-kai-poioteta-zoes/kinoumai-elektrika-iii",
    },
    {
        "label": "HDB Green Co-Financing SME",
        "expected": False,
        "url": "https://hdb.gr/growth-fund-green-co-financing-loans/",
    },
    {
        "label": "Exoikonomo Epixeiro",
        "expected": False,
        "url": "https://exoikonomoepixeiro.energy-invest.gov.gr/",
    },
    {
        "label": "HLEKTRA public buildings",
        "expected": False,
        "url": "https://hlektra.gov.gr/home",
    },
    {
        "label": "Piraeus business photovoltaic",
        "expected": False,
        "url": "https://www.piraeusbank.gr/el/epixeiriseis-epaggelmaties/xrimatodotiseis-programmata-xrimatodotisis/mikres-mesaies-epixeiriseis/prasines-xrimatodotiseis/peiraiws-epixeirein-fwtoboltaiko",
    },
    {
        "label": "Piraeus business net metering",
        "expected": False,
        "url": "https://www.piraeusbank.gr/el/epixeiriseis-epaggelmaties/xrimatodotiseis-programmata-xrimatodotisis/mikres-mesaies-epixeiriseis/prasines-xrimatodotiseis/peiraiws-net-metering",
    },
    # Positive controls: residential/home energy upgrades, renewables, or green home loans.
    {
        "label": "Alpha green home loan",
        "expected": True,
        "url": "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/alpha-prasines-luseis-katanalotiko-daneio-gia-to-spiti",
    },
    {
        "label": "Piraeus green consumer home repairs",
        "expected": True,
        "url": "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/daneia/proswpika/katanalwtiko-daneio-green-gia-prasines-episkeves",
    },
    {
        "label": "Piraeus green repair secured",
        "expected": True,
        "url": "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/prasino-episkeuastiko-daneio-me-eksasfalisi",
    },
    {
        "label": "Chania Anavathmizo",
        "expected": True,
        "url": "https://www.chaniabank.gr/idiotes/dania/anavathmizo-to-spiti-moy/",
    },
    {
        "label": "Epirus Anavathmizo",
        "expected": True,
        "url": "https://www.epirusbank.com/citizens/anavathmizo-to-spiti-mou",
    },
    {
        "label": "Eurobank Green Fast",
        "expected": True,
        "url": "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/green-fast-loan",
    },
    {
        "label": "NBG mixed home/car green loan",
        "expected": True,
        "url": "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/prasino-daneio-spitiou-h-autokinhtou",
    },
]


def write_log(message: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def json_request(method: str, path: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_classification(url: str) -> tuple[Path | None, dict]:
    root = ROOT / "output" / "va4_product_discoverer"
    candidates: list[Path] = []
    variants = [url, url.rstrip("/")]
    if not url.endswith("/"):
        variants.append(url + "/")
    for variant in dict.fromkeys(variants):
        key = f"{_make_safe_name(variant)}_{_short_hash(variant)}"
        candidates.extend(root.glob(f"*_{key}_classification.json"))
    if not candidates:
        return None, {}
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, {}
    classification = payload.get("classification")
    return path, classification if isinstance(classification, dict) else {}


def run_case(case: dict) -> dict:
    write_log(f"START {case['label']} expected={case['expected']} url={case['url']}")
    response = json_request(
        "POST",
        "/admin/crawl",
        {
            "urls": [case["url"]],
            "include_known_links": False,
            "model_fast": "llama3.2:latest",
            "model_classification": "llama3.1:8b-instruct-q4_K_M",
        },
    )
    job_id = response["job_id"]
    write_log(f"JOB {case['label']} id={job_id}")

    job = {}
    for _ in range(360):
        job = json_request("GET", f"/admin/jobs/{job_id}", timeout=20)
        if job.get("status") in {"completed", "failed"}:
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"Job polling timed out for {case['label']} ({job_id})")

    path, classification = latest_classification(case["url"])
    actual = classification.get("is_relevant")
    passed = actual is case["expected"] and job.get("status") == "completed"
    result = {
        "label": case["label"],
        "url": case["url"],
        "expected": case["expected"],
        "actual": actual,
        "passed": passed,
        "job_id": job_id,
        "job_status": job.get("status"),
        "saved_program_ids": job.get("saved_program_ids", []),
        "errors": job.get("errors", []),
        "message": job.get("message"),
        "classification_path": str(path) if path else "",
        "classification": classification,
        "last_log": (job.get("logs") or [""])[-1],
    }
    write_log(
        "DONE "
        f"{case['label']} expected={case['expected']} actual={actual} "
        f"passed={passed} saved={len(result['saved_program_ids'])} job={job_id}"
    )
    return result


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    health = json_request("GET", "/health")
    write_log(f"HEALTH {health}")
    results = []
    for case in CASES:
        result = run_case(case)
        results.append(result)
        SUMMARY_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for item in results if item["passed"])
    failed = len(results) - passed
    write_log(f"SUMMARY passed={passed} failed={failed} total={len(results)} summary={SUMMARY_PATH}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
