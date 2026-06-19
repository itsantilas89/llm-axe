from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = ROOT / ".codex_logs" / f"greece_completeness_va4_batch_{TS}.log"
SUMMARY_PATH = ROOT / ".codex_logs" / f"greece_completeness_va4_batch_{TS}_summary.json"

URLS = [
    "https://hdb.gr/anavathmizo-to-spiti-mou/",
    "https://greece20.gov.gr/?calls=programma-sygchrimatodotoymenon-stegastikon-daneion-anavathmizo-to-spiti-moy",
    "https://greece20.gov.gr/anavathmizo-to-spiti-moy/",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/programma-anabathmizo-to-spiti-mou",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/eksoikonomo-anakainizo-gia-neous",
    "https://www.nbg.gr/el/idiwtes/daneia/stegastika-daneia/episkeuastika-daneia/anavathmizw-to-spiti-mou",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/exoikonomo-2025",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/eksoikonomw-anakoinizw-daneio-gia-neous",
    "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/alpha-prasines-luseis-katanalotiko-daneio-gia-to-spiti",
    "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/programma-anavathmizo-to-spiti-mou",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/daneio-eksoikonomw",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/prasino-episkeuastiko-daneio-me-eksasfalisi",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/prasino-stegastiko-daneio-me-stathero-epitokio",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/prasino-stegastiko-daneio-sundedemeno-me-euribor",
    "https://www.crediabank.com/idiotes/daneia/hrimatodotika-programmata/exoikonomo-2025/",
    "https://www.crediabank.com/idiotes/daneia/eco-katanalotika/eco-lyseis-net-metering-egkatastasi-oikiakon-fv-sustimaton/",
    "https://www.crediabank.com/programma-exoikonomo-anakainizo-gia-neous/",
    "https://www.chaniabank.gr/idiotes/dania/anavathmizo-to-spiti-moy/",
    "https://www.bankofkarditsa.com.gr/el/idiotes/daneia/anavathmizo-to-spiti-mou",
    "https://www.bankofthessaly.gr/idiotes/prasina-daneia/",
    "https://www.bankofthessaly.gr/idiotes/prasina-daneia/exoikonomo-2025/",
    "https://www.bankofthessaly.gr/idiotes/prasina-daneia/anavathmizo-to-spiti-mou/",
    "https://www.bankofthessaly.gr/idiotes/prasina-daneia/exoikonomo-anakainizo-gia-neous/",
    "https://www.epirusbank.com/citizens/anavathmizo-to-spiti-mou",
]


def write(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def run_one(index: int, url: str) -> dict:
    write(f"\n[{index}/{len(URLS)}] START {url}")
    started = datetime.now(timezone.utc)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        str(ROOT / "llm_axe" / "va4_product_discoverer.py"),
        url,
        "--no-qa",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = proc.stdout or ""
        write(output[-6000:] if len(output) > 6000 else output)
        status = "ok" if proc.returncode == 0 else "failed"
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        write(f"[{index}/{len(URLS)}] END {status} rc={proc.returncode} seconds={seconds:.1f}")
        return {
            "url": url,
            "status": status,
            "returncode": proc.returncode,
            "seconds": seconds,
        }
    except Exception as exc:
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        write(f"[{index}/{len(URLS)}] END error {type(exc).__name__}: {exc}")
        return {
            "url": url,
            "status": "error",
            "returncode": None,
            "seconds": seconds,
            "error": str(exc),
        }


def main() -> int:
    write(f"Greece completeness batch started {TS}")
    results = []
    for index, url in enumerate(URLS, 1):
        result = run_one(index, url)
        results.append(result)
        SUMMARY_PATH.write_text(
            json.dumps({"started": TS, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write(f"Greece completeness batch summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
