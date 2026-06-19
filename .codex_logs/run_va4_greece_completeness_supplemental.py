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
LOG_PATH = ROOT / ".codex_logs" / f"greece_completeness_va4_supplemental_{TS}.log"
SUMMARY_PATH = ROOT / ".codex_logs" / f"greece_completeness_va4_supplemental_{TS}_summary.json"

URLS = [
    "https://exoikonomo2025.gov.gr/",
    "https://exoikonomoneon.gov.gr/",
    "https://stegasi.gov.gr/programs/anavathmizo-to-spiti-mou/",
    "https://stegasi.gov.gr/programs/exoikonomo-2025/",
    "https://www.gov.gr/ipiresies/periousia-kai-phorologia/diakheirise-akinetes-periousias/exoikonomo-2025",
    "https://www.gov.gr/ipiresies/polites-kai-kathemerinoteta/periballon-kai-poioteta-zoes/photoboltaika-ste-stege",
    "https://greece20.gov.gr/home-loans",
    "https://www.piraeusbank.gr/el/idiwtes/proionta-upiresies/stegastika-daneia/anabathmizo-to-spiti-mou",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/eksoikonomo-2025",
    "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/programma-eksoikonomo-2025",
    "https://www.alpha.gr/el/idiotes/daneia/episkevastika-daneia-gia-anakainisi/programma-eksoikonomo-anakainizo-gia-neous",
    "https://www.crediabank.com/idiotes/daneia/stegastika/programma-anavathmizo-to-spiti-mou/",
    "https://www.crediabank.com/idiotes/daneia/eco-katanalotika/eco-lyseis-energeiaki-anavathmisi-katoikias/",
    "https://www.crediabank.com/idiotes/daneia/stegastika/stegastiko-daneio-eco-home/",
    "https://www.crediabank.com/idiotes/daneia/hrimatodotika-programmata/programma-exoikonomo-anakainizo-gia-neous/",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/prasino-daneio",
    "https://www.nbg.gr/el/idiwtes/daneia/daneia-exoikonomisis-energeias/prasino-daneio-spitiou-h-autokinhtou",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/prasino-daneio-katoikias-eksoikonomisi-energeias",
    "https://www.eurobank.gr/el/retail/proionta-upiresies/proionta/daneia/prasina/green-fast-loan",
    "https://www.alpha.gr/el/idiotes/daneia/prasina-daneia",
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
    write(f"Greece completeness supplemental batch started {TS}")
    results = []
    for index, url in enumerate(URLS, 1):
        result = run_one(index, url)
        results.append(result)
        SUMMARY_PATH.write_text(
            json.dumps({"started": TS, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write(f"Greece completeness supplemental batch summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
