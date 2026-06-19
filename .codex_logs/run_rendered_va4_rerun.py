from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = ROOT / ".codex_logs" / f"rendered_va4_rerun_{TS}.log"
SUMMARY_PATH = ROOT / ".codex_logs" / f"rendered_va4_rerun_{TS}_summary.json"

URLS = [
    "https://stegasi.gov.gr/programs/anavathmizo-to-spiti-mou/",
    "https://stegasi.gov.gr/programs/exoikonomo-2025/",
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
    write(output[-8000:] if len(output) > 8000 else output)
    status = "ok" if proc.returncode == 0 else "failed"
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    write(f"[{index}/{len(URLS)}] END {status} rc={proc.returncode} seconds={seconds:.1f}")
    return {
        "url": url,
        "status": status,
        "returncode": proc.returncode,
        "seconds": seconds,
    }


def main() -> int:
    write(f"Rendered rerun started {TS}")
    results = []
    for index, url in enumerate(URLS, 1):
        result = run_one(index, url)
        results.append(result)
        SUMMARY_PATH.write_text(
            json.dumps({"started": TS, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write(f"Rendered rerun summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
