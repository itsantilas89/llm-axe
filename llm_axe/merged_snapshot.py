"""Shared helper for refreshing the canonical merged program snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def refresh_merged_snapshot(
    root_dir: str | Path | None = None,
    *,
    prefix: str = "merged",
    log: Callable[[str], None] | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Create a prepare-only merged snapshot from the latest VA4 outputs."""
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
    run_id = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir = root / "output" / "evaluation" / "qa_full_runs" / run_id
    cmd = [
        sys.executable,
        str(root / "evaluation" / "run_qa_evaluation_by_program.py"),
        "--run-id",
        run_id,
        "--prepare-only",
        "--force-prepare",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    emit(f"Ανανέωση merged snapshot: {run_id}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "run_id": run_id,
            "path": str(run_dir),
            "message": f"{type(exc).__name__}: {exc}",
        }

    output = (completed.stdout or "").strip()
    if output:
        for line in output.splitlines()[-20:]:
            emit(f"merge: {line}")

    status = "ok" if completed.returncode == 0 else "failed"
    result: dict[str, Any] = {
        "status": status,
        "run_id": run_id,
        "path": str(run_dir),
        "classification_dir": str(run_dir / "classifications"),
        "returncode": completed.returncode,
        "message": "merged snapshot refreshed" if status == "ok" else output[-1000:],
    }

    manifest_path = run_dir / "target_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            result["target_count"] = manifest.get("target_count")
            result["source_url_count"] = manifest.get("source_url_count")
    except Exception:
        pass
    return result
