from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_axe.models import OllamaChat
from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash

from evaluation.run_qa_evaluation_suite import (
    DEFAULT_MODEL,
    aggregate_qa_files,
    build_html_report,
    build_program_targets,
    load_latest_relevant_sources,
    load_questions,
    log,
    qa_payload_path,
    run_command,
    run_qa_mode,
    write_program_classifications,
    write_source_classifications,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def acquire_run_lock(run_dir: Path, force: bool) -> Path:
    lock_path = run_dir / "orchestrator.lock"
    if lock_path.exists():
        if force:
            lock_path.unlink()
        else:
            raise RuntimeError(
                f"Run lock exists: {lock_path}. Another orchestrator may already be running. "
                "If this is stale, rerun with --force-run-lock."
            )
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "started_at": now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        os.write(fd, payload)
    finally:
        os.close(fd)
    return lock_path


def target_from_classification(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extracted = payload.get("extracted_data") or {}
    url = str(payload.get("url") or extracted.get("source_url") or "")
    programme_name = str(extracted.get("programme_name") or path.stem)
    return {
        "timestamp": datetime.now(timezone.utc),
        "path": path,
        "payload": payload,
        "url": url,
        "programme_name": programme_name,
        "copied_path": path,
    }


def load_materialized_targets(run_dir: Path) -> list[dict[str, Any]]:
    classification_dir = run_dir / "classifications"
    return [target_from_classification(path) for path in sorted(classification_dir.glob("*_classification.json"))]


def program_tag(index: int, target: dict[str, Any]) -> str:
    url = target.get("url") or target.get("programme_name") or f"program_{index}"
    return f"{index:03d}_{_make_safe_name(str(url))}_{_short_hash(str(url))}"


def program_dir(run_dir: Path, index: int, target: dict[str, Any]) -> Path:
    return run_dir / "program_runs" / program_tag(index, target)


def response_count(path: Path) -> int:
    payload = read_json(path, {})
    responses = payload.get("responses") if isinstance(payload, dict) else None
    return len(responses) if isinstance(responses, list) else 0


def program_expected_counts(repeat_count: int) -> tuple[int, int]:
    return 15, 11 * repeat_count


def program_complete(target: dict[str, Any], run_dir: Path, repeat_count: int) -> bool:
    all15_expected, repeat_expected = program_expected_counts(repeat_count)
    all15_path = qa_payload_path(run_dir / "qa_all15", str(target.get("url") or ""), "all15")
    repeat_path = qa_payload_path(run_dir / "qa_repeat_q1_11_x5", str(target.get("url") or ""), "repeat_q1_11_x5")
    return response_count(all15_path) >= all15_expected and response_count(repeat_path) >= repeat_expected


def reset_program_outputs(target: dict[str, Any], run_dir: Path) -> None:
    url = str(target.get("url") or "")
    for output_dir, mode in (
        (run_dir / "qa_all15", "all15"),
        (run_dir / "qa_repeat_q1_11_x5", "repeat_q1_11_x5"),
    ):
        path = qa_payload_path(output_dir, url, mode)
        if path.exists():
            path.unlink()


def build_manifest(args: argparse.Namespace, run_id: str, source_count: int, targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": now_iso(),
        "runner": "per_program",
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
            "all15": 15,
            "repeat_questions": 11,
            "repeat_count": args.repeat_count,
        },
        "llm": {
            "model": args.llm_model,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "generation_mode": "per_program_full_pruned_json",
            "all15_batch_size": args.batch_size,
            "repeat_batch_size": args.repeat_batch_size,
            "prompt_context": "full_pruned_json_empty_fields_removed_no_string_truncation",
        },
        "source_url_count": source_count,
        "target_count": len(targets),
        "targets": [
            {
                "index": index,
                "programme_name": item["programme_name"],
                "url": item["url"],
                "source_path": str(item["path"]),
                "copied_path": str(item["copied_path"]),
                "merged_from_count": item["payload"].get("merged_from_count", 1),
                "source_urls": item["payload"].get("source_urls", [item["url"]]),
            }
            for index, item in enumerate(targets, 1)
        ],
    }


def prepare_run(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "target_manifest.json"
    if manifest_path.exists() and not args.force_prepare:
        targets = load_materialized_targets(run_dir)
        if targets:
            log(run_dir, f"Using existing prepared targets: {len(targets)}")
            return targets

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
    manifest = build_manifest(args, run_dir.name, len(source_classifications), copied_targets)
    write_json(manifest_path, manifest)
    log(run_dir, f"Prepared targets: {len(copied_targets)}")
    return copied_targets


def run_program_worker(args: argparse.Namespace, run_dir: Path, index: int) -> int:
    questions = load_questions(ROOT / args.questions_file)
    all15_questions = questions[:15]
    repeat_questions = questions[:11]
    targets = load_materialized_targets(run_dir)
    if index < 1 or index > len(targets):
        raise ValueError(f"Program index out of range: {index}; targets={len(targets)}")

    target = targets[index - 1]
    if args.reset_program_output:
        reset_program_outputs(target, run_dir)
    pdir = program_dir(run_dir, index, target)
    pdir.mkdir(parents=True, exist_ok=True)
    status_path = pdir / "status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "started_at": now_iso(),
            "program_index": index,
            "programme_name": target.get("programme_name", ""),
            "url": target.get("url", ""),
        },
    )
    log(pdir, f"START program {index}/{len(targets)}: {target.get('programme_name', '')}")
    try:
        llm = OllamaChat(model=args.llm_model)
        run_qa_mode(
            run_dir=pdir,
            targets=[target],
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
            run_dir=pdir,
            targets=[target],
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
        write_json(
            status_path,
            {
                "status": "completed",
                "started_at": read_json(status_path, {}).get("started_at", ""),
                "finished_at": now_iso(),
                "program_index": index,
                "programme_name": target.get("programme_name", ""),
                "url": target.get("url", ""),
            },
        )
        log(pdir, "DONE")
        return 0
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "started_at": read_json(status_path, {}).get("started_at", ""),
                "finished_at": now_iso(),
                "program_index": index,
                "programme_name": target.get("programme_name", ""),
                "url": target.get("url", ""),
                "error": str(exc),
            },
        )
        log(pdir, f"FAILED {exc}")
        return 1


def finalize_run(args: argparse.Namespace, run_dir: Path) -> int:
    manifest = read_json(run_dir / "target_manifest.json", {})
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

    if args.skip_qa_metrics:
        report_path = build_html_report(run_dir, manifest)
        log(run_dir, f"HTML report: {report_path}")
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


def run_orchestrator(args: argparse.Namespace, run_dir: Path) -> int:
    targets = prepare_run(args, run_dir)
    if not targets:
        log(run_dir, "No targets found.")
        return 1

    python = sys.executable
    script = Path(__file__).resolve()
    failures = []
    skipped = 0
    for index, target in enumerate(load_materialized_targets(run_dir), 1):
        pdir = program_dir(run_dir, index, target)
        pdir.mkdir(parents=True, exist_ok=True)
        if not args.force_program and program_complete(target, run_dir, args.repeat_count):
            skipped += 1
            log(run_dir, f"SKIP completed program {index}: {target.get('programme_name', '')}")
            continue

        cmd = [
            python,
            str(script),
            "--run-id",
            run_dir.name,
            "--output-root",
            args.output_root,
            "--classification-dir",
            args.classification_dir,
            "--scraped-dir",
            args.scraped_dir,
            "--questions-file",
            args.questions_file,
            "--worker-program-index",
            str(index),
            "--llm-model",
            args.llm_model,
            "--temperature",
            str(args.temperature),
            "--num-ctx",
            str(args.num_ctx),
            "--num-predict",
            str(args.num_predict),
            "--repeat-count",
            str(args.repeat_count),
            "--batch-size",
            str(args.batch_size),
            "--repeat-batch-size",
            str(args.repeat_batch_size),
        ]
        if args.reset_program_output:
            cmd.append("--reset-program-output")
        log(run_dir, f"RUN program {index}/{len(targets)}: {target.get('programme_name', '')}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        stdout_path = pdir / "worker.out.log"
        stderr_path = pdir / "worker.err.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=args.program_timeout_seconds if args.program_timeout_seconds > 0 else None,
                )
            except subprocess.TimeoutExpired:
                failures.append({"program_index": index, "status": "timeout"})
                write_json(
                    pdir / "status.json",
                    {
                        "status": "timeout",
                        "finished_at": now_iso(),
                        "program_index": index,
                        "programme_name": target.get("programme_name", ""),
                        "url": target.get("url", ""),
                    },
                )
                log(run_dir, f"TIMEOUT program {index}")
                continue
        if completed.returncode != 0:
            failures.append({"program_index": index, "status": "failed", "returncode": completed.returncode})
            log(run_dir, f"FAILED program {index} exit={completed.returncode}")

    write_json(
        run_dir / "program_run_summary.json",
        {
            "timestamp": now_iso(),
            "target_count": len(targets),
            "skipped_completed": skipped,
            "failures": failures,
        },
    )
    if failures and not args.finalize_with_failures:
        log(run_dir, f"Stopping before finalize because {len(failures)} program(s) failed/timed out.")
        return 1
    return finalize_run(args, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA evaluation as isolated per-program workers.")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat-batch-size", type=int, default=1)
    parser.add_argument("--max-programs", type=int, default=0)
    parser.add_argument("--program-timeout-seconds", type=int, default=0)
    parser.add_argument("--no-dedupe-title", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-program", action="store_true")
    parser.add_argument("--force-run-lock", action="store_true")
    parser.add_argument("--reset-program-output", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--finalize-with-failures", action="store_true")
    parser.add_argument("--skip-qa-metrics", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--worker-program-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"per_program_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_program_index:
        return run_program_worker(args, run_dir, args.worker_program_index)
    if args.finalize_only:
        return finalize_run(args, run_dir)

    lock_path = acquire_run_lock(run_dir, force=args.force_run_lock)
    try:
        if args.prepare_only:
            prepare_run(args, run_dir)
            log(run_dir, "Prepared only.")
            return 0
        return run_orchestrator(args, run_dir)
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
