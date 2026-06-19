from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _safe_name(value: str, fallback: str) -> str:
    value = (value or "").strip()
    return value if value else fallback


def _dedupe_labels(labels: list[str]) -> list[str]:
    counts = Counter(labels)
    seen: dict[str, int] = {}
    output = []
    for label in labels:
        if counts[label] == 1:
            output.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        output.append(f"{label} ({seen[label]})")
    return output


def _compute_qa_consistency_stats(report: dict) -> dict:
    results = report.get("results", []) if isinstance(report, dict) else []
    ok_items = [r for r in results if r.get("status") == "OK"]
    skip_items = [r for r in results if r.get("status") == "SKIP"]

    programs = []
    answer_sources = {}
    for idx, item in enumerate(ok_items, 1):
        name = _safe_name(item.get("programme_name", ""), f"program_{idx}")
        consistency = float(item.get("consistency_score", 0.0) or 0.0)
        programs.append(
            {
                "programme_name": name,
                "consistency_score": max(0.0, min(1.0, consistency)),
                "questions_tested": int(item.get("questions_tested", 0) or 0),
                "questions_answered": int(item.get("questions_answered", 0) or 0),
                "consistency_applicable": int(item.get("consistency_applicable", 0) or 0),
            }
        )

        for detail in item.get("details", []):
            src = _safe_name(detail.get("answer_source", "unknown"), "unknown")
            answer_sources[src] = answer_sources.get(src, 0) + 1

    avg_consistency = mean([p["consistency_score"] for p in programs]) if programs else 0.0

    return {
        "ok_count": len(ok_items),
        "skip_count": len(skip_items),
        "avg_consistency": avg_consistency,
        "programs": programs,
        "answer_sources": answer_sources,
    }


def _compute_semantic_stats(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {
            "available": False,
            "items_scored": 0,
            "global_metrics": {},
            "question_summary": {},
        }

    sem = report.get("semantic_validation", {})
    if not isinstance(sem, dict):
        return {
            "available": False,
            "items_scored": 0,
            "global_metrics": {},
            "question_summary": {},
        }

    global_metrics = sem.get("global_metrics", {})
    question_summary = sem.get("question_summary", {})

    return {
        "available": True,
        "items_scored": int(sem.get("items_scored", 0) or 0),
        "global_metrics": global_metrics if isinstance(global_metrics, dict) else {},
        "question_summary": question_summary if isinstance(question_summary, dict) else {},
    }


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _compute_repeatability_stats(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {
            "available": False,
            "groups_compared": 0,
            "avg_token_f1": 0.0,
            "avg_jaccard": 0.0,
            "by_program": [],
            "by_question": [],
        }

    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    by_program = report.get("by_program", [])
    by_question = report.get("by_question", [])

    return {
        "available": True,
        "groups_compared": int(summary.get("groups_compared", 0) or 0),
        "avg_token_f1": _bounded_score(float(summary.get("avg_token_f1", 0.0) or 0.0)),
        "avg_jaccard": _bounded_score(float(summary.get("avg_jaccard", 0.0) or 0.0)),
        "by_program": by_program if isinstance(by_program, list) else [],
        "by_question": by_question if isinstance(by_question, list) else [],
    }


def _compute_html_stats(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {"available": False, "coverage": 0.0, "found": 0, "missing": 0, "checked": 0}

    summary = report.get("summary", {})
    overall = summary.get("overall_coverage", {}) if isinstance(summary, dict) else {}
    found = int(overall.get("found", 0) or 0)
    missing = int(overall.get("missing", 0) or 0)
    checked = int(overall.get("total_checked", 0) or 0)
    coverage = (found / checked) if checked > 0 else 0.0

    return {
        "available": True,
        "coverage": coverage,
        "found": found,
        "missing": missing,
        "checked": checked,
    }


def _plot_kpi_overview(output_dir: Path, dpi: int, qa: dict, sem: dict, html: dict, repeat: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ["Answer-data agreement"]
    values = [qa["avg_consistency"]]
    colors = ["#2ca02c"]

    if repeat.get("available") and repeat.get("groups_compared", 0) > 0:
        labels.append("Same-question repeatability")
        values.append(repeat["avg_token_f1"])
        colors.append("#1f77b4")

    if sem.get("available"):
        token_f1 = float(sem["global_metrics"].get("token_f1", 0.0) or 0.0)
        labels.append("Answer/reference semantic F1")
        values.append(token_f1)
        colors.append("#9467bd")

    if html.get("available"):
        labels.append("HTML evidence found rate")
        values.append(html["coverage"])
        colors.append("#ff7f0e")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Evaluation Scores")
    ax.set_ylabel("Score (0-1, higher is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "overall_quality_scores.png", dpi=dpi)
    plt.close(fig)


def _plot_program_scores(output_dir: Path, dpi: int, qa: dict, top_n: int) -> None:
    import matplotlib.pyplot as plt

    programs = sorted(qa["programs"], key=lambda p: p["consistency_score"], reverse=True)
    if top_n > 0:
        programs = programs[:top_n]
    if not programs:
        return

    names = _dedupe_labels([p["programme_name"] for p in programs])
    consistency = [p["consistency_score"] for p in programs]

    x = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.25), 5.5))
    bars = ax.bar(x, consistency, width=0.62, label="Answer-data agreement", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer-Data Consistency by Program")
    ax.set_ylabel("Consistency score (0-1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, consistency):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "program_answer_data_consistency.png", dpi=dpi)
    plt.close(fig)


def _plot_answer_sources(output_dir: Path, dpi: int, qa: dict) -> None:
    import matplotlib.pyplot as plt

    if not qa["answer_sources"]:
        return

    items = sorted(qa["answer_sources"].items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#17becf")
    ax.set_title("Stored QA Answer Sources")
    ax.set_ylabel("Answers")
    ax.set_xlabel("Source")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, val in enumerate(values):
        ax.text(i, val + 0.1, str(val), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "answer_source_counts.png", dpi=dpi)
    plt.close(fig)


def _plot_semantic_global(output_dir: Path, dpi: int, sem: dict) -> None:
    import matplotlib.pyplot as plt

    if not sem.get("available"):
        return

    metrics = sem.get("global_metrics", {})
    selected = [
        "bleu1",
        "bleu2",
        "bleu4",
        "token_f1",
        "jaccard",
        "bertscore_f1",
    ]
    labels = [k for k in selected if k in metrics]
    values = [float(metrics[k] or 0.0) for k in labels]
    display_labels = {
        "bleu1": "BLEU-1",
        "bleu2": "BLEU-2",
        "bleu4": "BLEU-4",
        "token_f1": "Token-F1",
        "jaccard": "Jaccard",
        "bertscore_f1": "BERTScore F1",
    }

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([display_labels.get(label, label) for label in labels], values, color="#9467bd")
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer vs Reference Semantic Similarity")
    ax.set_ylabel("Similarity score (0-1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "semantic_similarity_overall.png", dpi=dpi)
    plt.close(fig)


def _plot_semantic_by_question(output_dir: Path, dpi: int, sem: dict) -> None:
    import matplotlib.pyplot as plt

    if not sem.get("available"):
        return

    qsum = sem.get("question_summary", {})
    if not qsum:
        return

    qids = list(qsum.keys())
    token_f1 = [float(qsum[q].get("token_f1", 0.0) or 0.0) for q in qids]
    bleu4 = [float(qsum[q].get("bleu4", 0.0) or 0.0) for q in qids]

    x = list(range(len(qids)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(qids) * 1.2), 5.5))
    ax.bar([i - width / 2 for i in x], token_f1, width=width, label="Token-F1", color="#8c564b")
    ax.bar([i + width / 2 for i in x], bleu4, width=width, label="BLEU-4", color="#e377c2")

    ax.set_xticks(x)
    ax.set_xticklabels(qids, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer vs Reference Similarity by Question")
    ax.set_ylabel("Similarity score (0-1)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "semantic_similarity_by_question.png", dpi=dpi)
    plt.close(fig)


def _plot_repeatability_by_question(output_dir: Path, dpi: int, repeat: dict, top_n: int) -> None:
    import matplotlib.pyplot as plt

    if not repeat.get("available") or repeat.get("groups_compared", 0) == 0:
        return

    items = repeat.get("by_question", [])
    if not items:
        return

    items = sorted(items, key=lambda item: float(item.get("avg_token_f1", 0.0) or 0.0))
    if top_n > 0:
        items = items[:top_n]

    labels = [_safe_name(item.get("question_id", ""), f"q{i + 1}") for i, item in enumerate(items)]
    token_f1 = [float(item.get("avg_token_f1", 0.0) or 0.0) for item in items]

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), 5.5))
    bars = ax.bar(x, token_f1, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Same-Question Repeatability by Question")
    ax.set_ylabel("Average pairwise Token-F1 (0-1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, token_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "same_question_repeatability.png", dpi=dpi)
    plt.close(fig)


def _plot_html_found_missing(output_dir: Path, dpi: int, html: dict) -> None:
    import matplotlib.pyplot as plt

    if not html.get("available") or html["checked"] == 0:
        return

    labels = ["Evidence found", "Evidence missing"]
    values = [html["found"], html["missing"]]
    colors = ["#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.45},
    )
    for text in texts + autotexts:
        text.set_fontsize(10)
    ax.set_title("HTML Evidence Check: Found vs Missing")

    fig.tight_layout()
    fig.savefig(output_dir / "html_evidence_found_missing.png", dpi=dpi)
    plt.close(fig)


def generate_visuals(
    qa_consistency_report: Path,
    qa_semantic_report: Path,
    qa_repeatability_report: Path,
    html_report: Path,
    output_dir: Path,
    dpi: int,
    top_n: int,
    export_pdf: bool = True,
) -> int:
    qa_raw = _load_json(qa_consistency_report)
    if qa_raw is None:
        print(f"ERROR: QA consistency report not found or invalid: {qa_consistency_report}")
        return 1

    sem_raw = _load_json(qa_semantic_report)
    repeat_raw = _load_json(qa_repeatability_report)
    html_raw = _load_json(html_report)

    qa = _compute_qa_consistency_stats(qa_raw)
    sem = _compute_semantic_stats(sem_raw)
    repeat = _compute_repeatability_stats(repeat_raw)
    html = _compute_html_stats(html_raw)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception as exc:
        print("ERROR: matplotlib is required for Flow F visualizations.")
        print(f"Install it with: .\\.venv\\Scripts\\python.exe -m pip install matplotlib\nDetails: {exc}")
        return 1

    _plot_kpi_overview(output_dir, dpi, qa, sem, html, repeat)
    _plot_program_scores(output_dir, dpi, qa, top_n)
    _plot_answer_sources(output_dir, dpi, qa)
    _plot_semantic_global(output_dir, dpi, sem)
    _plot_semantic_by_question(output_dir, dpi, sem)
    _plot_repeatability_by_question(output_dir, dpi, repeat, top_n)
    _plot_html_found_missing(output_dir, dpi, html)

    chart_names = [
        "overall_quality_scores.png",
        "program_answer_data_consistency.png",
        "answer_source_counts.png",
        "semantic_similarity_overall.png",
        "semantic_similarity_by_question.png",
        "same_question_repeatability.png",
        "html_evidence_found_missing.png",
    ]
    charts = [name for name in chart_names if (output_dir / name).exists()]

    summary = {
        "flow": "F_visual_analytics",
        "inputs": {
            "qa_consistency_report": str(qa_consistency_report),
            "qa_semantic_report": str(qa_semantic_report),
            "qa_repeatability_report": str(qa_repeatability_report),
            "html_report": str(html_report),
        },
        "kpis": {
            "qa_answer_data_consistency": qa["avg_consistency"],
            "qa_validated_programs": qa["ok_count"],
            "qa_skipped_programs": qa["skip_count"],
            "qa_repeatability_available": repeat["available"],
            "qa_repeatability_groups_compared": repeat["groups_compared"],
            "qa_repeatability_avg_token_f1": repeat["avg_token_f1"],
            "qa_repeatability_avg_jaccard": repeat["avg_jaccard"],
            "html_evidence_found_rate": html["coverage"],
            "semantic_items_scored": sem["items_scored"],
            "semantic_available": sem["available"],
        },
        "charts": charts,
    }

    with (output_dir / "visual_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if export_pdf:
        try:
            import img2pdf

            chart_paths = []
            for name in summary.get("charts", []):
                p = output_dir / name
                if p.exists():
                    chart_paths.append(str(p))

            if chart_paths:
                out_pdf = output_dir / "evaluation_plots.pdf"
                with open(out_pdf, "wb") as f:
                    f.write(img2pdf.convert(chart_paths))
                print(f"Combined PDF saved: {out_pdf}")
        except Exception:
            # If img2pdf isn't available or conversion fails, continue without failing.
            pass

    print("Flow F complete: visual analytics generated")
    print(f"Output directory: {output_dir}")
    print(f"QA answer-data consistency: {qa['avg_consistency']:.2%}")
    if repeat["available"] and repeat["groups_compared"] > 0:
        print(
            "QA same-question repeatability: "
            f"{repeat['avg_token_f1']:.2%} Token-F1 across {repeat['groups_compared']} groups"
        )
    elif repeat["available"]:
        print("Repeatability report found, but no repeated URL/question groups were comparable")
    elif not repeat["available"]:
        print("Repeatability report not found: same-question repeatability chart was skipped")
    if sem["available"]:
        print(f"Semantic items scored: {sem['items_scored']}")
    else:
        print("Semantic report not found: semantic charts were skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flow F: convert evaluation metrics into visual charts."
    )
    parser.add_argument(
        "--qa-consistency-report",
        default="output/evaluation/qa_consistency_report.json",
        help="Path to qa_consistency report JSON.",
    )
    parser.add_argument(
        "--qa-semantic-report",
        default="output/evaluation/qa_semantic_report.json",
        help="Path to qa_semantic report JSON (optional; charts skipped if missing).",
    )
    parser.add_argument(
        "--qa-repeatability-report",
        default="output/evaluation/qa_repeatability_report.json",
        help="Path to qa_repeatability report JSON (optional; chart skipped if missing).",
    )
    parser.add_argument(
        "--html-report",
        default="output/evaluation/html_json_report.json",
        help="Path to HTML validation report JSON (optional; charts skipped if missing).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/evaluation/plots",
        help="Directory for generated PNG charts.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Image DPI for charts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Top N programs to show in per-program chart (0 = all).",
    )
    parser.add_argument(
        "--no-export-pdf",
        action="store_true",
        help="Disable automatic export of generated charts into a single PDF (enabled by default).",
    )

    args = parser.parse_args()

    return generate_visuals(
        qa_consistency_report=Path(args.qa_consistency_report),
        qa_semantic_report=Path(args.qa_semantic_report),
        qa_repeatability_report=Path(args.qa_repeatability_report),
        html_report=Path(args.html_report),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        top_n=args.top_n,
        export_pdf=not args.no_export_pdf,
    )


if __name__ == "__main__":
    raise SystemExit(main())
