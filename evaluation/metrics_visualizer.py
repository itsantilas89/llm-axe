from __future__ import annotations

import argparse
import json
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


def _compute_qa_consistency_stats(report: dict) -> dict:
    results = report.get("results", []) if isinstance(report, dict) else []
    ok_items = [r for r in results if r.get("status") == "OK"]
    skip_items = [r for r in results if r.get("status") == "SKIP"]

    programs = []
    answer_sources = {}
    for idx, item in enumerate(ok_items, 1):
        name = _safe_name(item.get("programme_name", ""), f"program_{idx}")
        coverage = float(item.get("coverage", 0.0) or 0.0)
        consistency = float(item.get("consistency_score", 0.0) or 0.0)
        programs.append(
            {
                "programme_name": name,
                "coverage": max(0.0, min(1.0, coverage)),
                "consistency_score": max(0.0, min(1.0, consistency)),
                "questions_tested": int(item.get("questions_tested", 0) or 0),
                "questions_answered": int(item.get("questions_answered", 0) or 0),
            }
        )

        for detail in item.get("details", []):
            src = _safe_name(detail.get("answer_source", "unknown"), "unknown")
            answer_sources[src] = answer_sources.get(src, 0) + 1

    avg_coverage = mean([p["coverage"] for p in programs]) if programs else 0.0
    avg_consistency = mean([p["consistency_score"] for p in programs]) if programs else 0.0

    return {
        "ok_count": len(ok_items),
        "skip_count": len(skip_items),
        "avg_coverage": avg_coverage,
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


def _plot_kpi_overview(output_dir: Path, dpi: int, qa: dict, sem: dict, html: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ["QA Coverage", "QA Consistency", "HTML Coverage"]
    values = [qa["avg_coverage"], qa["avg_consistency"], html["coverage"]]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    if sem.get("available"):
        token_f1 = float(sem["global_metrics"].get("token_f1", 0.0) or 0.0)
        labels.append("Semantic Token-F1")
        values.append(token_f1)
        colors.append("#9467bd")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("KPI Overview")
    ax.set_ylabel("Score (0-1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "kpi_overview.png", dpi=dpi)
    plt.close(fig)


def _plot_program_scores(output_dir: Path, dpi: int, qa: dict, top_n: int) -> None:
    import matplotlib.pyplot as plt

    programs = sorted(qa["programs"], key=lambda p: p["consistency_score"], reverse=True)
    if top_n > 0:
        programs = programs[:top_n]
    if not programs:
        return

    names = [p["programme_name"] for p in programs]
    coverage = [p["coverage"] for p in programs]
    consistency = [p["consistency_score"] for p in programs]

    x = list(range(len(names)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.25), 5.5))
    ax.bar([i - width / 2 for i in x], coverage, width=width, label="Coverage", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], consistency, width=width, label="Consistency", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("QA Coverage vs Consistency by Program")
    ax.set_ylabel("Score (0-1)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "qa_program_scores.png", dpi=dpi)
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
    ax.set_title("Answer Source Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Source")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, val in enumerate(values):
        ax.text(i, val + 0.1, str(val), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "qa_answer_sources.png", dpi=dpi)
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

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#9467bd")
    ax.set_ylim(0, 1)
    ax.set_title("Semantic Global Metrics")
    ax.set_ylabel("Score (0-1)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "semantic_global_metrics.png", dpi=dpi)
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
    ax.set_ylim(0, 1)
    ax.set_title("Semantic Metrics by Question")
    ax.set_ylabel("Score (0-1)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "semantic_by_question.png", dpi=dpi)
    plt.close(fig)


def _plot_html_found_missing(output_dir: Path, dpi: int, html: dict) -> None:
    import matplotlib.pyplot as plt

    if not html.get("available") or html["checked"] == 0:
        return

    labels = ["Found", "Missing"]
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
    ax.set_title("HTML Validation: Found vs Missing")

    fig.tight_layout()
    fig.savefig(output_dir / "html_found_vs_missing.png", dpi=dpi)
    plt.close(fig)


def generate_visuals(
    qa_consistency_report: Path,
    qa_semantic_report: Path,
    html_report: Path,
    output_dir: Path,
    dpi: int,
    top_n: int,
) -> int:
    qa_raw = _load_json(qa_consistency_report)
    if qa_raw is None:
        print(f"ERROR: QA consistency report not found or invalid: {qa_consistency_report}")
        return 1

    sem_raw = _load_json(qa_semantic_report)
    html_raw = _load_json(html_report)

    qa = _compute_qa_consistency_stats(qa_raw)
    sem = _compute_semantic_stats(sem_raw)
    html = _compute_html_stats(html_raw)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception as exc:
        print("ERROR: matplotlib is required for Flow F visualizations.")
        print(f"Install it with: .\\.venv\\Scripts\\python.exe -m pip install matplotlib\nDetails: {exc}")
        return 1

    _plot_kpi_overview(output_dir, dpi, qa, sem, html)
    _plot_program_scores(output_dir, dpi, qa, top_n)
    _plot_answer_sources(output_dir, dpi, qa)
    _plot_semantic_global(output_dir, dpi, sem)
    _plot_semantic_by_question(output_dir, dpi, sem)
    _plot_html_found_missing(output_dir, dpi, html)

    summary = {
        "flow": "F_visual_analytics",
        "inputs": {
            "qa_consistency_report": str(qa_consistency_report),
            "qa_semantic_report": str(qa_semantic_report),
            "html_report": str(html_report),
        },
        "kpis": {
            "qa_avg_coverage": qa["avg_coverage"],
            "qa_avg_consistency": qa["avg_consistency"],
            "qa_validated_programs": qa["ok_count"],
            "qa_skipped_programs": qa["skip_count"],
            "html_coverage": html["coverage"],
            "semantic_items_scored": sem["items_scored"],
            "semantic_available": sem["available"],
        },
        "charts": [
            "kpi_overview.png",
            "qa_program_scores.png",
            "qa_answer_sources.png",
            "semantic_global_metrics.png",
            "semantic_by_question.png",
            "html_found_vs_missing.png",
        ],
    }

    with (output_dir / "visual_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    # By default export all generated charts into a single PDF for convenience.
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
    print(f"QA avg coverage: {qa['avg_coverage']:.2%}")
    print(f"QA avg consistency: {qa['avg_consistency']:.2%}")
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
        html_report=Path(args.html_report),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    raise SystemExit(main())
