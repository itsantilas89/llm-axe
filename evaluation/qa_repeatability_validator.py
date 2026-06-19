"""Same-question repeatability validator for stored QA responses.

This script compares answers produced for the same URL and same question across
multiple QA runs, or across repeated asks inside one QA run. It measures how
similar the model's answers are when the prompt is repeated.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text.strip().lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def normalize_question(question: str) -> str:
    question = re.sub(r"^\s*\d+\s*[\).:-]\s*", "", question or "")
    question = normalize_text(question)
    question = re.sub(r"[^\w\s]", " ", question, flags=re.UNICODE)
    return re.sub(r"\s+", " ", question).strip()


def base_question_id(question_id: str, fallback: str) -> str:
    question_id = (question_id or "").strip()
    match = re.match(r"^(q\d+)", question_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return question_id or fallback


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def token_overlap_scores(a: str, b: str) -> tuple[float, float, float, float]:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0, 0.0, 0.0, 0.0

    a_counter = Counter(a_tokens)
    b_counter = Counter(b_tokens)
    overlap = sum(min(count, b_counter.get(token, 0)) for token, count in a_counter.items())

    precision = overlap / max(1, sum(a_counter.values()))
    recall = overlap / max(1, sum(b_counter.values()))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    a_set = set(a_tokens)
    b_set = set(b_tokens)
    union = a_set | b_set
    jaccard = 0.0 if not union else len(a_set & b_set) / len(union)
    return precision, recall, f1, jaccard


def answer_pair_metrics(a: str, b: str) -> dict[str, float | bool]:
    precision, recall, f1, jaccard = token_overlap_scores(a, b)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "jaccard": jaccard,
        "exact_match": normalize_text(a) == normalize_text(b),
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_qa_records(qa_responses_dir: Path) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    files_scanned = 0
    file_errors = 0

    for qa_file in sorted(qa_responses_dir.glob("*_qa_responses.json")):
        files_scanned += 1
        try:
            payload = json.loads(qa_file.read_text(encoding="utf-8"))
        except Exception:
            file_errors += 1
            continue

        url = payload.get("url", "")
        programme_name = payload.get("programme_name", "")
        file_timestamp = payload.get("timestamp", "")

        for idx, response in enumerate(payload.get("responses", []), 1):
            question = str(response.get("question", "") or "").strip()
            answer = str(response.get("answer", "") or "").strip()
            if not question or not answer:
                continue

            records.append(
                {
                    "url": url,
                    "programme_name": programme_name,
                    "source_file": qa_file.name,
                    "file_timestamp": file_timestamp,
                    "question_id": base_question_id(str(response.get("question_id", "")), f"q{idx}"),
                    "question": question,
                    "question_key": normalize_question(question),
                    "answer": answer,
                    "answer_timestamp": response.get("timestamp", ""),
                    "repeat_index": response.get("repeat_index"),
                    "repeat_count": response.get("repeat_count"),
                }
            )

    return records, files_scanned, file_errors


def build_repeatability_report(
    qa_responses_dir: Path,
    min_repeats: int,
    stable_threshold: float,
    summary_only: bool,
) -> dict:
    records, files_scanned, file_errors = load_qa_records(qa_responses_dir)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["url"], record["question_key"])].append(record)

    details = []
    for group in grouped.values():
        if len(group) < min_repeats:
            continue

        pair_records = []
        for left, right in combinations(group, 2):
            metrics = answer_pair_metrics(left["answer"], right["answer"])
            pair_records.append(
                {
                    "left_source_file": left["source_file"],
                    "right_source_file": right["source_file"],
                    "left_question_id": left["question_id"],
                    "right_question_id": right["question_id"],
                    "metrics": metrics,
                }
            )

        if not pair_records:
            continue

        avg_token_f1 = mean([_safe_float(p["metrics"]["token_f1"]) for p in pair_records])
        avg_jaccard = mean([_safe_float(p["metrics"]["jaccard"]) for p in pair_records])
        exact_match_rate = mean([1.0 if p["metrics"]["exact_match"] else 0.0 for p in pair_records])
        representative = group[0]
        unique_answers = {normalize_text(item["answer"]) for item in group}

        details.append(
            {
                "url": representative["url"],
                "programme_name": representative["programme_name"],
                "question_id": representative["question_id"],
                "question": representative["question"],
                "answer_count": len(group),
                "pair_count": len(pair_records),
                "avg_token_f1": avg_token_f1,
                "avg_jaccard": avg_jaccard,
                "exact_match_rate": exact_match_rate,
                "unique_answer_count": len(unique_answers),
                "is_stable": avg_token_f1 >= stable_threshold,
                "source_files": sorted({item["source_file"] for item in group}),
                "answer_previews": [] if summary_only else [item["answer"][:240] for item in group],
                "pair_metrics": [] if summary_only else pair_records,
            }
        )

    details.sort(key=lambda item: item["avg_token_f1"])

    by_program = []
    program_groups: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        program_groups[item["url"]].append(item)

    for url, items in program_groups.items():
        by_program.append(
            {
                "url": url,
                "programme_name": items[0].get("programme_name", ""),
                "groups_compared": len(items),
                "answers_compared": sum(int(item["answer_count"]) for item in items),
                "avg_token_f1": mean([float(item["avg_token_f1"]) for item in items]),
                "avg_jaccard": mean([float(item["avg_jaccard"]) for item in items]),
                "stable_groups": sum(1 for item in items if item["is_stable"]),
            }
        )

    by_program.sort(key=lambda item: item["avg_token_f1"])

    question_groups: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        question_groups[normalize_question(item["question"])].append(item)

    by_question = []
    for _, items in question_groups.items():
        by_question.append(
            {
                "question_id": items[0].get("question_id", ""),
                "question": items[0].get("question", ""),
                "groups_compared": len(items),
                "answers_compared": sum(int(item["answer_count"]) for item in items),
                "avg_token_f1": mean([float(item["avg_token_f1"]) for item in items]),
                "avg_jaccard": mean([float(item["avg_jaccard"]) for item in items]),
                "stable_groups": sum(1 for item in items if item["is_stable"]),
            }
        )

    by_question.sort(key=lambda item: item["avg_token_f1"])

    groups_compared = len(details)
    summary = {
        "qa_files_scanned": files_scanned,
        "qa_file_errors": file_errors,
        "answers_scanned": len(records),
        "groups_compared": groups_compared,
        "min_repeats": min_repeats,
        "stable_threshold": stable_threshold,
        "avg_token_f1": mean([float(item["avg_token_f1"]) for item in details]) if details else 0.0,
        "avg_jaccard": mean([float(item["avg_jaccard"]) for item in details]) if details else 0.0,
        "avg_exact_match_rate": mean([float(item["exact_match_rate"]) for item in details]) if details else 0.0,
        "stable_groups": sum(1 for item in details if item["is_stable"]),
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "validation_type": "qa_repeatability_similarity",
        "input_dir": str(qa_responses_dir),
        "summary": summary,
        "by_program": by_program,
        "by_question": by_question,
        "details": [] if summary_only else details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure answer similarity for repeated asks of the same QA question."
    )
    parser.add_argument(
        "--qa-responses-dir",
        default="output/va4_product_discoverer",
        help="Directory containing *_qa_responses.json files.",
    )
    parser.add_argument(
        "--output",
        default="output/evaluation/qa_repeatability_report.json",
        help="Output repeatability report JSON path.",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=2,
        help="Minimum answers needed for a URL/question group to be compared.",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.80,
        help="Token-F1 threshold used to mark a repeated-answer group as stable.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit answer previews and pair-level details from the output JSON.",
    )
    args = parser.parse_args()

    qa_responses_dir = Path(args.qa_responses_dir)
    if not qa_responses_dir.exists():
        print(f"ERROR: QA responses directory not found: {qa_responses_dir}")
        return 1
    if args.min_repeats < 2:
        print("ERROR: --min-repeats must be >= 2.")
        return 1
    if not 0.0 <= args.stable_threshold <= 1.0:
        print("ERROR: --stable-threshold must be between 0 and 1.")
        return 1

    report = build_repeatability_report(
        qa_responses_dir=qa_responses_dir,
        min_repeats=args.min_repeats,
        stable_threshold=args.stable_threshold,
        summary_only=args.summary_only,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("QA repeatability similarity complete")
    print(f"QA files scanned: {summary['qa_files_scanned']}")
    print(f"Answers scanned: {summary['answers_scanned']}")
    print(f"Repeated URL/question groups: {summary['groups_compared']}")
    print(f"Average Token-F1: {summary['avg_token_f1']:.2%}")
    print(f"Average Jaccard: {summary['avg_jaccard']:.2%}")
    print(f"Stable groups: {summary['stable_groups']}/{summary['groups_compared']}")
    print(f"Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
