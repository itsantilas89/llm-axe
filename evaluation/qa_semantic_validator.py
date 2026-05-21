"""Semantic QA Validator: evaluate meaning-level similarity for QA answers.

This script reads a QA consistency report and computes semantic-style metrics
between each LLM answer and reference text derived from extracted JSON values.

Metrics:
- BLEU-1, BLEU-2, BLEU-4 (implemented locally, no external dependency)
- Token F1 (precision/recall harmonic mean)
- Jaccard similarity
- Optional BERTScore (if bert-score package is installed)

Example:
    python evaluation/qa_semantic_validator.py \
        --qa-report output/evaluation/qa_consistency_report.json \
        --output output/evaluation/qa_semantic_report.json \
        --summary-only

Optional BERTScore:
    python evaluation/qa_semantic_validator.py \
        --qa-report output/evaluation/qa_consistency_report.json \
        --enable-bertscore \
        --bertscore-model bert-base-multilingual-cased
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class PairMetrics:
    bleu1: float
    bleu2: float
    bleu4: float
    token_precision: float
    token_recall: float
    token_f1: float
    jaccard: float
    bert_precision: float | None = None
    bert_recall: float | None = None
    bert_f1: float | None = None


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    # Unicode-aware tokenization for Greek/English + numbers.
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def ngrams(tokens: List[str], n: int) -> List[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def clipped_ngram_precision(candidate: List[str], reference: List[str], n: int) -> float:
    cand_ngrams = Counter(ngrams(candidate, n))
    ref_ngrams = Counter(ngrams(reference, n))

    total = sum(cand_ngrams.values())
    if total == 0:
        return 0.0

    clipped_hits = 0
    for ng, count in cand_ngrams.items():
        clipped_hits += min(count, ref_ngrams.get(ng, 0))

    return clipped_hits / total


def brevity_penalty(candidate_len: int, reference_len: int) -> float:
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - (reference_len / candidate_len))


def bleu_n(candidate_text: str, reference_text: str, max_n: int) -> float:
    cand = tokenize(candidate_text)
    ref = tokenize(reference_text)

    if not cand or not ref:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p = clipped_ngram_precision(cand, ref, n)
        # Smoothing to avoid exact zeros collapsing all BLEU.
        if p == 0.0:
            p = 1e-9
        precisions.append(p)

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = brevity_penalty(len(cand), len(ref))
    return bp * geo_mean


def token_overlap_scores(candidate_text: str, reference_text: str) -> tuple[float, float, float, float]:
    cand_tokens = tokenize(candidate_text)
    ref_tokens = tokenize(reference_text)

    if not cand_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0, 0.0

    cand_counter = Counter(cand_tokens)
    ref_counter = Counter(ref_tokens)

    overlap = 0
    for t, c in cand_counter.items():
        overlap += min(c, ref_counter.get(t, 0))

    precision = overlap / max(1, sum(cand_counter.values()))
    recall = overlap / max(1, sum(ref_counter.values()))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    cand_set = set(cand_tokens)
    ref_set = set(ref_tokens)
    union = cand_set | ref_set
    jaccard = 0.0 if not union else len(cand_set & ref_set) / len(union)

    return precision, recall, f1, jaccard


def flatten_values(values: Iterable) -> List[str]:
    out: List[str] = []

    def _walk(v):
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            return
        if isinstance(v, list):
            for x in v:
                _walk(x)
            return
        if isinstance(v, dict):
            for x in v.values():
                _walk(x)
            return

        s = str(v).strip()
        if s:
            out.append(s)

    _walk(values)
    return out


def build_reference_text(extracted_values: list) -> str:
    flat = flatten_values(extracted_values)
    return " ".join(flat)


def compute_pair_metrics(candidate: str, reference: str) -> PairMetrics:
    bleu1 = bleu_n(candidate, reference, max_n=1)
    bleu2 = bleu_n(candidate, reference, max_n=2)
    bleu4 = bleu_n(candidate, reference, max_n=4)

    p, r, f1, jac = token_overlap_scores(candidate, reference)

    return PairMetrics(
        bleu1=bleu1,
        bleu2=bleu2,
        bleu4=bleu4,
        token_precision=p,
        token_recall=r,
        token_f1=f1,
        jaccard=jac,
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _compute_bertscore(
    candidates: list[str],
    references: list[str],
    model_name: str,
) -> tuple[list[float], list[float], list[float]]:
    from bert_score import score as bert_score_fn  # Optional dependency

    p, r, f1 = bert_score_fn(
        cands=candidates,
        refs=references,
        model_type=model_name,
        verbose=False,
    )
    return p.tolist(), r.tolist(), f1.tolist()


def _compute_embedding_similarities(candidates: list[str], references: list[str], model_name: str):
    """Compute cosine similarities between candidate and reference using sentence-transformers.

    Returns list of floats (cosine similarity per pair).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"Embedding model unavailable: {e}")

    model = SentenceTransformer(model_name)
    cand_emb = model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
    ref_emb = model.encode(references, convert_to_numpy=True, show_progress_bar=False)

    sims = (cand_emb * ref_emb).sum(axis=1) / (
        np.linalg.norm(cand_emb, axis=1) * np.linalg.norm(ref_emb, axis=1)
    )
    return sims.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute semantic metrics (BLEU/BERTScore/etc.) for QA answers vs JSON references."
    )
    parser.add_argument(
        "--qa-report",
        default="output/evaluation/qa_consistency_report.json",
        help="Input report from qa_consistency_validator.py",
    )
    parser.add_argument(
        "--output",
        default="output/evaluation/qa_semantic_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only global summary.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Limit number of QA items to score (0 = all).",
    )
    parser.add_argument(
        "--enable-bertscore",
        action="store_true",
        help="Enable BERTScore (requires bert-score package).",
    )
    parser.add_argument(
        "--bertscore-model",
        default="bert-base-multilingual-cased",
        help="Model name for BERTScore.",
    )
    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Enable embedding-based cosine similarity (requires sentence-transformers).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embeddings.",
    )

    args = parser.parse_args()

    qa_report_path = Path(args.qa_report)
    if not qa_report_path.exists():
        print(f"ERROR: QA report not found: {qa_report_path}")
        print("Run qa_consistency_validator first with --output.")
        return 1

    with open(qa_report_path, "r", encoding="utf-8") as f:
        qa_report = json.load(f)

    raw_items = []
    for result in qa_report.get("results", []):
        url = result.get("url", "")
        programme_name = result.get("programme_name", "")
        for detail in result.get("details", []):
            answer = (detail.get("llm_answer") or "").strip()
            extracted_values = detail.get("extracted_values") or []
            reference = build_reference_text(extracted_values)

            if not answer or not reference:
                continue

            raw_items.append(
                {
                    "url": url,
                    "programme_name": programme_name,
                    "question_id": detail.get("question_id", ""),
                    "question": detail.get("question", ""),
                    "answer": answer,
                    "reference": reference,
                    "answer_source": detail.get("answer_source", "unknown"),
                }
            )

    if args.max_items > 0:
        raw_items = raw_items[: args.max_items]

    if not raw_items:
        print("No comparable QA items found (missing answers or references).")
        return 1

    per_item = []
    for item in raw_items:
        m = compute_pair_metrics(item["answer"], item["reference"])
        rec = {
            "url": item["url"],
            "programme_name": item["programme_name"],
            "question_id": item["question_id"],
            "question": item["question"],
            "answer_source": item["answer_source"],
            "metrics": {
                "bleu1": m.bleu1,
                "bleu2": m.bleu2,
                "bleu4": m.bleu4,
                "token_precision": m.token_precision,
                "token_recall": m.token_recall,
                "token_f1": m.token_f1,
                "jaccard": m.jaccard,
            },
        }
        per_item.append(rec)

    bert_enabled = False
    bert_error = ""
    if args.enable_bertscore:
        try:
            cands = [x["answer"] for x in raw_items]
            refs = [x["reference"] for x in raw_items]
            bp, br, bf1 = _compute_bertscore(cands, refs, args.bertscore_model)
            bert_enabled = True
            for i, rec in enumerate(per_item):
                rec["metrics"]["bertscore_precision"] = bp[i]
                rec["metrics"]["bertscore_recall"] = br[i]
                rec["metrics"]["bertscore_f1"] = bf1[i]
        except Exception as e:
            bert_error = str(e)

    embed_enabled = False
    embed_error = ""
    if args.enable_embeddings:
        try:
            cands = [x["answer"] for x in raw_items]
            refs = [x["reference"] for x in raw_items]
            sims = _compute_embedding_similarities(cands, refs, args.embedding_model)
            embed_enabled = True
            for i, rec in enumerate(per_item):
                rec["metrics"]["embed_cosine"] = sims[i]
        except Exception as e:
            embed_error = str(e)

    agg = {
        "bleu1": mean([x["metrics"]["bleu1"] for x in per_item]),
        "bleu2": mean([x["metrics"]["bleu2"] for x in per_item]),
        "bleu4": mean([x["metrics"]["bleu4"] for x in per_item]),
        "token_precision": mean([x["metrics"]["token_precision"] for x in per_item]),
        "token_recall": mean([x["metrics"]["token_recall"] for x in per_item]),
        "token_f1": mean([x["metrics"]["token_f1"] for x in per_item]),
        "jaccard": mean([x["metrics"]["jaccard"] for x in per_item]),
    }

    if bert_enabled:
        agg["bertscore_precision"] = mean([x["metrics"]["bertscore_precision"] for x in per_item])
        agg["bertscore_recall"] = mean([x["metrics"]["bertscore_recall"] for x in per_item])
        agg["bertscore_f1"] = mean([x["metrics"]["bertscore_f1"] for x in per_item])

    if embed_enabled:
        agg["embed_cosine"] = mean([x["metrics"]["embed_cosine"] for x in per_item])

    by_question = defaultdict(list)
    for x in per_item:
        by_question[x["question_id"]].append(x)

    question_summary = {}
    for qid, items in by_question.items():
        q = {
            "count": len(items),
            "bleu1": mean([x["metrics"]["bleu1"] for x in items]),
            "bleu2": mean([x["metrics"]["bleu2"] for x in items]),
            "bleu4": mean([x["metrics"]["bleu4"] for x in items]),
            "token_f1": mean([x["metrics"]["token_f1"] for x in items]),
            "jaccard": mean([x["metrics"]["jaccard"] for x in items]),
        }
        if bert_enabled:
            q["bertscore_f1"] = mean([x["metrics"]["bertscore_f1"] for x in items])
        question_summary[qid] = q

    output = {
        "semantic_validation": {
            "input_report": str(qa_report_path),
            "items_scored": len(per_item),
            "bertscore_enabled": bert_enabled,
            "bertscore_model": args.bertscore_model if args.enable_bertscore else None,
            "bertscore_error": bert_error or None,
            "embeddings_enabled": embed_enabled,
            "embedding_model": args.embedding_model if args.enable_embeddings else None,
            "embeddings_error": embed_error or None,
            "global_metrics": agg,
            "question_summary": question_summary,
            "details": [] if args.summary_only else per_item,
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Semantic QA evaluation complete")
    print(f"Items scored: {len(per_item)}")
    print("Global metrics:")
    for k, v in agg.items():
        print(f"  - {k}: {v:.4f}")

    if args.enable_bertscore and not bert_enabled:
        print("BERTScore was requested but failed. See bertscore_error in output JSON.")

    print(f"Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
