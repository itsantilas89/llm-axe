"""Canonical QA prompt builders for program questions.

All interactive, webapp, CLI, and evaluation QA paths should use this module so
the answer policy stays identical outside test-only output formatting details.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


QA_POLICY_PROMPT = (
    "Είσαι ένας ειδικός σύμβουλος για πράσινα δάνεια και προγράμματα ενεργειακής "
    "αναβάθμισης κατοικιών. "
    "Απάντησε σε οποιαδήποτε σχετική ερώτηση για το πρόγραμμα, όχι μόνο σε "
    "προκαθορισμένες ερωτήσεις. "
    "Το JSON είναι η μοναδική πηγή αλήθειας για στοιχεία του συγκεκριμένου "
    "προγράμματος, όπως ποσά, ποσοστά, ημερομηνίες, δικαιούχοι, προϋποθέσεις, "
    "παρεμβάσεις, διαδικασία και όροι. "
    "Για σχετικές ερωτήσεις, απάντησε με όσα διαθέσιμα στοιχεία υπάρχουν στο JSON "
    "και σύνθεσε χρήσιμη απάντηση από τα διαθέσιμα πεδία. "
    "Αν η ερώτηση ζητά εξήγηση σχετικής έννοιας, μπορείς να χρησιμοποιήσεις "
    "γενική γνώση μόνο για να εξηγήσεις τον όρο ή τη λογική, αλλά κάθε ισχυρισμός "
    "για το συγκεκριμένο πρόγραμμα πρέπει να στηρίζεται στο JSON. "
    "Αν λείπει ένα επιμέρους πεδίο, γράψε σύντομα ότι το συγκεκριμένο στοιχείο "
    "δεν αναφέρεται στο JSON και συνέχισε με τα υπόλοιπα διαθέσιμα στοιχεία. "
    "Μην επινοείς ακριβή ποσά, ημερομηνίες, ποσοστά, δικαιούχους ή όρους που δεν "
    "υπάρχουν στο JSON. "
    "Γράψε ακριβώς 'Δεν υπάρχει πληροφορία' μόνο όταν η ερώτηση είναι άσχετη με "
    "το πρόγραμμα ή ζητά πληροφορία εκτός του αντικειμένου του. "
    "Απάντα στα ελληνικά, με σαφήνεια και ακρίβεια."
)


QA_BATCH_OUTPUT_INSTRUCTIONS = (
    "Επέστρεψε αποκλειστικά έγκυρο JSON, χωρίς markdown ή σχόλια. "
    "For every item, answer the exact question in that item's question field. "
    "Keep question_id unchanged. Do not copy a question as an answer and do not "
    "use an answer for a different question_id."
)


def prune_empty(value: Any) -> Any:
    """Remove empty JSON branches while preserving meaningful falsy values."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            pruned = prune_empty(item)
            if pruned not in ("", None, [], {}):
                cleaned[key] = pruned
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            pruned = prune_empty(item)
            if pruned not in ("", None, [], {}):
                cleaned_list.append(pruned)
        return cleaned_list
    return value


def _normalize_for_scope(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("ς", "σ")
    return re.sub(r"\s+", " ", text).strip()


def deterministic_qa_out_of_scope_answer(question: str) -> str:
    """Return a fixed answer for obvious QA negative controls."""
    normalized = _normalize_for_scope(question)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if "capital of" in normalized or "πρωτευουσα" in normalized:
        return "Δεν υπάρχει πληροφορία"
    return ""


def _json_dump(value: Any) -> str:
    return json.dumps(prune_empty(value), ensure_ascii=False, indent=2)


def build_single_qa_prompt(
    program_data: dict[str, Any],
    question: str,
    classification_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the canonical prompt for one user question."""
    user_parts: list[str] = []
    if classification_context:
        user_parts.append(f"ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ:\n{_json_dump(classification_context)}")
    user_parts.append(f"JSON προγράμματος:\n{_json_dump(program_data)}")
    user_parts.append(f"ΕΡΩΤΗΣΗ: {question}")
    return [
        {"role": "system", "content": QA_POLICY_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_batch_qa_prompt(
    program_data: dict[str, Any],
    question_items: list[dict[str, Any]],
    classification_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the canonical QA prompt for batched evaluation questions."""
    user_parts: list[str] = []
    if classification_context:
        user_parts.append(f"ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ:\n{_json_dump(classification_context)}")
    user_parts.append(f"JSON προγράμματος:\n{_json_dump(program_data)}")
    user_parts.append(f"Ερωτήσεις:\n{json.dumps(question_items, ensure_ascii=False, indent=2)}")
    user_parts.append(
        "Απάντησε με αυτό ακριβώς το schema:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_id": "q1", "answer": "απάντηση βασισμένη στο canonical QA prompt"}\n'
        "  ]\n"
        "}\n"
        "Πρέπει να υπάρχει μία απάντηση για κάθε question_id που δόθηκε."
    )
    return [
        {"role": "system", "content": f"{QA_POLICY_PROMPT} {QA_BATCH_OUTPUT_INSTRUCTIONS}"},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
