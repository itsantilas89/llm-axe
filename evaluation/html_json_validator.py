"""Validate whether JSON-recorded values exist in a given HTML document.

Example:
    python evaluation/html_json_validator.py \
        --html-file page.html \
        --json-file output/va4_product_discoverer/some_classification.json \
        --fields extracted_data,classification.key_features \
        --output output/evaluation/html_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# Domain-specific synonym groups (Greek green loans context)
SYNONYM_GROUPS = [
    # Core financial terms
    {"δάνειο", "loan", "δανειο", "δανια", "δανείου"},
    {"πρόγραμμα", "προγραμμα", "program", "προγραμα", "προγράμματα", "προγραματα"},
    {"επιδότηση", "επιχορήγηση", "grant", "χρηματοδότηση", "χρηματοδοτηση", "subsidy"},
    # Loan types / green programs
    {"ενεργειακή", "ενεργεια", "energy", "ενεργειακη", "ενεργειακές"},
    {"πράσινο", "green", "πρασινο", "πράσινη", "πράσινα", "πρασινη"},
    {"επιτόκιο", "επιτοκιο", "interest", "rate"},
    # Time/duration
    {"χρόνια", "χρονια", "years", "έτη", "ετη", "ετος"},
    {"μήνα", "μηνα", "month", "μήνες", "μηνες"},
    # Eligibility/requirements
    {"δικαιούχος", "δικαιουχος", "ιδιοκτήτης", "ιδιοκτητης", "owner", "eligible"},
    {"κατοικία", "κατοικια", "κατοικίες", "κατοικιες", "residence", "house", "home"},
    # Performance/efficiency
    {"απόδοση", "αποδοση", "efficiency", "performance", "απόδοσης"},
    {"ενεργειακή απόδοση", "ενεργειακη αποδοση", "energy efficiency"},
    {"μόνωση", "μονωση", "insulation", "θερμομόνωση", "θερμομονωση"},
    # Contact
    {"τηλέφωνο", "τηλεφωνο", "phone", "call", "τηλ"},
    {"email", "ηλεκτρονικό", "ηλεκτρονικο", "mail"},
]
# Build lookup: variant -> canonical
SYNONYM_LOOKUP = {}
for grp in SYNONYM_GROUPS:
    canonical = sorted(grp)[0]
    for tok in grp:
        SYNONYM_LOOKUP[tok] = canonical


def _apply_synonyms(tokens: Iterable[str]) -> list[str]:
    out = []
    for t in tokens:
        out.append(SYNONYM_LOOKUP.get(t, t))
    return out


def _greek_stem(token: str) -> str:
    # Extended rule-based Greek stemmer: strip common Greek suffixes
    # Order matters: longer suffixes first
    suffixes = [
        "ακότητα", "ότητα",  # -ακότητα, -ότητα (abstract nouns)
        "αγωγός", "αγωγή",   # -αγωγός, -αγωγή
        "ισμός",              # -ισμός (systems, ideologies)
        "ιστής", "ιστές",     # -ιστής, -ιστές (agent nouns)
        "αρχία",              # -αρχία (leadership)
        "σκοπός",             # compound suffix
        "κρατία",             # -κρατία (rule)
        "φορία",              # -φορία (action nouns)
        "γνώση",              # -γνώση (knowledge)
        "γραφία",             # -γραφία (writing)
        "λογία",              # -λογία (study)
        "μετρία",             # -μετρία (measurement)
        "νομία",              # -νομία (law)
        "πολία",              # -πολία (city)
        "φωνία",              # -φωνία (sound)
        # Common nominal suffixes
        "αία", "ειά",         # diminutive/affective
        "ίδα", "ίδες",        # plural
        "είς", "ής",          # masculine singular
        "ών", "ες",           # genitive plural / nominative plural
        "ος", "ους",          # nominative/genitive singular
        "ης",                 # adjective/nominal ending
        "ιο", "ια", "ιου",    # neuter forms
        "ου", "ας",           # genitive singular
        "η", "ο", "α",        # base singular
        "ει", "εις",          # verb endings
    ]
    for s in suffixes:
        if token.endswith(s) and len(token) - len(s) >= 3:
            return token[: -len(s)]
    return token


# Micro-dictionary of common Greek lemmas (problematic word forms)
GREEK_LEMMA_MAP = {
    "έχω": "εχω", "είμαι": "ειμαι", "κάνω": "κανω", "δίνω": "δινω",
    "πάω": "παω", "παίρνω": "παιρνω", "λέω": "λεω", "θέλω": "θελω",
    "κάνεις": "κανω", "έχεις": "εχω", "δίνεις": "δινω", "πας": "παω",
    "καν": "κανω", "εχ": "εχω", "δ": "δινω", "πα": "παω",
}


def _apply_lemmatization(token: str) -> str:
    """Apply lemma lookup first, then fallback to stemming."""
    token_norm = token.lower()
    if token_norm in GREEK_LEMMA_MAP:
        return GREEK_LEMMA_MAP[token_norm]
    return _greek_stem(token_norm)


def _token_set_with_stemming(tokens: Iterable[str]) -> set[str]:
    return set(_apply_lemmatization(t) for t in tokens)


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s)
        s = f" {s} "
        return set(s[i:i+n] for i in range(len(s)-n+1))

    A = ngrams(a)
    B = ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


@dataclass
class CheckResult:
    path: str
    value: str
    found: bool


def normalize_text(text: str) -> str:
    """Lowercase, remove accents, and collapse whitespace for robust matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s@.+%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cleanup_value(value: str) -> str:
    """Normalize noisy extracted values before matching to source text."""
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 \2", value)
    value = re.sub(r"\b(email|e-mail|phone|τηλ|τηλέφωνο|cat|κατ)\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


def _token_recall(haystack_tokens: set[str], needle_tokens: list[str]) -> float:
    if not needle_tokens:
        return 0.0
    hits = sum(1 for tok in needle_tokens if tok in haystack_tokens)
    return hits / len(needle_tokens)


def _normalize_email(s: str) -> str | None:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
    return m.group(0).lower() if m else None


def _normalize_phone(s: str) -> str | None:
    # keep only digits, normalize leading country code 30 (Greece)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    # strip leading country code 30 if present and length > 8
    if digits.startswith("30") and len(digits) > 8:
        digits = digits[2:]
    return digits


def _fuzzy_in_sentences(norm_html: str, norm_value: str, min_ratio: float = 0.82) -> bool:
    # Compare value against sentences in the HTML using SequenceMatcher
    if not norm_value:
        return False
    sentences = re.split(r"[.!?\n]", norm_html)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # quick length filter
        if abs(len(sent) - len(norm_value)) > max(30, len(norm_value) // 2):
            continue
        ratio = SequenceMatcher(None, norm_value, sent).ratio()
        if ratio >= min_ratio:
            return True
    return False


def _normalize_date_signature(token: str) -> str | None:
    parts = re.split(r"[-/.]", token)
    if len(parts) != 3:
        return None

    if len(parts[0]) == 4:
        yyyy, mm, dd = parts[0], parts[1], parts[2]
    elif len(parts[2]) == 4:
        dd, mm, yyyy = parts[0], parts[1], parts[2]
    else:
        return None

    if not (yyyy.isdigit() and mm.isdigit() and dd.isdigit()):
        return None

    if len(mm) == 1:
        mm = f"0{mm}"
    if len(dd) == 1:
        dd = f"0{dd}"

    return f"{yyyy}{mm}{dd}"


def _extract_numeric_signatures(text: str) -> set[str]:
    signatures: set[str] = set()
    for token in re.findall(r"\d+[\d.,/-]*", text):
        signatures.add(token)

        digits_only = re.sub(r"\D", "", token)
        if digits_only:
            signatures.add(digits_only)

        date_sig = _normalize_date_signature(token)
        if date_sig:
            signatures.add(date_sig)

    return signatures


def _is_value_found(
    norm_html: str,
    haystack_tokens: set[str],
    haystack_numeric_signatures: set[str],
    value: str,
    token_recall_threshold: float,
) -> bool:
    cleaned = _cleanup_value(value)
    norm_value = normalize_text(cleaned)
    if not norm_value:
        return False

    # 1) strict normalized substring
    if norm_value in norm_html:
        return True

    # 2) relaxed token recall for longer text fields
    needle_tokens = [t for t in norm_value.split() if len(t) >= 2]
    # apply synonym normalization and lemmatization to tokens
    if needle_tokens:
        needle_tokens_syn = _apply_synonyms(needle_tokens)
        needle_tokens_lem = [ _apply_lemmatization(t) for t in needle_tokens_syn ]
        # prepare haystack tokens with lemmatization
        hay_tokens_list = list(haystack_tokens)
        hay_tokens_syn = _apply_synonyms(hay_tokens_list)
        hay_tokens_lem_set = _token_set_with_stemming(hay_tokens_syn)

        # exact token recall on lemmatized tokens
        hits = sum(1 for tok in needle_tokens_lem if tok in hay_tokens_lem_set)
        recall = hits / len(needle_tokens_lem)
        adaptive_threshold = token_recall_threshold
        if len(needle_tokens_lem) <= 3:
            adaptive_threshold = min(0.5, token_recall_threshold)
        if recall >= adaptive_threshold:
            return True

        # n-gram jaccard between value and html as another fuzzy check
        jacc = _ngram_jaccard(norm_value, norm_html, n=3)
        if jacc >= 0.28:
            return True

    # 3) numeric fallback when all numbers from value appear in HTML
    needle_numeric_signatures = _extract_numeric_signatures(cleaned)
    if needle_numeric_signatures and needle_numeric_signatures.issubset(haystack_numeric_signatures):
        return True

    # 4) email/phone normalization checks
    email = _normalize_email(cleaned)
    if email:
        # collect emails from haystack
        hay_emails = set(m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", norm_html))
        if email in hay_emails:
            return True

    phone = _normalize_phone(cleaned)
    if phone:
        # haystack numeric signatures already include phones digits-only
        # compare normalized phones
        if phone in haystack_numeric_signatures:
            return True

    # 5) fuzzy sentence matching as last resort
    if _fuzzy_in_sentences(norm_html, norm_value, min_ratio=0.78):
        return True

    return False


def html_to_text(html: str) -> str:
    """Convert HTML to visible text using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def _scalar_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return None


def flatten_json_values(data: Any, prefix: str = "") -> List[tuple[str, str]]:
    """Return all scalar values from nested JSON with their key path."""
    out: List[tuple[str, str]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            out.extend(flatten_json_values(value, path))
        return out

    if isinstance(data, list):
        for idx, item in enumerate(data):
            path = f"{prefix}[{idx}]"
            out.extend(flatten_json_values(item, path))
        return out

    scalar = _scalar_to_text(data)
    if scalar is not None and prefix:
        out.append((prefix, scalar))
    return out


def path_is_included(path: str, field_prefixes: Iterable[str]) -> bool:
    for prefix in field_prefixes:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return True
    return False


def validate_values_in_html(
    html_text: str,
    pairs: Iterable[tuple[str, str]],
    field_prefixes: List[str],
    min_value_length: int,
    token_recall_threshold: float = 0.7,
) -> List[CheckResult]:
    norm_html = normalize_text(html_text)
    haystack_tokens = set(norm_html.split())
    haystack_numeric_signatures = _extract_numeric_signatures(html_text)
    results: List[CheckResult] = []

    for path, value in pairs:
        if field_prefixes and not path_is_included(path, field_prefixes):
            continue

        value = value.strip()
        if not value:
            continue
        if len(value) < min_value_length:
            continue

        found = _is_value_found(
            norm_html=norm_html,
            haystack_tokens=haystack_tokens,
            haystack_numeric_signatures=haystack_numeric_signatures,
            value=value,
            token_recall_threshold=token_recall_threshold,
        )
        results.append(CheckResult(path=path, value=value, found=found))

    return results


def summarize(results: List[CheckResult]) -> dict[str, Any]:
    total = len(results)
    found = sum(1 for r in results if r.found)
    missing = total - found
    coverage = (found / total) if total else 1.0

    return {
        "total_checked": total,
        "found": found,
        "missing": missing,
        "coverage": round(coverage, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether scalar values from a JSON file exist in an HTML document."
    )
    parser.add_argument("--json-file", required=True, help="Path to JSON file to validate.")

    html_group = parser.add_mutually_exclusive_group(required=True)
    html_group.add_argument("--html-file", help="Path to HTML file.")
    html_group.add_argument("--html-text", help="Raw HTML text.")

    parser.add_argument(
        "--fields",
        default="extracted_data",
        help=(
            "Comma-separated JSON path prefixes to validate. "
            "Example: extracted_data,classification.key_features. "
            "Use '*' to include all fields."
        ),
    )
    parser.add_argument(
        "--min-value-length",
        type=int,
        default=2,
        help="Ignore JSON values shorter than this length.",
    )
    parser.add_argument(
        "--token-recall-threshold",
        type=float,
        default=0.7,
        help="Token recall threshold for relaxed matching of long values (0-1).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for full validation report (JSON).",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="How many missing values to print to console.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with code 1 if at least one value is missing in HTML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"JSON file not found: {json_path}", file=sys.stderr)
        return 2

    if args.html_file:
        html_path = Path(args.html_file)
        if not html_path.exists():
            print(f"HTML file not found: {html_path}", file=sys.stderr)
            return 2
        html_raw = html_path.read_text(encoding="utf-8", errors="ignore")
    else:
        html_raw = args.html_text

    data = json.loads(json_path.read_text(encoding="utf-8"))
    pairs = flatten_json_values(data)

    if args.fields.strip() == "*":
        field_prefixes: List[str] = []
    else:
        field_prefixes = [f.strip() for f in args.fields.split(",") if f.strip()]

    html_visible_text = html_to_text(html_raw)
    results = validate_values_in_html(
        html_text=html_visible_text,
        pairs=pairs,
        field_prefixes=field_prefixes,
        min_value_length=args.min_value_length,
        token_recall_threshold=args.token_recall_threshold,
    )

    summary = summarize(results)
    missing = [r for r in results if not r.found]

    print("=== HTML vs JSON Validation ===")
    print(f"Checked values : {summary['total_checked']}")
    print(f"Found in HTML  : {summary['found']}")
    print(f"Missing        : {summary['missing']}")
    print(f"Coverage       : {summary['coverage']:.2%}")

    if missing:
        print("\nMissing examples:")
        for item in missing[: args.show_missing]:
            print(f"- {item.path} -> {item.value}")

    report = {
        "json_file": str(json_path),
        "fields": field_prefixes if field_prefixes else "*",
        "summary": summary,
        "results": [
            {"path": r.path, "value": r.value, "found": r.found}
            for r in results
        ],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved report: {output_path}")

    if args.fail_on_missing and summary["missing"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
