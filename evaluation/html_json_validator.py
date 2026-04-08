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
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
) -> List[CheckResult]:
    norm_html = normalize_text(html_text)
    results: List[CheckResult] = []

    for path, value in pairs:
        if field_prefixes and not path_is_included(path, field_prefixes):
            continue

        value = value.strip()
        if not value:
            continue
        if len(value) < min_value_length:
            continue

        norm_value = normalize_text(value)
        found = norm_value in norm_html
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
