"""Batch validation: Match classification.json files with scraped.txt and validate.

This script automatically pairs _classification.json files from va4_product_discoverer
with corresponding _scraped.txt files from va3_scraper_to_template and validates
whether the extracted values appear in the source text.

Example:
    python evaluation/batch_html_validator.py \
        --classification-dir output/va4_product_discoverer \
        --scraped-dir output/va3_scraper_to_template \
        --output output/evaluation/batch_validation_report.json \
        --summary-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import the validator
sys.path.insert(0, str(Path(__file__).parent))
from html_json_validator import (
    validate_values_in_html,
    html_to_text,
    flatten_json_values,
    summarize,
)


def extract_url_parts(filename: str) -> str:
    """Extract URL domain + path parts from filename (ignore hash).
    
    Examples:
        '20260226T012055Z_www.eurobank.gr_el_retail_proionta-upiresies_prasino-daneio-katoikias_abbe8e29_classification.json'
        -> 'www.eurobank.gr_el_retail_proionta-upiresies_prasino-daneio-katoikias'
    """
    # Remove timestamp and .json/.txt extension
    parts = filename.replace("_classification.json", "").replace("_scraped.txt", "").split("_")
    
    if len(parts) < 2:
        return ""
    
    # Remove timestamp (first part)
    parts = parts[1:]
    
    # Last part is often _llm_raw, _extracted, or a short hash - remove if it looks like a hash
    if parts and (len(parts[-1]) == 8 and parts[-1].isalnum() and not parts[-1].startswith("scraped")):
        parts = parts[:-1]
    
    # Join domain + path parts
    url_parts = "_".join(parts)
    return url_parts


def timestamp_from_filename(filename: str) -> datetime | None:
    match = re.match(r"^(\d{8}T\d{6}Z)_", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def find_matching_scraped(classification_file: Path, scraped_dir: Path) -> Optional[Path]:
    """Find the corresponding _scraped.txt file for a _classification.json file.
    
    Matches by domain + URL path, ignoring hashes and timestamps. If multiple
    scrapes exist for the same URL, prefer the closest scrape at or before the
    classification timestamp; otherwise use the newest matching scrape.
    """
    
    class_url_parts = extract_url_parts(classification_file.name)
    
    if not class_url_parts:
        return None
    
    matches: list[Path] = []
    for scraped_file in scraped_dir.glob("*_scraped.txt"):
        scraped_url_parts = extract_url_parts(scraped_file.name)
        
        # Check if URL parts match (can be partial thanks to _llm_raw prefixes)
        if class_url_parts in scraped_url_parts or scraped_url_parts in class_url_parts:
            matches.append(scraped_file)
    
    if not matches:
        return None

    class_timestamp = timestamp_from_filename(classification_file.name)
    if class_timestamp is not None:
        older_or_same = [
            scraped_file
            for scraped_file in matches
            if (timestamp_from_filename(scraped_file.name) or datetime.min.replace(tzinfo=timezone.utc)) <= class_timestamp
        ]
        if older_or_same:
            return max(
                older_or_same,
                key=lambda path: timestamp_from_filename(path.name) or datetime.min.replace(tzinfo=timezone.utc),
            )

    return max(
        matches,
        key=lambda path: timestamp_from_filename(path.name) or datetime.min.replace(tzinfo=timezone.utc),
    )


def validate_one_pair(
    classification_file: Path,
    scraped_file: Path,
    fields: str = "extracted_data",
    exclude_path_prefixes: list[str] | None = None,
    token_recall_threshold: float = 0.7,
) -> dict:
    """Validate a single classification.json against its scraped.txt."""
    
    try:
        # Load JSON
        classification_data = json.loads(classification_file.read_text(encoding="utf-8"))
        
        # Load and clean HTML (in this case, already cleaned text)
        scraped_text = scraped_file.read_text(encoding="utf-8", errors="ignore")
        html_visible_text = html_to_text(scraped_text)
        
        # Get scalar values to check
        pairs = flatten_json_values(classification_data)
        
        # Parse fields
        if fields.strip() == "*":
            field_prefixes = []
        else:
            field_prefixes = [f.strip() for f in fields.split(",") if f.strip()]
        
        # Validate
        results = validate_values_in_html(
            html_text=html_visible_text,
            pairs=pairs,
            field_prefixes=field_prefixes,
            min_value_length=2,
            token_recall_threshold=token_recall_threshold,
        )

        if exclude_path_prefixes:
            def _is_excluded(path: str) -> bool:
                return any(
                    path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
                    for prefix in exclude_path_prefixes
                )

            results = [r for r in results if not _is_excluded(r.path)]
        
        summary = summarize(results)
        
        return {
            "status": "ok",
            "classification_file": str(classification_file),
            "scraped_file": str(scraped_file),
            "summary": summary,
            "results": [
                {"path": r.path, "value": r.value, "found": r.found}
                for r in results
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "classification_file": str(classification_file),
            "scraped_file": str(scraped_file) if scraped_file else None,
            "error": str(e),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-validate HTML vs classification JSON files."
    )
    parser.add_argument(
        "--classification-dir",
        default="output/va4_product_discoverer",
        help="Directory containing _classification.json files.",
    )
    parser.add_argument(
        "--scraped-dir",
        default="output/va3_scraper_to_template",
        help="Directory containing _scraped.txt files.",
    )
    parser.add_argument(
        "--fields",
        default="extracted_data",
        help="Comma-separated JSON path prefixes to validate.",
    )
    parser.add_argument(
        "--exclude-fields",
        default="",
        help=(
            "Comma-separated JSON path prefixes to exclude from coverage KPI. "
            "Use empty string to include everything."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional output path for full validation report (JSON).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only summary, not per-file results.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of classifications to validate (0 = all).",
    )
    parser.add_argument(
        "--token-recall-threshold",
        type=float,
        default=0.7,
        help="Token recall threshold for relaxed matching of long values (0-1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    classification_dir = Path(args.classification_dir)
    scraped_dir = Path(args.scraped_dir)
    
    if not classification_dir.exists():
        print(f"Classification dir not found: {classification_dir}", file=sys.stderr)
        return 2
    
    if not scraped_dir.exists():
        print(f"Scraped dir not found: {scraped_dir}", file=sys.stderr)
        return 2
    
    # Collect all classification files
    classification_files = sorted(classification_dir.glob("*_classification.json"))
    
    if not classification_files:
        print(f"No _classification.json files found in {classification_dir}", file=sys.stderr)
        return 2
    
    if args.max_files > 0:
        classification_files = classification_files[: args.max_files]
    
    print(f"Found {len(classification_files)} classification files.")
    if args.exclude_fields.strip():
        print(f"Excluding fields from KPI: {args.exclude_fields}")
    print()
    
    validated = 0
    skipped = 0
    errors = 0
    
    results = []
    all_summaries = {"total_checked": 0, "found": 0, "missing": 0}
    
    for i, class_file in enumerate(classification_files, 1):
        try:
            classification_data = json.loads(class_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[{i}] ERROR {class_file.name} (cannot parse JSON: {exc})")
            errors += 1
            continue

        is_relevant = bool(classification_data.get("classification", {}).get("is_relevant", False))
        if not is_relevant:
            print(f"[{i}] SKIP  {class_file.name} (not relevant)")
            skipped += 1
            continue

        scraped_file = find_matching_scraped(class_file, scraped_dir)
        
        if not scraped_file:
            print(f"[{i}] SKIP  {class_file.name} (no matching scraped file)")
            skipped += 1
            continue
        
        print(f"[{i}] CHECK {class_file.name}")
        exclude_path_prefixes = [
            p.strip() for p in args.exclude_fields.split(",") if p.strip()
        ] if args.exclude_fields.strip() else []

        result = validate_one_pair(
            class_file,
            scraped_file,
            args.fields,
            exclude_path_prefixes=exclude_path_prefixes,
            token_recall_threshold=args.token_recall_threshold,
        )
        results.append(result)
        
        if result["status"] == "error":
            print(f"      ERROR: {result['error']}")
            errors += 1
        else:
            summary = result["summary"]
            found = summary["found"]
            total = summary["total_checked"]
            coverage = summary["coverage"]
            print(f"      OK: {found}/{total} found ({coverage:.0%} coverage)")
            
            # Aggregate
            all_summaries["total_checked"] += total
            all_summaries["found"] += found
            all_summaries["missing"] += summary["missing"]
            validated += 1
    
    print()
    print("=" * 80)
    print(f"Summary: {validated} validated, {skipped} skipped, {errors} errors")
    print(f"Overall coverage: {all_summaries['found']}/{all_summaries['total_checked']} "
          f"({all_summaries['found'] / all_summaries['total_checked']:.1%})" 
          if all_summaries["total_checked"] > 0 else "No data")
    print("=" * 80)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "summary": {
                "total_files": len(classification_files),
                "validated": validated,
                "skipped": skipped,
                "errors": errors,
                "overall_coverage": all_summaries,
            },
            "results": results if not args.summary_only else [],
        }
        
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved report: {output_path}")
    
    return 0 if (errors == 0 and validated > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
