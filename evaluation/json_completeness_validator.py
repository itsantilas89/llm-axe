from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CORE_FIELDS = {
    "programme_name",
    "description",
    "programme_objective",
    "funding_type",
    "maximum_funding_amount",
    "eligible_interventions",
}

QA_POLICY_FIELDS = {
    "completion_delay_consequences",
    "post_completion_obligations",
}

PUBLIC_PROGRAM_FIELDS = {
    "total_budget",
    "funding_sources",
    "managing_body",
    "application_start_date",
    "application_end_date",
    "completion_deadline",
    "eligibility_criteria",
    "eligible_parties",
    "energy_performance_targets",
}

BANK_LOAN_FIELDS = {
    "interest_rate",
    "loan_duration",
    "minimum_funding_amount",
    "maximum_funding_amount",
    "funding_coverage",
    "property_requirements",
    "eligible_interventions",
}


def has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_value(item) for item in value)
    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())
    return value not in (None, "")


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("ς", "σ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def program_kind(payload: dict[str, Any]) -> str:
    urls = payload.get("source_urls") or [payload.get("url", "")]
    url_text = normalize_text(" ".join(str(url) for url in urls))
    name = normalize_text((payload.get("extracted_data") or {}).get("programme_name", ""))

    public_url_markers = (
        "gov.gr",
        "greece20.gov.gr",
        "stegasi.gov.gr",
        "hdb.gr",
    )
    public_name_markers = (
        "exoikonomo",
        "αναβαθμιζω το σπιτι",
        "εξοικονομω",
        "φωτοβολταικα στη στεγη",
    )
    bank_markers = ("bank", "alpha.gr", "eurobank.gr", "nbg.gr", "piraeusbank.gr", "crediabank")

    if any(marker in url_text for marker in public_url_markers):
        return "public_program"
    if any(marker in url_text for marker in bank_markers):
        return "bank_loan"
    if any(marker in name for marker in public_name_markers):
        return "public_program"
    return "unknown"


def expected_fields_for(kind: str) -> set[str]:
    if kind == "public_program":
        return CORE_FIELDS | PUBLIC_PROGRAM_FIELDS | QA_POLICY_FIELDS
    if kind == "bank_loan":
        return CORE_FIELDS | BANK_LOAN_FIELDS | QA_POLICY_FIELDS
    return set(CORE_FIELDS)


def validate_classification(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extracted = payload.get("extracted_data") or {}
    kind = program_kind(payload)
    expected = expected_fields_for(kind)
    present = sorted(field for field in expected if has_value(extracted.get(field)))
    missing = sorted(field for field in expected if not has_value(extracted.get(field)))

    all_fields = sorted(extracted.keys())
    empty_all = sorted(field for field in all_fields if not has_value(extracted.get(field)))

    return {
        "file": str(path),
        "programme_name": extracted.get("programme_name", ""),
        "url": payload.get("url", ""),
        "source_urls": payload.get("source_urls", []),
        "merged_from_count": payload.get("merged_from_count", 1),
        "program_kind": kind,
        "expected_fields": sorted(expected),
        "expected_present": present,
        "expected_missing": missing,
        "expected_coverage": len(present) / len(expected) if expected else 0.0,
        "empty_fields_all": empty_all,
    }


def build_report(classification_dir: Path) -> dict[str, Any]:
    files = sorted(classification_dir.glob("*_classification.json"))
    records = []
    field_missing = Counter()
    kind_counts = Counter()
    examples = defaultdict(list)

    for path in files:
        try:
            record = validate_classification(path)
        except Exception as exc:
            records.append({"file": str(path), "status": "error", "error": str(exc)})
            continue

        record["status"] = "ok"
        records.append(record)
        kind_counts[record["program_kind"]] += 1
        for field in record["expected_missing"]:
            field_missing[field] += 1
            if len(examples[field]) < 5:
                examples[field].append(record["programme_name"] or path.name)

    ok_records = [record for record in records if record.get("status") == "ok"]
    avg_expected_coverage = (
        sum(float(record.get("expected_coverage", 0.0) or 0.0) for record in ok_records) / len(ok_records)
        if ok_records
        else 0.0
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "classification_dir": str(classification_dir),
        "program_count": len(ok_records),
        "error_count": len(records) - len(ok_records),
        "kind_counts": dict(kind_counts),
        "summary": {
            "avg_expected_coverage": avg_expected_coverage,
            "missing_expected_fields": [
                {
                    "field": field,
                    "missing_count": count,
                    "examples": examples[field],
                }
                for field, count in field_missing.most_common()
            ],
        },
        "results": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit expected JSON field completeness for classification files.")
    parser.add_argument("--classification-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(Path(args.classification_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON completeness report: {output}")
    print(f"Programs: {report['program_count']}")
    print(f"Average expected coverage: {report['summary']['avg_expected_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
