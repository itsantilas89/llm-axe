"""Q&A Consistency Validator - Validate Q&A responses against extracted program data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import re

from qa_questions import build_qa_questions


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove accents, collapse whitespace."""
    import unicodedata
    if not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_value(text: str) -> str:
    """Normalize a value for matching."""
    if not text:
        return ""
    text = normalize_text(text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def flatten_values(values: list) -> list[str]:
    """Flatten nested list/dict values to strings."""
    result = []
    for v in values:
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    result.extend(str(val) for val in item.values() if val)
                else:
                    result.append(str(item))
        elif isinstance(v, dict):
            result.extend(str(val) for val in v.values() if val)
        elif v:
            result.append(str(v))
    return result


def token_set(text: str) -> set[str]:
    """Create a normalized token set for lightweight similarity checks."""
    return set(re.findall(r"\w+", normalize_text(text), flags=re.UNICODE))


def question_overlap_score(q1: str, q2: str) -> float:
    """Compute token overlap between two questions."""
    tokens1 = set(q1.split())
    tokens2 = set(q2.split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def infer_question_spec(question_text: str, known_questions: list[dict]) -> dict:
    """
    Infer question → fields mapping for consistency checking.
    
    Two-tier fallback:
    1) Canonical matching - find closest question from qa_questions.py
    2) Keyword fallback (must reference real extracted_data keys)
    """
    norm_q = normalize_text(question_text)

    # 1) Canonical matching
    best = None
    best_score = 0.0
    for q in known_questions:
        score = question_overlap_score(norm_q, normalize_text(q.get("query", "")))
        if score > best_score:
            best_score = score
            best = q

    if best and best_score >= 0.35:
        return {
            "id": best.get("id", ""),
            "query": question_text,
            "fields": best.get("fields", []),
            "check_type": best.get("check_type", "substring"),
            "inference": f"canonical_match:{best_score:.2f}",
        }

    # 2) Keyword fallback - improved with domain-specific patterns
    keyword_specs = [
        # Green loans domain-specific patterns with expanded keywords
        ("interest_rate", ["interest_rate"], "substring", 
         ["επιτόκιο", "interest rate", "rate", "%", "προσδιοριζ"]),
        
        ("duration", ["loan_duration", "duration"], "numeric_range", 
         ["διάρκεια", "duration", "χρόνια", "έτη", "years", "ετών"]),
        
        ("programme_name", ["programme_name"], "substring", 
         ["ονομα", "ονομάζεται", "name", "program", "πρόγραμμα", "ονόμασε", "ονομάζουν"]),
        
        ("funding_amount", ["minimum_funding_amount", "maximum_funding_amount", "total_budget"], "numeric_range",
         ["ποσό", "ποσα", "amount", "funding", "χρηματοδ", "limit", "όριο", "budget", "minimum", "maximum", "χρηματοδότηση", "ανώτατο", "κατώτατο"]),
        
        ("deadlines", ["application_start_date", "application_end_date", "completion_deadline", "announcement_date"], "substring",
         ["προθεσμ", "deadline", "date", "ημερομην", "application end", "application start", "ημερομηνία", "εντόπισε", "αναφέρονται"]),
        
        ("eligibility", ["eligible_parties", "eligibility_criteria"], "list_contains",
         ["δικαιούχ", "eligible", "eligibility", "κριτήρια", "criteria", "requirements", "κατηγορίες", "κατηγορίας", "επιλέξιμ"]),
        
        ("interventions", ["eligible_interventions", "energy_performance_targets"], "list_contains",
         ["παρεμβάσεις", "interventions", "energy", "ενεργειακ", "ενεργειακή", "ενεργειακές", "αναβάθμιση"]),
        
        ("programme_description", ["programme_name", "description", "programme_objective", "additional_details"], "substring",
         ["περίγραψε", "describe", "στόχος", "objective", "benefits", "οφέλη", "σκοπός", "σκοπό", "περίγρ"]),
    ]

    inferred_fields: list[str] = []
    inferred_check_type = "substring"
    for _, fields, check_type, keywords in keyword_specs:
        # Strict matching: keyword must appear as normalized substring in question
        if any(normalize_text(k) in norm_q for k in keywords):
            for f in fields:
                if f not in inferred_fields:
                    inferred_fields.append(f)
            # Prefer numeric/list checks when discovered
            if check_type in {"numeric_range", "list_contains"}:
                inferred_check_type = check_type

    return {
        "id": "",
        "query": question_text,
        "fields": inferred_fields,
        "check_type": inferred_check_type,
        "inference": "keyword_fallback",
    }


def check_answer_against_fields(answer: str, field_values: list, check_type: str) -> tuple[bool | None, str]:
    """
    Check if LLM answer contains values from extracted fields.
    
    Returns: tuple[bool | None, str]
    - (True, reason): Answer matches field values
    - (False, reason): Answer contradicts field values  
    - (None, reason): Question not applicable (empty field_values)
    """
    if not answer or not answer.strip():
        return False, "Empty answer"
    
    normalized_answer = normalize_text(answer)
    
    # If answer is "no data available", treat as not applicable
    no_data_markers = ["δεν υπάρχει πληροφορία", "no data", "no information", "unknown", "δεν αναφέρεται"]
    if any(marker in normalized_answer for marker in no_data_markers):
        return None, "Field not available in source (LLM marked as no data)"
    
    # Filter out empty values
    non_empty_values = [v for v in field_values if v and str(v).strip()]
    
    if not non_empty_values:
        # Empty field_values means question is not applicable
        return None, "No extracted data to validate against (N/A)"
    
    if check_type == "substring":
        # Answer should contain one of the field values
        # Try exact match first
        for value in non_empty_values:
            norm_value = normalize_value(str(value))
            if norm_value and norm_value in normalize_value(answer):
                return True, f"Found '{value}' in answer"
        
        # Try token-level fuzzy matching
        field_words = set()
        for value in non_empty_values:
            tokens = re.findall(r"\w+", normalize_value(str(value)), flags=re.UNICODE)
            for token in tokens:
                if len(token) > 2:
                    field_words.add(token)

        answer_words = set(re.findall(r"\w+", normalize_value(answer), flags=re.UNICODE))
        if field_words & answer_words:
            matching_words = field_words & answer_words
            return True, f"Found field terms '{', '.join(list(matching_words)[:3])}' in answer"
        
        return False, f"None of {non_empty_values} found in answer"
    
    elif check_type == "list_contains":
        # For list fields, check if at least one item appears in answer
        for value in non_empty_values:
            if isinstance(value, list):
                for item in value:
                    norm_item = normalize_value(str(item))
                    if norm_item and norm_item in normalize_value(answer):
                        return True, f"Found list item '{item}' in answer"
                    # Token-level matching for list items
                    item_words = set(re.findall(r"\w+", norm_item, flags=re.UNICODE))
                    answer_words = set(re.findall(r"\w+", normalize_value(answer), flags=re.UNICODE))
                    if len(item_words) > 0 and item_words & answer_words:
                        matching = item_words & answer_words
                        return True, f"Found list item terms '{', '.join(list(matching)[:2])}' in answer"
            else:
                norm_value = normalize_text(str(value))
                if norm_value in normalized_answer:
                    return True, f"Found '{value}' in answer"
        return False, f"No list items found in answer"
    
    elif check_type == "numeric_range":
        # Extract numbers from answer and check against numeric fields
        answer_numbers = re.findall(r"\d+(?:[.,]\d+)?", normalize_value(answer))
        field_numbers = []
        for value in non_empty_values:
            field_numbers.extend(re.findall(r"\d+(?:[.,]\d+)?", normalize_value(str(value))))
        
        if not field_numbers:
            return False, "No numeric data to validate"
        
        # Check if any field number appears in answer
        for fn in field_numbers:
            if fn in answer_numbers:
                return True, f"Found amount '{fn}' in answer"
        
        return False, f"No amounts from {field_numbers} found in answer"
    
    else:
        return False, f"Unknown check type: {check_type}"


def validate_one_qa(
    classification_data: dict,
    num_questions: int = 3,
    allow_live_llm: bool = False,
    llm=None,
    stored_qa_responses: dict[str, dict] | None = None,
) -> dict:
    """Run Q&A validation for one classification file."""
    
    extracted_data = classification_data.get("extracted_data", {})
    classification = classification_data.get("classification", {})
    
    if not classification.get("is_relevant", False):
        return {
            "url": classification_data.get("url", ""),
            "status": "SKIP",
            "reason": "Not relevant"
        }
    
    # Generate questions. If stored QA responses exist, use those questions
    all_questions = build_qa_questions()
    questions = []
    if stored_qa_responses:
        for i, qtext in enumerate(stored_qa_responses.keys(), 1):
            spec = infer_question_spec(qtext, all_questions)
            questions.append({
                "id": f"q{i}",
                "query": qtext,
                "fields": spec.get("fields", []),
                "check_type": spec.get("check_type", "substring"),
                "inference": spec.get("inference", "unknown"),
            })
    else:
        questions = all_questions[:num_questions]
    
    results = {
        "url": classification_data.get("url", ""),
        "status": "OK",
        "programme_name": extracted_data.get("programme_name", ""),
        "questions_tested": len(questions),
        "questions_answered": 0,
        "questions_applicable": 0,
        "consistency_consistent": 0,
        "consistency_applicable": 0,
        "coverage": 0.0,
        "consistency_score": 0.0,
        "details": []
    }
    
    consistent_count = 0
    answered_count = 0
    applicable_count = 0
    url = classification_data.get("url", "")
    stored_qa_responses = stored_qa_responses or {}
    
    for qa in questions:
        try:
            answer = None
            source = ""

            # Get answer from stored QA responses
            if stored_qa_responses:
                for q_key, ans in stored_qa_responses.items():
                    if q_key == qa["query"]:
                        answer = ans
                        source = "stored_qa_responses"
                        break

            if not answer:
                results["details"].append({
                    "question_id": qa["id"],
                    "question": qa["query"],
                    "fields_checked": qa.get("fields", []),
                    "question_inference": qa.get("inference", "unknown"),
                    "check_type": qa.get("check_type", "substring"),
                    "extracted_values": [],
                    "llm_answer": "",
                    "llm_answer_preview": "",
                    "answer_source": "missing",
                    "is_consistent": False,
                    "reason": "No answer found (offline mode)",
                    "skipped": True,
                })
                continue

            answered_count += 1
            
            # Get field values
            field_values = [extracted_data.get(field) for field in qa.get("fields", [])]
            flat_field_values = flatten_values(field_values)

            is_applicable = True
            is_consistent = False
            reason = ""
            
            if not qa.get("fields"):
                is_applicable = False
                reason = "No mapped fields for question; excluded from consistency denominator"
            else:
                # Check consistency
                check_result = check_answer_against_fields(
                    answer, field_values, qa["check_type"]
                )
                is_consistent, reason = check_result
                
                if is_consistent is None:
                    # Question not applicable
                    is_applicable = False
                    is_consistent = False
                else:
                    # Question is applicable
                    applicable_count += 1
                    if is_consistent:
                        consistent_count += 1
            
            results["details"].append({
                "question_id": qa["id"],
                "question": qa["query"],
                "fields_checked": qa.get("fields", []),
                "question_inference": qa.get("inference", "canonical"),
                "check_type": qa.get("check_type", "substring"),
                "extracted_values": flat_field_values if is_applicable else [],
                "llm_answer": answer,
                "llm_answer_preview": answer[:200],
                "answer_source": source,
                "is_consistent": is_consistent,
                "consistency_applicable": is_applicable,
                "reason": reason
            })
        
        except Exception as e:
            results["details"].append({
                "question_id": qa["id"],
                "question": qa["query"],
                "error": str(e),
                "is_consistent": False
            })
    
    # Calculate scores
    if applicable_count > 0:
        results["consistency_score"] = consistent_count / applicable_count

    if questions:
        results["coverage"] = answered_count / len(questions)

    results["questions_answered"] = answered_count
    results["questions_applicable"] = applicable_count
    results["consistency_consistent"] = consistent_count
    results["consistency_applicable"] = applicable_count
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Q&A responses against extracted data")
    parser.add_argument("--classification-dir", required=True, help="Directory with classification JSONs")
    parser.add_argument("--qa-responses-dir", required=True, help="Directory with Q&A response JSONs")
    parser.add_argument("--output", required=True, help="Output report JSON file")
    parser.add_argument("--summary-only", action="store_true", help="Print summary only")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0=all)")
    
    args = parser.parse_args()
    
    print("📊 Q&A Consistency Validator")
    print(f"   Files: {len(list(Path(args.classification_dir).glob('*_classification.json')))}")
    print(f"   Mode: offline (stored Q&A only)")
    print(f"   Q&A responses dir: {args.qa_responses_dir}\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "validation_type": "qa_consistency",
        "total_files": 0,
        "results": []
    }
    
    classification_files = sorted(Path(args.classification_dir).glob("*_classification.json"))
    if args.max_files > 0:
        classification_files = classification_files[:args.max_files]
    
    valid_count = 0
    skipped_count = 0
    error_count = 0
    all_consistent = []
    all_applicable = []
    
    for i, cf in enumerate(classification_files, 1):
        print(f"[{i}/{len(classification_files)}] Processing {cf.name}...")
        
        try:
            # Load classification
            classification = json.loads(cf.read_text(encoding="utf-8"))
            
            # Find matching QA responses
            url = classification.get("url", "")
            qa_file = None
            for qa_f in Path(args.qa_responses_dir).glob("*_qa_responses.json"):
                qa_data = json.loads(qa_f.read_text(encoding="utf-8"))
                if qa_data.get("url") == url:
                    qa_file = qa_f
                    break
            
            stored_qa = {}
            if qa_file:
                qa_data = json.loads(qa_file.read_text(encoding="utf-8"))
                for resp in qa_data.get("responses", []):
                    q = resp.get("question", "")
                    a = resp.get("answer", "")
                    if q and a:
                        stored_qa[q] = a
                print(f"   ℹ Found {len(stored_qa)} stored Q&A responses")
            
            # Validate
            result = validate_one_qa(classification, stored_qa_responses=stored_qa)
            
            if result["status"] == "OK":
                valid_count += 1
                consistency = result["consistency_score"]
                applicable = result["consistency_applicable"]
                
                if consistency >= 0.75:
                    icon = "✅"
                elif consistency >= 0.5:
                    icon = "⚠"
                else:
                    icon = "❌"
                
                print(f"   {icon} Consistency: {consistency:.1%} | Coverage: {result['coverage']:.1%} ({result['questions_answered']}/{result['questions_tested']}) | Applicable: {applicable}")
                
                all_consistent.append(result["consistency_consistent"])
                all_applicable.append(result["consistency_applicable"])
            elif result["status"] == "SKIP":
                skipped_count += 1
                print(f"   ⊘ Not relevant, skipping Q&A")
            
            results["results"].append(result)
        
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error: {str(e)}")
    
    results["total_files"] = len(classification_files)
    
    # Calculate summary
    if all_applicable:
        total_applicable = sum(all_applicable)
        total_consistent = sum(all_consistent)
        avg_consistency = total_consistent / total_applicable if total_applicable else 0
    else:
        avg_consistency = 0
        total_applicable = 0
        total_consistent = 0
    
    print("\n" + "="*60)
    print("📈 Summary")
    print("="*60)
    print(f"Total programs validated: {valid_count}")
    print(f"Skipped (not relevant): {skipped_count}")
    print(f"Errors: {error_count}")
    print()
    print(f"Average Q&A consistency: {avg_consistency:.1%}")
    print(f"Average answer coverage: {sum(r['coverage'] for r in results['results'] if r.get('status') == 'OK') / valid_count if valid_count else 0:.1%}")
    print(f"Applicable consistency items: {total_applicable}")
    print(f"Highly consistent (≥75%): {sum(1 for r in results['results'] if r.get('consistency_score', 0) >= 0.75 and r.get('status') == 'OK')}/{valid_count}")
    
    # Save report
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
