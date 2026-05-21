from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _safe_import_ollama_chat():
    """Import OllamaChat lazily so batch QA can run without eager model loading."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_axe.models import OllamaChat

    return OllamaChat


def _save_qa_responses(
    url: str,
    programme_name: str,
    classification_file: Path,
    qa_responses: list[dict],
    output_dir: Path,
) -> Path:
    """Save batch-generated Q&A responses with the same naming pattern as VA4."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    path = output_dir / f"{timestamp}_{safe_name}_qa_responses.json"

    payload = {
        "timestamp": timestamp,
        "url": url,
        "programme_name": programme_name,
        "classification_file": classification_file.name,
        "qa_count": len(qa_responses),
        "source": "batch_qa_runner",
        "responses": qa_responses,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return path


def _build_prompt(program_data: dict, question: str) -> list[dict]:
    system_prompt = (
        "Είσαι ένας ειδικός σύμβουλος πράσινων δανείων. "
        "Απάντησε σύντομα και με ακρίβεια στη σχετική ερώτηση, "
        "βασιζόμενος ΜΟΝΟ στα δομημένα δεδομένα του προγράμματος. "
        "Αν δεν γνωρίζεις την απάντηση, πες 'Δεν υπάρχει πληροφορία'."
    )
    user_prompt = (
        f"Πληροφορίες προγράμματος:\n{json.dumps(program_data, ensure_ascii=False, indent=2)}\n\n"
        f"ΕΡΩΤΗΣΗ: {question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ask_llm(llm, program_data: dict, question: str) -> str:
    try:
        answer = llm.ask(_build_prompt(program_data, question), format="", temperature=0.2)
        return answer.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def run_batch(
    classification_dir: Path,
    output_dir: Path,
    llm_model: str,
    max_files: int,
    summary_only: bool,
    target_url: str,
    classification_file_name: str,
    custom_questions: list[str],
) -> int:
    classification_files = sorted(classification_dir.glob("*_classification.json"))
    if max_files > 0:
        classification_files = classification_files[:max_files]

    if not classification_files:
        print("❌ No classification files found")
        return 1

    if not custom_questions:
        print("❌ No questions provided. Use --question multiple times or --questions-file.")
        return 1

    try:
        OllamaChat = _safe_import_ollama_chat()
        llm = OllamaChat(model=llm_model)
    except Exception as exc:
        print(f"❌ Failed to initialize LLM: {exc}")
        return 1

    print("📊 QA Batch Runner")
    print(f"   Files: {len(classification_files)}")
    print(f"   Questions per program: {len(custom_questions)}")
    print(f"   Output dir: {output_dir}")
    print(f"   LLM: {llm_model}")
    print()

    total_programs = 0
    total_questions = 0
    total_saved_files = 0
    matched_targets = 0

    for index, classification_file in enumerate(classification_files, 1):
        print(f"[{index}/{len(classification_files)}] Processing {classification_file.name}...")

        try:
            with classification_file.open("r", encoding="utf-8") as handle:
                classification_data = json.load(handle)

            if classification_file_name and classification_file.name != classification_file_name:
                continue

            if target_url and target_url not in str(classification_data.get("url", "")):
                continue

            matched_targets += 1

            if not classification_data.get("classification", {}).get("is_relevant", False):
                print("   ⊘ Not relevant, skipping QA generation")
                continue

            extracted_data = classification_data.get("extracted_data", {}) or {}
            programme_name = extracted_data.get("programme_name", "")
            questions = custom_questions
            qa_responses = []

            print(f"   Program: {programme_name or '(no programme_name)'}")
            print(f"   URL: {classification_data.get('url', '')}")

            for idx, question in enumerate(questions, 1):
                question_id = f"q{idx}"
                answer = ask_llm(llm, extracted_data, question)
                qa_responses.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "batch_qa_runner",
                    }
                )

            saved_path = _save_qa_responses(
                url=classification_data.get("url", ""),
                programme_name=programme_name,
                classification_file=classification_file,
                qa_responses=qa_responses,
                output_dir=output_dir,
            )
            total_programs += 1
            total_questions += len(qa_responses)
            total_saved_files += 1

            print(f"   ✅ Saved {len(qa_responses)} answers -> {saved_path.name}")
            if not summary_only:
                for response in qa_responses[:2]:
                    print(f"      - {response['question_id']}: {response['answer'][:120]}")

        except Exception as exc:
            print(f"   ❌ Error: {exc}")

    if (target_url or classification_file_name) and matched_targets == 0:
        print("\n⚠ No matching classification found for the provided target filter.")
        if target_url:
            print(f"   URL contains filter: {target_url}")
        if classification_file_name:
            print(f"   Classification file: {classification_file_name}")

    print()
    print("=" * 60)
    print("📈 Summary")
    print("=" * 60)
    print(f"Validated programs: {total_programs}")
    print(f"QA files saved: {total_saved_files}")
    print(f"Total answers generated: {total_questions}")
    print(f"Output directory: {output_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stored Q&A responses from classification outputs without rerunning classification."
    )
    parser.add_argument(
        "--classification-dir",
        default="output/va4_product_discoverer",
        help="Directory containing _classification.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/va4_product_discoverer",
        help="Directory where _qa_responses.json files will be saved.",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.1:8b-instruct-q4_K_M",
        help="Ollama model used to generate answers.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of classification files to process (0 = all).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only summary, not per-answer previews.",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Run QA only for the exact URL string you provide (matched as a substring of the stored classification URL).",
    )
    parser.add_argument(
        "--classification-file",
        default="",
        help="Run QA only for one specific _classification.json filename.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Custom question to ask. Use multiple times to ask multiple custom questions.",
    )
    parser.add_argument(
        "--questions-file",
        default="",
        help="Optional text or JSON file with custom questions to ask. One question per line or a JSON list of strings.",
    )

    args = parser.parse_args()

    classification_dir = Path(args.classification_dir)
    output_dir = Path(args.output_dir)

    if not classification_dir.exists():
        print(f"❌ Classification directory not found: {classification_dir}")
        return 1

    questions = list(args.question)
    if args.questions_file:
        questions_path = Path(args.questions_file)
        if not questions_path.exists():
            print(f"❌ Questions file not found: {questions_path}")
            return 1
        raw = questions_path.read_text(encoding="utf-8").strip()
        if raw:
            if questions_path.suffix.lower() == ".json":
                loaded = json.loads(raw)
                if isinstance(loaded, list):
                    questions.extend(str(item).strip() for item in loaded if str(item).strip())
                else:
                    print("❌ Questions JSON must be a list of strings.")
                    return 1
            else:
                questions.extend(line.strip() for line in raw.splitlines() if line.strip())

    return run_batch(
        classification_dir=classification_dir,
        output_dir=output_dir,
        llm_model=args.llm_model,
        max_files=args.max_files,
        summary_only=args.summary_only,
        target_url=args.url,
        classification_file_name=args.classification_file,
        custom_questions=questions,
    )


if __name__ == "__main__":
    sys.exit(main())