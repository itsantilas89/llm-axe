# online_agent_v2.py
# -------------------
# Interactive terminal client that:
# 1) Accepts a URL from the user
# 2) Fetches and reads the page (or PDF)
# 3) Loads response_template.json
# 4) Prompts the LLM to fill the JSON object thoroughly
# 5) Sanitizes/validates the output against a strict schema (generic)
# 6) Prints and saves the result to ./outputs/<timestamp>_extracted.json
#
# Dependencies: llm_axe (this repo), requests

import os
import sys
import json
import time
import tempfile
import re
from datetime import datetime
from typing import Dict, Any, List

import requests

from llm_axe.models import OllamaChat
from llm_axe.core import read_website, read_pdf, make_prompt, safe_read_json


TEMPLATE_DEFAULT: List[Dict[str, Any]] = [{
    "programme_name": "",
    "description": "",
    "programme_objective": "",
    "eligible_parties": [],
    "eligibility_criteria": [],
    "property_requirements": [],
    "minimum_funding_amount": "",
    "maximum_funding_amount": "",
    "funding_type": "",
    "interest_rate": "",
    "funding_coverage": "",
    "total_budget": "",
    "funding_sources": [],
    "duration": "",
    "loan_duration": "",
    "completion_deadline": "",
    "application_start_date": "",
    "application_end_date": "",
    "energy_performance_targets": "",
    "eligible_interventions": [],
    "application_process": "",
    "managing_body": "",
    "announcement_date": "",
    "contact_info": [],
    "additional_details": ""
}]

# ---- Error + IO helpers ------------------------------------------------------

class ExtractionError(Exception):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_outputs_dir() -> str:
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_result(data: Any, url: str) -> str:
    out_dir = ensure_outputs_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = (
        url.replace("https://", "")
           .replace("http://", "")
           .replace("/", "_")
           .replace("?", "_")
           .replace("&", "_")
           .replace("=", "_")
    )
    path = os.path.join(out_dir, f"{ts}_{safe_name}_extracted.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def save_raw(raw: str, url: str, suffix: str = "raw_llm_output.txt") -> str:
    out_dir = ensure_outputs_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = (
        url.replace("https://", "")
           .replace("http://", "")
           .replace("/", "_")
           .replace("?", "_")
           .replace("&", "_")
           .replace("=", "_")
    )
    path = os.path.join(out_dir, f"{ts}_{safe_name}_{suffix}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw)
    return path

def load_response_template(path: str = "response_template.json") -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
        except Exception:
            pass
    return TEMPLATE_DEFAULT

# ---- Fetching/reading --------------------------------------------------------

def is_probable_pdf_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".pdf") or "content-type=application/pdf" in lower

def fetch_pdf_bytes(url: str, timeout: int = 20) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (compatible; llm-axe/2.0)"
    })
    resp.raise_for_status()
    return resp.content

def read_link(url: str, max_retries: int = 2, backoff_sec: float = 1.5) -> str:
    log("[DEBUG] Accessing the link")
    attempt = 0
    last_err = None

    # HTML path
    if not is_probable_pdf_url(url):
        while attempt <= max_retries:
            try:
                log("[DEBUG] Reading link (HTML mode)")
                text = read_website(url)
                if text and text.strip():
                    return text
                raise RuntimeError("Empty body")
            except Exception as e:
                last_err = e
                attempt += 1
                if attempt <= max_retries:
                    log("[DEBUG] Retrying")
                    time.sleep(backoff_sec)
        log(f"[DEBUG] HTML reader failed: {last_err}. Falling back to PDF detection if applicable.")

    # PDF path
    if is_probable_pdf_url(url):
        attempt = 0
        while attempt <= max_retries:
            try:
                log("[DEBUG] Reading link (PDF mode)")
                pdf_bytes = fetch_pdf_bytes(url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                try:
                    text = read_pdf(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if text and text.strip():
                    return text
                raise RuntimeError("Empty PDF text")
            except Exception as e:
                last_err = e
                attempt += 1
                if attempt <= max_retries:
                    log("[DEBUG] Retrying")
                    time.sleep(backoff_sec)

    raise RuntimeError(f"Failed to read the provided link after retries: {last_err}")

# ---- Prompt building (generic) ----------------------------------------------

def build_extraction_prompt(page_text: str, template: Dict[str, Any], source_url: str) -> List[Dict[str, str]]:
    system = (
        "You are an information extraction agent. "
        "Extract ONLY what is explicitly present in the provided content. "
        "Return EXACTLY the keys given in the schema. DO NOT add, rename, or remove keys. "
        "For any list field, return a FLAT ARRAY OF STRINGS (no objects). "
        "If a value is not present, use an empty string for scalars and an empty array for lists. "
        "Use ISO-8601 dates (YYYY-MM-DD) when dates are present. "
        "Do not fabricate contacts or organizations; include them only if explicitly present verbatim in the page."
    )
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    user = (
        f"SOURCE URL:\n{source_url}\n\n"
        f"PAGE CONTENT:\n{page_text}\n\n"
        f"RESPONSE SCHEMA (STRICT, EXACT KEYS):\n{template_str}\n\n"
        "Output must be a JSON array with exactly one object following the schema above. "
        "All list fields must be arrays of strings, each string representing one item. "
        "Return ONLY JSON, no prose."
        "Response must be in the language of the provided content, not translated to English, if foreign."
    )
    return [make_prompt("system", system), make_prompt("user", user)]

# ---- Sanitizer/validator (generic) ------------------------------------------

ALLOWED_KEYS = set(TEMPLATE_DEFAULT[0].keys())
ARRAY_FIELDS = {
    "eligible_parties",
    "eligibility_criteria",
    "property_requirements",
    "funding_sources",
    "eligible_interventions",
    "contact_info",
}
SCALAR_FIELDS = ALLOWED_KEYS - ARRAY_FIELDS

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){7,15}")
URL_RE = re.compile(r"https?://[^\s\)]+")
DATE_SLASH_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
MONEY_TOKEN_RE = re.compile(r"[€$]|EUR|USD|euro|dollar", re.IGNORECASE)

def _ensure_list_of_strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict):
                for k in ("description", "intervention", "requirement", "item", "value", "text", "name", "title"):
                    if k in item and isinstance(item[k], str) and item[k].strip():
                        out.append(item[k].strip())
                        break
                else:
                    parts = [str(v).strip() for v in item.values() if isinstance(v, (str, int, float)) and str(v).strip()]
                    if parts:
                        out.append(", ".join(parts))
            else:
                out.append(str(item))
        return out
    return [str(value)]

def _to_iso_dates_in_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    def repl(m):
        d, mth, y = m.groups()
        try:
            return datetime(int(y), int(mth), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            return m.group(0)
    return DATE_SLASH_RE.sub(repl, s)

def _present_in_page(val: str, page_text: str) -> bool:
    if not isinstance(val, str) or not val:
        return False
    return val in (page_text or "")

def _contacts_from_page(page_text: str) -> List[str]:
    emails = set(EMAIL_RE.findall(page_text or ""))
    phones = set(m.group(0).strip() for m in PHONE_RE.finditer(page_text or ""))
    urls = set(URL_RE.findall(page_text or ""))
    contacts = []
    for e in sorted(emails):
        contacts.append(f"email: {e}")
    for p in sorted(phones):
        contacts.append(f"phone: {p}")
    for u in sorted(urls):
        contacts.append(u)
    return contacts

def sanitize_and_validate(result_obj: Dict[str, Any], template_obj: Dict[str, Any], page_text: str) -> Dict[str, Any]:
    clean = {}

    # Drop unknown keys, coerce required keys
    for k in ALLOWED_KEYS:
        v = result_obj.get(k, template_obj.get(k, ""))

        if k in ARRAY_FIELDS:
            v_list = _ensure_list_of_strings(v)

            if k == "contact_info":
                # Only trust contacts found verbatim in page text
                v_list = _contacts_from_page(page_text)

            clean[k] = v_list
        else:
            s = v if isinstance(v, str) else (json.dumps(v, ensure_ascii=False) if v not in (None, "") else "")
            s = s.strip() if isinstance(s, str) else s

            if k in {"application_start_date", "application_end_date", "announcement_date", "completion_deadline", "duration", "loan_duration"}:
                s = _to_iso_dates_in_text(s)

            if k in {"minimum_funding_amount", "maximum_funding_amount", "total_budget", "funding_coverage", "interest_rate"}:
                # Keep only if anchored by a currency token or exact substring in the page
                if s and (MONEY_TOKEN_RE.search(page_text or "") or _present_in_page(s, page_text)):
                    pass
                else:
                    s = ""

            # Require page anchor for critical named entities to avoid hallucinations
            if k in {"managing_body", "programme_name"} and s and not _present_in_page(s, page_text):
                s = ""

            clean[k] = s

    return clean

# ---- LLM extraction ----------------------------------------------------------

def extract_json(llm: OllamaChat, page_text: str, template_obj: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    prompts = build_extraction_prompt(page_text, template_obj, url)
    log("[DEBUG] Filling the object")
    raw = llm.ask(prompts, format="json", temperature=0.1, stream=False)

    parsed = safe_read_json(raw)
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        raw_path = save_raw(raw, url)
        log(f"[DEBUG] LLM returned unparsable/unexpected JSON. Raw saved to: {raw_path}")
        raise ExtractionError("LLM did not return the expected JSON array with one object", raw)

    cleaned = sanitize_and_validate(parsed[0], template_obj, page_text)
    return [cleaned]

# ---- Main loop/CLI -----------------------------------------------------------

def interactive_loop(model_name: str = "mistral:latest") -> None:
    llm = OllamaChat(model=model_name)
    template_list = load_response_template()
    template_obj = template_list[0]

    log(f"LLM-AXE v2 ready on {model_name}. Type a URL, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("URL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        url = user_input

        try:
            content = read_link(url)
            log("[DEBUG] Content acquired; length: {} characters".format(len(content)))

            result = extract_json(llm, content, template_obj, url)
            saved_path = save_result(result, url)

            log("\n===== JSON RESULT =====")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            log("\n[DEBUG] Saved to: {}".format(saved_path))
            log("")
        except ExtractionError as ee:
            log("[ERROR] " + str(ee))
            log("\n===== RAW LLM OUTPUT (as returned) =====")
            print(ee.raw)
            raw_path = save_raw(ee.raw, url)
            log(f"[DEBUG] Raw output saved to: {raw_path}")
            log("")
        except Exception as e:
            log(f"[ERROR] {e}")

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
        llm = OllamaChat(model="mistral:latest")
        template_list = load_response_template()
        template_obj = template_list[0]

        try:
            content = read_link(url)
            log("[DEBUG] Content acquired; length: {} characters".format(len(content)))
            result = extract_json(llm, content, template_obj, url)
            saved_path = save_result(result, url)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            log("[DEBUG] Saved to: {}".format(saved_path))
        except ExtractionError as ee:
            log("[ERROR] " + str(ee))
            log("\n===== RAW LLM OUTPUT (as returned) =====")
            print(ee.raw)
            raw_path = save_raw(ee.raw, url)
            log(f"[DEBUG] Raw output saved to: {raw_path}")
        return

    interactive_loop()

if __name__ == "__main__":
    main()
