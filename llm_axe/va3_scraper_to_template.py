# va_scraper_v3.py
# -----------------
# Virtual Assistant v3 for financing scheme data extraction.
# Workflow:
# 1) Accepts a URL from the user.
# 2) Scrapes the webpage using BeautifulSoup, removing headers, footers, cookie banners, and non-content tags.
# 3) Saves the clean text to ./outputs/<timestamp>_scraped.txt.
# 4) Reads that text back.
# 5) Prompts the LLM to fill in the JSON template with all available details.
# 6) Saves the resulting JSON to ./outputs/<timestamp>_extracted.json.
#
# Dependencies: llm_axe (core + models), requests, beautifulsoup4

import os
import sys
import json
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from llm_axe.models import OllamaChat
from llm_axe.core import make_prompt, safe_read_json

# --------------------------------------------------------------------------
# Template Definition
# --------------------------------------------------------------------------

TEMPLATE_DEFAULT = [{
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


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_outputs_dir() -> str:
    """
    Ensure output directory exists at the project root:
    ./output/va3_scraper_to_template/
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    out_dir = os.path.join(project_root, "output", "va3_scraper_to_template")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_raw_text(content: str, url: str) -> str:
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
    path = os.path.join(out_dir, f"{ts}_{safe_name}_scraped.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def save_result(data, url: str) -> str:
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


# --------------------------------------------------------------------------
# Page Scraper
# --------------------------------------------------------------------------

def scrape_page(url: str) -> str:
    log("[DEBUG] Fetching page content")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    for tag in soup.select(".site-header, .site-footer, .footer, .header, #footer, #header"):
        tag.decompose()

    for tag in soup.select("[class*='cookie'], [id*='cookie'], [class*='consent'], [id*='consent'], "
                           "[class*='banner'], [class*='popup'], [class*='modal']"):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    log(f"[DEBUG] Scraped text length: {len(text)} characters")

    return text


# --------------------------------------------------------------------------
# Prompt Builder
# --------------------------------------------------------------------------

def build_prompt(page_text: str, template: dict, url: str):
    system = (
        "You are a structured information extraction assistant. "
        "Read the provided webpage text and extract financing scheme details strictly according to the schema below. "
        "Fill every field as completely as possible using only what appears in the text. "
        "Use multi-sentence text where relevant. "
        "If a value is unknown, leave it empty ('') or []. "
        "Do not hallucinate or fabricate missing details. "
        "Return a JSON array with exactly one object that matches the schema."
    )
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    user = (
        f"SOURCE URL:\n{url}\n\n"
        f"PAGE CONTENT:\n{page_text}\n\n"
        f"RESPONSE SCHEMA:\n{template_str}\n\n"
        "Return only valid JSON, no explanations."
    )
    return [make_prompt("system", system), make_prompt("user", user)]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_json(llm: OllamaChat, page_text: str, template: dict, url: str):
    log("[DEBUG] Building extraction prompt")
    prompts = build_prompt(page_text, template, url)
    raw = llm.ask(prompts, format="json", temperature=0.1)

    # --- NEW SECTION: sanitize the LLM output ---
    cleaned = raw.strip()
    # Remove Markdown fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove language labels like json
        cleaned = cleaned.replace("json", "", 1).strip()

    # If JSON block appears inside other text, extract the part between first { and last }
    if not cleaned.strip().startswith("["):
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]

    # --- Try parsing the cleaned text ---
    parsed = safe_read_json(cleaned)
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        raw_path = save_raw_text(raw, url + "_llm_raw")
        log(f"[DEBUG] Raw output saved to: {raw_path}")
        raise ValueError("Invalid JSON response after cleaning")
    return parsed


# --------------------------------------------------------------------------
# Main CLI Loop
# --------------------------------------------------------------------------

def interactive_loop():
    llm = OllamaChat(model="mistral:latest")
    template = TEMPLATE_DEFAULT[0]
    log("VA-Scraper v3 ready. Type a URL or 'exit' to quit.\n")

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
            log("[DEBUG] Starting scrape")
            text = scrape_page(url)
            text_path = save_raw_text(text, url)
            log(f"[DEBUG] Saved cleaned text to {text_path}")

            log("[DEBUG] Sending to LLM for extraction")
            llm_output = extract_json(llm, text, template, url)
            saved_path = save_result(llm_output, url)

            log("\n===== JSON RESULT =====")
            print(json.dumps(llm_output, ensure_ascii=False, indent=2))
            log(f"\n[DEBUG] Saved structured output to: {saved_path}\n")

        except Exception as e:
            log(f"[ERROR] {e}")


def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
        llm = OllamaChat(model="mistral:latest")
        template = TEMPLATE_DEFAULT[0]

        text = scrape_page(url)
        text_path = save_raw_text(text, url)
        log(f"[DEBUG] Saved cleaned text to {text_path}")

        llm_output = extract_json(llm, text, template, url)
        saved_path = save_result(llm_output, url)

        print(json.dumps(llm_output, ensure_ascii=False, indent=2))
        log(f"[DEBUG] Saved structured output to: {saved_path}")
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
