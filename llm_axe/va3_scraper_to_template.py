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
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from typing import List, Dict
from urllib.parse import urlparse, urlunparse
import hashlib
import socket

# Try normal package imports first. If the package-level import fails (for
# example because `llm_axe.__init__` imports missing modules), fall back to
# loading `models.py` and `core.py` directly from the same directory. This
# keeps this file runnable both as a script and when executed via importlib.
try:
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt, safe_read_json
except Exception:
    print("[WARN] package-level imports failed; loading llm_axe models/core directly", file=sys.stderr)
    import importlib.util as _il
    here = os.path.dirname(__file__)
    for _mod in ("models", "core"):
        _path = os.path.join(here, f"{_mod}.py")
        if os.path.exists(_path):
            spec = _il.spec_from_file_location(f"llm_axe.{_mod}", _path)
            module = _il.module_from_spec(spec)
            spec.loader.exec_module(module)
            # register in sys.modules so subsequent imports work normally
            sys.modules[f"llm_axe.{_mod}"] = module
    # re-import names from the loaded modules
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt, safe_read_json

# --------------------------------------------------------------------------
# Template Definition
# --------------------------------------------------------------------------

TEMPLATE_DEFAULT = [{
    "id": "",
    "source_url": "",
    "source_urls": [],
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
    "completion_delay_consequences": "",
    "application_start_date": "",
    "application_end_date": "",
    "energy_performance_targets": "",
    "eligible_interventions": [],
    "application_process": "",
    "post_completion_obligations": "",
    "managing_body": "",
    "announcement_date": "",
    "contact_info": [],
    "additional_details": ""
}]


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def _safe_console_print(message: object) -> None:
    text = str(message)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text, flush=True)

def log(msg: str) -> None:
    _safe_console_print(msg)

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
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Use canonical safe name and append a short hash to reduce collisions
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    path = os.path.join(out_dir, f"{ts}_{safe_name}_scraped.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def save_result(data, url: str) -> str:
    out_dir = ensure_outputs_dir()
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Use canonical safe name and append a short hash to reduce collisions
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    path = os.path.join(out_dir, f"{ts}_{safe_name}_extracted.json")
    data = [
        enrich_program_identity(item, url) if isinstance(item, dict) else item
        for item in data
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _safe_program_id(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:90]


def stable_program_id(programme_name: str, url: str) -> str:
    base = str(url or programme_name or datetime.utcnow().isoformat())
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    title_slug = _safe_program_id(programme_name)
    return f"{title_slug[:60]}-{digest}" if title_slug else f"program-{digest}"


def enrich_program_identity(program: dict, url: str) -> dict:
    """Fill stable metadata fields that should not be left to the LLM."""
    source_url = str(program.get("source_url") or "").strip()
    if not source_url:
        source = program.get("source")
        if isinstance(source, str):
            source_url = source.strip()
        elif isinstance(source, list) and source:
            source_url = str(source[0] or "").strip()
    if not source_url:
        source_url = str(url or "").strip()

    if source_url:
        program["source_url"] = source_url

    source_urls = program.get("source_urls")
    if not isinstance(source_urls, list):
        source_urls = []
    if source_url and source_url not in source_urls:
        source_urls.insert(0, source_url)
    program["source_urls"] = source_urls

    current_id = _safe_program_id(str(program.get("id") or ""))
    if not current_id:
        programme_name = str(
            program.get("programme_name")
            or program.get("program_name")
            or program.get("title")
            or ""
        ).strip()
        program["id"] = stable_program_id(programme_name, source_url)
    else:
        program["id"] = current_id
    return program


# --------------------------------------------------------------------------
# Topic -> Sources Recommendation
# --------------------------------------------------------------------------

def load_trusted_sources() -> Dict[str, List[str]]:
    """Load local trusted sources mapping from `trusted_sources.json`.

    Returns a mapping of topic keys to lists of URLs. If the file cannot be
    read, returns an empty dict.
    """
    try:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "trusted_sources.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sanitize_token(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s).strip().lower()


def _make_safe_name(url: str, max_len: int = 120) -> str:
    """Create a filesystem-safe short name from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_")
    if path:
        candidate = f"{host}_{path}"
    else:
        candidate = host
    # replace other unsafe chars
    candidate = re.sub(r"[^A-Za-z0-9_\-\.]+", "-", candidate)
    if len(candidate) > max_len:
        # truncate preserving start and end
        half = max_len // 2 - 3
        candidate = candidate[:half] + "..." + candidate[-half:]
    return candidate


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _build_sources_prompt(topic: str, max_results: int = 6) -> List[dict]:
    # Strong system instructions: prefer official/authoritative domains
    system = (
        "You are a domain-aware research assistant that recommends authoritative web sources for a given research topic. "
        "When possible, prefer official and institutionally-backed pages such as government sites (domains like .gov, .gov.gr, or government agencies), "
        "university and academic pages, recognized banks and financial institutions, and reputable research organisations. "
        "Avoid forums, social media posts, and personal blogs unless no authoritative sources exist for the topic. "
        "Return only a JSON array (no surrounding text)."
    )

    user = (
        f"Given the topic below, return up to {max_results} candidate sources as a JSON array. "
        "Each array item must be an object with these fields:\n"
        "- source: full URL string to the page\n"
        "- title: one-line human-friendly title for display\n"
        "- type: one of [gov, bank, research, news, other] indicating source kind\n"
        "- country: ISO country code if applicable (e.g., GR for Greece) or empty string\n"
        "- file: OPTIONAL - if you are recommending a local file, provide a filesystem path here (either absolute or relative). If present, `source` may be empty.\n"
        "- reason: one short sentence why this source is relevant (optional)\n"
        "Prefer pages in Greek when they are available for the topic; otherwise include English-language pages. "
        "If you cannot find suitable URLs, return an empty array. Do NOT include duplicates. "
        f"TOPIC:\n{topic}\n"
    )

    return [make_prompt("system", system), make_prompt("user", user)]


def query_llm_for_sources(llm, topic: str, max_results: int = 6) -> List[Dict[str, str]]:
    """Ask the LLM to propose candidate source URLs for a topic and
    convert the result to the internal recommendation format.
    """
    prompts = _build_sources_prompt(topic, max_results=max_results)
    try:
        raw = llm.ask(prompts, format="json", temperature=0.2)
    except Exception as e:
        log(f"[ERROR] LLM query failed: {e}")
        return []

    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    # Extract JSON array if embedded
    if not cleaned.strip().startswith("["):
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    parsed = safe_read_json(cleaned)
    results: List[Dict[str, str]] = []

    # Accept dicts that include a 'sources' key
    if isinstance(parsed, dict) and "sources" in parsed and isinstance(parsed["sources"], list):
        parsed = parsed["sources"]

    # If parsed is not a non-empty list, attempt one lenient retry and save raw outputs for debugging
    if not isinstance(parsed, list) or not parsed:
        # save raw response for debugging
        try:
            out_dir = ensure_outputs_dir()
            from datetime import timezone
            debug_path = os.path.join(out_dir, f"llm_sources_raw_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt")
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(cleaned)
        except Exception:
            debug_path = None

        # Retry once with a slightly higher temperature and more results
        try:
            prompts2 = _build_sources_prompt(topic, max_results=max_results * 2)
            raw2 = llm.ask(prompts2, format="json", temperature=0.5)
            cleaned2 = (raw2 or "").strip()
            if cleaned2.startswith("```"):
                cleaned2 = cleaned2.strip("`")
                cleaned2 = cleaned2.replace("json", "", 1).strip()
            if not cleaned2.strip().startswith("["):
                start = cleaned2.find("[")
                end = cleaned2.rfind("]")
                if start != -1 and end != -1:
                    cleaned2 = cleaned2[start:end + 1]
            parsed2 = safe_read_json(cleaned2)
            if isinstance(parsed2, dict) and "sources" in parsed2 and isinstance(parsed2["sources"], list):
                parsed2 = parsed2["sources"]
            if isinstance(parsed2, list) and parsed2:
                parsed = parsed2
                # save retry raw
                try:
                    if debug_path:
                        with open(debug_path.replace('.txt', '_retry.txt'), "w", encoding="utf-8") as df:
                            df.write(cleaned2)
                except Exception:
                    pass
        except Exception:
            pass

    # tolerate a single object returned instead of an array
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        # log the cleaned LLM output for debugging
        log(f"[WARN] LLM did not return a JSON array. Cleaned output:\n{cleaned}")
        return results

    # If the model returned an empty list, surface the cleaned/raw output to help debugging
    if isinstance(parsed, list) and len(parsed) == 0:
        log("[WARN] LLM returned an empty array for sources. Cleaned output below:")
        log(cleaned)
        return results

    out_dir = ensure_outputs_dir()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        url = item.get("source") or item.get("url")
        file_path = item.get("file") or item.get("path")
        # Determine whether this item is a web URL or a local file suggestion
        is_url = False
        is_file = False
        if url:
            parsed_u = urlparse(url)
            if parsed_u.scheme and parsed_u.scheme.lower() in ("http", "https"):
                # normalize and require host resolvability
                norm = _normalize_url(url)
                if _is_host_resolvable(norm):
                    is_url = True
                    url = norm
                else:
                    # skip unreachable URLs
                    continue
            else:
                # Could be a bare file path returned in 'source'
                if os.path.exists(url):
                    is_file = True
                else:
                    # not a valid http/https url and not an existing path -> skip
                    continue
        elif file_path:
            # accept only existing local file suggestions
            if os.path.exists(file_path):
                is_file = True
                url = file_path
            else:
                continue

        if not (is_url or is_file):
            continue
        title = item.get("title") or _short_display(url)
        reason = item.get("reason", "")

        safe = _make_safe_name(url)
        # append a short hash to reduce collisions
        safe = f"{safe}_{_short_hash(url)}"
        suggested_raw = os.path.join(out_dir, f"{safe}_scraped.txt")
        suggested_json = os.path.join(out_dir, f"{safe}_extracted.json")
        # For file suggestions, allow the command to pass a file path; for URLs pass the URL
        cmd_arg = f"file://{os.path.abspath(url)}" if is_file and not urlparse(url).scheme else url
        command = f'"{sys.executable}" "{os.path.abspath(__file__)}" "{cmd_arg}"'

        results.append({
            "source": url,
            "trusted_key": "model_suggested",
            "display_name": title,
            "reason": reason,
            "suggested_raw": suggested_raw,
            "suggested_json": suggested_json,
            "command": command,
        })

    return results


def _short_display(url: str, max_len: int = 60) -> str:
    """Return a user-friendly short display string for a URL."""
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"
    disp = host + path
    if len(disp) > max_len:
        return disp[: max_len - 3] + "..."
    return disp


def _is_host_resolvable(url: str) -> bool:
    """Return True if the hostname in `url` resolves via DNS."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.split(":")[0]
        if not host:
            return False
        # getaddrinfo will raise if the host cannot be resolved
        socket.getaddrinfo(host, None)
        return True
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    """Normalize a URL: ensure scheme and convert internationalized hostnames to ASCII (IDNA/punycode).

    If the input is a bare hostname or contains non-ascii characters, convert them so DNS and requests can resolve.
    """
    try:
        p = urlparse(url)
        scheme = p.scheme
        netloc = p.netloc
        path = p.path

        # handle cases where LLM returned a bare host or path instead of full URL
        if not netloc and path and ("/" not in path or path.count("/") == 1 and not path.startswith("/")):
            # treat the path as netloc when scheme is missing
            netloc = path
            path = ""

        if not scheme:
            scheme = "https"

        # IDNA-encode hostname if it contains non-ascii
        host_port = netloc.split("@")[-1]
        if ":" in host_port:
            host, port = host_port.split(":", 1)
        else:
            host, port = host_port, None

        if host and any(ord(c) > 127 for c in host):
            host_ascii = host.encode("idna").decode("ascii")
        else:
            host_ascii = host

        if port:
            netloc = f"{host_ascii}:{port}"
        else:
            netloc = host_ascii

        normalized = urlunparse((scheme, netloc, path or p.path, p.params, p.query, p.fragment))
        return normalized
    except Exception:
        return url


def recommend_sources_for_topic(topic: str) -> List[Dict[str, str]]:
    """Given a broad `topic` string, return a list of suggested sources and
    suggested output filenames for scraping and templating.

    The function looks up `trusted_sources.json` for matching keys (substring
    / token overlap). If no local matches are found, it returns a best-effort
    fallback list (empty list) so the caller can decide what to do.
    """
    topic_clean = _sanitize_token(topic)
    topic_tokens = set(topic_clean.split())
    trusted = load_trusted_sources()

    matches: List[Dict[str, str]] = []

    # Score each trusted key by token overlap with the topic
    scored = []
    for key, urls in trusted.items():
        key_tokens = set(_sanitize_token(key).split())
        overlap = len(topic_tokens & key_tokens)
        scored.append((overlap, key, urls))

    scored.sort(reverse=True, key=lambda x: x[0])

    output_dir = ensure_outputs_dir()

    for overlap, key, urls in scored:
        if overlap <= 0:
            continue
        for url in urls:
            # append short hash to reduce collisions between similar URLs
            safe = f"{_make_safe_name(url)}_{_short_hash(url)}"
            suggested_raw = os.path.join(output_dir, f"{safe}_scraped.txt")
            suggested_json = os.path.join(output_dir, f"{safe}_extracted.json")
            command = f'"{sys.executable}" "{os.path.abspath(__file__)}" "{url}"'
            matches.append({
                "source": url,
                "trusted_key": key,
                "display_name": _short_display(url),
                "suggested_raw": suggested_raw,
                "suggested_json": suggested_json,
                "command": command,
            })

    # If no matches found, provide fallback: return all known URLs (unscored)
    if not matches and trusted:
        for key, urls in trusted.items():
            for url in urls:
                # append short hash to reduce collisions between similar URLs
                safe = f"{_make_safe_name(url)}_{_short_hash(url)}"
                suggested_raw = os.path.join(output_dir, f"{safe}_scraped.txt")
                suggested_json = os.path.join(output_dir, f"{safe}_extracted.json")
                command = f'"{sys.executable}" "{os.path.abspath(__file__)}" "{url}"'
                matches.append({
                    "source": url,
                    "trusted_key": key,
                    "display_name": _short_display(url),
                    "suggested_raw": suggested_raw,
                    "suggested_json": suggested_json,
                    "command": command,
                })

    return matches


# --------------------------------------------------------------------------
# Page Scraper
# --------------------------------------------------------------------------

def _html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    for tag in soup.select(".site-header, .site-footer, .footer, .header, #footer, #header"):
        tag.decompose()

    for tag in soup.select("[class*='cookie'], [id*='cookie'], [class*='consent'], [id*='consent'], "
                           "[class*='banner'], [class*='popup'], [class*='modal']"):
        tag.decompose()

    return " ".join(soup.stripped_strings)


def _looks_like_dynamic_shell(text: str, html: str = "") -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) < 350:
        return True
    lower = cleaned.lower()
    shell_markers = [
        "enable javascript",
        "please enable javascript",
        "root",
        "__next",
        "vite",
        "nuxt",
    ]
    if any(marker in lower for marker in shell_markers) and len(cleaned) < 1200:
        return True
    html_lower = (html or "").lower()
    return len(cleaned) < 900 and any(marker in html_lower for marker in ("__next", "vite", "webpack", "data-reactroot"))


def _scrape_page_with_browser(url: str, timeout: int = 45) -> str:
    log("[DEBUG] Falling back to browser-rendered scrape")
    import time
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception as exc:
        log(f"[WARN] Selenium is not available for rendered scrape: {exc}")
        return ""

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1366,1200")
        options.add_argument("--lang=el-GR")
        options.add_argument("user-agent=Mozilla/5.0")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        WebDriverWait(driver, min(timeout, 20)).until(
            lambda d: len(d.find_element(By.TAG_NAME, "body").text.strip()) > 300
        )
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        rendered_text = _html_to_clean_text(driver.page_source or "")
        if _looks_like_dynamic_shell(rendered_text, driver.page_source or ""):
            rendered_text = driver.find_element(By.TAG_NAME, "body").get_attribute("innerText") or ""
            rendered_text = re.sub(r"\s+", " ", rendered_text).strip()

        log(f"[DEBUG] Browser-rendered text length: {len(rendered_text)} characters")
        return rendered_text
    except Exception as exc:
        log(f"[WARN] Browser-rendered scrape failed: {exc}")
        return ""
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def scrape_page(url: str, timeout: int = 30) -> str:
    log("[DEBUG] Fetching page content")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()

    text = _html_to_clean_text(r.text)
    log(f"[DEBUG] Scraped text length: {len(text)} characters")

    if _looks_like_dynamic_shell(text, r.text):
        rendered_text = _scrape_page_with_browser(url)
        if len(rendered_text) > len(text):
            text = rendered_text
            log(f"[DEBUG] Using browser-rendered text: {len(text)} characters")

    return text


# --------------------------------------------------------------------------
# Prompt Builder
# --------------------------------------------------------------------------

def build_prompt(page_text: str, template: dict, url: str):
    system = (
        "You are a JSON data extractor. "
        "Extract data from the provided Greek/English text into the given JSON schema. "
        "Output ONLY the filled JSON. No markdown, no explanation, no extra text. "
        "Rules: Use '' for missing strings, [] for missing arrays. "
        "Project scope: extract residential/home energy-upgrade, home renewable-energy, and green home-loan information. "
        "Ignore electric-vehicle-only, mobility-only, business/SME, commercial, industrial, and public-building details unless they are directly tied to residential home energy upgrades. "
        "For completion_delay_consequences, extract only explicit consequences or handling rules for late/non-completion of works, failure to meet completion deadlines, or failure to meet required energy targets. "
        "For post_completion_obligations, extract only explicit beneficiary/borrower obligations after project completion, final disbursement, or loan disbursement. "
        "VERBATIM MODE: every non-empty value must be copied exactly from the source text as-is. "
        "Do NOT paraphrase, summarize, translate, transliterate, normalize, or infer. "
        "When a field is explicitly stated in the text, fill it with the shortest exact copied span that answers the field. "
        "Only leave a field empty when the information is genuinely not present. "
        "Keep original language/script, punctuation, accents, symbols, and casing from source snippets. "
        "Do NOT invent data."
    )
    template_str = json.dumps(template, ensure_ascii=False, indent=2)
    
    user = (
        f"URL: {url}\n\n"
        f"TEXT:\n{page_text}\n\n"
        f"SCHEMA:\n{template_str}\n\n"
        "Fill the schema using ONLY exact spans from TEXT. "
        "Focus on the residential/home energy part of the source. "
        "Do not fill fields from electric vehicle, business, SME, public-building, or commercial sections when they are outside that scope. "
        "For list/object items, each field value must be an exact substring from TEXT. "
        "Prefer short exact snippets, amounts, dates, bullet items, and named programme titles when they appear in TEXT. "
        "If uncertain or not exact, keep it empty. Return ONLY the JSON."
    )
    return [make_prompt("system", system), make_prompt("user", user)]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_json(llm, page_text: str, template: dict, url: str):
    log("[DEBUG] Building extraction prompt")
    
    max_retries = 3
    last_error = None
    prompts = build_prompt(page_text, template, url)  # Build once, reuse across retries
    
    for attempt in range(max_retries):
        try:
            # Increase temperature on retries
            temp = 0.1 + (attempt * 0.15)
            input_chars = sum(len(m.get("content", "")) for m in prompts)
            num_ctx = min(16384, max(4096, (input_chars // 3) + 2048))
            log(f"[INFO] LLM inference ({getattr(llm, '_model', '?')}, attempt {attempt+1}/{max_retries}, ~{input_chars} input chars)...")
            import time as _t; _t0 = _t.monotonic()
            raw = llm.ask(
                prompts, format="json", temperature=temp,
                num_predict=1536,
                num_ctx=num_ctx
            )
            log(f"[INFO] ✓ LLM responded in {_t.monotonic()-_t0:.1f}s ({len(raw)} chars output)")
            
            # --- Sanitize the LLM output ---
            cleaned = raw.strip()
            
            # Remove markdown code blocks
            if cleaned.startswith("```"):
                # Find content between ``` markers
                lines = cleaned.split('\n')
                if len(lines) > 2:
                    # Remove first and last line (the ``` markers)
                    cleaned = '\n'.join(lines[1:-1])
                    # Remove 'json' label if present at start
                    if cleaned.strip().startswith('json'):
                        cleaned = cleaned.strip()[4:].strip()
            
            # Extract JSON from text - be smart about finding the top-level structure
            # First, check if response starts with [ or { (after stripping)
            cleaned_stripped = cleaned.strip()
            
            # Try to extract based on what the response actually is
            if cleaned_stripped.startswith('['):
                # It's an array - find the matching closing bracket
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start != -1 and end != -1 and start < end:
                    cleaned = cleaned[start:end + 1]
            elif cleaned_stripped.startswith('{'):
                # It's an object - find the matching closing brace
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and start < end:
                    cleaned = cleaned[start:end + 1]
            else:
                # No clear structure - try array first, then object
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start != -1 and end != -1 and start < end:
                    cleaned = cleaned[start:end + 1]
                else:
                    # Fall back to object
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1 and start < end:
                        cleaned = cleaned[start:end + 1]
            
            # Debug: Show what we're trying to parse
            if len(cleaned) < 500:
                log(f"[DEBUG] Parsing JSON string: {cleaned[:200]}...")
            else:
                log(f"[DEBUG] Parsing JSON string ({len(cleaned)} chars): {cleaned[:100]}...{cleaned[-50:]}")
            
            # Parse JSON
            parsed = safe_read_json(cleaned)
            
            # Debug: Show parse result
            if parsed is None:
                log(f"[DEBUG] Parse result: None (JSON parsing failed)")
                log(f"[DEBUG] String that failed to parse ({len(cleaned)} chars): {cleaned[:200]}...")
            else:
                log(f"[DEBUG] Parse result type: {type(parsed)}, keys: {list(parsed.keys())[:5] if isinstance(parsed, dict) else 'N/A'}")
            
            # Handle None result
            if parsed is None:
                raise ValueError("JSON parsing returned None - LLM output was not valid JSON")
            
            # Convert single object to array if needed
            if isinstance(parsed, dict):
                log(f"[DEBUG] Converting dict to array")
                parsed = [parsed]
            
            # Validate structure
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                log(f"[DEBUG] Successfully extracted JSON (attempt {attempt + 1})")
                return parsed
            else:
                raise ValueError(f"Invalid JSON structure: expected array of objects, got {type(parsed)}")
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                log(f"[WARN] Extraction attempt {attempt + 1} failed: {e}")
                log(f"[INFO] Retrying with temperature={0.1 + ((attempt+1) * 0.15):.2f}...")
            else:
                # Save raw output for debugging
                try:
                    raw_path = save_raw_text(raw if 'raw' in locals() else 'No output', url + "_llm_raw")
                    log(f"[DEBUG] Raw LLM output saved to: {raw_path}")
                except Exception as save_err:
                    log(f"[WARN] Failed to save raw debug output: {save_err}")
    
    # All retries failed
    raise ValueError(f"Failed to extract valid JSON after {max_retries} attempts. Last error: {last_error}")


# --------------------------------------------------------------------------
# Main CLI Loop
# --------------------------------------------------------------------------

def interactive_loop():
    llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
    template = TEMPLATE_DEFAULT[0]
    log("VA-Scraper v3 ready. Type a URL, a broad topic, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter URL or broad topic (or 'exit')> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        # If the input looks like a URL or an existing local file, proceed with scraping+extraction.
        if (user_input.lower().startswith("http://") or user_input.lower().startswith("https://")
                or user_input.startswith("file://") or os.path.exists(user_input)):
            url = user_input
            try:
                log("[DEBUG] Starting scrape")
                url_to_use = None  # Initialize before branching
                # If it's a local file, read it directly instead of scraping
                if url.startswith("file://"):
                    fp = url[len("file://"):]
                    with open(fp, "r", encoding="utf-8") as f:
                        text = f.read()
                elif os.path.exists(url):
                    with open(url, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    url_to_use = _normalize_url(url)
                    text = scrape_page(url_to_use)
                # use normalized URL when saving/extracting to improve consistency
                save_key = url_to_use if url_to_use else url
                text_path = save_raw_text(text, save_key)
                log(f"[DEBUG] Saved cleaned text to {text_path}")

                log("[DEBUG] Sending to LLM for extraction")
                llm_output = extract_json(llm, text, template, save_key)
                saved_path = save_result(llm_output, save_key)

                log("\n===== JSON RESULT =====")
                _safe_console_print(json.dumps(llm_output, ensure_ascii=False, indent=2))
                log(f"\n[DEBUG] Saved structured output to: {saved_path}\n")

            except Exception as e:
                log(f"[ERROR] {e}")
        else:
            # Treat input as a broad topic and recommend sources/files.
            topic = user_input
            log(f"[INFO] Asking LLM for recommended sources for topic: {topic}")
            # Ask the LLM first for suggested sources; fall back to local trusted list
            try:
                recs = query_llm_for_sources(llm, topic, max_results=6)
            except Exception as e:
                log(f"[ERROR] LLM-based discovery failed: {e}")
                recs = []

            if not recs:
                log("[INFO] No model-suggested sources found for this topic. Falling back to local trusted sources.")
                # Fall back to the local trusted list so the user still gets suggestions
                try:
                    recs = recommend_sources_for_topic(topic)
                except Exception as e:
                    log(f"[ERROR] local fallback failed: {e}")
                    recs = []

            log("\n===== RECOMMENDED SOURCES =====")
            for i, r in enumerate(recs, start=1):
                _safe_console_print(f"{i}. {r.get('display_name', r['source'])}")
                _safe_console_print(f"   url: {r['source']}")
                _safe_console_print(f"   trusted_key: {r.get('trusted_key', '')}")
                _safe_console_print(f"   local_raw: {r['suggested_raw']}")
                _safe_console_print(f"   local_json: {r['suggested_json']}")
                _safe_console_print(f"   run command: {r['command']}")
            # Allow the user to pick one of the recommended sources and run it immediately
            pick = input("Pick index to scrape (or press Enter to skip)> ").strip()
            if pick:
                if pick.isdigit():
                    idx = int(pick)
                    if 1 <= idx <= len(recs):
                        # try the chosen suggestion, and on failure offer to try the next one
                        i = idx - 1
                        while i < len(recs):
                            sel = recs[i]
                            sel_url = sel["source"]
                            try:
                                log(f"[INFO] Starting scrape for recommended source #{i+1}")
                                # If this is a web URL, normalize host (IDN -> punycode) and check resolution
                                if sel_url.startswith("http://") or sel_url.startswith("https://"):
                                    sel_norm = _normalize_url(sel_url)
                                    if not _is_host_resolvable(sel_norm):
                                        raise RuntimeError(f"Host not resolvable for URL: {sel_norm}")
                                    # ask for confirmation before scraping
                                    confirm = input("Proceed to scrape this URL? [y/N]> ").strip().lower()
                                    if confirm not in {"y", "yes"}:
                                        log("[INFO] Skipping scrape by user choice.")
                                        raise RuntimeError("User skipped")
                                    text = scrape_page(sel_norm)
                                    save_key = sel_norm
                                # support file:// or filesystem paths returned by the model
                                elif sel_url.startswith("file://"):
                                    fp = sel_url[len("file://"):]
                                    with open(fp, "r", encoding="utf-8") as f:
                                        text = f.read()
                                    save_key = sel_url
                                elif os.path.exists(sel_url):
                                    with open(sel_url, "r", encoding="utf-8") as f:
                                        text = f.read()
                                    save_key = sel_url
                                else:
                                    # unknown form; try scraping normalized url
                                    sel_norm = _normalize_url(sel_url)
                                    text = scrape_page(sel_norm)
                                    save_key = sel_norm

                                text_path = save_raw_text(text, save_key)
                                log(f"[DEBUG] Saved cleaned text to {text_path}")

                                log("[DEBUG] Sending to LLM for extraction")
                                llm_output = extract_json(llm, text, template, save_key)
                                saved_path = save_result(llm_output, save_key)

                                log("\n===== JSON RESULT =====")
                                _safe_console_print(json.dumps(llm_output, ensure_ascii=False, indent=2))
                                log(f"\n[DEBUG] Saved structured output to: {saved_path}\n")
                                # success - break out
                                break
                            except Exception as e:
                                log(f"[ERROR] {e}")
                                # if there is another suggestion, ask whether to try it
                                if i + 1 < len(recs):
                                    try_next = input("Scrape failed. Try next suggestion? [y/N]> ").strip().lower()
                                    if try_next in {"y", "yes"}:
                                        i += 1
                                        continue
                                # either no more suggestions or user declined
                                break
                    else:
                        log("[INFO] Selection out of range; skipping.")
                else:
                    log("[INFO] Invalid selection; skipping.")
            else:
                log("\n[INFO] To scrape one of these, run this script with the URL as the first argument.")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # If arg appears to be a URL or an existing file, run a scrape+extract. Otherwise treat as topic.
        if (arg.lower().startswith("http://") or arg.lower().startswith("https://")
                or arg.startswith("file://") or os.path.exists(arg)):
            url = arg
            llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
            template = TEMPLATE_DEFAULT[0]

            # Support local files passed as file:// or plain paths
            if url.startswith("file://"):
                fp = url[len("file://"):]
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
                save_key = url
            elif os.path.exists(url):
                with open(url, "r", encoding="utf-8") as f:
                    text = f.read()
                save_key = url
            else:
                save_key = _normalize_url(url)
                text = scrape_page(save_key)

            text_path = save_raw_text(text, save_key)
            log(f"[DEBUG] Saved cleaned text to {text_path}")

            llm_output = extract_json(llm, text, template, save_key)
            saved_path = save_result(llm_output, save_key)

            _safe_console_print(json.dumps(llm_output, ensure_ascii=False, indent=2))
            log(f"[DEBUG] Saved structured output to: {saved_path}")
        else:
            topic = arg
            # Try to use the LLM to produce recommendations (useful for CLI usage)
            try:
                llm = OllamaChat(model="llama3.1:8b-instruct-q4_K_M")
                recs = query_llm_for_sources(llm, topic, max_results=8)
            except Exception as e:
                log(f"[ERROR] LLM-based discovery failed: {e}")
                recs = []
            # Print recommendations as JSON so it can be consumed programmatically.
            _safe_console_print(json.dumps(recs, ensure_ascii=False, indent=2))
    else:
        interactive_loop()


if __name__ == "__main__":
    main()

