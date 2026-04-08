# va4_product_discoverer.py
# -------------------------
# Virtual Assistant v4 for product/program discovery and classification.
# 
# Workflow:
# 1) Accepts a URL from the user.
# 2) Calls va3_scraper_to_template to scrape and extract structured data.
# 3) Uses LLM to classify the product/program into relevant categories.
# 4) If the category is of interest, enables interactive Q&A about the content.
#
# Categories of Interest:
# - Στεγαστικά δάνεια (Housing loans)
# - Ενεργειακή αναβάθμιση (Energy upgrade)
# - Ανακαινίσεις σπιτιών (Home renovations)
# - ΑΠΕ σε σπίτι (Renewable energy for homes)
#
# Categories NOT of Interest:
# - Ηλεκτρικά οχήματα (Electric vehicles)
# - Αγορές σπιτιών/ακινήτων (Property purchases)
# - ΑΠΕ εκτός σπιτιού (Renewable energy not for homes)

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Optional, Tuple

# Import va3 components
try:
    from llm_axe.va3_scraper_to_template import (
        scrape_page, extract_json, save_raw_text, save_result,
        TEMPLATE_DEFAULT, _normalize_url, _is_host_resolvable
    )
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.simple_logger import log_experiment
except Exception:
    print("[WARN] package-level imports failed; loading modules directly", file=sys.stderr)
    import importlib.util as _il
    here = os.path.dirname(__file__)
    for _mod in ("models", "core", "va3_scraper_to_template", "simple_logger"):
        _path = os.path.join(here, f"{_mod}.py")
        if os.path.exists(_path):
            spec = _il.spec_from_file_location(f"llm_axe.{_mod}", _path)
            module = _il.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[f"llm_axe.{_mod}"] = module
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.va3_scraper_to_template import (
        scrape_page, extract_json, save_raw_text, save_result,
        TEMPLATE_DEFAULT, _normalize_url, _is_host_resolvable
    )
    from llm_axe.simple_logger import log_experiment

# --------------------------------------------------------------------------
# Category Definitions
# --------------------------------------------------------------------------

CATEGORIES_OF_INTEREST = {
    "energy_upgrade": "Ενεργειακή αναβάθμιση (Energy upgrade/efficiency programs)",
    "home_renewables": "ΑΠΕ σε σπίτι (Renewable energy for homes - solar panels, heat pumps, insulation)",
    "green_housing_loan": "Πράσινο στεγαστικό δάνειο (Green housing loan with mandatory energy upgrades)"
}

CATEGORIES_NOT_OF_INTEREST = {
    "housing_loan": "Στεγαστικό δάνειο ΧΩΡΙΣ ενεργειακή διάσταση (Regular housing loan without energy requirements)",
    "home_renovation": "Ανακαινίσεις χωρίς ενεργειακή διάσταση (Home renovations without energy focus)",
    "electric_vehicles": "Ηλεκτρικά οχήματα (Electric vehicles)",
    "property_purchase": "Αγορές σπιτιών/ακινήτων (Property purchases/real estate acquisition)",
    "commercial_renewables": "ΑΠΕ εκτός σπιτιού (Renewable energy for commercial/industrial use)",
    "other": "Άλλο (Other topics not related to energy efficiency)"
}

ALL_CATEGORIES = {**CATEGORIES_OF_INTEREST, **CATEGORIES_NOT_OF_INTEREST}

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_outputs_dir() -> str:
    """Ensure output directory exists at ./output/va4_product_discoverer/"""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    out_dir = os.path.join(project_root, "output", "va4_product_discoverer")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_classification_result(url: str, extracted_data: dict, classification: dict) -> str:
    """Save classification result with metadata."""
    out_dir = ensure_outputs_dir()
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    path = os.path.join(out_dir, f"{ts}_{safe_name}_classification.json")
    
    result = {
        "timestamp": ts,
        "url": url,
        "classification": classification,
        "extracted_data": extracted_data
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path

def _create_minimal_data_from_text(text: str, url: str) -> dict:
    """Create minimal data structure when full extraction fails.
    
    Returns dict with programme_name + description derived from URL/text.
    Used as fallback when LLM extraction fails after all retries.
    """
    
    # Extract a title (first meaningful line or from URL)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    title = lines[0] if lines else url.split('/')[-1]
    if len(title) > 200:
        title = title[:200] + "..."
    
    # Try to extract a better title from URL path
    url_parts = url.rstrip('/').split('/')
    if len(url_parts) > 1:
        # Get last meaningful part (not index.html, etc)
        last_part = url_parts[-1]
        if last_part and last_part not in ['index.html', 'index.php', '']:
            # Clean up URL slug: replace - and _ with spaces, capitalize
            url_title = last_part.replace('-', ' ').replace('_', ' ').title()
            if 5 < len(url_title) < 200:
                title = url_title
    
    # Create minimal structure from template to stay in sync
    minimal = {k: ([] if isinstance(v, list) else "") for k, v in TEMPLATE_DEFAULT[0].items()}
    minimal["programme_name"] = title
    minimal["description"] = text[:500] if len(text) > 500 else text
    minimal["additional_details"] = f"Raw text from {url} (extraction failed)"
    return minimal

# --------------------------------------------------------------------------
# Classification Logic
# --------------------------------------------------------------------------

def build_classification_prompt(extracted_data: dict) -> List[dict]:
    """Build prompt for LLM to classify the program/product."""
    
    categories_list = "\n".join([f"- {key}: {desc}" for key, desc in ALL_CATEGORIES.items()])
    
    system = (
        "Είσαι ένας ειδικός αναλυτής προγραμμάτων ενεργειακής αναβάθμισης. "
        "Η δουλειά σου είναι να εντοπίζεις προγράμματα που αφορούν ενεργειακή αναβάθμιση κατοικιών και ΑΠΕ. "
        "ΣΗΜΑΝΤΙΚΟ: Πράσινα στεγαστικά δάνεια (που ΑΠΑΙΤΟΥΝ ενεργειακές επεμβάσεις) = RELEVANT. "
        "Απλά στεγαστικά για αγορά σπιτιού χωρίς ενεργειακές προϋποθέσεις = NOT relevant. "
        "ΟΛΟΚΛΗΡΩΣ άσχετα πράγματα (software, hardware, εφαρμογές, κλπ) = NOT relevant. "
        "Αν έχεις αμφιβολία αν είναι πράσινο/ενεργειακό, βάλε is_relevant=false. "
        "Απάντησε ΜΟΝΟ με JSON format."
    )
    
    # Use all extracted fields for maximum classification context.
    full_data = extracted_data if isinstance(extracted_data, dict) else {}

    user = (
        f"Ανάλυσε το παρακάτω πρόγραμμα/προϊόν και κατηγοριοποίησέ το.\n\n"
        f"ΔΙΑΘΕΣΙΜΕΣ ΚΑΤΗΓΟΡΙΕΣ:\n{categories_list}\n\n"
        f"ΔΕΔΟΜΕΝΑ ΠΡΟΓΡΑΜΜΑΤΟΣ:\n{json.dumps(full_data, ensure_ascii=False, indent=2)}\n\n"
        "Επίστρεψε JSON με τη μορφή:\n"
        "{\n"
        '  "is_relevant": true/false,  // true αν το πρόγραμμα αφορά ενεργειακή αναβάθμιση ή ΑΠΕ σε κατοικίες\n'
        '  "primary_category": "energy_upgrade",  // η κύρια κατηγορία (key από τη λίστα)\n'
        '  "secondary_categories": [],  // προαιρετικές δευτερεύουσες κατηγορίες\n'
        '  "confidence": 0.95,  // βαθμός βεβαιότητας (0-1)\n'
        '  "reasoning": "Σύντομη εξήγηση γιατί το κατηγοριοποίησες έτσι",\n'
        '  "key_features": ["χαρακτηριστικό 1", "χαρακτηριστικό 2"]  // κύρια χαρακτηριστικά\n'
        "}\n\n"
        "✅ RELEVANT (is_relevant=true):\n"
        "• Ενεργειακές επεμβάσεις σε ΚΑΤΟΙΚΙΕΣ: θερμομόνωση, κουφώματα, φωτοβολταϊκά, αντλίες θερμότητας, ηλιακοί\n"
        "• Πράσινα δάνεια με ΥΠΟΧΡΕΩΤΙΚΕΣ ενεργειακές επεμβάσεις (π.χ. Εξοικονομώ, Αναβαθμίζω)\n"
        "• Δάνεια που ΑΠΑΙΤΟΥΝ ελάχιστη ενεργειακή κλάση\n\n"
        "❌ NOT RELEVANT (is_relevant=false):\n"
        "• Απλά στεγαστικά ΧΩΡΙΣ ενεργειακές προϋποθέσεις\n"
        "• Γενικές ανακαινίσεις, καταναλωτικά δάνεια, εμπορικές ΑΠΕ\n"
        "• Ηλεκτρικά οχήματα, software, άσχετα θέματα\n\n"
        "⚠️ ΚΑΝΟΝΕΣ:\n"
        "1. Πρέπει να αφορά ενεργειακή αναβάθμιση ΚΑΤΟΙΚΙΩΝ\n"
        "2. Στεγαστικά χωρίς ενεργειακό κριτήριο → not relevant\n"
        "3. Αν δεν είναι σαφές → is_relevant=false\n"
        "4. is_relevant=true ΜΟΝΟ ΑΝ primary_category ∈ {energy_upgrade, home_renewables, green_housing_loan}"
    )
    
    return [make_prompt("system", system), make_prompt("user", user)]


def _apply_deterministic_classification_guard(classification: dict, extracted_data: dict, source_url: str = "") -> dict:
    """Apply lightweight rule-based guardrails to reduce obvious false negatives.

    This guard is intentionally conservative: it only flips to relevant when there are
    strong green-loan/energy signals from URL/name/fields.
    """
    out = dict(classification or {})

    # Normalize potentially malformed LLM fields to safe scalar types.
    if not isinstance(out.get("primary_category"), str):
        out["primary_category"] = "other"
    if not isinstance(out.get("is_relevant"), bool):
        out["is_relevant"] = bool(out.get("is_relevant", False))
    try:
        out["confidence"] = float(out.get("confidence", 0.0) or 0.0)
    except Exception:
        out["confidence"] = 0.0
    # Avoid mutating shared list objects that may come from caller payloads.
    if isinstance(out.get("key_features"), list):
        out["key_features"] = list(out["key_features"])
    primary = (out.get("primary_category") or "").strip()

    # Normalize malformed category values.
    if primary not in ALL_CATEGORIES:
        out["primary_category"] = "other"
        if out.get("is_relevant"):
            out["is_relevant"] = False
            out["reasoning"] = "Auto-corrected: invalid primary_category"

    # Enforce consistency: relevant must map to one of the energy/home categories of interest.
    if out.get("is_relevant") and out.get("primary_category") not in CATEGORIES_OF_INTEREST:
        out["is_relevant"] = False
        out["reasoning"] = "Auto-corrected: non-interest category cannot be relevant"

    def _to_lower_text(value) -> str:
        """Convert any JSON-like value to lowercase text safely."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, (list, tuple, set)):
            return " ".join(_to_lower_text(v) for v in value if v is not None)
        if isinstance(value, dict):
            # Include both keys and values to preserve useful signal words.
            parts = []
            for k, v in value.items():
                parts.append(_to_lower_text(k))
                parts.append(_to_lower_text(v))
            return " ".join(p for p in parts if p)
        return str(value).lower()

    name = _to_lower_text((extracted_data or {}).get("programme_name", ""))
    description = _to_lower_text((extracted_data or {}).get("description", ""))
    objective = _to_lower_text((extracted_data or {}).get("programme_objective", ""))
    funding_type = _to_lower_text((extracted_data or {}).get("funding_type", ""))
    energy_targets = _to_lower_text((extracted_data or {}).get("energy_performance_targets", ""))
    url_lower = _to_lower_text(source_url)

    interventions = (extracted_data or {}).get("eligible_interventions", [])
    has_interventions = isinstance(interventions, list) and len(interventions) > 0

    text_blob = " ".join([name, description, objective, funding_type, energy_targets, url_lower])

    strong_energy_signals = any(k in text_blob for k in [
        "ενεργειακ", "αναβαθμ", "εξοικονομ", "heat pump", "φωτοβολτα", "θερμομον", "κουφωμ", "ηλιακ",
        "energy", "energeiak", "anavathm", "retrofit", "insulation", "solar"
    ])
    strong_finance_signals = any(k in text_blob for k in [
        "δάνει", "loan", "στεγαστ", "mortgage", "χρηματοδ"
    ])

    # IMPORTANT: we only care about residential/home programs.
    strong_home_signals = any(k in text_blob for k in [
        "κατοικ", "σπίτι", "σπιτι", "οικί", "οικια",
        "home", "housing", "residential", "house", "apartment",
        "στεγαστ"
    ])

    # Guard against commercial/enterprise contexts.
    commercial_signals = any(k in text_blob for k in [
        "επιχειρ", "business", "commercial", "βιομηχαν", "industrial",
        "ξενοδοχ", "hotel", "factory", "γραφεί", "office"
    ])

    # Exclude domains you explicitly do NOT care about.
    vehicle_signals = any(k in text_blob for k in [
        "αμάξ", "αυτοκίνη", "οχημ", "ev", "electric vehicle", "ηλεκτρικ", "φορτισ"
    ])
    land_purchase_signals = any(k in text_blob for k in [
        "οικόπεδ", "οικοπεδ", "αγορά κατοικ", "αγορά ακιν", "property purchase", "real estate acquisition",
        "αγορά πρώτης κατοικίας", "αγορά σπιτιού", "buy home", "buy house"
    ])

    # Require explicit home energy-upgrade intent (not just generic housing/loan words).
    home_upgrade_signals = any(k in text_blob for k in [
        "ενεργειακ αναβαθ", "αναβάθμιση κατοικ", "αναβαθμιση κατοικ", "εξοικονομ", "θερμομον", "κουφωμ",
        "αντλία θερμότ", "αντλια θερμοτ", "φωτοβολτα", "ηλιακ", "energy upgrade", "home retrofit", "home energy"
        , "energy class", "energeiaki", "anavathm"
    ]) or has_interventions or bool(energy_targets.strip())

    # Only override when home energy-upgrade signals are clear and excluded domains are absent.
    if (
        (not out.get("is_relevant"))
        and strong_energy_signals
        and strong_finance_signals
        and strong_home_signals
        and home_upgrade_signals
        and not commercial_signals
        and not vehicle_signals
        and not land_purchase_signals
    ):
        out["is_relevant"] = True
        if out.get("primary_category") in ("other", "housing_loan", ""):
            # Green housing loan if loan-like; else generic energy upgrade.
            out["primary_category"] = "green_housing_loan" if ("δάνει" in text_blob or "loan" in text_blob) else "energy_upgrade"

        base_reason = (out.get("reasoning") or "").strip()
        override_reason = "Deterministic override: strong home energy-upgrade + financing signals"
        out["reasoning"] = f"{base_reason} | {override_reason}" if base_reason else override_reason

        confidence = float(out.get("confidence", 0.0) or 0.0)
        out["confidence"] = max(confidence, 0.72 if not has_interventions else 0.8)

        key_features = out.get("key_features", [])
        if not isinstance(key_features, list):
            key_features = []
        key_features.append("deterministic_energy_financing_home_override")
        out["key_features"] = key_features

    return out

def _get_program_name(extracted_data: dict) -> str:
    """Extract program name with fallback chain and validation.
    
    Validation:
    - Rejects if Cyrillic mixed with Latin (encoding corruption)
    - Rejects if < 5 chars (too short to be valid name)
    - Rejects if contains garbage keywords (ηγέτη, πρόεδρος, κλπ - hallucinations)
    
    Tries fields: programme_name → program_name → name → title
    Returns 'Άγνωστο πρόγραμμα' if all validation fails.
    """
    # Garbage keywords that indicate hallucination or wrong extraction
    GARBAGE_KEYWORDS = [
        "ηγέτη", "ηгέτη", "ηγετη",  # Hallucination: "leader" makes no sense in program names
        "cookie", "consent", "gdpr",  # UI elements from cookie banners
        "navigation", "menu", "footer", "sidebar", "header",  # HTML structure elements
        "αποδοχή", "απόρριψη", "συγκατάθεση",  # Cookie banner text
        "click here", "read more", "learn more",  # UI links
        "πολιτική απορρήτου", "όροι χρήσης"  # Legal pages
    ]
    
    # Try multiple field names
    candidates = [
        extracted_data.get('programme_name', ''),
        extracted_data.get('program_name', ''),
        extracted_data.get('name', ''),
        extracted_data.get('title', '')
    ]
    
    for name in candidates:
        if name and isinstance(name, str) and name.strip():
            name_clean = name.strip()
            name_lower = name_clean.lower()
            
            # Reject if too short (< 5 chars)
            if len(name_clean) < 5:
                continue
            
            # Reject if contains garbage keywords
            if any(garbage in name_lower for garbage in GARBAGE_KEYWORDS):
                log(f"[WARN] Rejected extracted name as garbage: '{name_clean}'")
                continue
            
            # Reject if contains non-Greek/English/common characters (mixed cyrillic)
            # Check for cyrillic characters that shouldn't be in Greek text
            cyrillic_pattern = r'[а-яА-ЯёЁ]'  # Russian cyrillic
            if re.search(cyrillic_pattern, name_clean):
                log(f"[WARN] Rejected name with mixed Cyrillic: '{name_clean}'")
                continue
            
            return name_clean
    
    return "Άγνωστο πρόγραμμα"

def _get_program_description(extracted_data: dict) -> str:
    """Extract program description with fallback chain.
    
    Tries fields: description → programme_objective → objective
    Returns empty string if no description found.
    """
    candidates = [
        extracted_data.get('description', ''),
        extracted_data.get('programme_objective', ''),
        extracted_data.get('objective', '')
    ]
    for desc in candidates:
        if desc and isinstance(desc, str) and desc.strip():
            return desc.strip()
    return ""

def _is_extracted_data_valid(extracted_data: dict) -> bool:
    """Validate if extracted data is meaningful before LLM classification.
    
    Pre-filters garbage extractions to avoid wasting LLM resources.
    
    Returns True ONLY if:
    - Dict is not completely empty
    - At least 1 field has content
    - programme_name > 10 chars OR description > 50 chars
    
    This catches minimal/fallback extractions before they reach classifier.
    """
    if not extracted_data:
        return False
    
    # Check if all fields are empty
    non_empty_fields = sum(1 for v in extracted_data.values() if v and str(v).strip())
    if non_empty_fields == 0:
        log("[DEBUG] Extracted data is completely empty")
        return False
    
    # Check if we have at least a decent programme_name or description
    name = extracted_data.get('programme_name', '').strip()
    desc = extracted_data.get('description', '').strip()
    
    # At least one meaningful field (>10 chars)
    has_meaningful_data = (len(name) > 10) or (len(desc) > 50)
    
    if not has_meaningful_data:
        log(f"[DEBUG] Extracted data lacks meaningful content (name={len(name)}c, desc={len(desc)}c)")
        return False
    
    return True


def _extract_relevant_text(text: str, window: int = 500) -> str:
    """Extract all keyword-relevant sections from text.
    
    Instead of blindly taking the first N chars (which may be boilerplate),
    finds ALL positions where energy/financing keywords appear and keeps
    a window of text around each one.  Overlapping windows are merged.
    
    Always includes the first 300 chars (title/header area).
    """
    # Short texts don't need filtering
    if len(text) <= 5000:
        return text
    
    _relevance_keywords = [
        # Energy
        "ενεργειακ", "θερμομόνωσ", "μόνωσ", "κουφώματ", "κουφωμ",
        "φωτοβολτα", "ηλιακ", "αντλία θερμότ", "αντλια θερμοτ",
        "εξοικονομ", "αναβαθμίζω", "αναβάθμισ", "αναβαθμισ",
        "πράσιν", "πρασιν", "green loan", "heat pump",
        "ηλεκτρα", "ήλεκτρα",
        # Financing
        "δάνειο", "δανειο", "επιδότ", "επιδοτ", "επιχορήγ",
        "χρηματοδότ", "χρηματοδοτ", "αποπληρωμ",
        # Template fields (capture key data areas)
        "ποσό", "ποσο", "ποσοστό", "επιτόκιο", "επιτοκιο",
        "δικαιούχ", "δικαιουχ", "κριτήρι", "κριτηρι",
        "παρέμβασ", "παρεμβασ", "επέμβασ", "επεμβασ",
        "προθεσμία", "προθεσμια", "διάρκεια", "διαρκεια",
        "προϋπόθεσ", "προυποθεσ", "προϋπολογισμ",
        "αίτηση", "αιτηση", "υποβολ",
    ]
    
    text_lower = text.lower()
    
    # Collect all keyword hit positions
    positions = set()
    for kw in _relevance_keywords:
        start = 0
        while True:
            idx = text_lower.find(kw, start)
            if idx == -1:
                break
            positions.add(idx)
            start = idx + len(kw)
    
    if not positions:
        # No keywords found — fall back to first 5000 chars
        log("[DEBUG] No relevance keywords found, using first 5000 chars")
        return text[:5000]
    
    # Build intervals: [max(0, pos-window) .. pos+window] for each hit
    intervals = []
    for pos in sorted(positions):
        lo = max(0, pos - window)
        hi = min(len(text), pos + window)
        intervals.append((lo, hi))
    
    # Always include title/header (first 300 chars)
    intervals.insert(0, (0, min(300, len(text))))
    
    # Merge overlapping intervals
    intervals.sort()
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    
    # Build final text from merged intervals
    parts = []
    for lo, hi in merged:
        parts.append(text[lo:hi])
    
    result = "\n[...]\n".join(parts)
    log(f"[INFO] Smart text extraction: {len(text)} → {len(result)} chars ({len(merged)} relevant sections)")
    return result


def _has_energy_keywords(text: str) -> bool:
    """Check if scraped text contains any energy-related keywords."""
    # Energy-related keywords in Greek and English
    ENERGY_KEYWORDS = [
        # Core energy terms
        "ενεργεια", "ενεργειακ", "energy class", "energy performance", "energy efficiency",
        # Insulation and building envelope
        "θερμομονωσ", "μονωσ", "κουφωμ", "insulation",
        # Renewables
        "φωτοβολτα", "φ/β", " pv ", "ηλιακ", "solar panel", "αιολικ",
        "αντλια θερμοτητ", "heat pump", "ανανεωσιμ", "renewable",
        "βιομαζα", "γεωθερμ", "geothermal", "biomass",
        # HVAC systems
        "καυστηρ", "θερμανσ", "ψυξ", "κλιματισμ",
        "heating system", "cooling system", " hvac ",
        # Energy programs & upgrades
        "εξοικονομ", "αναβαθμιζω", "αναβαθμ", "αναβάθμ",
        "απε", "πρασιν", "πράσιν", "green loan",
        "ηλεκτρα", "ήλεκτρα", "ilektra", "elektra",
        "σπιτι μου", "σπίτι μου", "anav",
        # Funding/interventions
        "επιδοτ", "επεμβασ", "retrofit", "αναβαθμισ", "εκσυγχρονισμ",
        "renovation", "ανακαινισ", "upgrades",
        # Energy certification
        "ενεργειακη κλασ", "ενεργειακη πιστοποι", "energy certificate",
        "κλιματικ", "εκπομπ co2", "carbon",
        # Photovoltaic variants
        "photovoltaic", "φωτοβολταϊκ"
    ]
    
    # Financing keywords - MUST be present for energy programs
    # NOTE: "πρόγραμμα" alone is too generic (used for nutrition programs, exercise programs, etc.)
    # Must combine with specific financing terms to avoid false positives
    FINANCING_KEYWORDS = [
        "δάνειο", "δανειο", "loan", "επιδότ", "subsidy", "επιχορήγησ",
        "χρηματοδότ", "funding", "χρηματ", "ενυπόθηκ",
        "κρέδιτ", "credit",
        # Only financing-specific compound terms:
        "προγραμμα χρηματοδότ", "financing program",
        "προγραμμα δανει", "loan program",
        "επιστροφή κόστους", "αποπληρωμ"
    ]
    
    text_lower = text.lower()
    
    # Check for energy keywords
    has_energy = any(keyword in text_lower for keyword in ENERGY_KEYWORDS)
    
    # Check for financing keywords
    has_financing = any(keyword in text_lower for keyword in FINANCING_KEYWORDS)
    
    # Log what was found (important for debugging)
    if has_energy:
        for keyword in ENERGY_KEYWORDS:
            if keyword in text_lower:
                log(f"[DEBUG] ✓ Energy keyword found: '{keyword}'")
                break
    else:
        log(f"[DEBUG] ✗ No energy keywords found")
    
    if has_financing:
        for keyword in FINANCING_KEYWORDS:
            if keyword in text_lower:
                log(f"[DEBUG] ✓ Financing keyword found: '{keyword}'")
                break
    else:
        log(f"[DEBUG] ✗ No financing keywords found")
    
    # Both conditions must be met
    if has_energy and has_financing:
        return True
    
    if has_energy and not has_financing:
        log("[WARN] Found energy keyword but NO financing keywords - likely not a financial program")
    elif not has_energy and has_financing:
        log("[WARN] Found financing keyword but NO energy keywords")
    else:
        log("[WARN] No energy-related keywords found in scraped text")
    
    return False

def prescreen_with_llm(llm, text: str, url: str) -> bool:
    """Lightweight LLM pre-screening: is this page about energy financing?
    
    Uses a very short prompt (~200 tokens) to quickly determine if the page
    is about energy financing programs before doing the expensive 23-field extraction.
    
    Returns True if page appears relevant, False otherwise.
    """
    # Build a smart snippet: find where energy keywords appear and grab context around them.
    # Bank pages often have 1000+ chars of navigation/FAQ boilerplate before the actual content.
    SNIPPET_SIZE = 1500
    snippet = text[:SNIPPET_SIZE]  # default: first N chars
    
    if len(text) > SNIPPET_SIZE:
        # Find the first energy keyword that's past the title area (>300 chars).
        # Keywords at char 0-300 are usually just the page title, followed by
        # hundreds of chars of navbar/FAQ boilerplate before real content starts.
        _prescreen_keywords = [
            "ενεργειακ", "θερμομόνωσ", "φωτοβολτα", "αντλία θερμότ",
            "εξοικονομ", "αναβαθμίζω", "αναβάθμισ", "πράσιν", "green loan",
            "heat pump", "insulation", "renewable", "ηλιακ", "κουφωμ",
            "δάνειο", "επιδότ", "χρηματοδότ", "επιχορήγ",
        ]
        text_lower = text.lower()
        TITLE_ZONE = 300  # skip keywords in the title/header area
        earliest = len(text)
        for kw in _prescreen_keywords:
            # Search past the title zone first
            idx = text_lower.find(kw, TITLE_ZONE)
            if idx != -1 and idx < earliest:
                earliest = idx
        
        if earliest < len(text):
            # Take a window starting 200 chars before the keyword
            start = max(0, earliest - 200)
            # Always prepend the title (first line) for context
            first_line = text.split('\n', 1)[0][:200]
            body_snippet = text[start:start + SNIPPET_SIZE - len(first_line) - 1]
            snippet = first_line + "\n" + body_snippet
            log(f"[DEBUG] Pre-screen snippet: title + body from char {start} (keyword at {earliest})")
        else:
            log("[DEBUG] Pre-screen snippet: using first 1500 chars (no keywords found past title)")
    
    prompt_system = (
        "Είσαι ένας ταχύς φιλτράρισμα-bot. Απαντάς ΜΟΝΟ με JSON: {\"relevant\": true/false, \"reason\": \"...\"}\n"
        "RELEVANT = πρόγραμμα/δάνειο/επιδότηση για ενεργειακή αναβάθμιση κατοικιών (θερμομόνωση, φωτοβολταϊκά, αντλίες θερμότητας, κλπ)\n"
        "NOT RELEVANT = blog, άρθρο, ειδήσεις, διατροφή, γενικά θέματα, εμπορικά προϊόντα, software"
    )
    
    prompt_user = (
        f"URL: {url}\n\n"
        f"ΑΠΟΣΠΑΣΜΑ:\n{snippet}\n\n"
        f"Αυτή η σελίδα αφορά πρόγραμμα/δάνειο/επιδότηση ενεργειακής αναβάθμισης κατοικιών;"
    )
    
    try:
        prompts = [
            make_prompt("system", prompt_system),
            make_prompt("user", prompt_user),
        ]
        raw = llm.ask(prompts, format="json", temperature=0.0, num_predict=100)
        
        # Parse response
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            if len(lines) > 2:
                cleaned = '\n'.join(lines[1:-1])
        
        # Find JSON
        if not cleaned.strip().startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]
        
        result = json.loads(cleaned)
        is_relevant = result.get("relevant", False)
        reason = result.get("reason", "")
        
        log(f"[PRE-SCREEN] LLM says: {'✓ RELEVANT' if is_relevant else '✗ NOT RELEVANT'}")
        if reason:
            log(f"[PRE-SCREEN] Reason: {reason}")
        
        return is_relevant
        
    except Exception as e:
        # If pre-screening fails, let it through (fail open)
        log(f"[WARN] LLM pre-screening failed: {e} — proceeding with extraction")
        return True

def classify_product(llm, extracted_data: dict, source_url: str = "", max_retries: int = 2, log_it: bool = True) -> Tuple[dict, Optional[str]]:
    """
    Use LLM to classify the product/program. Logs to logs/{experiment_id}.json
    
    NOTE: Keyword pre-check is done in process_url() BEFORE extraction.
    If we reach here, the text has already passed the keyword + LLM pre-screen.
    
    Args:
        llm: LLM instance
        extracted_data: Extracted program data
        max_retries: Max classification attempts
        log_it: Whether to save experiment log
    
    Returns:
        Tuple[classification_result, experiment_id]
    """
    log("[DEBUG] Classifying product/program...")
    
    model_name = getattr(llm, "_model", "unknown")
    experiment_id = None
    prompt_text = ""
    result = None
    actual_temperature = 0.1  # Track actual temperature used
    
    # Try LLM classification with retries
    for attempt in range(max_retries):
        try:
            prompts = build_classification_prompt(extracted_data)
            actual_temperature = 0.1 if attempt == 0 else 0.3  # Track actual temperature
            
            # Save prompt only on first attempt
            if attempt == 0:
                prompt_text = prompts[1].get("content", "")[:300] if len(prompts) > 1 else ""
            
            raw = llm.ask(prompts, format="json", temperature=actual_temperature, num_predict=512)
            
            # Clean response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json", "", 1).strip()
            
            # Extract JSON
            if not cleaned.strip().startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1:
                    cleaned = cleaned[start:end + 1]
            
            result = json.loads(cleaned)
            
            # Validate required fields
            if "is_relevant" in result and "primary_category" in result:
                # First normalize malformed types/values from LLM output.
                result = _apply_deterministic_classification_guard(result, extracted_data, source_url)

                # STRICT: is_relevant only true for energy categories
                if result["is_relevant"] and result["primary_category"] not in CATEGORIES_OF_INTEREST:
                    result["is_relevant"] = False
                    result["reasoning"] = f"Auto-corrected: {result['primary_category']} is not energy-related"
                
                # Log and return
                if log_it:
                    try:
                        experiment_id = log_experiment(
                            model=model_name,
                            temperature=actual_temperature,  # Use actual temperature from this attempt
                            hyperparameters={"max_retries": max_retries, "attempt": attempt + 1},
                            prompt=prompt_text,
                            sources=[_get_program_name(extracted_data)],
                            terminal_output=f"Category: {result['primary_category']} | Confidence: {result.get('confidence', 0):.0%} | Relevant: {result['is_relevant']}"
                        )
                        log(f"[LOG] Experiment: {experiment_id}")
                    except Exception as log_err:
                        log(f"[WARN] Failed to log experiment: {log_err}")
                        experiment_id = None
                
                return result, experiment_id
            else:
                raise ValueError("Missing required fields")
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            if attempt < max_retries - 1:
                log(f"[WARN] Attempt {attempt + 1} failed, retrying...")
            else:
                log(f"[ERROR] Failed after {max_retries} attempts: {e}")
                if log_it:
                    try:
                        experiment_id = log_experiment(
                            model=model_name,
                            temperature=actual_temperature,  # Use actual temperature from last attempt
                            hyperparameters={"max_retries": max_retries},
                            prompt=prompt_text,
                            sources=[_get_program_name(extracted_data)],
                            terminal_output=f"ERROR: Classification failed - {str(e)[:100]}"
                        )
                        log(f"[LOG] Error experiment: {experiment_id}")
                    except Exception as log_err:
                        log(f"[WARN] Failed to log experiment: {log_err}")
                        experiment_id = None
    
    # Fallback
    fallback = {
        "is_relevant": False,
        "primary_category": "other",
        "secondary_categories": [],
        "confidence": 0.0,
        "reasoning": "Failed to classify after retries",
        "key_features": []
    }
    
    if log_it and not experiment_id:
        try:
            experiment_id = log_experiment(
                model=model_name,
                temperature=actual_temperature,
                hyperparameters={"max_retries": max_retries},
                prompt=prompt_text,
                sources=[_get_program_name(extracted_data)],
                terminal_output="FALLBACK: Max retries exceeded"
            )
        except Exception as log_err:
            log(f"[WARN] Failed to log fallback experiment: {log_err}")
            experiment_id = None
    
    return fallback, experiment_id

# --------------------------------------------------------------------------
# Interactive Q&A
# --------------------------------------------------------------------------

def build_qa_prompt(extracted_data: dict, classification: dict, question: str) -> List[dict]:
    """Build prompt for Q&A about the program."""
    
    system = (
        "Είσαι ένας εξειδικευμένος βοηθός για χρηματοδοτικά προγράμματα. "
        "Απαντάς ερωτήσεις με βάση τα δεδομένα που έχεις διαθέσιμα. "
        "Απάντα στα ελληνικά, με σαφήνεια και ακρίβεια. "
        "Αν δεν ξέρεις την απάντηση ή δεν υπάρχει στα δεδομένα, πες το ξεκάθαρα."
    )
    
    context = (
        f"ΚΑΤΗΓΟΡΙΑ: {ALL_CATEGORIES.get(classification.get('primary_category', 'other'), 'Άγνωστη')}\n"
        f"ΒΕΒΑΙΟΤΗΤΑ: {classification.get('confidence', 0.0):.0%}\n\n"
        f"ΔΕΔΟΜΕΝΑ ΠΡΟΓΡΑΜΜΑΤΟΣ:\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n\n"
        f"ΕΡΩΤΗΣΗ: {question}"
    )
    
    return [make_prompt("system", system), make_prompt("user", context)]

def interactive_qa(llm, extracted_data: dict, classification: dict, url: str):
    """Start interactive Q&A session about the classified program."""
    
    log("\n" + "="*70)
    log("ΔΙΑΔΡΑΣΤΙΚΟ ΣΥΣΤΗΜΑ ΕΡΩΤΗΣΕΩΝ")
    log("="*70)
    log(f"\nURL: {url}")
    program_name = _get_program_name(extracted_data)
    # Only show program name if it's not the fallback
    if program_name != "Άγνωστο πρόγραμμα":
        log(f"Πρόγραμμα: {program_name}")
    log(f"Κατηγορία: {ALL_CATEGORIES.get(classification.get('primary_category'), 'Άγνωστη')}")
    log(f"Βεβαιότητα: {classification.get('confidence', 0.0):.0%}")
    
    # Show description if available and substantive
    description = _get_program_description(extracted_data)
    if description and len(description) > 50:
        desc_preview = description[:150] + "..." if len(description) > 150 else description
        log(f"Περιγραφή: {desc_preview}\n")
    
    if classification.get('key_features'):
        log(f"Κύρια Χαρακτηριστικά:")
        for feat in classification['key_features']:
            log(f"  • {feat}")
    
    log("\nΜπορείς να κάνεις ερωτήσεις για το πρόγραμμα.")
    log("Γράψε 'exit' ή 'quit' για έξοδο.\n")
    
    while True:
        try:
            question = input("❓ Ερώτηση> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("\n[INFO] Έξοδος από το διαδραστικό σύστημα.")
            break
        
        if not question:
            continue
        
        if question.lower() in {"exit", "quit", "q", "έξοδος"}:
            log("[INFO] Έξοδος από το διαδραστικό σύστημα.")
            break
        
        try:
            prompts = build_qa_prompt(extracted_data, classification, question)
            answer = llm.ask(prompts, format="", temperature=0.3)
            log(f"\n💡 Απάντηση:\n{answer}\n")
        except Exception as e:
            log(f"[ERROR] Σφάλμα κατά την απάντηση: {e}\n")

# --------------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------------

def process_url(url: str, llm_fast, llm_smart=None, enable_qa: bool = True) -> Tuple[dict, dict, Optional[str]]:
    """
    Complete workflow: scrape, pre-screen, extract, classify, and optionally start Q&A.
    
    Uses two LLM instances for optimal speed/quality trade-off:
        - llm_fast  (e.g. llama3.2 3B): Pre-screen + JSON Extraction
        - llm_smart (e.g. llama3.1 8B): Classification + Q&A
    
    If llm_smart is None, llm_fast is used for all steps (backward compat).
    
    Flow:
        1) Scrape webpage
        2) Pre-screen: keyword filter (free) + LLM quick check (llm_fast)
        3) Extract: fill 23-field JSON template (llm_fast)
        4) Classify: determine category and relevance (llm_smart)
        5) Q&A: interactive questions about the program (llm_smart)
    
    Returns:
        Tuple of (extracted_data, classification, experiment_id)
    """
    if llm_smart is None:
        llm_smart = llm_fast
    log(f"\n{'='*70}")
    log(f"ΕΠΕΞΕΡΓΑΣΙΑ URL: {url}")
    log(f"{'='*70}\n")
    
    # Step 1: Scrape + Pre-screen + Extract
    log("[1/3] Scraping, pre-screening και εξαγωγή δεδομένων...")
    max_retries = 2
    extracted_data = None
    text = ""  # Initialize text for scoping (used later in classify)
    scraped_text_saved = False
    
    for attempt in range(max_retries):
        try:
            # Normalize URL
            if url.startswith("http://") or url.startswith("https://"):
                url_normalized = _normalize_url(url)
                if not _is_host_resolvable(url_normalized):
                    raise RuntimeError(f"Host not resolvable: {url_normalized}")
                text = scrape_page(url_normalized)
            elif url.startswith("file://"):
                fp = url[len("file://"):]
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
                url_normalized = url
            elif os.path.exists(url):
                with open(url, "r", encoding="utf-8") as f:
                    text = f.read()
                url_normalized = url
            else:
                raise ValueError(f"Invalid URL or file path: {url}")

            # Save scraped/source text once so evaluation can validate JSON values
            # against the actual source text.
            if not scraped_text_saved:
                try:
                    scraped_path = save_raw_text(text, url_normalized)
                    log(f"[DEBUG] Saved scraped text to: {scraped_path}")
                    scraped_text_saved = True
                except Exception as save_err:
                    log(f"[WARN] Failed to save scraped text: {save_err}")
            
            # ── Pre-screening: reject irrelevant pages BEFORE expensive extraction ──
            
            # 1) Keyword pre-filter (free, no LLM)
            if not _has_energy_keywords(text):
                log("[REJECT] Keyword pre-filter: No energy + financing keywords found")
                rejection = {
                    "is_relevant": False,
                    "primary_category": "other",
                    "secondary_categories": [],
                    "confidence": 0.99,
                    "reasoning": "Αυτόματη απόρριψη: Δεν βρέθηκαν ενεργειακά + χρηματοδοτικά keywords.",
                    "key_features": ["keyword_prefilter_reject"]
                }
                experiment_id = None
                try:
                    experiment_id = log_experiment(
                        model=getattr(llm_fast, "_model", "unknown"),
                        temperature=0.0,
                        hyperparameters={"filter": "keyword_prescreen"},
                        prompt="[Keyword Pre-screen]",
                        sources=[url_normalized],
                        terminal_output="REJECTED at keyword pre-screen (before extraction)"
                    )
                    log(f"[LOG] Experiment: {experiment_id}")
                except Exception as log_err:
                    log(f"[WARN] Failed to log: {log_err}")
                return {}, rejection, experiment_id
            
            # 2) Lightweight LLM pre-screening (cheap, ~200 tokens)
            if not prescreen_with_llm(llm_fast, text, url_normalized):
                log("[REJECT] LLM pre-screen: Page is not about energy financing programs")
                rejection = {
                    "is_relevant": False,
                    "primary_category": "other",
                    "secondary_categories": [],
                    "confidence": 0.95,
                    "reasoning": "LLM pre-screening: Η σελίδα δεν αφορά ενεργειακό πρόγραμμα/δάνειο.",
                    "key_features": ["llm_prescreen_reject"]
                }
                experiment_id = None
                try:
                    experiment_id = log_experiment(
                        model=getattr(llm_fast, "_model", "unknown"),
                        temperature=0.0,
                        hyperparameters={"filter": "llm_prescreen"},
                        prompt="[LLM Pre-screen]",
                        sources=[url_normalized],
                        terminal_output="REJECTED at LLM pre-screen (before extraction)"
                    )
                    log(f"[LOG] Experiment: {experiment_id}")
                except Exception as log_err:
                    log(f"[WARN] Failed to log: {log_err}")
                return {}, rejection, experiment_id
            
            log("[✓] Pre-screening passed — proceeding with full extraction")
            
            # ── Full extraction (expensive LLM call) ──
            # Smart text selection: keep all relevant sections around keywords
            extract_text = _extract_relevant_text(text)
            
            # Extract structured data with retry
            template = TEMPLATE_DEFAULT[0]
            try:
                extracted_data_list = extract_json(llm_fast, extract_text, template, url_normalized)
                extracted_data = extracted_data_list[0] if extracted_data_list else {}
                
                # DEBUG: Show what keys we got
                log(f"[DEBUG] Extracted data has {len(extracted_data)} keys: {list(extracted_data.keys())[:10]}...")
                
                # Save extracted data for inspection (even if incomplete)
                saved_path = save_result(extracted_data_list, url_normalized)
                log(f"[DEBUG] Saved extracted data to: {saved_path}")
                
                # Display what was extracted
                program_name = _get_program_name(extracted_data)
                description = _get_program_description(extracted_data)
                
                # If extraction gave garbage, try to get a better name from URL
                if program_name == "Άγνωστο πρόγραμμα":
                    url_parts = url_normalized.rstrip('/').split('/')
                    if len(url_parts) > 1:
                        last_part = url_parts[-1]
                        if last_part and last_part not in ['index.html', 'index.php', '']:
                            # Clean up URL slug
                            url_name = last_part.replace('-', ' ').replace('_', ' ').title()
                            if 5 < len(url_name) < 200:
                                program_name = f"{url_name} (από URL)"
                                # Update extracted_data with better name for downstream use
                                extracted_data['programme_name'] = program_name
                                log(f"[INFO] Using URL-derived name: {program_name}")
                
                # Only show name/desc if we have a real program name (not fallback)
                has_real_name = program_name != "Άγνωστο πρόγραμμα"
                has_real_desc = description and len(description) > 50  # Substantive description
                
                if has_real_name:
                    log(f"✓ Εξαγωγή ολοκληρώθηκε: {program_name}")
                    if has_real_desc:
                        log(f"  Περιγραφή: {description[:80]}...")
                elif has_real_desc:
                    # Show description only if it's substantial and we don't have a name
                    desc_preview = description[:100] + "..." if len(description) > 100 else description
                    log(f"✓ Εξαγωγή ολοκληρώθηκε: {desc_preview}")
                else:
                    # No good name or description - just show field count
                    non_empty = sum(1 for v in extracted_data.values() if v and str(v).strip())
                    template_count = len(TEMPLATE_DEFAULT[0]) if TEMPLATE_DEFAULT else 0
                    log(f"✓ Εξαγωγή ολοκληρώθηκε: {non_empty}/{template_count} πεδία συμπληρώθηκαν")
                
                # Show key extracted info
                key_fields = []
                if extracted_data.get('funding_type'):
                    funding = extracted_data.get('funding_type')
                    if funding:
                        key_fields.append(f"Τύπος: {funding}")
                if extracted_data.get('maximum_funding_amount'):
                    amount = extracted_data.get('maximum_funding_amount')
                    if amount:
                        key_fields.append(f"Max ποσό: {amount}")
                if extracted_data.get('eligible_interventions'):
                    interventions = extracted_data.get('eligible_interventions', [])
                    if isinstance(interventions, list) and interventions:
                        key_fields.append(f"Επεμβάσεις: {len(interventions)}")
                
                if key_fields:
                    log(f"  → {', '.join(key_fields)}")
                
                break  # Success, exit retry loop
            except Exception as extraction_error:
                if attempt < max_retries - 1:
                    log(f"[WARN] Extraction attempt {attempt + 1} failed: {extraction_error}")
                    log(f"[INFO] Retrying with higher temperature...")
                    # Retry will happen in next iteration
                else:
                    # Last attempt failed, create minimal data from scraped text
                    log(f"[WARN] All extraction attempts failed. Creating minimal dataset...")
                    extracted_data = _create_minimal_data_from_text(text, url_normalized)
                    break
        
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"[WARN] Attempt {attempt + 1} failed: {e}")
                log(f"[INFO] Retrying...")
            else:
                log(f"[ERROR] Αποτυχία scraping/extraction μετά από {max_retries} προσπάθειες: {e}")
                raise
    
    # Validate extracted_data is not completely empty
    if not extracted_data or all(not str(v).strip() for v in extracted_data.values() if v):
        raise RuntimeError("Failed to extract any meaningful data after all retries")
    
    # Validation: Check if extracted data is meaningful before classification
    if not _is_extracted_data_valid(extracted_data):
        log("[REJECT] Extracted data lacks meaningful content (empty dict or insufficient quality fields)")
        # Skip LLM classification - return negative classification immediately
        classification = {
            "is_relevant": False,
            "primary_category": "other",
            "secondary_categories": [],
            "confidence": 0.95,
            "reasoning": "Η εξαγομένη δεδομένα δεν περιέχουν νόημα για κατηγοριοποίηση.",
            "key_features": ["empty_extraction"]
        }
        
        # Log the experiment (with exception handling)
        experiment_id = None
        try:
            experiment_id = log_experiment(
                model=getattr(llm_fast, "_model", "unknown"),
                temperature=0.0,
                hyperparameters={"max_retries": max_retries, "validation_filter": True},
                prompt="[Data Validation Pre-filter]",
                sources=[_get_program_name(extracted_data)],
                terminal_output="REJECTED: Data validation failed\nNo meaningful content in extraction"
            )
            log(f"[LOG] Experiment: {experiment_id}")
        except Exception as log_err:
            log(f"[WARN] Failed to log experiment: {log_err}")
        
        # Save the classification result (with exception handling)
        try:
            save_classification_result(url_normalized, extracted_data, classification)
            log("[DEBUG] Validation rejection saved to output folder")
        except Exception as save_err:
            log(f"[WARN] Failed to save validation rejection result: {save_err}")
        
        # Return early with rejection (3-tuple: extracted_data, classification, experiment_id)
        return extracted_data, classification, experiment_id
    
    # Step 2: Classify the product
    log("\n[2/3] Κατηγοριοποίηση προγράμματος...")
    try:
        classification, experiment_id = classify_product(llm_smart, extracted_data, source_url=url_normalized)
        
        # Ensure experiment_id is not None for logging references
        if experiment_id is None:
            log(f"[WARN] No experiment ID generated for classification")
        
        is_relevant = classification.get("is_relevant", False)
        primary_cat = classification.get("primary_category", "other")
        confidence = classification.get("confidence", 0.0)
        reasoning = classification.get("reasoning", "")
        
        log(f"\n{'='*70}")
        log("ΑΠΟΤΕΛΕΣΜΑ ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗΣ")
        log(f"{'='*70}")
        log(f"Κατηγορία: {ALL_CATEGORIES.get(primary_cat, 'Άγνωστη')}")
        log(f"Ενδιαφέρον: {'✓ ΝΑΙ' if is_relevant else '✗ ΟΧΙ'}")
        log(f"Βεβαιότητα: {confidence:.0%}")
        log(f"Αιτιολόγηση: {reasoning}")
        
        if classification.get("secondary_categories"):
            log(f"Δευτερεύουσες: {', '.join([ALL_CATEGORIES.get(c, c) for c in classification['secondary_categories']])}")
        
        log(f"{'='*70}\n")
        
        # Save classification with experiment_id matching
        save_path = save_classification_result(url_normalized, extracted_data, classification)
        if experiment_id:
            log(f"[DEBUG] Experiment Log: {experiment_id}")
        log(f"[DEBUG] Classification Output: {save_path}")
        
    except Exception as e:
        log(f"[ERROR] Αποτυχία κατηγοριοποίησης: {e}")
        raise
    
    # Step 3: Interactive Q&A if relevant
    if is_relevant and enable_qa:
        log("\n[3/3] Το πρόγραμμα ΜΑΣ ΕΝΔΙΑΦΕΡΕΙ! Ενεργοποίηση διαδραστικού συστήματος...")
        try:
            interactive_qa(llm_smart, extracted_data, classification, url_normalized)
        except Exception as e:
            log(f"[ERROR] Σφάλμα στο διαδραστικό σύστημα: {e}")
    else:
        if not is_relevant:
            log("\n[3/3] Το πρόγραμμα ΔΕΝ μας ενδιαφέρει. Παράλειψη Q&A.")
        else:
            log("\n[3/3] Q&A απενεργοποιημένο.")
    
    # Return 3-tuple: (extracted_data, classification, experiment_id)
    # experiment_id is now available from both validation rejection and successful classification paths
    return extracted_data, classification, experiment_id

# --------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------

def main():
    """Main entry point for va4_product_discoverer."""
    
    import argparse
    parser = argparse.ArgumentParser(
        description="VA4 Product Discoverer: Scrape, classify, and explore financing programs"
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL to process (or file path)"
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Disable interactive Q&A even for relevant programs"
    )
    parser.add_argument(
        "--model-fast",
        default="llama3.2:latest",
        help="Fast LLM for pre-screen + extraction (default: llama3.2:latest)"
    )
    parser.add_argument(
        "--model-smart",
        default="llama3.1:8b-instruct-q4_K_M",
        help="Smart LLM for classification + Q&A (default: llama3.1:8b-instruct-q4_K_M)"
    )
    
    args = parser.parse_args()
    
    # Initialize dual LLMs
    log(f"[INFO] LLM fast  (pre-screen + extraction): {args.model_fast}")
    log(f"[INFO] LLM smart (classification + Q&A):     {args.model_smart}")
    llm_fast  = OllamaChat(model=args.model_fast)
    llm_smart = OllamaChat(model=args.model_smart)
    
    if args.url:
        # Single URL mode
        try:
            process_url(args.url, llm_fast, llm_smart, enable_qa=not args.no_qa)
        except Exception as e:
            log(f"\n[ERROR] Αποτυχία επεξεργασίας: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        log("="*70)
        log("VA4 PRODUCT DISCOVERER - ΔΙΑΔΡΑΣΤΙΚΟ ΜΟΔΕ")
        log("="*70)
        log("\nΕισάγετε URL για ανάλυση ή 'exit' για έξοδο.\n")
        
        while True:
            try:
                user_input = input("🔗 URL> ").strip()
            except (EOFError, KeyboardInterrupt):
                log("\n\n[INFO] Έξοδος.")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in {"exit", "quit", "q", "έξοδος"}:
                log("[INFO] Έξοδος.")
                break
            
            try:
                process_url(user_input, llm_fast, llm_smart, enable_qa=not args.no_qa)
            except Exception as e:
                log(f"\n[ERROR] Αποτυχία επεξεργασίας: {e}\n")
                continue

if __name__ == "__main__":
    main()
