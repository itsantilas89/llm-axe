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
import unicodedata
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# Import va3 components
try:
    from llm_axe.va3_scraper_to_template import (
        scrape_page, extract_json, save_raw_text, save_result,
        TEMPLATE_DEFAULT, _normalize_url, _is_host_resolvable,
        enrich_program_identity,
    )
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.qa_prompting import build_single_qa_prompt, deterministic_qa_out_of_scope_answer
    from llm_axe.merged_snapshot import refresh_merged_snapshot
    from llm_axe.simple_logger import log_experiment
except Exception:
    print("[WARN] package-level imports failed; loading modules directly", file=sys.stderr)
    import importlib.util as _il
    here = os.path.dirname(__file__)
    for _mod in ("models", "core", "qa_prompting", "merged_snapshot", "va3_scraper_to_template", "simple_logger"):
        _path = os.path.join(here, f"{_mod}.py")
        if os.path.exists(_path):
            spec = _il.spec_from_file_location(f"llm_axe.{_mod}", _path)
            module = _il.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[f"llm_axe.{_mod}"] = module
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.qa_prompting import build_single_qa_prompt, deterministic_qa_out_of_scope_answer
    from llm_axe.merged_snapshot import refresh_merged_snapshot
    from llm_axe.va3_scraper_to_template import (
        scrape_page, extract_json, save_raw_text, save_result,
        TEMPLATE_DEFAULT, _normalize_url, _is_host_resolvable,
        enrich_program_identity,
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
DEFAULT_MODEL_FAST = "llama3.2:latest"
DEFAULT_MODEL_SMART = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_MODEL_QA = DEFAULT_MODEL_SMART
DATE_FIELDS = {
    "announcement_date",
    "application_start_date",
    "application_end_date",
    "completion_deadline",
}

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

def list_latest_relevant_classification_urls(output_dir: Optional[str] = None) -> List[str]:
    """Return URLs whose latest VA4 classification is relevant."""
    out_dir = output_dir or ensure_outputs_dir()
    latest_by_url: dict[str, dict] = {}

    try:
        filenames = sorted(os.listdir(out_dir))
    except OSError:
        return []

    for filename in filenames:
        if not filename.endswith("_classification.json"):
            continue

        path = os.path.join(out_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict) or not isinstance(payload.get("classification"), dict):
            continue

        extracted = payload.get("extracted_data")
        if not isinstance(extracted, dict):
            extracted = {}

        url = (payload.get("url") or extracted.get("source_url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue

        key = url.rstrip("/").casefold()
        timestamp = _classification_output_timestamp(path, payload)
        current = latest_by_url.get(key)
        if current is None or timestamp > current["timestamp"]:
            latest_by_url[key] = {
                "url": url,
                "timestamp": timestamp,
                "is_relevant": payload["classification"].get("is_relevant") is True,
            }

    return [
        item["url"]
        for item in sorted(latest_by_url.values(), key=lambda entry: entry["url"].casefold())
        if item["is_relevant"]
    ]

def _classification_output_timestamp(path: str, payload: dict) -> datetime:
    timestamp = payload.get("timestamp")
    parsed = _parse_output_timestamp(timestamp if isinstance(timestamp, str) else "")
    if parsed is not None:
        return parsed

    match = re.match(r"^(\d{8}T\d{6}Z)_", os.path.basename(path))
    if match:
        parsed = _parse_output_timestamp(match.group(1))
        if parsed is not None:
            return parsed

    try:
        return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    except OSError:
        return datetime.min.replace(tzinfo=timezone.utc)

def _parse_output_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if re.match(r"^\d{8}T\d{6}Z$", value):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _text_for_matching(value) -> str:
    """Return lowercase, accent-insensitive text from JSON-like values."""
    if value is None:
        return ""
    if isinstance(value, dict):
        value = " ".join(
            f"{_text_for_matching(k)} {_text_for_matching(v)}"
            for k, v in value.items()
        )
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(_text_for_matching(v) for v in value)
    else:
        value = str(value)

    normalized = unicodedata.normalize("NFD", value.casefold())
    without_accents = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    without_accents = without_accents.replace("ς", "σ")
    return re.sub(r"\s+", " ", without_accents).strip()


def _text_for_position_matching(value: str) -> str:
    """Accent-insensitive text that preserves character positions."""
    normalized = unicodedata.normalize("NFD", str(value).casefold())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _derive_title_from_url(url: str) -> str:
    slug = (url or "").rstrip("/").split("/")[-1]
    if not slug:
        return ""
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return " ".join(word[:1].upper() + word[1:] for word in slug.split())


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(_text_for_matching(needle) in text for needle in needles)


HOME_GREEN_SCOPE_TERMS = {
    "home": [
        "κατοικ", "σπιτι", "σπιτιου", "οικια", "οικιακ", "στεγη", "στεγαστικ",
        "διαμερισμα", "νοικοκυρι", "home", "house", "housing", "household",
        "residential", "apartment", "roof", "eco home", "home upgrade",
    ],
    # Generic "home" often appears as a website route (/home) and must not be enough
    # to make public/business energy programs look residential.
    "home_specific": [
        "κατοικ", "σπιτι", "σπιτιου", "οικια", "οικιακ", "στεγη", "στεγαστικ",
        "διαμερισμα", "νοικοκυρι", "household", "residential", "apartment",
        "roof", "eco home", "home upgrade", "home energy", "home retrofit",
        "green home", "green housing",
    ],
    "energy": [
        "ενεργεια", "ενεργειακ", "εξοικονομ", "αναβαθμ", "πρασιν",
        "θερμομον", "μονωση", "κουφωμα", "κουφωματα", "φωτοβολται",
        "φωτοβολταϊ", "ηλιακ", "αντλια θερμο", "κλιματισ", "θερμαν",
        "ψυξη", "απε", "ανανεωσιμ", "energy", "energy efficiency",
        "energy upgrade", "green", "retrofit", "insulation", "solar",
        "photovoltaic", "heat pump", "heating", "cooling", "renewable",
    ],
    "finance": [
        "δανει", "δανειο", "επιδοτ", "επιχορηγ", "χρηματοδοτ",
        "χρηματοδοτηση", "τοκ", "επιτοκ", "loan", "mortgage", "funding",
        "subsidy", "grant", "co financing", "credit", "interest",
    ],
    "home_program": [
        "εξοικονομω", "αναβαθμιζω το σπιτι μου", "αναβαθμιζω",
        "prasino daneio katoikias", "prasino episkeuastiko",
        "prasina daneia", "green home loan", "green housing loan",
        "eco home", "exoikonomo", "anavathmizo", "energeiaki anavathmisi katoikias",
    ],
    "home_appliance": [
        "οικιακες συσκευες", "λευκες οικιακες συσκευες", "συσκευες θερμανσης",
        "συσκευες ψυξης", "κλιματιστικ", "θερμοπομπ", "household appliance",
        "home appliance", "green appliances",
    ],
    "business": [
        "επιχειρ", "επαγγελματ", "μμε", "sme", "smes", "business",
        "enterprise", "commercial", "industrial", "industry", "hotel",
        "ξενοδοχ", "γραφει", "office", "factory",
    ],
    "public": [
        "δημοσι", "δημοσιο", "public sector", "public building", "hlektra",
        "ηλεκτρα", "municipal", "δημων", "δημους",
    ],
    "vehicle": [
        "αυτοκινη", "οχημ", "ηλεκτροκινη", "electric vehicle", "vehicle",
        "car", "ev ", "scooter", "ποδηλατ", "πατιν", "μηχανη",
        "autokinito", "ilektriko autokinito", "ilektriko-autokinito",
    ],
    "vehicle_dominant": [
        "δανειο για ηλεκτρικο αυτοκινητο",
        "δανειο για αγορα ηλεκτρικου αυτοκινητου",
        "δανειο ηλεκτρικο αυτοκινητο",
        "ηλεκτρικου αυτοκινητου",
        "ηλεκτρικο αυτοκινητο",
        "υβριδικο η ηλεκτρικο αυτοκινητο",
        "υβριδικου η ηλεκτρικου αυτοκινητου",
        "οικολογικο οχημα",
        "οικολογικου οχηματος",
        "πρασινο οχημα",
        "πρασινου οχηματος",
        "ev-loan",
        "ev loan",
        "ilektriko autokinito",
        "ilektriko-autokinito",
        "ilektrikou autokinitou",
        "ilektrikou-autokinitou",
        "electric car",
        "electric vehicle loan",
    ],
}


def _home_green_scope_signals(*values) -> dict:
    text = _text_for_matching(" ".join(_text_for_matching(v) for v in values if v is not None))
    has_home = _contains_any(text, HOME_GREEN_SCOPE_TERMS["home"])
    has_home_specific = _contains_any(text, HOME_GREEN_SCOPE_TERMS["home_specific"])
    has_energy = _contains_any(text, HOME_GREEN_SCOPE_TERMS["energy"])
    has_finance = _contains_any(text, HOME_GREEN_SCOPE_TERMS["finance"])
    has_home_program = _contains_any(text, HOME_GREEN_SCOPE_TERMS["home_program"])
    has_home_appliance = _contains_any(text, HOME_GREEN_SCOPE_TERMS["home_appliance"])
    has_business = _contains_any(text, HOME_GREEN_SCOPE_TERMS["business"])
    has_public = _contains_any(text, HOME_GREEN_SCOPE_TERMS["public"])
    has_vehicle = _contains_any(text, HOME_GREEN_SCOPE_TERMS["vehicle"])
    has_vehicle_dominant = _contains_any(text, HOME_GREEN_SCOPE_TERMS["vehicle_dominant"])
    has_mixed_home_vehicle_product = _contains_any(
        text,
        [
            "σπιτιου η αυτοκινητου",
            "σπιτι η αυτοκινητου",
            "home or car",
            "spitiou h autokinhtou",
        ],
    )

    home_energy = (has_home_specific and has_energy) or has_home_program or has_home_appliance
    scoped_candidate = has_finance and home_energy
    vehicle_only = (
        has_vehicle
        and not has_home_appliance
        and (
            not home_energy
            or (has_vehicle_dominant and not has_mixed_home_vehicle_product)
        )
    )
    business_or_public_only = (has_business or has_public) and not (has_home_specific or has_home_appliance)

    return {
        "text": text,
        "has_home": has_home,
        "has_home_specific": has_home_specific,
        "has_energy": has_energy,
        "has_finance": has_finance,
        "has_home_program": has_home_program,
        "has_home_appliance": has_home_appliance,
        "has_business": has_business,
        "has_public": has_public,
        "has_vehicle": has_vehicle,
        "has_vehicle_dominant": has_vehicle_dominant,
        "has_mixed_home_vehicle_product": has_mixed_home_vehicle_product,
        "home_energy": home_energy,
        "scoped_candidate": scoped_candidate,
        "vehicle_only": vehicle_only,
        "business_or_public_only": business_or_public_only,
        "in_scope": scoped_candidate and not vehicle_only and not business_or_public_only,
    }


def _is_home_green_finance_candidate(*values) -> tuple[bool, str, dict]:
    signals = _home_green_scope_signals(*values)
    if signals["in_scope"]:
        return True, "home energy finance signals found", signals
    if signals["business_or_public_only"]:
        return False, "business/public context without residential home signal", signals
    if signals["vehicle_only"]:
        return False, "vehicle-only green finance context", signals
    if not signals["has_finance"]:
        return False, "missing financing signal", signals
    if not signals["home_energy"]:
        return False, "missing residential home energy-upgrade signal", signals
    return False, "outside scoped home green finance rules", signals


def _best_relevant_excerpt(text: str, max_chars: int = 3000) -> str:
    if not text:
        return ""
    excerpt = _extract_relevant_text(text, window=900)
    return excerpt[:max_chars]


def _first_source_title(text: str) -> str:
    head = re.sub(r"\s+", " ", (text or "").strip())
    if not head:
        return ""
    match = re.match(r"^(.{5,160}?)\s+\|\s+", head)
    if match:
        title = match.group(1).strip()
    else:
        title = head[:120].strip()
    title = re.split(
        r"\s+(Επιλέξτε γλώσσα|Top Menu|Main Menu|Page Contents|Footer|Αναζήτηση|Χρηματοδοτούμε|μάθετε περισσότερα|Μάθετε περισσότερα)\b",
        title,
    )[0].strip()
    return title if 5 <= len(title) <= 180 else ""


def _source_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    chunks = re.split(r"(?<=[.!;;])\s+", clean)
    sentences = []
    for chunk in chunks:
        chunk = chunk.strip()
        if 40 <= len(chunk) <= 600:
            sentences.append(chunk)
    return sentences


def _source_policy_windows(text: str, needles: list[str], radius: int = 360) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []

    normalized = _text_for_matching(clean)
    windows = []
    for needle in needles:
        needle_norm = _text_for_matching(needle)
        if not needle_norm:
            continue
        start = 0
        while True:
            idx = normalized.find(needle_norm, start)
            if idx == -1:
                break
            lo = max(0, idx - radius)
            hi = min(len(clean), idx + len(needle_norm) + radius)
            left_boundary = max(clean.rfind(".", 0, idx), clean.rfind(";", 0, idx), clean.rfind(";", 0, idx))
            right_candidates = [
                pos
                for pos in (
                    clean.find(".", idx + len(needle_norm)),
                    clean.find(";", idx + len(needle_norm)),
                    clean.find(";", idx + len(needle_norm)),
                )
                if pos != -1
            ]
            if left_boundary != -1:
                lo = left_boundary + 1
            if right_candidates:
                hi = min(right_candidates) + 1
            snippet = _trim_source_snippet(clean[lo:hi])
            if 45 <= len(snippet) <= 900:
                windows.append(snippet)
            start = idx + len(needle_norm)
    return windows


def _best_policy_snippet(text: str, field: str) -> str:
    if not text:
        return ""

    if field == "completion_delay_consequences":
        needles = [
            "δεν ολοκληρω",
            "μη ολοκλήρωση",
            "μη ολοκληρωση",
            "δεν έχουν ολοκληρωθεί",
            "δεν εχουν ολοκληρωθει",
            "δεν έχει πετύχει",
            "δεν εχει πετυχει",
            "δεν τον έχετε πετύχει",
            "δεν τον εχετε πετυχει",
            "εκπρόθεσ",
            "εκπροθεσ",
            "υπερημερία",
            "υπερημερια",
            "απένταξ",
            "απενταξ",
            "ανακαλείται",
            "ανακαλειται",
            "ανάκληση",
            "ανακληση",
            "επιστροφή",
            "επιστροφη",
            "κυρώσ",
            "κυρωσ",
        ]
        required_groups = [
            [
                "ολοκληρω",
                "υλοποι",
                "εργασ",
                "εργου",
                "προθεσμ",
                "ενεργειακο στοχο",
                "ενεργειακη κατηγορια",
                "τον εχετε πετυχει",
                "πετυχει",
            ],
            [
                "ανακαλ",
                "απενταξ",
                "επιστροφ",
                "κυρωσ",
                "δεν θα μπορειτε να λαβετε",
                "τοκοι υπερημεριας",
                "επιπτωσ",
                "επιβαρυν",
            ],
        ]
        blockers = ["10 λεπτα", "online αιτηση", "ηλεκτρονικη υπογραφη", "ειδοποιητηριο", "χωρισ επιβαρυνση"]
    elif field == "post_completion_obligations":
        needles = [
            "μετά την ολοκλήρωση",
            "μετα την ολοκληρωση",
            "μετά την εκταμίευση",
            "μετα την εκταμιευση",
            "τελική εκταμίευση",
            "τελικη εκταμιευση",
            "μετά την υπογραφή",
            "μετα την υπογραφη",
            "υποχρεούται",
            "υποχρεουται",
            "υποχρέωση",
            "υποχρεωση",
            "αποστολή απόδειξης",
            "αποστολη αποδειξης",
            "αποδείξεις που ανεβάζετε",
            "αποδειξεις που ανεβαζετε",
            "παραστατικά",
            "παραστατικα",
        ]
        required_groups = [
            [
                "μετα την ολοκληρωση",
                "μετα την εκταμιευση",
                "τελικη εκταμιευση",
                "μετα την υπογραφη",
                "υποχρεουται",
                "υποχρεωση",
                "πρεπει",
                "απαραιτητο",
                "χρειαζεται",
            ],
            [
                "αποδειξ",
                "παραστατικ",
                "δικαιολογητικ",
                "πεα",
                "ενεργειακη επιθεωρηση",
                "μηχανικ",
                "πιστοποιητικ",
                "ασφαλισ",
                "προσημειωση",
                "ελεγχει την προοδο",
            ],
        ]
        blockers = ["10 λεπτα", "online αιτηση", "εχει περιθωριο"]
    else:
        return ""

    best = ""
    best_score = 0
    candidates = _source_sentences(text) + _source_policy_windows(text, needles)
    seen = set()
    for candidate in candidates:
        candidate = _trim_source_snippet(candidate)
        key = _text_for_matching(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if any(blocker in key for blocker in blockers):
            continue
        group_hits = [any(marker in key for marker in group) for group in required_groups]
        if (
            field == "completion_delay_consequences"
            and "υπερημερι" in key
            and not any(marker in key for marker in ["ολοκληρω", "εργασ", "ενεργειακ", "προθεσμ"])
        ):
            continue
        if field == "completion_delay_consequences" and not all(group_hits):
            continue
        if field == "post_completion_obligations":
            has_post_context = any(
                marker in key
                for marker in [
                    "μετα την ολοκληρωση",
                    "μετα την εκταμιευση",
                    "τελικη εκταμιευση",
                    "μετα την υπογραφη",
                    "μετα απο τον ελεγχο",
                    "μεσα σε 30 ημερες απο την υπογραφη",
                    "παραμεινει το επιτοκιο",
                    "υποχρεουται",
                ]
            )
            has_obligation_payload = group_hits[1] or any(
                marker in key
                for marker in [
                    "ειναι απαραιτητο να",
                    "πρεπει να μας στειλετε",
                    "υποχρεουται",
                    "υποχρεωση",
                    "προσκομιση",
                ]
            )
            if (
                any(marker in key for marker in ["καταργειται", "επιβαρυνεται", "ανακαλειται", "απενταξ"])
                and not any(marker in key for marker in ["πρεπει", "απαραιτητο", "υποχρε", "προσκομιση", "παραστατικ", "αποδειξ"])
            ):
                continue
            if not (has_post_context and has_obligation_payload):
                continue
        score = 0
        for hit in group_hits:
            if hit:
                score += 3
        if any(_text_for_matching(needle) in key for needle in needles):
            score += 2
        if _contains_any(key, HOME_GREEN_SCOPE_TERMS["vehicle"]) and not _contains_any(key, HOME_GREEN_SCOPE_TERMS["home"]):
            score -= 4
        if _contains_any(key, ["cookie", "javascript", "αναζητηση", "top menu", "footer"]):
            score -= 3
        if score > best_score:
            best = candidate
            best_score = score

    return best if best_score >= 5 else ""


def _iter_policy_value_candidates(value) -> list[str]:
    if value in ("", None, [], {}):
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        return []

    candidates = []
    for item in values:
        candidate = _trim_source_snippet(re.sub(r"\s+", " ", item or "").strip())
        if 20 <= len(candidate) <= 900:
            candidates.append(candidate)
    return candidates


def _policy_value_score(field: str, value: str, source_text: str) -> int:
    if not value or not source_text:
        return 0
    if not _value_has_source_support(value, source_text, min_recall=0.80):
        return 0

    key = _text_for_matching(value)
    if _contains_any(key, ["cookie", "javascript", "αναζητηση", "top menu", "footer"]):
        return 0

    if field == "completion_delay_consequences":
        if any(marker in key for marker in ["δωρεαν δαπανη", "απαντηση σε 48", "εκταμιευση σε 24"]):
            return 0
        has_completion_context = any(
            marker in key
            for marker in [
                "δεν ολοκληρω",
                "μη ολοκληρω",
                "μετα την ολοκληρωση των εργασιων",
                "ενεργειακο στοχο",
                "ενεργειακη κατηγορια",
                "ανεγερση επισκευη",
                "ανεγερση/επισκευη",
                "προθεσμ",
            ]
        )
        has_consequence = any(
            marker in key
            for marker in [
                "ανακαλ",
                "δεν θα μπορειτε να λαβετε",
                "καταργειται",
                "επιβαρυν",
                "απενταξ",
                "επιστροφ",
                "κυρωσ",
            ]
        )
        if "υπερημερι" in key and not any(marker in key for marker in ["ολοκληρω", "εργασ", "ενεργειακ", "προθεσμ"]):
            return 0
        if not (has_completion_context and has_consequence):
            return 0
        score = 7
        score += 3 if _source_contains_normalized(source_text, value) else 0
        score += 2 if "δεν" in key or "μη" in key or "εφοσον" in key else 0
        return score

    if field == "post_completion_obligations":
        if (
            any(marker in key for marker in ["καταργειται", "επιβαρυνεται", "ανακαλειται", "απενταξ"])
            and not any(marker in key for marker in ["πρεπει", "απαραιτητο", "υποχρε", "προσκομιση", "παραστατικ", "αποδειξ"])
        ):
            return 0
        has_post_context = any(
            marker in key
            for marker in [
                "μετα την ολοκληρωση",
                "μετα την εκταμιευση",
                "τελικη εκταμιευση",
                "μετα την υπογραφη",
                "μετα απο τον ελεγχο",
                "εντος 6 μηνων απο την εκταμιευση",
                "μεσα σε 30 ημερες απο την υπογραφη",
            ]
        )
        has_obligation_payload = any(
            marker in key
            for marker in [
                "αποδειξ",
                "παραστατικ",
                "προσκομιση",
                "πιστοποιητικ",
                "πεα",
                "ειναι απαραιτητο να",
                "πρεπει να",
                "υποχρε",
                "ενημερο",
                "παραμεινει το επιτοκιο",
            ]
        )
        if not (has_post_context and has_obligation_payload):
            return 0
        score = 7
        score += 3 if _source_contains_normalized(source_text, value) else 0
        score += 2 if any(marker in key for marker in ["πρεπει", "απαραιτητο", "υποχρε"]) else 0
        return score

    return 0


def _best_llm_policy_value(value, field: str, source_text: str) -> str:
    best = ""
    best_score = 0
    best_len = 0
    for candidate in _iter_policy_value_candidates(value):
        score = _policy_value_score(field, candidate, source_text)
        if score <= 0:
            continue
        candidate_len = len(candidate)
        if score > best_score or (score == best_score and (not best_len or candidate_len < best_len)):
            best = candidate
            best_score = score
            best_len = candidate_len
    return best


def _marker_scoped_snippets(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    snippets = []
    for marker in [
        "Με το πράσινο",
        "Με το νέο",
        "Στην Eurobank",
        "Πρώτη η Εθνική",
        "Το πρόγραμμα",
        "Το Πρόγραμμα",
        "Αποκτήστε",
    ]:
        idx = clean.find(marker)
        if idx == -1:
            continue
        end = clean.find(".", idx + 120)
        while end != -1 and ("π." in clean[max(0, end - 3): end + 1] or ".χ." in clean[max(0, end - 3): end + 1]):
            end = clean.find(".", end + 1)
        if end == -1:
            end = min(len(clean), idx + 500)
        snippet = clean[idx:end + 1].strip()
        if 40 <= len(snippet) <= 700:
            snippets.append(snippet)
    return snippets


def _trim_source_snippet(sentence: str) -> str:
    for noise in (
        "Επιλέξτε γλώσσα",
        "Top Menu",
        "Main Menu",
        "Extra Button Menu",
        "Page Contents",
        "Footer",
        "Αναζήτηση",
    ):
        sentence = sentence.replace(noise, " ")
    sentence = re.sub(r"\s+", " ", sentence).strip()
    markers = [
        "Με το πράσινο",
        "Με το νέο",
        "Στην Eurobank",
        "Πρώτη η Εθνική",
        "Το πρόγραμμα",
        "Το Πρόγραμμα",
        "Αποκτήστε",
        "Μπορείτε",
    ]
    for marker in markers:
        idx = sentence.find(marker)
        if idx > 0:
            return sentence[idx:].strip()
    return sentence.strip()


def _best_scoped_source_sentence(text: str, url: str) -> str:
    best = ""
    best_score = -1
    title = _first_source_title(text)
    title_norm = _text_for_matching(title)
    clean = re.sub(r"\s+", " ", text or "").strip()
    title_snippets = []
    if title:
        idx = clean.find(title)
        if idx != -1:
            title_snippets.append(clean[idx: idx + 900].strip())

    for sentence in title_snippets + _marker_scoped_snippets(text) + _source_sentences(text):
        sentence = _trim_source_snippet(sentence)
        ok, _, signals = _is_home_green_finance_candidate(url, sentence)
        if not (ok or signals.get("home_energy")):
            continue
        sentence_norm = _text_for_matching(sentence)
        score = 0
        score += 4 if ok else 0
        score += 2 if signals.get("has_home") else 0
        score += 2 if signals.get("has_energy") else 0
        score += 1 if signals.get("has_finance") else 0
        score += 6 if title_norm and title_norm in sentence_norm else 0
        score += 2 if sentence.startswith(("Με το", "Αποκτήστε", "Το πρόγραμμα", "Το Πρόγραμμα")) else 0
        score -= 3 if signals.get("has_vehicle") else 0
        score -= 4 if signals.get("has_business") or signals.get("has_public") else 0
        score -= 5 if _contains_any(sentence, ["Top Menu", "Footer", "Αναζήτηση", "Επιλέξτε γλώσσα"]) else 0
        if _contains_any(sentence_norm, ["fast loan", "δειτε επισης", "σχετικα προιοντα"]) and not (
            title_norm and title_norm in sentence_norm
        ):
            score -= 10
        if score > best_score:
            best = sentence
            best_score = score
    return best


def _should_replace_scoped_text(value: str, programme_name: str) -> bool:
    text = _text_for_matching(value)
    if not text:
        return True
    name = _text_for_matching(programme_name)
    if "fast loan" in text and "fast loan" in name:
        return False
    related_footer = _contains_any(
        text,
        [
            "fast loan",
            "προσωπικο δανειο",
            "αναβαθμιζω το σπιτι μου",
            "σπιτι μου ii",
            "δειτε επισης",
            "σχετικα προιοντα",
        ],
    )
    if related_footer and (not name or name not in text):
        return True
    return False


def _source_product_title(text: str) -> str:
    head = re.sub(r"\s+", " ", (text or "")[:260]).strip()
    title = _first_source_title(text)
    if not title:
        title = ""
    else:
        title = re.split(r"\s*\|\s*", title, maxsplit=1)[0].strip()
        title = re.sub(r"\s+", " ", title)
        for marker in (
            " - CrediaBank",
            " Νέα & Δελτία Τύπου",
            " Αρχική",
            " Εκδήλωση Ενδιαφέροντος",
            " Χρηματοδοτούμε",
            " μάθετε περισσότερα",
            " Μάθετε περισσότερα",
        ):
            idx = title.find(marker)
            if idx > 10:
                title = title[:idx].strip(" -–")

    title_norm = _text_for_matching(title)
    if (
        5 <= len(title) <= 180
        and "πρασιν" in title_norm
        and "δανει" in title_norm
    ):
        return title

    quoted = re.search(r"Πρόγραμμα\s+[\"«“](.+?)[\"»”]", head, flags=re.IGNORECASE)
    if quoted:
        candidate = re.sub(r"\s+", " ", quoted.group(1)).strip(" -–")
        if 5 <= len(candidate) <= 180:
            return candidate

    return title if 5 <= len(title) <= 180 else ""


def _should_use_source_product_title(current_name: str, source_title: str, url: str) -> bool:
    if not source_title:
        return False
    current_norm = _text_for_matching(current_name)
    source_norm = _text_for_matching(source_title)
    if not current_norm:
        return True
    if (
        current_norm in source_norm
        and len(source_title) > len(current_name or "") + 15
        and _contains_any(source_norm, HOME_GREEN_SCOPE_TERMS["finance"])
        and _contains_any(source_norm, HOME_GREEN_SCOPE_TERMS["energy"])
    ):
        return True
    if (
        source_norm in current_norm
        and len(current_name or "") > len(source_title) + 25
        and _contains_any(
            current_norm,
            ["χρηματοδοτουμε", "μαθετε περισσοτερα", "mobile app", "κλεισιμο"],
        )
    ):
        return True
    if current_norm in source_norm or source_norm in current_norm:
        return False

    slug_norm = _text_for_matching(_derive_title_from_url(url))
    if len(slug_norm) >= 8 and slug_norm in source_norm and slug_norm not in current_norm:
        return True
    slug_terms = {
        "exoikonomo": ["εξοικονομ"],
        "eksoikonomo": ["εξοικονομ"],
        "exoikonomisi": ["εξοικονομ"],
        "eksoikonomisi": ["εξοικονομ"],
        "anavathmizo": ["αναβαθμιζ"],
        "anabathmizo": ["αναβαθμιζ"],
        "anakainizo": ["ανακαινιζ"],
        "prasini": ["πρασιν"],
        "prasino": ["πρασιν"],
        "spiti": ["σπιτι"],
        "spitiou": ["σπιτι"],
        "katoikias": ["κατοικ"],
        "stegi": ["στεγη", "στεγ"],
        "steger": ["στεγη", "στεγ"],
        "fotovoltaika": ["φωτοβολταικ"],
        "photoboltaika": ["φωτοβολταικ"],
        "neous": ["νεους"],
        "estia": ["εστια"],
    }
    source_hits = 0
    current_hits = 0
    for token in [item for item in slug_norm.split() if len(item) >= 4]:
        aliases = [token, *slug_terms.get(token, [])]
        in_source = any(alias in source_norm for alias in aliases)
        in_current = any(alias in current_norm for alias in aliases)
        source_hits += 1 if in_source else 0
        current_hits += 1 if in_current else 0
    if source_hits >= 2 and source_hits > current_hits:
        return True
    return False


def _should_use_product_intro(current_value: str, source_title: str, url: str) -> bool:
    value_norm = _text_for_matching(current_value)
    source_norm = _text_for_matching(source_title)
    slug_norm = _text_for_matching(_derive_title_from_url(url))
    if not value_norm:
        return True
    if len(slug_norm) >= 8 and slug_norm in source_norm and slug_norm not in value_norm:
        return True
    return False


def _best_product_intro(text: str, source_title: str, url: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""

    source_norm = _text_for_matching(source_title)
    slug_norm = _text_for_matching(_derive_title_from_url(url))
    candidates = []
    for term in (source_title, _derive_title_from_url(url)):
        term = str(term or "").strip()
        if not term:
            continue
        idx = clean.find(term)
        if idx != -1:
            candidates.append(clean[idx: idx + 1000].strip())
    candidates.extend(_marker_scoped_snippets(text))
    candidates.extend(_source_sentences(text))

    best = ""
    best_score = -1
    for candidate in candidates:
        candidate = _trim_source_snippet(candidate)
        if len(candidate) < 40:
            continue
        candidate_norm = _text_for_matching(candidate)
        score = 0
        score += 8 if source_norm and source_norm in candidate_norm else 0
        score += 8 if slug_norm and slug_norm in candidate_norm else 0
        score += 2 if _contains_any(candidate_norm, HOME_GREEN_SCOPE_TERMS["finance"]) else 0
        score += 2 if _contains_any(candidate_norm, HOME_GREEN_SCOPE_TERMS["energy"]) else 0
        score += 1 if _contains_any(candidate_norm, HOME_GREEN_SCOPE_TERMS["home"]) else 0
        score -= 6 if _contains_any(candidate_norm, ["δειτε επισης", "σχετικα προιοντα"]) else 0
        if score > best_score:
            best = candidate
            best_score = score

    if not best:
        return ""
    for marker in ("Εάν χρειάζεστε", "Ανακαλύψτε", "Το αποτέλεσμα"):
        idx = best.find(marker)
        if idx > 120:
            best = best[:idx].strip()
            break
    if len(best) > 900:
        boundary = max(best.rfind(".", 0, 900), best.rfind(";", 0, 900), best.rfind(";", 0, 900))
        best = best[:boundary + 1] if boundary > 300 else best[:900]
    return best.strip()


def _source_sentence_from_marker(clean: str, marker: str, max_chars: int = 850) -> str:
    idx = clean.find(marker)
    if idx == -1:
        return ""
    segment = clean[idx: idx + max_chars].strip()
    boundary = max(segment.rfind(".", 0, max_chars), segment.rfind(";", 0, max_chars), segment.rfind(";", 0, max_chars))
    if boundary > 80:
        segment = segment[: boundary + 1]
    return re.sub(r"\s+", " ", segment).strip()


def _best_exact_program_intro(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    for marker in (
        "Με το Green Στεγαστικό/Επισκευαστικό δάνειο",
        "Αποκτήστε πράσινο δάνειο για το σπίτι",
        "Μπορείτε να λάβετε επιδότηση",
    ):
        sentence = _source_sentence_from_marker(clean, marker)
        if 60 <= len(sentence) <= 900:
            return sentence
    return ""


def _looks_like_boilerplate_intro(value: str) -> bool:
    normalized = _text_for_matching(value)
    if not normalized:
        return True
    return _contains_any(
        normalized,
        [
            "piraeus app",
            "e banking",
            "ψηφιακο βοηθο",
            "created with sketchtool",
            "el english",
            "ρωτηστε τον ψηφιακο",
            "ερωτησεις απαντησεις",
            "χαμηλης ενεργειας",
        ],
    )


def _extract_exact_interventions_from_source(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    clean_norm = _text_for_matching(clean)

    alpha_items = [
        "Αγοράσετε και εγκαταστήσετε οικιακά φωτοβολταϊκά, αντλίες θερμότητας, ηλιακό θερμοσίφωνα ή σύστημα",
        "Αλλάξετε κουφώματα",
        "Κάνετε θερμομόνωση",
        "Αγοράσετε οικιακές συσκευές, όπως κλιματιστικά με χαμηλή ενεργειακή κατανάλωση",
    ]
    if _contains_any(clean_norm, ["Τι ανάγκες μπορώ να καλύψω"]) and _contains_any(clean_norm, ["Μόνο για ενεργειακή αναβάθμιση"]):
        found = [item for item in alpha_items if _text_for_matching(item) in clean_norm]
        if found:
            return found

    gov_items = [
        "Αντικατάστασης κουφωμάτων",
        "Τοποθέτησης/αναβάθμισης θερμομόνωσης",
        "Αναβάθμισης συστήματος θέρμανσης/ψύξης",
        "Τοποθέτησης συστήματος ζεστού νερού χρήσης (ΖΝΧ) με χρήση Ανανεώσιμων Πηγών Ενέργειας (ΑΠΕ)",
        "Λοιπές παρεμβάσεις εξοικονόμησης ενέργειας, όπως η εγκατάσταση έξυπνου συστήματος διαχείρισης (smart home) και η εγκατάσταση συστήματος αποθήκευσης ενέργειας (μπαταρίες)",
    ]
    if _contains_any(clean_norm, ["Οι επιλέξιμες δαπάνες που μπορούν να επιδοτηθούν είναι"]):
        found = [item for item in gov_items if _text_for_matching(item) in clean_norm]
        if found:
            return found

    return []


def _extract_property_requirements_from_source(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    clean_norm = _text_for_matching(clean)
    if not _contains_any(clean_norm, ["Γενικές προϋποθέσεις κατοικίας"]):
        return []
    candidates = [
        "Υφίσταται νόμιμα / Δεν έχει κριθεί κατεδαφιστέα.",
        "Χρησιμοποιείται ως κύρια κατοικία.",
        "Έχει καταταχθεί βάσει του Πρώτου Πιστοποιητικού Ενεργειακής Απόδοσης (Α’ Π.Ε.Α.) σε κατηγορία χαμηλότερη ή ίση της Γ.",
    ]
    return [item for item in candidates if _text_for_matching(item) in clean_norm]


def _extract_completion_deadline_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(
        r"καταληκτική\s+ημερομηνία\s+ολοκλήρωσης\s+του\s+προγράμματος,\s+είναι\s+η\s+([^.;;]+)",
        clean,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _energy_targets_look_like_interventions(value) -> bool:
    raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value or "")
    normalized = _text_for_matching(raw)
    if not normalized:
        return False
    if re.search(r"\d|%", raw) or any(marker in raw for marker in ("Α+", "Β+", "Α’", "Γ")):
        return False
    return _contains_any(
        normalized,
        ["φωτοβολταικ", "αντλιες θερμοτητας", "θερμοσιφων", "κουφωμα", "οικιακες συσκευες"],
    )


def _eligible_parties_looks_like_document_artifact(value, source_text: str) -> bool:
    values = value if isinstance(value, list) else [value]
    normalized_values = [_text_for_matching(item) for item in values if str(item or "").strip()]
    if normalized_values == [_text_for_matching("Ελεύθερος επαγγελματίας")] and _contains_any(
        _text_for_matching(source_text),
        ["Πράσινο δάνειο για το σπίτι"],
    ):
        return True
    return False


def _clean_extracted_value(value):
    if isinstance(value, list):
        cleaned = []
        for item in value:
            item = _clean_extracted_value(item)
            if item in ("", [], {}):
                continue
            cleaned.append(item)
        return cleaned
    if isinstance(value, dict):
        cleaned = {k: _clean_extracted_value(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in ("", [], {})}
    if isinstance(value, str):
        return value.strip()
    return value


def _source_contains_normalized(source_text: str, value) -> bool:
    source_norm = _text_for_matching(source_text)
    value_norm = _text_for_matching(value)
    return bool(value_norm) and value_norm in source_norm


def _number_signatures(value) -> list[str]:
    signatures = []
    for token in re.findall(r"\d+(?:[.,]\d+)?%?", str(value or "")):
        signature = re.sub(r"\D", "", token)
        if signature:
            signatures.append(signature)
    return signatures


def _significant_value_tokens(value) -> list[str]:
    stopwords = {
        "και", "για", "απο", "από", "στο", "στη", "στην", "στις", "στον",
        "του", "της", "των", "τον", "την", "το", "τα", "με", "σε", "ως",
        "εως", "έως", "μεχρι", "μέχρι", "που", "είναι", "ειναι",
        "the", "and", "for", "with", "from", "that", "this", "must", "have",
        "your", "you", "can", "are", "is", "of", "to", "in", "on", "up",
    }
    tokens = re.findall(r"[\w%.,]+", _text_for_matching(value), flags=re.UNICODE)
    return [
        token
        for token in tokens
        if token not in stopwords and (len(token) >= 4 or re.search(r"\d", token))
    ]


def _value_has_source_support(value, source_text: str, min_recall: float = 0.72) -> bool:
    if value in ("", None, [], {}):
        return True
    if _source_contains_normalized(source_text, value):
        return True

    source_norm = _text_for_matching(source_text)
    if not source_norm:
        return False

    value_numbers = _number_signatures(value)
    if value_numbers:
        source_numbers = set(_number_signatures(source_text))
        if not all(number in source_numbers for number in value_numbers):
            return False

    tokens = _significant_value_tokens(value)
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in source_norm)
    return hits / len(tokens) >= min_recall


_GREEK_MONTH_RE = (
    r"(?:Ιανουαρίου|Ιανουαριου|Φεβρουαρίου|Φεβρουαριου|Μαρτίου|Μαρτιου|"
    r"Απριλίου|Απριλιου|Μαΐου|Μαιου|Ιουνίου|Ιουνιου|Ιουλίου|Ιουλιου|"
    r"Αυγούστου|Αυγουστου|Σεπτεμβρίου|Σεπτεμβριου|Οκτωβρίου|Οκτωβριου|"
    r"Νοεμβρίου|Νοεμβριου|Δεκεμβρίου|Δεκεμβριου)"
)
_DATE_TOKEN_RE = re.compile(
    rf"\b\d{{1,2}}(?:\s*(?:η|ης|ος))?\s+{_GREEK_MONTH_RE},?\s+\d{{4}}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    flags=re.IGNORECASE,
)


def _has_date_token(value) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_DATE_TOKEN_RE.search(value))


def _clean_date_token(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .,;:-–—")
    cleaned = re.sub(r"(\d{1,2})\s+(η|ης|ος)\b", r"\1\2", cleaned, flags=re.IGNORECASE)
    return cleaned


def _normalize_date_field_value(value: str) -> str:
    match = _DATE_TOKEN_RE.search(value or "")
    if not match:
        return _clean_date_token(value)
    return _clean_date_token(match.group(0))


def _date_context_score(context_norm: str, field: str) -> int:
    score = 0
    has_application = _contains_any(context_norm, ["αιτησ", "υποβολ", "πλατφορμα"])
    has_until = _contains_any(context_norm, ["εωσ", "μεχρι", "ληξη", "προθεσμι", "καταληκτικ", "παραταση", "παρατεινεται"])
    has_from = _contains_any(context_norm, ["εναρξη", "ξεκινα", "ανοιγει", "απο"])
    has_completion = _contains_any(
        context_norm,
        [
            "ολοκληρω",
            "υλοποιηση",
            "περατωση",
            "φυσικου αντικειμενου",
            "καταληκτικη ημερομηνια του προγραμματος",
            "επιλεξιμοτητα δαπανων",
            "προθεσμια ολοκληρωσης",
            "προθεσμια υλοποιησης",
        ],
    )

    if field == "application_start_date":
        score += 5 if _contains_any(context_norm, ["εναρξη", "ξεκινα", "ανοιγει"]) else 0
        score += 4 if has_application and has_from else 0
        score -= 4 if has_until else 0
        score -= 5 if has_completion else 0
    elif field == "application_end_date":
        score += 5 if has_application and has_until else 0
        score += 4 if _contains_any(context_norm, ["ληξη αιτησεων", "καταληκτικη ημερομηνια αιτησεων", "προθεσμια υποβολης"]) else 0
        score += 2 if has_application and _contains_any(context_norm, ["παραταση", "παρατεινεται"]) else 0
        score -= 5 if has_completion else 0
    elif field == "completion_deadline":
        score += 6 if has_completion else 0
        score += 2 if has_until else 0
        score -= 3 if has_application and not has_completion else 0
    elif field == "announcement_date":
        score += 6 if _contains_any(context_norm, ["δελτιο τυπου", "δημοσιευ", "αναρτηθηκε", "ημερομηνια δημοσιευσης"]) else 0
        score -= 4 if has_application or has_completion else 0
    return score


def _extract_program_dates_from_source(text: str) -> dict[str, str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return {}

    candidates: dict[str, list[tuple[int, int, str]]] = {field: [] for field in DATE_FIELDS}
    for match in _DATE_TOKEN_RE.finditer(clean):
        token = _clean_date_token(match.group(0))
        if not token:
            continue
        context = clean[max(0, match.start() - 220): match.end() + 220]
        context_norm = _text_for_matching(context)
        if _contains_any(context_norm, ["cookies", "javascript", "mobile app", "κλεισιμο"]):
            continue
        for field in DATE_FIELDS:
            score = _date_context_score(context_norm, field)
            if score >= 4:
                candidates[field].append((score, match.start(), token))

    out: dict[str, str] = {}
    for field, items in candidates.items():
        if not items:
            continue
        # Highest context score wins; for ties keep the earliest source mention.
        score, _, token = sorted(items, key=lambda item: (-item[0], item[1]))[0]
        if score >= 4:
            out[field] = token
    return out


def _date_field_invalid(value) -> bool:
    if value in ("", None, [], {}):
        return False
    if not isinstance(value, str):
        return True
    raw = value.strip()
    if not raw:
        return False
    if raw.lower().startswith(("http://", "https://", "www.")):
        return True
    return not _has_date_token(raw)


def _repair_date_fields_from_source(clean: dict, source_text: str) -> None:
    if not source_text:
        return

    for field in DATE_FIELDS:
        value = clean.get(field)
        if _date_field_invalid(value):
            clean[field] = ""
        elif isinstance(value, str):
            clean[field] = _normalize_date_field_value(value)

    source_dates = _extract_program_dates_from_source(source_text)
    for field, source_value in source_dates.items():
        current = clean.get(field)
        if (
            current in ("", None, [], {})
            or _date_field_invalid(current)
            or not _value_has_source_support(current, source_text)
        ):
            clean[field] = source_value


def _looks_like_unsupported_hallucination(value, source_text: str) -> bool:
    if value in ("", None, [], {}):
        return False
    if _value_has_source_support(value, source_text):
        return False

    normalized = _text_for_matching(value)
    hallucination_markers = (
        "national energy",
        "environment agency",
        "homeowners",
        "residential building",
        "online application form",
        "january 1st",
        "february 1st",
        "june 30th",
        "baseline value",
        "bedrooms",
        "bathrooms",
        "single family home",
        "apartment complex",
    )
    if any(marker in normalized for marker in hallucination_markers):
        return True

    raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    latin_count = len(re.findall(r"[A-Za-z]", raw))
    greek_count = len(re.findall(r"[\u0370-\u03ff]", raw))
    return latin_count >= 24 and latin_count > greek_count * 2


def _clean_source_supported_list(value, source_text: str, min_recall: float = 0.72) -> list:
    values = value if isinstance(value, list) else [value]
    cleaned = []
    for item in values:
        if item in ("", None, [], {}):
            continue
        if _looks_like_unsupported_hallucination(item, source_text):
            continue
        if _value_has_source_support(item, source_text, min_recall=min_recall):
            cleaned.append(item)
    return cleaned


def _clear_unsupported_hallucinated_fields(clean: dict, source_text: str) -> None:
    if not source_text:
        return

    list_fields = {
        "contact_info",
        "eligibility_criteria",
        "eligible_parties",
        "energy_performance_targets",
        "property_requirements",
    }
    scalar_fields = {
        "announcement_date",
        "application_start_date",
        "application_end_date",
        "completion_deadline",
        "completion_delay_consequences",
        "application_process",
        "post_completion_obligations",
        "managing_body",
        "duration",
        "funding_coverage",
    }

    for field in list_fields:
        if field not in clean:
            continue
        filtered = _clean_source_supported_list(clean.get(field), source_text)
        clean[field] = filtered if filtered else ""

    for field in scalar_fields:
        value = clean.get(field)
        if value in ("", None, [], {}):
            continue
        if _looks_like_unsupported_hallucination(value, source_text):
            clean[field] = ""
            continue
        if (
            field in {"completion_delay_consequences", "post_completion_obligations"}
            and not _value_has_source_support(value, source_text)
        ):
            clean[field] = ""
            continue
        if (
            field in {
                "duration",
                "funding_coverage",
                "application_start_date",
                "application_end_date",
                "announcement_date",
                "completion_deadline",
                "completion_delay_consequences",
                "post_completion_obligations",
            }
            and not _value_has_source_support(value, source_text)
            and re.search(r"\d|%", str(value))
        ):
            clean[field] = ""


def _extract_interventions_from_source(text: str, url: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    for marker in (
        "Επιλέξιμες δαπάνες",
        "Επιλεξιμες δαπανες",
        "αφορούν τις κάτωθι παρεμβάσεις",
        "αφορουν τις κατωθι παρεμβασεις",
    ):
        idx = clean.find(marker)
        if idx == -1:
            continue
        segment = clean[idx: idx + 2400]
        numbered = re.findall(
            r"(?:^|\s)\d+\.\d+\.?\s*([^\d]+?)(?=\s+\d+\.\d+\.|\s+Σημειώσεις:|\s+Σημειωσεις:|$)",
            segment,
        )
        interventions = []
        for item in numbered:
            item = item.strip(" .:-–—")
            item = re.sub(r"\s+", " ", item)
            if 5 <= len(item) <= 280:
                interventions.append(item)
        if interventions:
            return list(dict.fromkeys(interventions))

    clean_norm = _text_for_matching(clean)
    canonical_interventions = [
        ("πρασινες οικιακες συσκευες", "Πράσινες οικιακές συσκευές"),
        ("λευκες οικιακες συσκευες", "Λευκές οικιακές συσκευές"),
        ("συσκευες θερμανσης", "Συσκευές θέρμανσης και κλιματισμού"),
        ("μονωσ", "Μονώσεις"),
        ("ενεργειακα το σπιτι", "Ενεργειακή αναβάθμιση κατοικίας"),
        ("αντικατασταση κουφω", "Αντικατάσταση κουφωμάτων"),
        ("θερμομον", "Τοποθέτηση / αντικατάσταση κελύφους για θερμομόνωση"),
        ("συστηματα ψυξης", "Συστήματα ψύξης/θέρμανσης"),
        ("αναβαθμιση συστηματος θερμανσης", "Αναβάθμιση συστήματος θέρμανσης / ψύξης"),
        ("συστημα θερμανσης / ψυξης", "Διαχείριση συστήματος θέρμανσης / ψύξης"),
        ("ζεστου νερου χρησης", "Αναβάθμιση συστήματος ζεστού νερού χρήσης μέσω ΑΠΕ"),
        ("ηλιακου θερμοσιφωνα", "Τοποθέτηση ηλιακού θερμοσίφωνα"),
        ("πρασινης στεγης", "Κατασκευή πράσινης στέγης"),
        ("οικιακο φωτοβολταικο", "Εγκατάσταση οικιακού φωτοβολταϊκού συστήματος"),
        ("φωτοβολται", "Εγκατάσταση φωτοβολταϊκών ή άλλων ΑΠΕ"),
        ("ανεμογεννητρια", "Οικιακή ανεμογεννήτρια"),
        ("γεωθερμια", "Γεωθερμία"),
        ("αποθηκευσης ενεργειας", "Εγκατάσταση συστήματος αποθήκευσης ενέργειας"),
        ("φωτισμος led", "Φωτισμός - έξυπνες πρίζες, φωτισμός LED"),
        ("απομακρυσμενο ελεγχο", "Απομακρυσμένος έλεγχος - π.χ. αισθητήρες WiFi"),
    ]
    canonical = [
        label
        for needle, label in canonical_interventions
        if _text_for_matching(needle) in clean_norm
    ]
    if len(canonical) >= 3:
        return list(dict.fromkeys(canonical))

    section_interventions = []
    for start_marker, end_marker in (
        ("καλύπτει ενδεικτικά:", "Για έξυπνα συστήματα"),
        ("καλυπτει ενδεικτικα:", "Για εξυπνα συστηματα"),
        ("όπως για:", "Αρκεί τα συστήματα"),
        ("οπως για:", "Αρκει τα συστηματα"),
    ):
        idx = clean.find(start_marker)
        if idx == -1:
            continue
        end = clean.find(end_marker, idx + len(start_marker))
        if end == -1:
            end = idx + 700
        segment = clean[idx + len(start_marker): end]
        for item in re.split(r"(?<=\.)\s+", segment):
            item = item.strip(" .:-–—")
            item = re.sub(r"\s+", " ", item)
            if 5 <= len(item) <= 180 and _contains_any(item, HOME_GREEN_SCOPE_TERMS["energy"]):
                section_interventions.append(item)
    if section_interventions:
        return list(dict.fromkeys(section_interventions))

    marker_positions = [clean.find(marker) for marker in ("π.χ.", "όπως") if clean.find(marker) != -1]
    from_example_list = bool(marker_positions)
    if marker_positions:
        idx = min(marker_positions)
        segment = clean[idx + 4: idx + 360]
    else:
        sentence = _best_scoped_source_sentence(text, url)
        if not sentence:
            return []
        segment = sentence

    if not segment:
        return []
    for marker in ("π.χ.", "όπως", "παρεμβάσεις"):
        idx = segment.find(marker)
        if idx != -1:
            segment = segment[idx + len(marker):]
            break
    segment = re.split(r"[.;;]", segment, maxsplit=1)[0]
    parts = re.split(r",|\s+ή\s+|\s+και\s+", segment)
    interventions = []
    for part in parts:
        part = part.strip(" .:-–—")
        part = re.sub(r"^(τοποθετείτε|τοποθετειτε|να)\s+", "", part, flags=re.IGNORECASE).strip()
        part_norm = _text_for_matching(part)
        if any(
            marker in part_norm
            for marker in (
                "συναλλακτικη συμπεριφορα",
                "παρεχομενη εξασφαλιση",
                "το ποσο",
                "διαρκεια του δανειου",
                "πιστοληπτικη",
            )
        ):
            continue
        if 5 <= len(part) <= 120 and (from_example_list or _contains_any(part, HOME_GREEN_SCOPE_TERMS["energy"])):
            interventions.append(part)
    if not interventions:
        clean_norm = _text_for_matching(clean)
        if (
            all(token in clean_norm for token in ("υψηλη", "ενεργειακη", "κατηγορια"))
        ) and any(marker in clean_norm for marker in ("αγορασετε", "κατασκευασετε", "ανακαινισετε")):
            interventions.append("Αγορά, κατασκευή ή ανακαίνιση κατοικίας υψηλής ενεργειακής κατηγορίας (Α+, Α, Β+)")
        if "net metering" in clean_norm and "φωτοβολται" in clean_norm:
            interventions.append("Εγκατάσταση οικιακού φωτοβολταϊκού συστήματος με net metering")
    return list(dict.fromkeys(interventions))


def _extract_amount_phrases_from_source(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "")
    phrases = re.findall(
        r"(?:Έως|έως|Πάνω από|πάνω από|Μέχρι|μέχρι)\s+[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?\s*€",
        clean,
    )
    filtered = []
    for phrase in phrases:
        phrase = phrase.strip()
        amount = _amount_token_to_number(phrase)
        if _looks_like_program_budget_amount(phrase):
            continue
        if amount is not None and amount >= 100000:
            continue
        filtered.append(phrase)
    return list(dict.fromkeys(filtered))


def _extract_loan_amount_range_from_source(text: str) -> tuple[str, str]:
    clean = re.sub(r"\s+", " ", text or "")
    euro = r"(?:€|ευρώ|ευρω)"
    number = r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?"
    amount_with_euro = rf"(?:{euro}\s*)?\(?\s*{number}\s*\)?\s*(?:{euro})?"
    amount_requiring_euro = rf"(?:{euro}\s*\(?\s*{number}\s*\)?|\(?\s*{number}\s*\)?\s*{euro})"
    patterns = [
        rf"(?:ΠΟΣΟ\s*)?({amount_requiring_euro})\s*[-–—]\s*({amount_requiring_euro})",
        rf"(?:ΠΟΣΟ\s*)?({number})\s*[-–—]\s*({amount_requiring_euro})",
        rf"(?:Από|από)\s+({amount_with_euro})\s+(?:έως|εως|μέχρι|μεχρι)\s+(?:και\s+)?({amount_requiring_euro})",
        rf"(?:Από|από)\b[^.;:]{{0,100}}?({amount_requiring_euro})\s+(?:έως|εως|μέχρι|μεχρι)\s+(?:και\s+)?[^.;:]{{0,100}}?({amount_requiring_euro})",
        rf"({number})\s+(?:έως|εως|μέχρι|μεχρι)\s+(?:και\s+)?({amount_requiring_euro})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            context = clean[max(0, match.start() - 160): match.end() + 160]
            minimum = _normalize_amount_token(match.group(1))
            maximum = _normalize_amount_token(match.group(2))
            if _looks_like_program_budget_context(context) or _looks_like_program_budget_amount(maximum):
                continue
            return minimum, maximum
    return "", ""


def _extract_minimum_loan_amount_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "")
    euro = r"(?:€|ευρώ|ευρω)"
    number = r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?"
    amount_requiring_euro = rf"(?:{euro}\s*\(?\s*{number}\s*\)?|\(?\s*{number}\s*\)?\s*{euro})"
    patterns = [
        rf"(?:ΠΟΣΟ|Ποσό|ποσό|ποσο)\s+(?:Από|από|απο)\s+({amount_requiring_euro})",
        rf"(?:Ύψος|Υψος|ύψος|υψος)\s+χρηματοδότησης\s+(?:Από|από|απο)\s+({amount_requiring_euro})",
        rf"(?:Από|από|απο)\s+({amount_requiring_euro})\s+και\s+μέχρι\s*:\s*100%",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            context = clean[max(0, match.start() - 120): match.end() + 180]
            if _looks_like_program_budget_context(context):
                continue
            return _normalize_amount_token(match.group(1))
    return ""


def _amount_number(value: str) -> Optional[float]:
    return _amount_token_to_number(str(value or ""))


def _source_range_should_override(current_minimum: str, current_maximum: str, source_minimum: str, source_maximum: str) -> bool:
    source_max = _amount_number(source_maximum)
    current_max = _amount_number(current_maximum)
    current_min = _amount_number(current_minimum)
    if source_max is None:
        return False
    if not current_maximum:
        return True
    if current_max is not None and current_max < source_max:
        return True
    if current_min is not None and current_max is not None and current_min == current_max and current_max < source_max:
        return True
    return False


def _bad_related_context(context: str) -> bool:
    text = _text_for_matching(context)
    return any(
        marker in text
        for marker in (
            "δειτε επισησ",
            "δειτε επισης",
            "σχετικα προιοντα",
            "related",
            "μπορει να σας ενδιαφερει",
        )
    )


def _clean_source_sentence(value: str, max_chars: int = 260) -> str:
    protected = re.sub(r"\bΝ\.\s*128", "Ν§128", value or "", flags=re.IGNORECASE)
    sentence = re.sub(r"\s+", " ", protected).strip(" .:-–—")
    sentence = re.split(r"(?<=[.;;])\s+", sentence, maxsplit=1)[0].strip(" .:-–—")
    sentence = sentence.replace("Ν§128", "Ν.128")
    for marker in (
        " ΕΙΔΟΣ ΕΠΙΤΟΚΙΟΥ",
        " ΕΠΙΤΟΚΙΟ",
        " ΔΙΑΡΚΕΙΑ",
        " ΠΟΣΟ",
        " Αποκτήστε",
        " Αποκτήστε Online",
        " Μπορείτε",
        " Πλεονεκτήματα",
        " Αναλυτικά χαρακτηριστικά",
        " Σημείωση:",
        " Σημειώσεις:",
    ):
        idx = sentence.find(marker)
        if idx > 0:
            sentence = sentence[:idx].strip(" .:-–—")
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0].strip()
    return sentence


def _extend_truncated_legal_tail(raw_candidate: str, tail: str) -> str:
    candidate = raw_candidate or ""
    match = re.match(r"(\.128/(?:75|1975))", tail or "", flags=re.IGNORECASE)
    if match and candidate.rstrip().endswith("Ν"):
        return candidate.rstrip() + match.group(1)
    return candidate


def _extract_interest_rate_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "")
    if not clean:
        return ""
    patterns = [
        r"ΕΠΙΤΟΚΙΟ\s+([^.;;\n]{1,180})",
        r"(?:Ποιο είναι το επιτόκιο του δανείου\?|\bεπιτόκιο του δανείου\b)[^.;;]{0,80}?(?:(?:είναι|:)\s*)?([^.;;]{1,240})",
        r"((?:κυμαινόμενο|κυμαινομενο|σταθερό|σταθερο|άτοκο|ατοκο)[^.;;]{0,240}(?:Euribor|euribor|επιτόκιο|επιτοκιο|%|άτοκο|ατοκο)[^.;;]{0,120})",
        r"((?:[0-9]+(?:[.,][0-9]+)?%)\s*[^.;;]{0,220}(?:επιτόκιο|επιτοκιο|σταθερό|σταθερο|κυμαινόμενο|κυμαινομενο)[^.;;]{0,80})",
    ]
    signal = re.compile(r"(?:%|euribor|επιτόκ|επιτοκ|σταθερ|κυμαιν|άτοκ|ατοκ)", flags=re.IGNORECASE)
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            context = clean[max(0, match.start() - 140): match.end() + 140]
            if _bad_related_context(context):
                continue
            raw_candidate = _extend_truncated_legal_tail(match.group(1), clean[match.end(): match.end() + 40])
            candidate = _clean_source_sentence(raw_candidate, max_chars=320)
            if candidate and signal.search(candidate):
                return candidate
    return ""


def _extract_loan_duration_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "")
    if not clean:
        return ""
    patterns = [
        r"ΔΙΑΡΚΕΙΑ\s+([^.;;\n]{1,120})",
        r"(?:διάρκεια|διαρκεια)\s*(?:του δανείου|του δανειου)?\s*(?::|είναι|ειναι)?\s*((?:από|απο)?\s*[0-9][^.;;]{0,120}(?:έτη|ετη|χρόνια|χρονια|μήνες|μηνες))",
        r"((?:Από|από|ΑΠΟ|απο)\s+[0-9]+\s+(?:έως|εως|μέχρι|μεχρι)\s+[0-9]+\s+(?:έτη|ετη|χρόνια|χρονια|μήνες|μηνες))",
        r"([0-9]+(?:,\s*[0-9]+)*(?:\s+ή\s+[0-9]+)?\s+(?:έτη|ετη|χρόνια|χρονια|μήνες|μηνες))",
    ]
    duration_signal = re.compile(r"(?:\d).{0,120}(?:έτη|ετη|χρόνια|χρονια|μήνες|μηνες)", flags=re.IGNORECASE)
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            context = clean[max(0, match.start() - 140): match.end() + 140]
            if _bad_related_context(context):
                continue
            candidate = _clean_source_sentence(match.group(1), max_chars=180)
            if candidate and duration_signal.search(candidate):
                return candidate
    return ""


def _extract_funding_coverage_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "")
    if not clean:
        return ""
    section_idx = clean.find("Ύψος χρηματοδότησης")
    if section_idx == -1:
        section_idx = clean.find("Υψος χρηματοδότησης")
    if section_idx != -1:
        section = clean[section_idx: section_idx + 1200]
        end_idx = section.find("Διάρκεια")
        if end_idx > 80:
            section = section[:end_idx]
        if "80%" in section and "90%" in section:
            candidate = re.sub(r"\s+", " ", section).strip(" .;:;")
            if len(candidate) > 900:
                candidate = candidate[:900].strip(" .;:;")
            if candidate:
                return candidate
    patterns = [
        r"((?:Ύψος|Υψος)\s+χρηματοδότησης[^.;;]{0,220}(?:έως|εως|μέχρι|μεχρι)\s+(?:το\s+)?[0-9]+%[^.;;]{0,320})",
        r"((?:ποσοστό|ποσοστο)\s+χρηματοδότησης[^.;;]{0,220}(?:έως|εως|μέχρι|μεχρι)\s+(?:το\s+)?[0-9]+%[^.;;]{0,260})",
        r"((?:χρηματοδότηση|χρηματοδοτηση)[^.;;]{0,180}(?:έως|εως|μέχρι|μεχρι)\s+(?:και\s+)?(?:σ?το\s+)?[0-9]+%[^.;;]{0,220})",
        r"((?:Εξασφαλίστε|εξασφαλίστε|εξασφαλιστε)[^.;;]{0,100}100%\s+(?:άτοκο|ατοκο)\s+δάνειο[^.;;]{0,180})",
        r"((?:Άτοκο|Ατοκο|άτοκο|ατοκο)[^.;;]{0,100}100%\s+του\s+ποσού[^.;;]{0,180})",
        r"((?:έως|εως|μέχρι|μεχρι)\s+(?:το\s+)?[0-9]+%\s+(?:της|του)\s+(?:αξίας|αξιας|προϋπολογισμού|προυπολογισμου|κόστους|κοστους)[^.;;]{0,240})",
        r"((?:χρηματοδότηση|χρηματοδοτηση)[^.;;]{0,140}(?:έως|εως|μέχρι|μεχρι)\s*:?\s+(?:και\s+)?(?:σ?το\s+)?100%[^.;;]{0,180})",
        r"((?:Κάλυψε|κάλυψε|καλυψε)[^.;;]{0,80}(?:100%)[^.;;]{0,180})",
        r"((?:έως|εως|μέχρι|μεχρι)\s*:?\s+(?:και\s+)?(?:σ?το\s+)?100%\s+(?:του|της)[^.;;]{0,180})",
        r"((?:100%\s+επιδοτ[^.;;]{0,180}(?:τόκ|τοκ|επιτοκ)[^.;;]{0,120}))",
        r"((?:επιτόκιο|επιτοκιο)[^.;;]{0,100}επιδοτείται\s+(?:κατά\s+)?100%[^.;;]{0,160})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            context = clean[max(0, match.start() - 160): match.end() + 160]
            if _bad_related_context(context):
                continue
            raw_candidate = match.group(1)
            tail = clean[match.end(): match.end() + 40]
            raw_candidate = _extend_truncated_legal_tail(raw_candidate, tail)
            amount_tail = re.match(r"([.,]\d{3}\s*(?:€|ευρώ|ευρω))", tail, flags=re.IGNORECASE)
            if amount_tail and re.search(r"\d$", raw_candidate):
                raw_candidate += amount_tail.group(1)
            candidate = _clean_source_sentence(raw_candidate, max_chars=520)
            if candidate and re.search(r"[0-9]+\s*%", candidate):
                return candidate
    return ""


def _source_field_should_replace(field: str, current_value, source_value: str, source_text: str) -> bool:
    current = re.sub(r"\s+", " ", str(current_value or "")).strip()
    source = re.sub(r"\s+", " ", str(source_value or "")).strip()
    if not source:
        return False
    if not current:
        return True
    if _looks_like_unsupported_hallucination(current, source_text):
        return True

    current_supported = _value_has_source_support(current, source_text)
    source_supported = _value_has_source_support(source, source_text)
    if source_supported and not current_supported:
        return True

    current_norm = _text_for_matching(current)
    source_norm = _text_for_matching(source)
    if field == "interest_rate":
        if current.endswith("Ν") and "Ν.128" in source:
            return True
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?%", current) and "εκπτωση" in _text_for_matching(
            _context_around_value(source_text, current, radius=180)
        ):
            return True
        if ("euribor" in source_norm and "euribor" not in current_norm) or ("ατοκ" in source_norm and "ατοκ" not in current_norm):
            return True
        if len(current) > len(source) + 60 and re.search(r"%|euribor|επιτοκ|σταθερ|κυμαιν", source_norm):
            return True
    elif field in {"loan_duration", "duration"}:
        if len(current) > len(source) + 35 and re.search(r"\d", source):
            return True
    elif field == "funding_coverage":
        if len(current) > len(source) + 60 and re.search(r"[0-9]+\s*%|άτοκο|ατοκο|επιδοτ", source):
            return True
    elif field in {"completion_delay_consequences", "post_completion_obligations"}:
        return True
    return False


def _normalize_amount_token(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if not value:
        return ""
    number = re.sub(r"(?:€|ευρώ|ευρω)", "", value, flags=re.IGNORECASE).strip()
    number = number.strip("() ")
    return f"{number}€"


def _clean_amount_field_value(value) -> str:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return ""
    euro = r"(?:€|ευρώ|ευρω)"
    number = r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?"
    amount = rf"(?:{euro}\s*)?\(?\s*{number}\s*\)?\s*(?:{euro})?"
    match = re.search(
        rf"(?:Από|από|Έως|έως|Μέχρι|μέχρι|Πάνω από|πάνω από)?\s*({amount})",
        raw,
        flags=re.IGNORECASE,
    )
    if match and re.search(euro, match.group(1), flags=re.IGNORECASE):
        return _normalize_amount_token(match.group(1))
    return raw


def _amount_field_lacks_currency(value) -> bool:
    raw = str(value or "").strip()
    return bool(raw) and re.search(r"\d", raw) is not None and not re.search(r"(?:€|ευρώ|ευρω)", raw, flags=re.IGNORECASE)


def _add_currency_if_source_has_same_amount(value, text: str) -> str:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not _amount_field_lacks_currency(raw):
        return raw
    if "%" in raw:
        return raw
    amount = re.search(r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?", raw)
    if not amount:
        return raw
    token = amount.group(0)
    euro = r"(?:€|ευρώ|ευρω)"
    if re.search(rf"(?:{euro}\s*)?{re.escape(token)}\s*(?:{euro})", text or "", flags=re.IGNORECASE):
        return raw.replace(token, f"{token}€", 1)
    return raw


def _same_min_max_looks_like_single_threshold(minimum: str, maximum: str, text: str) -> bool:
    if not minimum or not maximum:
        return False
    if _normalize_amount_token(minimum) != _normalize_amount_token(maximum):
        return False
    token = re.sub(r"(?:€|ευρώ|ευρω)", "", str(maximum), flags=re.IGNORECASE).strip()
    token = token.strip("() ")
    if not token or "%" in token:
        return False
    amount = rf"(?:€\s*)?{re.escape(token)}\s*(?:€|ευρώ|ευρω)?"
    explicit_range = re.search(
        rf"(?:Από|από)\s+{amount}\s+(?:έως|εως|μέχρι|μεχρι)\s+{amount}",
        text or "",
        flags=re.IGNORECASE,
    )
    if explicit_range:
        return False
    return re.search(
        rf"(?:Έως|έως|Μέχρι|μέχρι|Πάνω από|πάνω από)\s+{amount}",
        text or "",
        flags=re.IGNORECASE,
    ) is not None


def _looks_like_percentage_only(value) -> bool:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    return bool(raw) and "%" in raw and not re.search(r"(?:€|ευρώ|ευρω)", raw, flags=re.IGNORECASE)


def _amount_token_to_number(value: str) -> Optional[float]:
    match = re.search(r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?", value or "")
    if not match:
        return None
    token = match.group(0)
    if "." in token and "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        parts = token.split(",")
        token = "".join(parts) if len(parts[-1]) == 3 else token.replace(",", ".")
    elif "." in token:
        parts = token.split(".")
        token = "".join(parts) if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) else token
    try:
        return float(token)
    except ValueError:
        return None


def _looks_like_program_budget_amount(value: str) -> bool:
    text = _text_for_matching(value)
    amount = _amount_token_to_number(value)
    return (
        any(marker in text for marker in ("εκ", "million", "συνολο χορηγησεων", "προυπολογισμ"))
        or (amount is not None and amount >= 1000000)
    )


def _looks_like_program_budget_context(value: str) -> bool:
    text = _text_for_matching(value)
    amount = _amount_token_to_number(value)
    budget_markers = (
        "συνολικος προυπολογισμος",
        "προυπολογισμος ταμειου",
        "δημοσια δαπανη",
        "συνολικη δημοσια δαπανη",
        "συνολικη επενδυση",
        "διαθεσιμοι ποροι",
        "program budget",
        "programme budget",
        "total budget",
        "budget:",
    )
    return any(marker in text for marker in budget_markers) or (
        amount is not None and amount >= 1000000 and "προυπολογισμ" in text
    )


def _looks_like_per_property_or_loan_budget_context(value: str) -> bool:
    text = _text_for_matching(value)
    per_item_markers = (
        "επιλεξιμος προυπολογισμος",
        "τελικος επιλεξιμος προυπολογισμος",
        "προυπολογισμος παρεμβασεων",
        "προυπολογισμος εργασιων",
        "ανα κατοικια",
        "ανα αιτηση",
        "ανα ωφελουμενο",
        "ποσο δανειου",
        "υψος δανειου",
        "δανειο εως",
        "χρηματοδοτηση εως",
        "ανωτατο ποσο",
        "μεγιστο ποσο",
    )
    return any(marker in text for marker in per_item_markers)


def _total_budget_value_has_large_unit(value: str) -> bool:
    text = _text_for_matching(value)
    return any(
        marker in text
        for marker in (
            "εκατ",
            "εκατομμυρ",
            "εκ ευρω",
            "δισ",
            "δις",
            "million",
            "billion",
            "bn",
        )
    )


def _total_budget_value_is_suspicious(value: str, source_text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    if not raw:
        return False
    amount = _amount_token_to_number(raw)
    if amount is None:
        return False
    if _total_budget_value_has_large_unit(raw):
        return False
    source_context = _context_around_value(source_text, raw, radius=140)
    if source_context and _looks_like_program_budget_context(source_context):
        return False
    if source_context and _looks_like_per_property_or_loan_budget_context(source_context):
        return True
    return amount < 1000000


def _context_around_value(text: str, value: str, radius: int = 180) -> str:
    if not text or not value:
        return ""
    normalized_text = re.sub(r"\s+", " ", text)
    token = re.sub(r"(?:€|ευρώ|ευρω)", "", str(value), flags=re.IGNORECASE).strip()
    token = token.strip("() ")
    candidates = [str(value).strip(), token]
    for candidate in candidates:
        if not candidate:
            continue
        idx = normalized_text.find(candidate)
        if idx != -1:
            return normalized_text[max(0, idx - radius): idx + len(candidate) + radius]
    return ""


def _normalize_total_budget_phrase(value: str) -> str:
    phrase = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;")
    phrase = re.sub(r"^(?:ύψους|υψους|περίπου|περιπου|σε|στα|στο)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase


def _extract_total_budget_from_source(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "")
    if not clean:
        return ""

    amount = (
        r"(?:€\s*)?"
        r"[0-9]+(?:[.,][0-9]{3})*(?:[.,][0-9]+)?"
        r"\s*(?:€|ευρώ|ευρω|εκατ\.?\s*ευρώ|εκατομμύρια\s*ευρώ|εκατομμυρια\s*ευρω|δισ\.?\s*ευρώ|δις\.?\s*ευρώ|billion|million)?"
    )
    strong_patterns = [
        rf"(?:συνολικός|συνολικος|συνολική|συνολικη)\s+(?:προϋπολογισμός|προυπολογισμος|δημόσια\s+δαπάνη|δημοσια\s+δαπανη|επένδυση|επενδυση)[^.;:,\n]{{0,120}}?({amount})",
        rf"(?:προϋπολογισμός|προυπολογισμος)\s+(?:του\s+)?(?:προγράμματος|προγραμματος)[^.;:,\n]{{0,120}}?({amount})",
        rf"(?:διαθέσιμοι|διαθεσιμοι)\s+πόροι[^.;:,\n]{{0,120}}?({amount})",
        rf"(?:budget|program budget|programme budget|total budget)[^.;:,\n]{{0,120}}?({amount})",
        rf"({amount})[^.;:,\n]{{0,120}}?(?:συνολικός|συνολικος)\s+(?:προϋπολογισμός|προυπολογισμος)",
    ]

    for pattern in strong_patterns:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            phrase = _normalize_total_budget_phrase(match.group(1))
            context = clean[max(0, match.start() - 160): match.end() + 160]
            if not phrase or _looks_like_per_property_or_loan_budget_context(context):
                continue
            if _looks_like_program_budget_context(context) or _looks_like_program_budget_amount(phrase):
                return phrase
    return ""


def _looks_like_home_purchase_amount(value: str) -> bool:
    amount = _amount_token_to_number(value)
    return amount is not None and amount >= 100000


def _interventions_need_source_fill(value) -> bool:
    if not value:
        return True
    if not isinstance(value, list):
        return True
    normalized = [_text_for_matching(item) for item in value if str(item).strip()]
    if len(set(normalized)) < len(normalized):
        return True
    if not any(_contains_any(item, HOME_GREEN_SCOPE_TERMS["energy"]) for item in value):
        return True
    return any(
        "[...]" in str(item)
        or "…" in str(item)
        or ") επενδύσεις" in _text_for_matching(item)
        or "(φωτοβολτα" in _text_for_matching(item)
        or _text_for_matching(item).endswith(" κ")
        for item in value
    )


def _replace_composite_home_purchase_with_energy_section(clean: dict, text: str) -> bool:
    """Repair composite pages where the model picked a home-purchase subprogram."""
    name = str(clean.get("programme_name") or "")
    name_text = _text_for_matching(name)
    if "σπιτι μου" not in name_text or "αναβαθμιζω" in name_text:
        return False

    match = re.search(r"Αναβαθμίζω\s+το\s+σπίτι\s+μου", text or "", flags=re.IGNORECASE)
    if not match:
        return False

    clean["programme_name"] = match.group(0)
    segment = re.sub(r"\s+", " ", (text or "")[match.start(): match.start() + 1400]).strip()
    stop_markers = ["Οι επιλέξιμες δαπάνες", "Ο συνολικός προϋπολογισμός"]
    stops = [segment.find(marker) for marker in stop_markers if segment.find(marker) > 80]
    if stops:
        segment = segment[: min(stops)].strip()
    if segment:
        clean["description"] = segment
        clean["programme_objective"] = segment
    return True


def _fill_missing_core_fields_from_source(extracted_data: dict, text: str, url: str) -> dict:
    """Fill missing high-signal fields using exact spans from scraped source text."""
    clean = extracted_data if isinstance(extracted_data, dict) else {}
    clean = {k: _clean_extracted_value(v) for k, v in clean.items()}
    _clear_unsupported_hallucinated_fields(clean, text)
    forced_energy_section = _replace_composite_home_purchase_with_energy_section(clean, text)
    forced_source_title = False
    source_title = _source_product_title(text)

    if _should_use_source_product_title(str(clean.get("programme_name") or ""), source_title, url):
        clean["programme_name"] = source_title
        forced_source_title = True

    scoped_sentence = _best_scoped_source_sentence(text, url)
    product_intro = _best_product_intro(text, source_title, url) if source_title else ""
    if product_intro:
        if forced_source_title or _should_use_product_intro(str(clean.get("description") or ""), source_title, url):
            clean["description"] = product_intro
        if forced_source_title or _should_use_product_intro(str(clean.get("programme_objective") or ""), source_title, url):
            clean["programme_objective"] = product_intro

    if scoped_sentence:
        programme_name = str(clean.get("programme_name") or "")
        if not forced_source_title and _should_replace_scoped_text(str(clean.get("description") or ""), programme_name):
            clean["description"] = scoped_sentence
        if not forced_source_title and _should_replace_scoped_text(str(clean.get("programme_objective") or ""), programme_name):
            clean["programme_objective"] = scoped_sentence

    exact_intro = _best_exact_program_intro(text)
    if exact_intro:
        if _looks_like_boilerplate_intro(str(clean.get("description") or "")):
            clean["description"] = exact_intro
        if _looks_like_boilerplate_intro(str(clean.get("programme_objective") or "")):
            clean["programme_objective"] = exact_intro

    funding_type_text = _text_for_matching(clean.get("funding_type") or "")
    loan_source_text = _text_for_matching(" ".join([source_title, scoped_sentence, url]))
    if (
        _contains_any(
            loan_source_text,
            ["δάνειο", "δανειο", "loan", "mortgage", "daneio", "daneia", "dania", "stegastiko", "stegastika"],
        )
        and (
            not funding_type_text
            or funding_type_text in {"grant", "subsidy", "επιχορηγηση", "επιδοτηση"}
        )
    ):
        clean["funding_type"] = "Δάνειο"

    exact_interventions = _extract_exact_interventions_from_source(text)
    if exact_interventions:
        clean["eligible_interventions"] = exact_interventions
    elif _interventions_need_source_fill(clean.get("eligible_interventions")):
        interventions = _extract_interventions_from_source(text, url)
        if interventions:
            clean["eligible_interventions"] = interventions
        else:
            existing_interventions = clean.get("eligible_interventions") or []
            if not isinstance(existing_interventions, list):
                existing_interventions = [existing_interventions]
            if not any(_contains_any(item, HOME_GREEN_SCOPE_TERMS["energy"]) for item in existing_interventions):
                clean["eligible_interventions"] = []

    property_requirements = _extract_property_requirements_from_source(text)
    if property_requirements:
        clean["property_requirements"] = property_requirements

    if _energy_targets_look_like_interventions(clean.get("energy_performance_targets")):
        clean["energy_performance_targets"] = ""

    if _eligible_parties_looks_like_document_artifact(clean.get("eligible_parties"), text):
        clean["eligible_parties"] = []

    minimum_amount, maximum_amount = _extract_loan_amount_range_from_source(text)
    if minimum_amount and (
        forced_energy_section
        or not str(clean.get("minimum_funding_amount") or "").strip()
        or _amount_field_lacks_currency(clean.get("minimum_funding_amount"))
        or _source_range_should_override(
            str(clean.get("minimum_funding_amount") or ""),
            str(clean.get("maximum_funding_amount") or ""),
            minimum_amount,
            maximum_amount,
        )
    ):
        clean["minimum_funding_amount"] = minimum_amount
    if maximum_amount and (
        forced_energy_section
        or _looks_like_home_purchase_amount(str(clean.get("maximum_funding_amount") or ""))
        or
        not str(clean.get("maximum_funding_amount") or "").strip()
        or _looks_like_program_budget_amount(str(clean.get("maximum_funding_amount") or ""))
        or _amount_field_lacks_currency(clean.get("maximum_funding_amount"))
        or _source_range_should_override(
            str(clean.get("minimum_funding_amount") or ""),
            str(clean.get("maximum_funding_amount") or ""),
            minimum_amount,
            maximum_amount,
        )
    ):
        clean["maximum_funding_amount"] = maximum_amount

    minimum_only = _extract_minimum_loan_amount_from_source(text)
    if minimum_only and (
        not str(clean.get("minimum_funding_amount") or "").strip()
        or _amount_field_lacks_currency(clean.get("minimum_funding_amount"))
    ):
        clean["minimum_funding_amount"] = minimum_only
    if (
        minimum_only
        and _normalize_amount_token(str(clean.get("maximum_funding_amount") or "")) == _normalize_amount_token(minimum_only)
        and not maximum_amount
    ):
        clean["maximum_funding_amount"] = ""

    amount_phrases = _extract_amount_phrases_from_source(text)
    if amount_phrases and not str(clean.get("maximum_funding_amount") or "").strip():
        clean["maximum_funding_amount"] = amount_phrases[0]
    elif _looks_like_program_budget_amount(str(clean.get("maximum_funding_amount") or "")):
        clean["maximum_funding_amount"] = ""
    elif (
        str(clean.get("maximum_funding_amount") or "").strip()
        and not maximum_amount
        and not _value_has_source_support(clean.get("maximum_funding_amount"), text)
    ):
        clean["maximum_funding_amount"] = ""

    source_interest_rate = _extract_interest_rate_from_source(text)
    if source_interest_rate and _source_field_should_replace(
        "interest_rate",
        clean.get("interest_rate"),
        source_interest_rate,
        text,
    ):
        clean["interest_rate"] = source_interest_rate
    if str(clean.get("interest_rate") or "").rstrip().endswith("του Ν"):
        legal_tail = re.search(r"του\s+Ν\.\s*128/(?:75|1975)", text or "", flags=re.IGNORECASE)
        if legal_tail:
            clean["interest_rate"] = re.sub(r"του\s+Ν\s*$", legal_tail.group(0), str(clean["interest_rate"]).rstrip())

    source_loan_duration = _extract_loan_duration_from_source(text)
    if source_loan_duration and _source_field_should_replace(
        "loan_duration",
        clean.get("loan_duration"),
        source_loan_duration,
        text,
    ):
        clean["loan_duration"] = source_loan_duration

    source_funding_coverage = _extract_funding_coverage_from_source(text)
    if source_funding_coverage and _source_field_should_replace(
        "funding_coverage",
        clean.get("funding_coverage"),
        source_funding_coverage,
        text,
    ):
        clean["funding_coverage"] = source_funding_coverage
    elif (
        source_funding_coverage
        and "80%" in source_funding_coverage
        and "80%" not in str(clean.get("funding_coverage") or "")
    ):
        clean["funding_coverage"] = source_funding_coverage

    clean["minimum_funding_amount"] = _clean_amount_field_value(clean.get("minimum_funding_amount"))
    clean["maximum_funding_amount"] = _clean_amount_field_value(clean.get("maximum_funding_amount"))
    clean["minimum_funding_amount"] = _add_currency_if_source_has_same_amount(clean.get("minimum_funding_amount"), text)
    clean["maximum_funding_amount"] = _add_currency_if_source_has_same_amount(clean.get("maximum_funding_amount"), text)

    source_total_budget = _extract_total_budget_from_source(text)
    current_total_budget = str(clean.get("total_budget") or "").strip()
    if source_total_budget and (
        not current_total_budget
        or _total_budget_value_is_suspicious(current_total_budget, text)
        or _looks_like_per_property_or_loan_budget_context(_context_around_value(text, current_total_budget))
    ):
        clean["total_budget"] = source_total_budget
    elif current_total_budget and _total_budget_value_is_suspicious(current_total_budget, text):
        clean["total_budget"] = ""

    clean["total_budget"] = _add_currency_if_source_has_same_amount(clean.get("total_budget"), text)
    if _same_min_max_looks_like_single_threshold(
        str(clean.get("minimum_funding_amount") or ""),
        str(clean.get("maximum_funding_amount") or ""),
        text,
    ):
        clean["minimum_funding_amount"] = ""
    for amount_key in ("minimum_funding_amount", "maximum_funding_amount"):
        if _looks_like_percentage_only(clean.get(amount_key)):
            if not str(clean.get("funding_coverage") or "").strip():
                clean["funding_coverage"] = clean.get(amount_key)
            clean[amount_key] = ""

    _repair_date_fields_from_source(clean, text)
    source_completion_deadline = _extract_completion_deadline_from_source(text)
    if source_completion_deadline:
        clean["completion_deadline"] = source_completion_deadline
    for policy_field in ("completion_delay_consequences", "post_completion_obligations"):
        llm_policy_value = _best_llm_policy_value(clean.get(policy_field), policy_field, text)
        source_policy_value = _best_policy_snippet(text, policy_field)
        clean[policy_field] = llm_policy_value or source_policy_value or ""
    return clean


def _save_qa_responses(url: str, qa_responses: List[dict]) -> str:
    """Save Q&A responses collected during interactive session.
    
    Args:
        url: URL of the program
        qa_responses: List of {"question": str, "answer": str, "timestamp": str} dicts
    
    Returns:
        Path to saved file
    """
    out_dir = ensure_outputs_dir()
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from llm_axe.va3_scraper_to_template import _make_safe_name, _short_hash
    safe_name = f"{_make_safe_name(url)}_{_short_hash(url)}"
    path = os.path.join(out_dir, f"{ts}_{safe_name}_qa_responses.json")
    
    result = {
        "timestamp": ts,
        "url": url,
        "qa_count": len(qa_responses),
        "responses": qa_responses
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    log(f"[INFO] Q&A responses saved to: {path}")
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
    url_title = _derive_title_from_url(url)
    if 5 < len(url_title) < 200:
        title = url_title
    
    # Create minimal structure from template to stay in sync
    minimal = {k: ([] if isinstance(v, list) else "") for k, v in TEMPLATE_DEFAULT[0].items()}
    minimal["programme_name"] = title
    minimal["description"] = _best_relevant_excerpt(text, max_chars=1200) or (text[:500] if len(text) > 500 else text)
    minimal["source_url"] = url
    minimal["source_urls"] = [url] if url else []
    minimal["additional_details"] = f"Raw text from {url} (extraction failed)"
    return minimal


def _minimal_rejected_data(url: str) -> dict:
    minimal = {k: ([] if isinstance(v, list) else "") for k, v in TEMPLATE_DEFAULT[0].items()}
    minimal["source_url"] = url
    minimal["source_urls"] = [url] if url else []
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
    scope_ok, scope_reason, scope = _is_home_green_finance_candidate(text_blob, extracted_data, source_url)

    if scope_ok:
        if not out.get("is_relevant"):
            out["is_relevant"] = True
            base_reason = (out.get("reasoning") or "").strip()
            override_reason = f"Deterministic override: {scope_reason}"
            out["reasoning"] = f"{base_reason} | {override_reason}" if base_reason else override_reason
        if out.get("primary_category") in ("other", "housing_loan", ""):
            out["primary_category"] = "energy_upgrade" if scope.get("has_home_appliance") else "green_housing_loan"
        out["confidence"] = max(float(out.get("confidence", 0.0) or 0.0), 0.8 if has_interventions else 0.74)
        key_features = out.get("key_features", [])
        if not isinstance(key_features, list):
            key_features = []
        if "deterministic_home_green_finance_scope" not in key_features:
            key_features.append("deterministic_home_green_finance_scope")
        out["key_features"] = key_features
        return out

    if out.get("is_relevant") and (scope.get("business_or_public_only") or scope.get("vehicle_only")):
        out["is_relevant"] = False
        out["primary_category"] = "other"
        out["confidence"] = max(float(out.get("confidence", 0.0) or 0.0), 0.85)
        out["reasoning"] = f"Auto-corrected: outside scoped home green finance ({scope_reason})"
        return out

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


def _deterministic_classification_if_obvious(extracted_data: dict, source_url: str = "") -> Optional[dict]:
    """Return a conservative deterministic classification for clear scope cases."""
    if not isinstance(extracted_data, dict):
        return None

    scope_ok, scope_reason, scope = _is_home_green_finance_candidate(source_url, extracted_data)
    if scope.get("business_or_public_only") or scope.get("vehicle_only"):
        return {
            "is_relevant": False,
            "primary_category": "other",
            "secondary_categories": [],
            "confidence": 0.95,
            "reasoning": f"Deterministic scope reject: {scope_reason}",
            "key_features": ["deterministic_scope_reject"],
        }

    if not scope_ok:
        return None

    interventions = extracted_data.get("eligible_interventions", [])
    has_interventions = isinstance(interventions, list) and bool(interventions)
    text = scope.get("text", "")
    loan_like = _contains_any(text, ["δάνει", "δανει", "loan", "mortgage", "στεγαστ", "credit"])
    renewables_like = _contains_any(text, ["φωτοβολτα", "solar", "heat pump", "αντλια θερμο", "απε"])

    primary_category = "green_housing_loan" if loan_like else "energy_upgrade"
    secondary_categories = []
    if renewables_like:
        secondary_categories.append("home_renewables")
    if primary_category != "energy_upgrade" and (has_interventions or scope.get("has_energy")):
        secondary_categories.append("energy_upgrade")

    return {
        "is_relevant": True,
        "primary_category": primary_category,
        "secondary_categories": secondary_categories,
        "confidence": 0.92 if has_interventions else 0.84,
        "reasoning": f"Deterministic scope pass: {scope_reason}",
        "key_features": ["deterministic_home_green_finance_scope"],
    }

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
    
    Empty fields are allowed; the record only needs enough source-backed
    substance to classify and display the parts that were actually found.
    """
    if not extracted_data:
        return False

    cleaned = _clean_extracted_value(extracted_data)

    def has_value(value) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(has_value(item) for item in value)
        if isinstance(value, dict):
            return any(has_value(item) for item in value.values())
        return value is not None and value != ""
    
    non_empty_fields = sum(1 for v in cleaned.values() if has_value(v))
    if non_empty_fields == 0:
        log("[DEBUG] Extracted data is completely empty")
        return False

    name = str(cleaned.get("programme_name") or "").strip()
    desc = str(cleaned.get("description") or "").strip()
    objective = str(cleaned.get("programme_objective") or "").strip()
    interventions = cleaned.get("eligible_interventions") or []
    criteria = cleaned.get("eligibility_criteria") or []
    property_reqs = cleaned.get("property_requirements") or []
    funding_fields = [
        cleaned.get("minimum_funding_amount"),
        cleaned.get("maximum_funding_amount"),
        cleaned.get("funding_type"),
        cleaned.get("interest_rate"),
        cleaned.get("funding_coverage"),
        cleaned.get("total_budget"),
    ]

    evidence_score = 0
    evidence_score += 2 if len(name) > 8 else 0
    evidence_score += 2 if len(desc) > 40 else 0
    evidence_score += 2 if len(objective) > 40 else 0
    evidence_score += 2 if has_value(interventions) else 0
    evidence_score += 1 if has_value(criteria) else 0
    evidence_score += 1 if has_value(property_reqs) else 0
    evidence_score += sum(1 for value in funding_fields if has_value(value))

    if evidence_score < 3:
        log(
            "[DEBUG] Extracted data lacks enough source-backed fields "
            f"(non_empty={non_empty_fields}, score={evidence_score}, "
            f"name={len(name)}c, desc={len(desc)}c, objective={len(objective)}c)"
        )
        return False
    
    return True


def _extract_relevant_text(text: str, window: int = 900) -> str:
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
    
    _relevance_keywords.extend(
        HOME_GREEN_SCOPE_TERMS["home"]
        + HOME_GREEN_SCOPE_TERMS["energy"]
        + HOME_GREEN_SCOPE_TERMS["finance"]
        + HOME_GREEN_SCOPE_TERMS["home_program"]
        + HOME_GREEN_SCOPE_TERMS["home_appliance"]
    )

    text_lower = _text_for_position_matching(text)
    
    # Collect all keyword hit positions
    positions = set()
    for kw in _relevance_keywords:
        kw = _text_for_position_matching(kw)
        if not kw:
            continue
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
    max_extract_chars = min(24000, max(9000, len(text) // 2))
    if len(result) > max_extract_chars:
        priority_keyword_groups = (
            HOME_GREEN_SCOPE_TERMS["home_program"],
            HOME_GREEN_SCOPE_TERMS["home"],
            HOME_GREEN_SCOPE_TERMS["energy"],
            HOME_GREEN_SCOPE_TERMS["home_appliance"],
            HOME_GREEN_SCOPE_TERMS["finance"],
            [
                "ποσό",
                "ποσο",
                "ποσοστό",
                "ποσοστο",
                "επιτόκιο",
                "επιτοκιο",
                "δικαιούχ",
                "δικαιουχ",
                "κριτήρι",
                "κριτηρι",
                "παρέμβασ",
                "παρεμβασ",
                "επέμβασ",
                "επεμβασ",
                "προθεσμία",
                "προθεσμια",
                "διάρκεια",
                "διαρκεια",
                "προϋπόθεσ",
                "προυποθεσ",
                "προϋπολογισμ",
                "αιτηση",
                "αίτηση",
                "υποβολ",
            ],
        )
        focused_positions = []
        for keyword_group in priority_keyword_groups:
            group_positions = []
            for kw in keyword_group:
                kw = _text_for_position_matching(kw)
                if not kw:
                    continue
                start = 0
                while len(group_positions) < 5:
                    idx = text_lower.find(kw, start)
                    if idx == -1:
                        break
                    if all(abs(idx - existing) > 500 for existing in group_positions):
                        group_positions.append(idx)
                    start = idx + len(kw)
                if len(group_positions) >= 5:
                    break
            focused_positions.extend(group_positions)

        focused_intervals = [(0, min(1600, len(text)))]
        for pos in sorted(set(focused_positions)):
            focused_intervals.append((max(0, pos - window), min(len(text), pos + window)))

        focused_intervals.sort()
        focused_merged = []
        for lo, hi in focused_intervals:
            if not focused_merged or lo > focused_merged[-1][1]:
                focused_merged.append((lo, hi))
            else:
                focused_merged[-1] = (focused_merged[-1][0], max(focused_merged[-1][1], hi))

        result = "\n[...]\n".join(text[lo:hi] for lo, hi in focused_merged)
        if len(result) > max_extract_chars:
            packed_parts = []
            used = 0
            separator = "\n[...]\n"
            for lo, hi in focused_merged:
                part = text[lo:hi]
                extra = len(part) + (len(separator) if packed_parts else 0)
                if used + extra > max_extract_chars:
                    remaining = max_extract_chars - used - (len(separator) if packed_parts else 0)
                    if remaining > 700:
                        packed_parts.append(part[:remaining])
                    break
                packed_parts.append(part)
                used += extra
            result = separator.join(packed_parts) if packed_parts else result[:max_extract_chars]
        log(f"[INFO] Focused extraction cap applied: {len(text)} -> {len(result)} chars")
    log(f"[INFO] Smart text extraction: {len(text)} → {len(result)} chars ({len(merged)} relevant sections)")
    return result


def _has_energy_keywords(text: str, url: str = "") -> bool:
    """Check if scraped text contains any energy-related keywords."""
    scope_ok, scope_reason, scope = _is_home_green_finance_candidate(url, text)
    if scope_ok:
        log(f"[DEBUG] Scoped keyword pre-filter passed: {scope_reason}")
        return True
    if scope.get("business_or_public_only") or scope.get("vehicle_only"):
        log(f"[DEBUG] Scoped keyword pre-filter rejected: {scope_reason}")
        return False

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
    
    ENERGY_KEYWORDS.extend(
        HOME_GREEN_SCOPE_TERMS["energy"]
        + HOME_GREEN_SCOPE_TERMS["home_program"]
        + HOME_GREEN_SCOPE_TERMS["home_appliance"]
    )
    FINANCING_KEYWORDS.extend(HOME_GREEN_SCOPE_TERMS["finance"])

    text_lower = _text_for_matching(text)
    
    # Check for energy keywords
    has_energy = any(_text_for_matching(keyword) in text_lower for keyword in ENERGY_KEYWORDS)
    
    # Check for financing keywords
    has_financing = any(_text_for_matching(keyword) in text_lower for keyword in FINANCING_KEYWORDS)
    
    # Log what was found (important for debugging)
    if has_energy:
        for keyword in ENERGY_KEYWORDS:
            if _text_for_matching(keyword) in text_lower:
                log(f"[DEBUG] ✓ Energy keyword found: '{keyword}'")
                break
    else:
        log(f"[DEBUG] ✗ No energy keywords found")
    
    if has_financing:
        for keyword in FINANCING_KEYWORDS:
            if _text_for_matching(keyword) in text_lower:
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
    scope_ok, scope_reason, scope = _is_home_green_finance_candidate(url, text)
    if scope_ok:
        log(f"[PRE-SCREEN] Deterministic scope pass: {scope_reason}")
        return True
    if scope.get("business_or_public_only") or scope.get("vehicle_only"):
        log(f"[PRE-SCREEN] Deterministic scope reject: {scope_reason}")
        return False

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
        _prescreen_keywords.extend(
            HOME_GREEN_SCOPE_TERMS["home"]
            + HOME_GREEN_SCOPE_TERMS["energy"]
            + HOME_GREEN_SCOPE_TERMS["finance"]
            + HOME_GREEN_SCOPE_TERMS["home_program"]
            + HOME_GREEN_SCOPE_TERMS["home_appliance"]
        )
        text_lower = _text_for_position_matching(text)
        TITLE_ZONE = 300  # skip keywords in the title/header area
        earliest = len(text)
        for kw in _prescreen_keywords:
            kw = _text_for_position_matching(kw)
            if not kw:
                continue
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

    deterministic = _deterministic_classification_if_obvious(extracted_data, source_url)
    if deterministic is not None:
        experiment_id = None
        log(f"[DEBUG] Deterministic classification: {deterministic['reasoning']}")
        if log_it:
            try:
                experiment_id = log_experiment(
                    model="deterministic_scope_guard",
                    temperature=0.0,
                    hyperparameters={"mode": "pre_llm_classification"},
                    prompt="",
                    sources=[_get_program_name(extracted_data)],
                    terminal_output=(
                        f"Category: {deterministic['primary_category']} | "
                        f"Confidence: {deterministic.get('confidence', 0):.0%} | "
                        f"Relevant: {deterministic['is_relevant']}"
                    ),
                )
                log(f"[LOG] Experiment: {experiment_id}")
            except Exception as log_err:
                log(f"[WARN] Failed to log deterministic experiment: {log_err}")
        return deterministic, experiment_id

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
    classification_context = {
        "primary_category": classification.get("primary_category", "other"),
        "category_label": ALL_CATEGORIES.get(classification.get("primary_category", "other"), "Άγνωστη"),
        "confidence": classification.get("confidence", 0.0),
        "is_relevant": classification.get("is_relevant"),
        "secondary_categories": classification.get("secondary_categories", []),
        "key_features": classification.get("key_features", []),
    }
    return build_single_qa_prompt(extracted_data, question, classification_context)

def ask_program_question(
    llm,
    extracted_data: dict,
    classification: dict,
    question: str,
    temperature: float = 0.3,
    num_ctx: Optional[int] = None,
) -> str:
    """Ask the configured Q&A LLM using the canonical VA4 prompt."""
    out_of_scope_answer = deterministic_qa_out_of_scope_answer(question)
    if out_of_scope_answer:
        return out_of_scope_answer

    prompts = build_qa_prompt(extracted_data, classification, question)
    kwargs = {"format": "", "temperature": temperature}
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx
    return llm.ask(prompts, **kwargs)

def interactive_qa(llm, extracted_data: dict, classification: dict, url: str):
    """Start interactive Q&A session about the classified program.
    
    Stores Q&A pairs in a JSON file for later evaluation by qa_consistency_validator.
    """
    
    qa_responses = []  # Collect Q&A pairs for storage
    
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
            answer = ask_program_question(llm, extracted_data, classification, question)
            log(f"\n💡 Απάντηση:\n{answer}\n")
            
            # Store Q&A pair for later evaluation
            qa_responses.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            log(f"[ERROR] Σφάλμα κατά την απάντηση: {e}\n")
    
    # Save Q&A responses to file
    if qa_responses:
        _save_qa_responses(url, qa_responses)

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
            if not _has_energy_keywords(text, url_normalized):
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
                try:
                    rejected_data = _minimal_rejected_data(url_normalized)
                    save_classification_result(
                        url_normalized,
                        rejected_data,
                        rejection,
                    )
                    log("[DEBUG] Keyword pre-screen rejection classification saved to output folder")
                except Exception as save_err:
                    log(f"[WARN] Failed to save keyword rejection result: {save_err}")
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
                try:
                    rejected_data = _minimal_rejected_data(url_normalized)
                    save_classification_result(
                        url_normalized,
                        rejected_data,
                        rejection,
                    )
                    log("[DEBUG] LLM pre-screen rejection classification saved to output folder")
                except Exception as save_err:
                    log(f"[WARN] Failed to save LLM rejection result: {save_err}")
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
                extracted_data = _fill_missing_core_fields_from_source(extracted_data, text, url_normalized)
                extracted_data = enrich_program_identity(extracted_data, url_normalized)
                
                # DEBUG: Show what keys we got
                log(f"[DEBUG] Extracted data has {len(extracted_data)} keys: {list(extracted_data.keys())[:10]}...")
                
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
                
                if (
                    isinstance(extracted_data.get("programme_name"), str)
                    and extracted_data["programme_name"].strip().endswith("URL)")
                ):
                    log("[WARN] Discarding URL-derived programme_name from extracted JSON")
                    extracted_data["programme_name"] = ""
                    program_name = _get_program_name(extracted_data)

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
                    log(f"[ERROR] All extraction attempts failed; not creating fallback JSON.")
                    raise extraction_error
        
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

        if is_relevant:
            try:
                saved_path = save_result([extracted_data], url_normalized)
                log(f"[DEBUG] Saved relevant extracted data to: {saved_path}")
            except Exception as save_err:
                log(f"[WARN] Failed to save relevant extracted data: {save_err}")
        else:
            log("[DEBUG] Not saving extracted program JSON for irrelevant classification")
        
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
        default=DEFAULT_MODEL_FAST,
        help=f"Fast LLM for pre-screen + extraction (default: {DEFAULT_MODEL_FAST})"
    )
    parser.add_argument(
        "--model-smart",
        default=DEFAULT_MODEL_SMART,
        help=f"Smart LLM for classification + Q&A (default: {DEFAULT_MODEL_SMART})"
    )
    parser.add_argument(
        "--no-merge-refresh",
        action="store_true",
        help="Skip automatic prepare-only merged snapshot refresh after URL processing"
    )
    parser.add_argument(
        "--merge-refresh-timeout-seconds",
        type=int,
        default=600,
        help="Timeout for automatic merged snapshot refresh (default: 600 seconds)"
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
            if not args.no_merge_refresh:
                snapshot = refresh_merged_snapshot(
                    prefix="cli_merged",
                    log=log,
                    timeout_seconds=args.merge_refresh_timeout_seconds,
                )
                if snapshot.get("status") == "ok":
                    log(
                        "[INFO] Merged snapshot refreshed: "
                        f"{snapshot.get('run_id')} "
                        f"({snapshot.get('target_count', '?')} programs, "
                        f"{snapshot.get('source_url_count', '?')} source URLs)"
                    )
                else:
                    log(f"[WARN] Merged snapshot refresh failed: {snapshot.get('message', '')}")
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
                if not args.no_merge_refresh:
                    snapshot = refresh_merged_snapshot(
                        prefix="cli_merged",
                        log=log,
                        timeout_seconds=args.merge_refresh_timeout_seconds,
                    )
                    if snapshot.get("status") == "ok":
                        log(
                            "[INFO] Merged snapshot refreshed: "
                            f"{snapshot.get('run_id')} "
                            f"({snapshot.get('target_count', '?')} programs, "
                            f"{snapshot.get('source_url_count', '?')} source URLs)"
                        )
                    else:
                        log(f"[WARN] Merged snapshot refresh failed: {snapshot.get('message', '')}")
            except Exception as e:
                log(f"\n[ERROR] Αποτυχία επεξεργασίας: {e}\n")
                continue

if __name__ == "__main__":
    main()
