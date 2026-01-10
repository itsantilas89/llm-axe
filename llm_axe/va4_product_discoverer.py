# va4_product_discoverer.py
# Simple product discovery agent: finds relevant green loan products from trusted bank URLs
# Uses va3 for scraping and extraction

import os, sys, json, time, re
from datetime import datetime
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

try:
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.va3_scraper_to_template import scrape_page as va3_scrape, extract_json as va3_extract, TEMPLATE_DEFAULT
except:
    import importlib.util as _il
    here = os.path.dirname(__file__)
    for _mod in ("models", "core", "va3_scraper_to_template"):
        _path = os.path.join(here, f"{_mod}.py")
        if os.path.exists(_path):
            spec = _il.spec_from_file_location(f"llm_axe.{_mod}", _path)
            sys.modules[f"llm_axe.{_mod}"] = _il.module_from_spec(spec)
            sys.modules[f"llm_axe.{_mod}"].loader.exec_module(sys.modules[f"llm_axe.{_mod}"])
    from llm_axe.models import OllamaChat
    from llm_axe.core import make_prompt
    from llm_axe.va3_scraper_to_template import scrape_page as va3_scrape, extract_json as va3_extract, TEMPLATE_DEFAULT

log = lambda msg: print(msg, flush=True)

def load_trusted_sources() -> Dict[str, List[str]]:
    """Load trusted sources from trusted_sources.json"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "trusted_sources.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def scrape_page(url: str) -> Tuple[str, str]:
    """Get clean text and HTML from URL"""
    try:
        text = va3_scrape(url)
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        return text, html
    except Exception as e:
        log(f"[ERROR] {e}")
        return "", ""

def extract_product_links(base_url: str, html: str) -> List[Dict[str, str]]:
    """Extract only product/loan-related links"""
    soup = BeautifulSoup(html, "html.parser")
    links, seen = [], set()
    base_host = urlparse(base_url).netloc
    
    # MUST HAVE keywords in URL or text
    must_have = ["loan", "daneio", "δανειο", "product", "proi", "προϊ", "stegastik", "στεγαστικ",
                 "energy", "energe", "ενεργε", "solar", "photovoltaic", "fotovolt", "φωτοβολτα",
                 "renovation", "anaplasi", "αναπλασ", "green", "prasino", "πρασιν"]
    
    # Skip patterns
    skip = ["about", "contact", "terms", "privacy", "help", "faq", "sitemap", "news", 
            "blog", "career", "login", "register", "search", "category", "tag", "archive"]
    
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith("#"): continue
        
        url = urljoin(base_url, href)
        url_lower = url.lower()
        text_lower = a.get_text(strip=True).lower()
        
        # Skip if wrong domain, seen, or generic
        if urlparse(url).netloc != base_host or url in seen: continue
        if any(s in url_lower for s in skip): continue
        
        # ONLY keep if any word starts with relevant keywords
        url_words = url_lower.replace('/', ' ').replace('-', ' ').split()
        text_words = text_lower.split()
        if not any(any(word.startswith(kw) for word in url_words + text_words) for kw in must_have):
            continue
        
        seen.add(url)
        links.append({"url": url, "text": a.get_text(strip=True)[:100]})
    
    return links  # No limit

def is_relevant(llm, url: str, text: str) -> bool:
    """Check if product is relevant to energy efficiency"""
    prompt = [
        make_prompt("system", "You are an expert at identifying green loan products. Respond ONLY 'yes' or 'no'."),
        make_prompt("user", f"Is this related to energy efficiency/green loans?\nURL: {url}\nText: {text[:500]}\nAnswer:")
    ]
    try:
        return llm.ask(prompt, temperature=0.2).strip().lower().startswith("yes")
    except:
        return False

def extract_data(llm, url: str, text: str) -> Dict:
    """Extract structured data using va3"""
    try:
        result = va3_extract(llm, text, TEMPLATE_DEFAULT[0], url)
        return result[0] if isinstance(result, list) and result else (result if isinstance(result, dict) else {})
    except:
        return {}

def save_result(bank: str, name: str, url: str, data: Dict) -> str:
    """Deprecated: per-product file saving disabled to reduce file clutter."""
    return ""

def discover_products(bank: str, url: str, llm) -> List[Dict]:
    """Discover and extract green loan products from a bank URL"""
    log(f"\n{'='*60}\nProcessing: {bank}\nURL: {url}\n{'='*60}")
    
    # Scrape entry page
    text, html = scrape_page(url)
    if not text:
        return []
    
    # Extract all links
    links = extract_product_links(url, html)
    log(f"[INFO] Found {len(links)} links")
    
    results = []
    for i, link in enumerate(links, 1):  # Check all relevant links
        prod_url = link["url"]
        
        # Skip generic links
        if any(x in prod_url.lower() for x in ["page=", "search", "filter", "login"]):
            continue
        
        log(f"\n[{i}] {prod_url}")
        
        # Scrape product page
        prod_text, _ = scrape_page(prod_url)
        if not prod_text:
            continue
        
        # Check relevance
        if is_relevant(llm, prod_url, prod_text):
            log(f"    ✓ RELEVANT - extracting data...")
            data = extract_data(llm, prod_url, prod_text)
            
            if data:
                data["source_url"] = prod_url
                data["bank_name"] = bank
                name = data.get("programme_name") or link["text"][:50]
                # Per-product saving disabled; accumulate for single-run output
                results.append(data)
        else:
            log(f"    ✗ Not relevant")
        
        time.sleep(1)
    
    return results

def main():
    """Main entry point"""
    llm = OllamaChat(model="llama3.2:1b")
    sources = load_trusted_sources()
    
    if "greek_banks_green_loans_stegastika" not in sources:
        log("[ERROR] No sources found")
        return
    
    banks = ["Eurobank", "Crediabank", "Piraeus Bank", "National Bank of Greece", "Alpha Bank"]
    urls = sources["greek_banks_green_loans_stegastika"]
    all_results = []
    
    log("🚀 Starting discovery from trusted bank URLs...")
    log(f"📋 Will check {len(banks)} banks for green loan products\n")
    
    for bank, url in zip(banks, urls):
        try:
            results = discover_products(bank, url, llm)
            all_results.extend(results)
        except Exception as e:
            log(f"[ERROR] {bank}: {e}")
        time.sleep(2)
    
    # Save a single combined results file in output/
    if all_results:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "va4_product_discoverer")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        results_file = os.path.join(out_dir, f"{ts}_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "banks_checked": len(banks),
                "total_products": len(all_results),
                "products": all_results
            }, f, ensure_ascii=False, indent=2)
        log(f"\n✅ Discovery complete! Found {len(all_results)} green loan products")
        log(f"📄 Results: {results_file}")
    else:
        log("\n⚠️ No products discovered")

if __name__ == "__main__":
    main()
