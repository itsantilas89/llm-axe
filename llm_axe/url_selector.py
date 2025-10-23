"""
url_selector.py
---------------
Module providing deterministic URL selection for llm_axe OnlineAgent.

Functions
---------
load_sources(path="trusted_sources.json")
    Load the trusted URL registry from JSON.

match_sources(prompt, sources, threshold=0.25)
    Perform a fuzzy semantic match between user prompt and source keys.
    Returns a list of candidate URLs.

Usage
-----
Called internally by OnlineAgent.search() to prefer trusted URLs
before falling back to general web search.
"""

import json
from difflib import SequenceMatcher

def load_sources(path="trusted_sources.json"):
    """Load the trusted source definitions from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_sources(prompt, sources, threshold=0.25):
    """
    Match the user prompt against known source domains.

    Parameters
    ----------
    prompt : str
        The natural-language query from the user.
    sources : dict
        Mapping of domain keys to lists of URLs.
    threshold : float
        Minimum similarity score for inclusion.

    Returns
    -------
    list[str]
        URLs whose domain key is semantically close to the prompt.
    """
    prompt_low = prompt.lower()
    matches = []
    for domain, urls in sources.items():
        score = SequenceMatcher(None, prompt_low, domain.lower()).ratio()
        if score >= threshold:
            matches.extend(urls)
    return matches
