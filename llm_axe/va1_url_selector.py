"""
online_agent_unified.py
-----------------------
Unified terminal client for llm_axe with trusted-source selection and online retrieval.

Combines:
- online_agent.py: interactive Ollama-based chat agent.
- url_selector.py: deterministic trusted source matching and selection.

Features
--------
- Trusted URL loading and fuzzy matching.
- Automatic handling of URLs or natural-language queries.
- Persistent in-session conversation memory.
- Generic structure usable for any downstream data extraction or content retrieval.

Dependencies
------------
llm_axe (core + models)
trusted_sources.json  # must exist in same directory

Usage
-----
python online_agent_unified.py
"""

import json
from difflib import SequenceMatcher
from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat


# --------------------------------------------------------------------------
# Trusted Source Handling
# --------------------------------------------------------------------------

def load_sources(path: str = "trusted_sources.json") -> dict:
    """Load trusted source definitions."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Trusted sources file not found: {path}. Using empty source list.")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load trusted sources: {e}")
        return {}


def match_sources(prompt: str, sources: dict, threshold: float = 0.25) -> list:
    """
    Fuzzy match the user query against known domains.

    Parameters
    ----------
    prompt : str
        The natural-language query from the user.
    sources : dict
        Mapping of domain keys to lists of URLs.
    threshold : float
        Minimum similarity ratio for inclusion.

    Returns
    -------
    list[str]
        Candidate URLs with domain similarity above threshold.
    """
    prompt_low = prompt.lower()
    matches = []
    for domain, urls in sources.items():
        score = SequenceMatcher(None, prompt_low, domain.lower()).ratio()
        if score >= threshold:
            matches.extend(urls)
    return matches


# --------------------------------------------------------------------------
# Unified Online Agent
# --------------------------------------------------------------------------

def main():
    """Launch an interactive loop with trusted source matching and online access."""
    llm = OllamaChat(model="llama3.2:3b")
    agent = OnlineAgent(llm)
    history = []

    model_name = llm._model
    print(f"Unified LLM-AXE Agent for {model_name}. Type 'exit' to quit.\n")

    trusted_sources = load_sources()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = None

        # Case 1: explicit URL provided by user
        if "http://" in user_input or "https://" in user_input:
            response = agent.search(user_input, history=history)

        # Case 2: query matches a trusted domain
        elif trusted_sources:
            matches = match_sources(user_input, trusted_sources)
            if matches:
                print(f"[INFO] Matched trusted sources: {matches}")
                response = agent.search(matches[0], history=history)

        # Case 3: generic chat with LLM
        if response is None:
            response = llm.ask(
                history + [{"role": "user", "content": user_input}],
                temperature=0.7,
            )

        # Store and display conversation
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        print(f"{model_name}:", response, "\n")


if __name__ == "__main__":
    main()
