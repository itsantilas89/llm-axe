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
import time
import threading
from difflib import SequenceMatcher
from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat

# Thread-safe printing helper to avoid interleaved messages from multiple threads
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(ts, *args, **kwargs)


# --------------------------------------------------------------------------
# Trusted Source Handling
# --------------------------------------------------------------------------

def load_sources(path: str = "trusted_sources.json") -> dict:
    """Load trusted source definitions."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        safe_print(f"[WARN] Trusted sources file not found: {path}. Using empty source list.")
        return {}
    except Exception as e:
        safe_print(f"[ERROR] Failed to load trusted sources: {e}")
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


def run_with_warning(func, *args, warning_after: float = 30.0, warning_msg: str = None, activity_msg: str = None, activity_steps: list | None = None, status_interval: float = 10.0, **kwargs):
    """Run `func` in a thread and, if it runs long, print periodic status messages.

    Behaviour:
    - Start `func` in a daemon thread.
    - If the thread is still running after `warning_after` seconds, start printing
      `activity_msg` (or `warning_msg` if activity_msg is None) every `status_interval`
      seconds until the call finishes.

    Returns the function result or raises the original exception.
    """
    result = {}

    def target():
        try:
            result['value'] = func(*args, **kwargs)
        except Exception as e:
            result['exc'] = e

    t = threading.Thread(target=target, daemon=True)

    # record start time immediately before starting worker so heartbeat can report elapsed
    start_time = time.monotonic()
    t.start()

    def heartbeat():
        # Wait until initial grace period
        time.sleep(warning_after)
        # Determine message mode: single message or ordered steps
        if activity_steps:
            steps = list(activity_steps)
        elif activity_msg:
            steps = [activity_msg]
        elif warning_msg:
            steps = [warning_msg]
        else:
            steps = ["[INFO] Please wait, your request is being processed..."]

        printed = 0
        # First, print each step in order once at each interval
        while t.is_alive():
            if printed < len(steps):
                safe_print(steps[printed])
                printed += 1
            else:
                # After exhausting the steps, print a concise still-working message with elapsed time
                elapsed = int(time.monotonic() - start_time)
                safe_print(f"[INFO] Still working, elapsed {elapsed}s. I'm still processing and will finish soon...")
            time.sleep(status_interval)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    # Wait for the worker to finish (no timeout)
    t.join()

    if 'exc' in result:
        raise result['exc']
    return result.get('value')


# --------------------------------------------------------------------------
# Unified Online Agent
# --------------------------------------------------------------------------

def main():
    """Launch an interactive loop with trusted source matching and online access."""
    try:
        llm = OllamaChat(model="gemma3")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        print("[HINT] Ensure a local LLM service is running and the requested model is available.")
        print("[HINT] Check your LLM client configuration, network connectivity, and that any required runtime is installed.")
        return
    agent = OnlineAgent(llm)
    history = []

    model_name = llm._model
    safe_print(f"Unified LLM-AXE Agent for {model_name}. Type 'exit' to quit.\n")

    trusted_sources = load_sources(path="llm_axe/trusted_sources.json")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = None

        # Case 1: explicit URL provided by user
        if "http://" in user_input or "https://" in user_input:
            safe_print("[DEBUG] Detected URL in input — will call agent.search()")
            try:
                start = time.monotonic()
                response = run_with_warning(
                    agent.search,
                    user_input,
                    history=history,
                    warning_after=30.0,
                    activity_steps=[
                        f"[INFO] Started downloading the page...",
                        f"[INFO] Downloading content from {user_input} and cleaning HTML...",
                        f"[INFO] Extracting text from {user_input}, finding relevant sections...",
                        f"[INFO] Performing further processing / data extraction from {user_input}...",
                        f"[INFO] Finalizing and preparing results from {user_input}...",
                    ],
                    status_interval=45.0
                )
                print(f"[DEBUG] agent.search completed in {time.monotonic() - start:.2f}s")
            except Exception as e:
                safe_print(f"[ERROR] agent.search failed: {e}")
                response = None

        # Case 2: query matches a trusted domain
        elif trusted_sources:
            matches = match_sources(user_input, trusted_sources)
            if matches:
                safe_print(f"[DEBUG] Matched trusted sources: {matches}")
                safe_print(f"[DEBUG] Using first matched source: {matches[0]} — calling agent.search()")
                try:
                    start = time.monotonic()
                    response = run_with_warning(
                        agent.search,
                        matches[0],
                        history=history,
                        warning_after=30.0,
                        activity_steps=[
                            f"[INFO] Connecting to source: {matches[0]}",
                            f"[INFO] Downloading page from {matches[0]} and extracting text...",
                            f"[INFO] Grouping and filtering content from {matches[0]}...",
                            f"[INFO] Processing results from {matches[0]} for presentation...",
                        ],
                        status_interval=45.0
                    )
                    print(f"[DEBUG] agent.search completed in {time.monotonic() - start:.2f}s")
                except Exception as e:
                    safe_print(f"[ERROR] agent.search failed: {e}")
                    response = None

        # Case 3: generic chat with LLM
        if response is None:
            safe_print("[DEBUG] No trusted source used, sending prompt to LLM (llm.ask)")
            try:
                start = time.monotonic()
                response = run_with_warning(
                    llm.ask,
                    history + [{"role": "user", "content": user_input}],
                    temperature=0.7,
                    warning_after=30.0,
                    activity_steps=[
                        "[INFO] Sending prompt to the model...",
                        "[INFO] The model is encoding the prompt and preparing context...",
                        "[INFO] The model is generating tokens...",
                        "[INFO] Collecting response and finishing formatting...",
                    ],
                    status_interval=45.0
                )
                print(f"[DEBUG] llm.ask completed in {time.monotonic() - start:.2f}s")
            except Exception as e:
                safe_print(f"[ERROR] llm.ask failed: {e}")
                response = "[ERROR] failed to get response"

        # Store and display conversation
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        safe_print(f"{model_name}:", response, "\n")


if __name__ == "__main__":
    main()
