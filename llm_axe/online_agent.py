"""
online_agent.py
---------------
Interactive terminal client for llm_axe using Ollama backend.

Features
--------
- Persistent conversation memory within one session.
- Automatic detection of URLs for online retrieval through OnlineAgent.
- Direct LLM chat for all other messages.

Intended as a demonstration and testing interface
for the web-integrated Virtual Assistant project.
"""

from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat

def main():
    """Launch an interactive chat loop with internet-aware capabilities."""
    llm = OllamaChat(model="gemma:1b")
    model_name = llm._model
    print(f"Interactive LLM-AXE for {model_name}. Type 'exit' to quit.\n")

    agent = OnlineAgent(llm)
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        if "http://" in user_input or "https://" in user_input:
            response = agent.search(user_input, history=history)
        else:
            response = llm.ask(
                history + [{"role": "user", "content": user_input}],
                temperature=0.7,
            )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        print(f"{model_name}:", response, "\n")

if __name__ == "__main__":
    main()
