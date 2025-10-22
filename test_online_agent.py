from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat

def main():
    llm = OllamaChat(model="llama3.2:3b")       ## You may change to your desired model
    agent = OnlineAgent(llm)
    history = []

    print("Interactive LLM-AXE session with model LLaMA 3.2-3B. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # if message contains a URL, use OnlineAgent.search()
        if "http://" in user_input or "https://" in user_input:
            response = agent.search(user_input, history=history)
        else:
            # normal chat flow
            response = llm.ask(
                history + [{"role": "user", "content": user_input}],
                temperature=0.7,
            )

        # record conversation
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print("Model:", response, "\n")

if __name__ == "__main__":
    main()
