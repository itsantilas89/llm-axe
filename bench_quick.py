"""Quick benchmark: measure LLM response time for tiny + medium prompts."""
import time
from llm_axe.models import OllamaChat
from llm_axe.core import make_prompt

model = "llama3.1:8b-instruct-q4_K_M"
print(f"Model: {model}")
llm = OllamaChat(model=model)

# Test 1: Tiny prompt
prompts = [
    make_prompt("system", "Reply with ONLY JSON."),
    make_prompt("user", 'Return: {"test": true}'),
]
t0 = time.time()
r = llm.ask(prompts, format="json", temperature=0.0, num_predict=50)
t1 = time.time()
print(f"\n[Test 1] Tiny prompt (cold load)")
print(f"  Time: {t1-t0:.1f}s")
print(f"  Response: {r[:100]}")

# Test 2: Same tiny prompt (warm)
t0 = time.time()
r = llm.ask(prompts, format="json", temperature=0.0, num_predict=50)
t1 = time.time()
print(f"\n[Test 2] Tiny prompt (warm)")
print(f"  Time: {t1-t0:.1f}s")

# Test 3: Medium prompt (~2000 chars input)  
text = "Ενεργειακή αναβάθμιση κατοικιών. " * 80  # ~2700 chars
prompts_med = [
    make_prompt("system", "Extract data as JSON. Return ONLY JSON."),
    make_prompt("user", f"TEXT:\n{text}\n\nReturn: " + '{"name": "", "description": ""}'),
]
t0 = time.time()
r = llm.ask(prompts_med, format="json", temperature=0.1, num_predict=512)
t1 = time.time()
print(f"\n[Test 3] Medium prompt (~2700 chars input, 512 max output)")
print(f"  Time: {t1-t0:.1f}s")
print(f"  Response: {r[:150]}")

print("\nDone!")
