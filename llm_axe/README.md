# llm_axe Internal Documentation

This document summarizes the purpose and core functionality of each Python module
inside the `llm_axe` package.  
It complements the external README by describing implementation roles and
internal dependencies.

---

## core.py
Contains foundational utilities:
- **Prompt creation** (`make_prompt`, `get_yaml_prompt`)
- **Web access** (`internet_search`, `read_website`, `selenium_reader`)
- **Data processing** (PDF reading, sentence splitting, embedding relevance)
- **Schema generation** for function-calling agents.

All other modules import from this file.

---

## models.py
Wraps external LLM providers.  
Currently implements:
- `OllamaChat` — lightweight interface to the local Ollama REST API.
Additional model backends can be added by following the same structure.

---

## agents.py
Implements specialized agent classes built around the LLM abstraction:
- `Agent`, `OnlineAgent`, `WebsiteReaderAgent`, `PythonAgent`, `FunctionCaller`,
  `DataExtractor`, `PdfReader`, `ObjectDetectorAgent`.
Each class exposes an `.ask()` or `.search()` method that orchestrates prompts,
context, and response formatting.

---

## system_prompts.yaml
Houses standardized system-level prompt templates used by all agents.
Each key corresponds to an `AgentType` or workflow component.

---

## url_selector.py
(Added by project extension)
Encapsulates logic for selecting relevant URLs from a trusted source registry.
Used by `OnlineAgent` to replace or augment open web searches.

---

## online_agent.py
(Added by project extension)
Example entry point for interactive sessions and web integration.
Demonstrates conversation memory, URL detection, and dynamic online retrieval.