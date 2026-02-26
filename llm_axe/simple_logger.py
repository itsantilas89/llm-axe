"""
Minimal Experiment Logger

Saves experiment details to logs/ as JSON files for tracking and reproducibility.
"""

import json
import os
from datetime import datetime
import uuid


def log_experiment(
    model: str,
    temperature: float,
    hyperparameters: dict,
    prompt: str,
    sources: list,
    terminal_output: str,
    template_output_file: str = None
) -> str:
    """
    Log an experiment to logs/{experiment_id}.json
    
    Args:
        model: LLM model name (e.g., "llama3.2")
        temperature: Temperature parameter used
        hyperparameters: Dict of other hyperparameters
        prompt: The prompt sent to the model
        sources: List of source URLs or references
        terminal_output: The terminal/console output
        template_output_file: Optional filename of the template JSON output
    
    Returns:
        experiment_id for naming convention match
    """
    # Generate unique experiment ID: exp_YYYYMMDD_HHMMSS_xxxxxxxx
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    exp_id = f"exp_{timestamp}_{unique_id}"
    
    # Create logs directory in project root (same as va4_product_discoverer)
    # This ensures consistency across all modules
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Build log entry
    log_entry = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "hyperparameters": hyperparameters,
        "prompt": prompt,
        "sources": sources,
        "terminal_output": terminal_output,
        "template_output_file": template_output_file,
    }
    
    # Save to JSON
    log_file = os.path.join(log_dir, f"{exp_id}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)
    
    return exp_id
