# projects/procedures/src/procedures/ollama.py
"""
Ollama-backed query helpers for structured procedure generation and repair.

These functions implement the `QueryFn`/`RepairFn` style expected by the GA core:
    - `query(...)`: general-purpose text → JSON call, optionally constrained by a JSON Schema.
    - `hard_query(...)`: strict/low-temperature variant useful for repair passes.
    - `query_repair_structured(...)`: loop that validates a procedure JSON and issues
    minimal repair prompts until it passes (or exhausts retries).
    - `create_and_validate_procedure_structured(...)`: one-shot convenience that creates
    a procedure JSON for a task and validates/repairs it before returning.

All functions are pure client wrappers around `ollama.generate`.
They return **strings** (for raw model responses) or **dicts** (for JSON payloads).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import ollama  # pip install ollama

from procedures.builders import create_and_validate_procedure_structured
from procedures.repairs import repair_procedure_structured
from procedures.pipelines import run_full_procedure_structured as _run_full
from procedures.schemas import get_schema

__all__ = [
    "query", "hard_query",
    "repair_fn_ollama", "create_and_validate_procedure_ollama",
    "run_full_procedure_ollama",
]

def hard_query(
    prompt: str,
    model: str,
    fmt: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 1234,
) -> str:
    """Call Ollama with *strict* decoding settings.

    Use this when you need the model to adhere tightly to `fmt` (a JSON Schema),
    e.g., repair steps. Temperature is set to 0.

    Args:
        prompt: Full prompt text.
        model: Ollama model name (e.g., ``"gemma3:latest"``).
        fmt: Optional JSON Schema dict for structured output.
        seed: Optional random seed for reproducibility.

    Returns:
        The raw string response (usually JSON text when `fmt` is provided).
    """
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format=fmt,  # Ollama expects a schema dict here
        options={"temperature": 0, "seed": seed},
    )
    return res["response"]


def query(
    prompt: str,
    model: str,
    fmt: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 1234,
) -> str:
    """General-purpose Ollama call.

    This is your default structured call. When `fmt` is supplied, Ollama
    will attempt to return an object matching that schema.

    Args:
        prompt: Full prompt text.
        model: Ollama model name.
        fmt: Optional JSON Schema dict used as a response format.
        seed: Optional random seed for reproducibility.

    Returns:
        Raw string response (often JSON text when `fmt` is provided).
    """
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format=fmt,
        options={"temperature": 1, "seed": seed},
    )
    return res["response"]

def repair_fn_ollama(proc: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Adapter for GA's `repair_fn(proc, model)` using Ollama backend."""
    return repair_procedure_structured(proc, model=model, query_fn=hard_query, print_diagnostics=False)

def create_and_validate_procedure_ollama(idx: int, task: str, *, model: str, seed: int = 1234, print_diagnostics: bool = False) -> Dict[str, Any]:
    """Convenience creator that uses Ollama under the hood."""
    return create_and_validate_procedure_structured(idx, task, model=model, query_fn=query, seed=seed, print_diagnostics=print_diagnostics)

def run_full_procedure_ollama(
    idx: int,
    question: str,
    *,
    model: str = "gemma3:latest",
    answer_schema_name: str = "gsm",
    run_steps_fn=None,
    seed: Optional[int] = 1234,
    print_diagnostics: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Ollama-backed 'create → repair → execute' helper for one question.

    Args mirror the backend-agnostic version; this just wires in the Ollama `query`
    and fetches the answer schema by name.
    """
    from procedures.ollama import query  # local import to avoid cycles
    if run_steps_fn is None:
        raise ValueError("run_steps_fn must be provided (executor for your steps).")
    schema = get_schema(answer_schema_name)
    return _run_full(
        idx,
        question,
        model=model,
        query_fn=query,
        run_steps_fn=run_steps_fn,
        answer_schema=schema,
        seed=seed,
        print_diagnostics=print_diagnostics,
    )