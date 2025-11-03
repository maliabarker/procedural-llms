# projects/procedures/src/procedures/runners.py
"""Step executors for global-state procedures.

This module executes a Procedure (as JSON/dict) step-by-step by:
  1) Assembling the *visible inputs* for each step from the global state,
  2) Building an execution prompt with :func:`procedures.prompts.create_execution_prompt`,
  3) Calling a provided `query_fn` (LLM backend) with a JSON Schema for the step's outputs,
  4) Parsing and merging returned outputs back into the global state.

Design
------
- Backend-agnostic: pass any `query_fn(prompt, model, fmt, seed) -> str`.
- Strict by default: raises if a required output is missing.
- Final step uses your `answer_schema` (e.g., GSM/ARC) instead of a generic schema.

Typical usage
-------------
>>> from procedures.runners import run_steps_stateful_minimal
>>> from procedures.query_backends.ollama import query
>>> state = run_steps_stateful_minimal(proc, "2+3=?", gsm_schema, "gemma3:latest", query_fn=query)
>>> state["final_answer"]
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Callable

from procedures.prompts import create_execution_prompt

JSON = Dict[str, Any]
Fields = List[Mapping[str, Any]]
QueryFn = Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str]


# ----------------------- small field helpers (no src.* deps) -----------------------

def _field_names(fields: Fields) -> List[str]:
    """Return the list of ``name`` values from a list of input/output field dicts."""
    return [str(f.get("name")) for f in (fields or []) if "name" in f]

def _field_descriptions(fields: Fields) -> Dict[str, str]:
    """Return a mapping of field name → description (missing desc → '')."""
    return {str(f.get("name")): str(f.get("description", "")) for f in (fields or []) if "name" in f}


# ------------------------------- schema helper ------------------------------------

def create_output_schema(step: Mapping[str, Any]) -> JSON:
    """Create a permissive JSON Schema for this step's declared outputs.

    The schema enforces *keys and presence* (``required``), but allows the value
    type to be number/string/boolean. Use a stricter schema if your step needs it.

    Parameters
    ----------
    step
        A step object with an ``"output"`` list of field dicts
        (each dict must include ``{"name": ..., "description": ...}``).

    Returns
    -------
    dict
        JSON Schema enforcing required output keys and simple scalar types.
    """
    required_keys = _field_names(step.get("output", []))
    valid_types = {"oneOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}]}
    return {
        "type": "object",
        "properties": {name: valid_types for name in required_keys},
        "required": required_keys,
        "additionalProperties": False,
    }


# ----------------------------------- runner ---------------------------------------

def run_steps_stateful_minimal(
    proc: Mapping[str, Any],
    problem_text: str,
    answer_schema: JSON,
    model: str,
    *,
    query_fn: Optional[QueryFn] = None,
    seed: Optional[int] = 1234,
    print_bool: bool = False,
    strict_missing: bool = True,
) -> JSON:
    """Execute a global-state procedure and return the final state.

    For each step, this:
      • Collects only the needed inputs from the current global state,
      • Builds an execution prompt and a strict JSON Schema for outputs,
      • Calls `query_fn` to obtain a JSON object,
      • Merges declared outputs back into the global state.

    The final step uses ``answer_schema`` (e.g., GSM/ARC) so you can grade results.

    Parameters
    ----------
    proc
        Procedure JSON with keys: ``"steps"`` (list), and step fields including
        ``"id"``, ``"inputs"`` (list of fields), ``"stepDescription"``, and ``"output"``.
    problem_text
        The original task text; becomes ``state["problem_text"]`` and must be the only
        input of Step 1 per global-state rules.
    answer_schema
        JSON Schema dict for the final step's output object.
    model
        Backend model name passed to `query_fn`.
    query_fn
        Callable with signature ``(prompt, model, fmt, seed) -> str``. If omitted,
        we lazily import Ollama's default `query` (requires procedures[llm]).
    seed
        Optional random seed for the backend (if supported).
    print_bool
        If True, prints visible inputs and outputs for each step (debugging).
    strict_missing
        If True, raises when the model omits a required output key; if False, leaves it unset.

    Returns
    -------
    dict
        The final global state containing all produced variables (including ``final_answer``).

    Raises
    ------
    RuntimeError
        If a step input cannot be resolved from the global state, or a required output
        is missing and ``strict_missing=True``.
    ValueError
        If the backend response is not valid JSON.
    """
    # Lazy default backend to avoid hard dependency here.
    if query_fn is None:
        try:
            from procedures.query_backends.ollama import query as _default_query  # type: ignore
        except Exception as e:
            raise ImportError(
                "No `query_fn` provided and Ollama backend is unavailable. "
                "Install the LLM extra (`pip install -e projects/procedures[llm]`) "
                "or pass a custom `query_fn`."
            ) from e
        query_fn = _default_query

    state: JSON = {"problem_text": problem_text}
    steps: Iterable[Mapping[str, Any]] = proc.get("steps", [])

    for step in steps:
        need = _field_names(step.get("inputs", []))

        # Assemble *only* the inputs the step declared
        visible_inputs: JSON = {}
        for name in need:
            if name == "problem_text":
                visible_inputs[name] = problem_text
            elif name in state:
                visible_inputs[name] = state[name]
            else:
                raise RuntimeError(
                    f"Unresolvable input '{name}' for step id={step.get('id')}: "
                    "no prior producer in state."
                )

        is_last = (step.get("id") == len(proc.get("steps", [])))

        if is_last:
            schema = answer_schema
            expected = list(answer_schema.get("properties", {}).keys())
            output_desc = {k: answer_schema["properties"][k].get("description", "") for k in expected}
        else:
            expected = _field_names(step.get("output", []))
            output_desc = _field_descriptions(step.get("output", []))
            schema = create_output_schema(step)

        action = step.get("stepDescription") or step.get("step_description") or ""
        step_prompt = create_execution_prompt(
            visible_inputs, action, schema, expected, output_desc, is_final=is_last
        )

        # Backend call
        raw = query_fn(step_prompt, model, schema, seed)
        try:
            out = json.loads(raw) if isinstance(raw, str) else raw  # type: ignore[assignment]
        except Exception as e:
            raise ValueError(f"Non-JSON response for step id={step.get('id')}: {e}") from e

        # Merge declared outputs only
        for name in expected:
            if name in out:
                state[name] = out[name]
            elif strict_missing:
                raise RuntimeError(
                    f"Model omitted required output '{name}' for step id={step.get('id')}"
                )

        if print_bool:
            print(f"[step {step.get('id')}] inputs: {visible_inputs}")
            print(f"[step {step.get('id')}] outputs: {{k: state[k] for k in {expected} if k in state}}")

    return state
