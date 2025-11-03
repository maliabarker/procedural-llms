from __future__ import annotations
from typing import Any, Dict, Callable, Optional, Tuple
import json

from evoproc_procedures.models import Procedure
from evoproc_procedures.prompts import create_procedure_prompt
from evoproc_procedures.repairs import repair_procedure_structured

JSON = Dict[str, Any]
QueryFn = Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str]
RunStepsFn = Callable[..., Dict[str, Any]]  # see notes in docstring

def run_full_procedure_structured(
    idx: int,
    question: str,
    *,
    model: str,
    query_fn: QueryFn,
    run_steps_fn: RunStepsFn,
    answer_schema: Dict[str, Any],
    seed: Optional[int] = 1234,
    print_diagnostics: bool = False,
) -> Tuple[JSON, Dict[str, Any]]:
    """Create → repair → execute a structured procedure for one task.

    Steps
    -----
    1) Build a creation prompt (global-state constraints) for `question`.
    2) Ask the LLM (via `query_fn`) to emit a **Procedure** JSON that validates
       `Procedure.model_json_schema()`.
    3) Iteratively repair with `repair_procedure_structured(...)` until it validates.
    4) Execute the procedure with `run_steps_fn` to obtain the final answer/state.

    Args
    ----
    idx:
        An index for logging (e.g., dataset row).
    question:
        Natural-language task text (fed as `problem_text`).
    model:
        Backend model name (passed through to `query_fn`).
    query_fn:
        Callable `(prompt, model, fmt, seed) -> str`. Provide your backend.
    run_steps_fn:
        Your executor. Two common signatures:
          (a) `run_steps_fn(proc, question, answer_schema, model, print_bool=False) -> state`
          (b) `run_steps_fn(proc, *, inputs: dict, answer_schema: dict, model: str) -> state`
        This wrapper will try (a) first, then (b).
    answer_schema:
        JSON Schema for the final answer object (e.g., GSM/ARC).
    seed:
        Optional seed for determinism (if backend supports it).
    print_diagnostics:
        If True, prints repair diagnostics each iteration.

    Returns
    -------
    (procedure_json, state_dict)
        Validated procedure JSON and the execution state (should include the final answer).

    Raises
    ------
    ValueError
        If the model response cannot be parsed as JSON.
    RuntimeError
        If the procedure cannot be repaired to pass validation.
    """
    # 1) Create prompt + schema
    prompt = create_procedure_prompt(question)
    schema = Procedure.model_json_schema()

    # 2) Structured generation
    raw = query_fn(prompt, model, schema, seed)
    try:
        proc: JSON = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        raise ValueError(f"[{idx}] Non-JSON response from model: {e}") from e

    # 3) Repair until valid
    proc = repair_procedure_structured(
        proc, model=model, query_fn=query_fn, max_tries=10, print_diagnostics=print_diagnostics
    )

    # 4) Execute (support either executor signature)
    try:
        state = run_steps_fn(proc, question, answer_schema, model, print_diagnostics)  # type: ignore[arg-type]
    except TypeError:
        # Fallback to keyword-style API
        state = run_steps_fn(
            proc, inputs={"problem_text": question}, answer_schema=answer_schema, model=model  # type: ignore[call-arg]
        )

    return proc, state
