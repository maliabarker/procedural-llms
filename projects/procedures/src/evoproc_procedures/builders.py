"""High-level builders for creating + validating procedures (backend-agnostic)."""

from __future__ import annotations
from typing import Any, Dict, Callable, Optional
import json

from evoproc_procedures.models import Procedure
from evoproc_procedures.prompts import create_procedure_prompt
from evoproc_procedures.repairs import repair_procedure_structured

JSON = Dict[str, Any]
QueryFn = Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str]


def create_and_validate_procedure_structured(
    idx: int,
    task: str,
    *,
    model: str,
    query_fn: QueryFn,
    seed: Optional[int] = 1234,
    print_diagnostics: bool = False,
) -> JSON:
    """Create a procedure for `task` and repair it until valid.

    Steps:
      1) Build a creation prompt (`create_procedure_prompt(task)`).
      2) Ask for a structured Procedure object using `query_fn` with the Pydantic schema.
      3) Parse the JSON and run `repair_procedure_structured(...)`.

    Args:
        idx: Index/id for logging.
        task: Natural-language task description.
        model: LLM model name.
        query_fn: Backend function to call the LLM.
        seed: Random seed for the backend (if supported).
        print_diagnostics: If True, print repair diagnostics.

    Returns:
        A structurally valid procedure JSON.

    Raises:
        ValueError: If the model response is not valid JSON.
    """
    # 1) Build prompt
    prompt = create_procedure_prompt(task)

    # 2) Structured call
    schema = Procedure.model_json_schema()
    raw = query_fn(prompt, model, schema, seed)
    try:
        proc = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        raise ValueError(f"[{idx}] Non-JSON response from model: {e}") from e

    # 3) Repair until valid
    proc = repair_procedure_structured(
        proc, model=model, query_fn=query_fn, print_diagnostics=print_diagnostics
    )
    return proc
