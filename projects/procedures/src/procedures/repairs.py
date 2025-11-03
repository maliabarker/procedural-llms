"""LLM-assisted repair helpers (backend-agnostic).

These functions *use* your pure validators to diagnose issues, then call a
provided `query_fn` to ask an LLM to minimally fix the JSON so it validates.

They don't import a specific backend; you pass a `query_fn` with signature:
    query_fn(prompt: str, model: str, fmt: dict | None = None, seed: int | None = None) -> str
"""

from __future__ import annotations
from typing import Any, Dict, Callable, Optional
import json

from procedures.models import Procedure
from llm_procedure_generation_ga.validators import validate_procedure_structured

JSON = Dict[str, Any]
QueryFn = Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str]


def repair_procedure_structured(
    proc: JSON,
    *,
    model: str,
    query_fn: QueryFn,
    max_tries: int = 10,
    print_diagnostics: bool = False,
) -> JSON:
    """Iteratively repair a procedure JSON until it passes validation.

    Args:
        proc: Procedure JSON to repair.
        model: LLM model name.
        query_fn: Backend function to call the LLM (see signature above).
        max_tries: Maximum number of repair attempts.
        print_diagnostics: If True, prints diagnostics each loop.

    Returns:
        A structurally valid procedure JSON.

    Raises:
        RuntimeError: If the procedure could not be validated after `max_tries`.
    """
    schema_json = Procedure.model_json_schema()

    for _ in range(max_tries):
        diags = validate_procedure_structured(proc)
        if not diags:
            return proc

        if print_diagnostics:
            print("[repair] diagnostics:")
            for d in diags:
                print(" -", d)

        diag_str = "\n- ".join(str(i) for i in diags)
        repair_prompt = (
            "You will make the requested minimal structural fixes to the following "
            "procedure JSON so that it validates against the schema.\n\n"
            "# Schema (verbatim)\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}\n\n"
            "# Fix Instructions\n"
            f"- {diag_str}\n\n"
            "# Procedure JSON\n"
            f"{json.dumps(proc, ensure_ascii=False)}\n\n"
            "Return ONLY the corrected JSON object. No commentary."
        )

        raw = query_fn(repair_prompt, model, schema_json, 1234)
        try:
            proc = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            # keep looping; try again
            continue

    raise RuntimeError("Could not satisfy validator after retries.")
