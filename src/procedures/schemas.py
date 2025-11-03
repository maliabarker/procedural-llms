"""
Answer schemas for final LLM outputs.

These JSON Schemas constrain the structure of the *final* answer object the
LLM should return. They are suitable both for:
    1) passing as `fmt`/`response_format` to your LLM call, and
    2) local validation via `jsonschema`.

Guidelines
----------
    - Keep answers compact and *deterministic* for downstream comparison.
    - Prefer including a bounded `confidence` when available (0..1).
    - For numeric tasks, return both a human `final_answer` string (traceable) and a
    machine `final_answer_numerical` for exact comparison.
"""

from __future__ import annotations

from typing import Dict, Any, Final, Iterable, Optional
from jsonschema import Draft202012Validator, ValidationError

JSON = Dict[str, Any]

# Common header for all schemas (explicit draft + helpful metadata)
_BASE: Final[JSON] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
}

__all__ = [
    "general_answer_schema",
    "bool_answer_schema",
    "GSM_answer_schema",
    "ARC_answer_schema",
    "get_schema",
    "list_schemas",
    "validate_answer",
]


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

#: Minimal answer container; useful for quick smoke tests or free-form tasks.
#: 
#: Structure:
#:   - answer (str): Human-readable final answer text.
general_answer_schema: Final[JSON] = {
    **_BASE,
    "title": "GeneralAnswer",
    "description": "Minimal answer envelope with only a string answer.",
    "type": "object",
    "properties": {
        "final_answer": {"type": "string", "description": "Final answer text."}
    },
    "required": ["final_answer"],
    "additionalProperties": False,
}

#: Boolean answer with both a natural-language answer and a strict bool.
#:
#: Structure:
#:   - answer (str): Human-readable justification or 'yes'/'no' text.
#:   - answer_bool (bool): Machine-usable boolean.
#:
#: Tip:
#:   If you want the string and bool to agree, enforce it in code or add a
#:   post-check (schema can’t easily express cross-field string-to-bool logic).
bool_answer_schema: Final[JSON] = {
    **_BASE,
    "title": "BooleanAnswer",
    "description": "Answer with both a textual rationale and a strict boolean.",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "string",
            "description": "Final answer text (e.g., 'yes'/'no' with rationale).",
            "maxLength": 1000,
        },
        "final_answer_bool": {
            "type": "boolean",
            "description": "Machine-usable boolean answer.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Optional model confidence for the boolean (0..1).",
        },
    },
    "required": ["final_answer", "final_answer_bool"],
    "additionalProperties": False,
}

#: GSM-style numeric answer:
#:
#: Structure:
#:   - answer (str): Full reasoning / final statement in natural language.
#:   - answer_numerical (number): Final numeric value for exact comparison.
#:   - confidence (number, optional): 0..1 calibrated confidence.
#:
#: Notes:
#:   - Keep numbers finite; avoid 'NaN'/'Infinity' (validator checks in code).
#:   - If units matter, add an optional 'units' enum (e.g., 'USD', 'km', ...).
GSM_answer_schema: Final[JSON] = {
    **_BASE,
    "title": "GSMNumericAnswer",
    "description": "Natural-language answer plus a strict numeric extraction.",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "string",
            "description": "Final answer statement (end with the result).",
            "maxLength": 1000,
        },
        "final_answer_numerical": {
            "type": "number",
            "description": "Final numeric value (finite).",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Optional model confidence (0..1).",
        },
        "units": {
            "type": "string",
            "description": "Optional units for the numeric value (e.g., 'USD', 'km').",
        },
    },
    "required": ["final_answer", "final_answer_numerical"],
    "additionalProperties": False,
}

#: ARC-style multiple choice:
#:
#: Structure:
#:   - answer (str enum): One of 'A', 'B', 'C', 'D'.
#:   - confidence (number, optional): 0..1 confidence.
#:   - choice_rationale (string, optional): Brief justification for the choice.
ARC_answer_schema: Final[JSON] = {
    **_BASE,
    "title": "ARCChoiceAnswer",
    "description": "Multiple-choice answer constrained to a single letter.",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
            "description": "Chosen option label.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Optional model confidence (0..1).",
        },
        "choice_rationale": {
            "type": "string",
            "description": "Optional brief explanation of the choice.",
            "maxLength": 1000,
        },
    },
    "required": ["final_answer"],
    "additionalProperties": False,
}


# -----------------------------------------------------------------------------
# Helpers (optional but handy in pipelines & tests)
# -----------------------------------------------------------------------------

_SCHEMAS: Final[dict[str, JSON]] = {
    "general": general_answer_schema,
    "bool": bool_answer_schema,
    "gsm": GSM_answer_schema,
    "arc": ARC_answer_schema,
}

def list_schemas() -> Iterable[str]:
    """Return the available schema keys (``general``, ``bool``, ``gsm``, ``arc``)."""
    return _SCHEMAS.keys()

def get_schema(name: str) -> JSON:
    """Get a schema by short name.

    Args:
        name: One of ``general``, ``bool``, ``gsm``, ``arc``.

    Raises:
        KeyError: If the name is unknown.
    """
    return _SCHEMAS[name]

def validate_answer(payload: JSON, schema: JSON, *, check_finite_number: bool = True) -> Optional[str]:
    """Validate a payload against a JSON Schema; return an error message or ``None``.

    This is a convenience wrapper for quick tests. For richer error reporting,
    catch ``jsonschema.ValidationError`` yourself and format as needed.

    Args:
        payload: Parsed JSON object from model output.
        schema: The JSON Schema dict to validate against.
        check_finite_number: If True, reject non-finite numbers in numeric fields.

    Returns:
        None if valid, otherwise a short error message.
    """
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as e:
        return f"schema validation error: {e.message}"

    if check_finite_number and "final_answer_numerical" in payload:
        v = payload["final_answer_numerical"]
        # Reject NaN/Infinity which are not valid JSON numbers but can appear in some parses.
        if not isinstance(v, (int, float)):
            return "final_answer_numerical must be a number"
        if v != v or v in (float("inf"), float("-inf")):
            return "final_answer_numerical must be finite"

    return None
