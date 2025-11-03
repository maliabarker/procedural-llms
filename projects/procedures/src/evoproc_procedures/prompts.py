"""Prompt builders for procedure creation and step execution.

These helpers produce **LLM prompts** used by the GA scaffold and your runners:

- :func:`create_direct_prompt` — a simple “ask the question directly” prompt
  (handy for baselines and ARC-style multiple choice).
- :func:`create_procedure_prompt` — asks the model to emit a *global-state*
  procedure JSON that validates your Pydantic schema.
- :func:`create_execution_prompt` — runs a single step by showing inputs,
  the step action, and the required outputs, while reminding the model to
  return **strict JSON** conforming to a supplied schema.

All functions are pure string builders and have no network side-effects.
They are safe to use in tests and with any LLM backend.

Example
-------
>>> from evoproc_procedures.prompts import create_procedure_prompt
>>> p = create_procedure_prompt("Natalia sold clips to 48 friends in April...")
>>> isinstance(p, str)
True
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

# Import from your plugin package, not from "src."
from evoproc_procedures.models import Procedure

__all__ = [
    "create_direct_prompt",
    "create_procedure_prompt",
    "create_execution_prompt",
]


def create_direct_prompt(item: str) -> str:
    """Build a direct Q→A prompt (baseline / ARC-style).

    Args:
        item: The task text (e.g., question stem or problem statement).

    Returns:
        A concise prompt string suitable for a direct answer call.
    """
    # Keep it minimal to reduce unintended reasoning scaffolds in baselines.
    return f"Solve this problem: {item}"


def create_procedure_prompt(item: str, example_prompt: Optional[str] = None) -> str:
    """Ask the LLM to synthesize a *global-state* procedure JSON.

    The model is instructed to return **exactly one** JSON object that validates
    the Pydantic schema produced by :class:`uhj_procedures.models.Procedure`.

    Args:
        item: Natural-language task to decompose.
        example_prompt: Optional extra hint or in-context example description
            (kept as a single string you can preformat upstream).

    Returns:
        A prompt string suitable for a structured generation call.

    Notes:
        - The schema is injected verbatim via ``Procedure.model_json_schema()``.
        - Constraints match your GA scaffold: step 1 uses ``problem_text`` only,
          later steps may read any earlier variables (global state), and the
          final step emits ``final_answer`` **as a description only**.
    """
    schema_json = json.dumps(Procedure.model_json_schema(), ensure_ascii=False)
    example = f"\n# Example (hint)\n{example_prompt}\n" if example_prompt else ""

    # No fenced code blocks; some providers echo fences literally.
    return (
        "Decompose the task into small, single-action steps to solve the problem.\n"
        f"# Task\n{item}\n"
        f"{example}"
        "# Output Contract\n"
        "Return exactly one JSON object that validates against this schema (verbatim):\n"
        f"{schema_json}\n"
        "## Global IO Constraints (must follow)\n"
        "- Global state: Steps may read any variable produced by earlier steps.\n"
        "- Step 1 inputs: exactly ['problem_text'].\n"
        "- Inputs resolvable: Every step input must come from problem_text or from some earlier "
        "step's outputs (by name).\n"
        "- Variable names: snake_case; consistent across steps.\n"
        "- Descriptions: concise and concrete.\n"
        "- No numeric results: do not compute or reveal numeric values or the final answer.\n"
        "- Final step: outputs exactly ['final_answer'] (description only).\n"
        "## Step rules\n"
        "- 'step_description' is a single, imperative action.\n"
        "## Validation Checklist (self-check before returning)\n"
        "- JSON validates against schema.\n"
        "- Each step has id, input(s), step_description, output(s).\n"
        "- Step 1 input is exactly problem_text.\n"
        "- All step inputs are available in the global state (problem_text or prior outputs).\n"
        "- Final step outputs exactly final_answer with a descriptive definition only.\n"
        "Return ONLY the JSON object. Do not include commentary."
    )


def create_execution_prompt(
    visible_inputs: Dict[str, Any],
    action: str,
    schema: Dict[str, Any],
    expected_outputs: Iterable[str],
    output_descriptions: Optional[Dict[str, str]] = None,
    *,
    is_final: bool = False,
) -> str:
    """Build a prompt to execute a **single step** with strict JSON output.

    Args:
        visible_inputs: The subset of global state the step may read (already extracted variables).
        action: The step's imperative instruction (e.g., \"extract the two numbers a and b\").
        schema: JSON Schema that the **output object** must validate against
            (for regular steps use your step-output schema; for the last step
            you can pass your answer schema, e.g. GSM/ARC).
        expected_outputs: Names of keys the model must return in the JSON object.
        output_descriptions: Optional mapping of key → human description (shown to the model).
        is_final: If ``True``, annotate that these outputs constitute the final answer.

    Returns:
        A prompt string that instructs the model to return only a JSON object
        matching ``schema`` and containing exactly ``expected_outputs``.
    """
    output_lines = []
    output_descriptions = output_descriptions or {}
    for name in expected_outputs:
        desc = output_descriptions.get(name, "")
        output_lines.append(f"- {name}: {desc}".rstrip())

    outputs_block = "\n".join(output_lines) if output_lines else "(see schema)"
    inputs_block = json.dumps(visible_inputs, ensure_ascii=False, indent=2)
    schema_block = json.dumps(schema, ensure_ascii=False, indent=2)

    final_note = " (final_answer)" if is_final else ""
    return (
        f"{action}\n"
        f"# Inputs (JSON)\n{inputs_block}\n"
        "# Required Outputs\n"
        f"Return a JSON object with exactly these keys{final_note}:\n"
        f"{outputs_block}\n\n"
        "# Format\n"
        "- Return ONLY a JSON object that conforms to the provided schema.\n"
        "- Do NOT include extra keys.\n"
        "- Do NOT include commentary.\n\n"
        "# Schema (verbatim)\n"
        f"{schema_block}"
    )
