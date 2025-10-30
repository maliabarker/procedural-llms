import json
from typing import Any, Dict
from src.helpers import _names, _descriptions
from functions.prompts import create_execution_prompt
from functions.query import query

def run_steps_stateful_minimal(proc: Dict[str, Any], problem_text: str, answer_schema: Dict[str, Any], model: str, *, print_bool: bool = False):
    state: Dict[str, Any] = {"problem_text": problem_text}

    for step in proc["steps"]:
        need = _names(step["inputs"])

        # Build the *visible* inputs for this step from global state (no extras!)
        visible_inputs: Dict[str, Any] = {}
        for name in need:
            if name == "problem_text":
                visible_inputs[name] = problem_text
            elif name in state:
                visible_inputs[name] = state[name]
            else:
                raise RuntimeError(
                    f"Unresolvable input '{name}' for step id={step['id']}. "
                    "No prior producer in state."
                )

        is_last = (step["id"] == len(proc["steps"]))
        # Build the output schema
        if is_last:
            schema = answer_schema
            expected_outputs = list(answer_schema["properties"].keys())
            output_desc = {k: answer_schema["properties"][k].get("description", "")
                           for k in expected_outputs}
        else:
            expected_outputs = _names(step["output"])
            output_desc = _descriptions(step["output"])
            schema = create_output_schema(step)

        action = step["stepDescription"]

        step_prompt = create_execution_prompt(
            visible_inputs, action, schema,
            expected_outputs, output_desc, is_final=is_last
        )

        raw = query(step_prompt, model, schema)
        out = json.loads(raw) if isinstance(raw, str) else raw

        # Update global state: only declared outputs
        for name in expected_outputs:
            if name in out:
                state[name] = out[name]
            # If an output is missing, you can choose to raise or backfill/pass-through.
            # Here we raise for strictness:
            else:
                raise RuntimeError(
                    f"Model omitted required output '{name}' for step id={step['id']}"
                )

        if print_bool:
            print(f"Step {step['id']} visible inputs: {visible_inputs}")
            print(f"Step {step['id']} outputs: { {k: state[k] for k in _names(step['output'])} }")

    # Expect final step produced 'final_answer' inside state; your caller can return it
    return state

def create_output_schema(step):
    # Used to create a format for the LLM answer (passed into format option of LLM call) 
    # with desired outputs from procedure step
    required_keys = _names(step["output"])
    valid_types = {
        "oneOf": [
            {"type": "number"},
            {"type": "string"},
            {"type": "boolean"}
        ]
    }
    schema = {
        "type": "object",
        "properties": {name: valid_types for name in required_keys},  # allow any type
        "required": required_keys,
        "additionalProperties": False
    }
    return schema