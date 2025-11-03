# Global-State Procedures

Global-state procedures are lightweight, JSON-structured plans where **every step can read any variable produced by earlier steps** (plus the original `problem_text`). This keeps procedures simple to write and easy to execute deterministically.

## Why global state?

- **Fewer wiring bugs:** No long chains of pass-through arguments.
- **Readable steps:** Each step declares only the variables it actually needs.
- **Deterministic execution:** A single global `state` drives evaluation.

## Hard rules (validated)

These are enforced by {mod}`llm_procedure_generation_ga.validators`:

1. **Step 1 input** must be exactly `["problem_text"]`.  
   *(No other inputs exist yet.)*
2. The **final step** must output exactly `["final_answer"]`.  
   *(A description of how to compute it—no numeric work here.)*
3. **Resolvable inputs:** Every step input must either be `problem_text` or a variable produced by an earlier step.
4. **No silent redefinitions:** Reusing the same variable name later is flagged (prefer `normalized_total`, `count_after_discount`, etc).
5. **No dead outputs:** Outputs never used by later steps are flagged for removal.

{note}
These constraints are diagnostic-driven: the validator returns structured findings you can feed to an auto-repair prompt.

## Minimal schema (Pydantic)

The canonical model is provided by the plugin’s `{mod}procedures.models`:

```python
class StepInputField(BaseModel):
    name: str
    description: str

class StepOutputField(BaseModel):
    name: str
    description: str

class Step(BaseModel):
    id: int
    inputs: List[StepInputField]
    stepDescription: str
    output: List[StepOutputField]

class Procedure(BaseModel):
    NameDescription: str
    steps: List[Step]
```

Call ```Procedure.model_json_schema()``` when prompting the LLM so it returns strictly-valid JSON.

## A good procedure (toy)

```json
{
  "NameDescription": "Solve small arithmetic word problems",
  "steps": [
    {
      "id": 1,
      "inputs": [{"name": "problem_text", "description": "original question"}],
      "stepDescription": "Extract primitive facts (numbers, units, relations) from the text.",
      "output": [{"name": "facts", "description": "structured facts"}]
    },
    {
      "id": 2,
      "inputs": [{"name": "facts", "description": "structured facts"}],
      "stepDescription": "Plan the arithmetic operations needed to obtain the result.",
      "output": [{"name": "plan", "description": "ordered operations"}]
    },
    {
      "id": 3,
      "inputs": [{"name": "plan", "description": "ordered operations"}],
      "stepDescription": "Describe the final answer (do not compute numeric value).",
      "output": [{"name": "final_answer", "description": "answer description"}]
    }
  ]
}
```

## A bad procedure (violations)

- Step 1 asks for facts (not allowed yet).
- Final step outputs result instead of final_answer.
- An unused foo output.

```json
{
  "NameDescription": "Bad example",
  "steps": [
    {
      "id": 1,
      "inputs": [{"name": "problem_text"}, {"name": "facts"}],
      "stepDescription": "Do everything at once.",
      "output": [{"name": "foo"}]
    },
    {
      "id": 2,
      "inputs": [{"name": "foo"}],
      "stepDescription": "Compute number.",
      "output": [{"name": "result"}]
    }
  ]
}
```

Running `validate_procedure_structured(...)` will produce diagnostics such as:

- REWRITE_FIRST_STEP (fatal): step 1 inputs wrong.
- ADD_FINAL_STEP (fatal): missing final_answer.
- PATCH_LOCALLY (repairable): remove unused foo.

## Execution model

Use `{mod}procedures.runners`:

1. Build visible inputs for the current step from the global state.
2. Prompt the LLM with a strict per-step JSON Schema (or your final-answer schema on the last step).
3. Merge only the declared outputs back into state.

```python
from procedures.runners import run_steps_stateful_minimal
from procedures.schemas import get_schema
from procedures.query_backends.ollama import query

state = run_steps_stateful_minimal(
    proc,
    problem_text=question,
    answer_schema=get_schema("gsm"),
    model="gemma3:latest",
    query_fn=query,
    print_bool=True,
)
print(state.get("final_answer"), state.get("answer_numerical"))
```

## Prompting contract (creation)

When you ask the LLM to create a procedure, include:

- `Procedure.model_json_schema()` verbatim in the prompt.
- The global-state rules (step 1 input, final step output).
- A “no numeric computation” reminder.

The helper `{func}procedures.prompts.create_procedure_prompt` already injects these.

## Common pitfalls & fixes

- “ModuleNotFoundError: procedures…”
    Ensure your docs environment installs both packages and that conf.py adds both src/ paths.
- Final step computes numbers
    Tighten your prompt: “Final step outputs a descriptive final_answer only—no numeric computation.”
- Missing outputs
    The runner raises if the model omits a required key—keep that strict for reliability.

