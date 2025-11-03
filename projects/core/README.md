# core

A lightweight **genetic algorithm (GA)** scaffold for evolving **LLM-generated, structured procedures**.  
Bring your own:
- **procedure schema** (e.g., Pydantic model → `.model_json_schema()`),
- **validator/repair loop** (to enforce structure & global-state rules),
- **query backend** (Ollama/OpenAI/httpx—anything that returns JSON).

This core package stays backend-agnostic and domain-neutral.  
See the companion package **`procedures`** for concrete models, answer schemas, prompt builders, a stateful runner, and an Ollama adapter.

---

## ✨ Features

- **GA loop**: initialize → evaluate → select → reproduce (crossover/mutation) → next generation.
- **LLM-driven operators**: crossover & mutation prompts emit **one** JSON object validating your schema.
- **Structural hygiene**: validator-driven scoring out of the box.
- **Task-eval scoring**: plug in a runner & evaluator for dataset-grade fitness.
- **Diversity**: random immigrants to escape local optima.
- **Determinism**: propagate seeds into your backend for reproducible runs.

---

## 📦 Install

Mono-repo layout:

```bash
pip install -e projects/core
```
Single-repo layout:
```bash
pip install -e .
```
Python: 3.10+

This package has no heavy runtime deps; it expects you to provide a query_fn and (optionally) run_steps_fn.

## 🧱 Package layout
```bash
llm_procedure_generation_ga/
└── src/llm_procedure_generation_ga/
    ├── __init__.py
    ├── ga_scaffold_structured.py   # GAConfig, Individual, ProcedureGA, operators
    ├── validators.py               # global-state validator suite + diagnostics
    ├── scorers.py                  # StructuralHygieneScorer, TaskEvalScorer, adapters
    └── helpers.py                  # small utilities (naming, canonicalization, etc.)
```

## 🚀 Quickstart (minimal)

You provide:
- a schema function (e.g., Procedure.model_json_schema()),
- a query_fn(prompt, model, fmt, seed) -> str` that returns JSON,
- a validator and repair_fn.

```python
from llm_procedure_generation_ga.ga_scaffold_structured import ProcedureGA, GAConfig
from llm_procedure_generation_ga.validators import validate_procedure_structured

# --- Your app code (examples) ---
def schema_json_fn():
    # Return your Procedure JSON Schema (dict)
    from procedures.models import Procedure
    return Procedure.model_json_schema()

def create_procedure_prompt(question: str) -> str:
    # Prompt the model to return ONE JSON object validating your schema
    from procedures.prompts import create_procedure_prompt as _p
    return _p(question)

def query(prompt: str, model: str, fmt=None, seed: int | None = None) -> str:
    # Any backend; must return a JSON string/object that matches `fmt`
    from procedures.query_backends.ollama import query as _q
    return _q(prompt, model, fmt, seed=seed)

def repair_fn(proc_json: dict, model: str) -> dict:
    # Optional LLM-mediated repair loop; can be identity if your generation is strict
    from procedures.builders import query_repair_structured
    return query_repair_structured(proc_json, model)

# --- GA setup ---
ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=schema_json_fn,
    validate_fn=validate_procedure_structured,
    repair_fn=repair_fn,
    cfg=GAConfig(population_size=6, max_generations=3, seed=42),
)

best, history = ga.run(
    task_description="Natalia sold clips to 48 friends in April, then half as many in May. How many altogether?",
    # If you supply these three, GA switches to task-eval scoring:
    final_answer_schema=None,   # e.g., procedures.schemas.get_schema("gsm")
    eval_fn=None,               # (state, proc) -> float
    run_steps_fn=None,          # executes a proc & returns `state`
    print_progress=True,
)

print("best fitness:", best.fitness)
print("best proc steps:", len(best.proc.get("steps", [])))
```
Want end-to-end execution and dataset scoring (e.g., GSM8K)?
Use the plugin pieces from `procedures`: answer schemas, runner, and an `eval_fn`.

## 🧠 Global-state design (enforced by validators)
- Step 1 must take exactly `["problem_text"]`.
- Later steps may read any variable produced by earlier steps (**global state**).
- The **final step** must output **exactly** `["final_answer"]` (a description of the solution, not necessarily a numeric result).

See `validators.py` for:
- `validate_first_step_inputs`
- `validate_final_step_output`
- `validate_inputs_resolvable_from_prior`
- `validate_no_redefine_existing_vars`
- `validate_unused_outputs`
- `validate_procedure_structured` (master composition)

Diagnostics are structured (`severity`, `action`, `message`, `details`) so you can drive **auto-repair** prompts.

## 🧪 Scoring modes
1) Structural hygiene (default)

Uses the validator suite to assign a scalar fitness (adapter provided in scorers.py).
Great for bootstrapping while your executors are still WIP.

2) Task-eval scoring

Provide all three to ga.run(...):
- `final_answer_schema` (JSON Schema for the runner’s final step),
- `run_steps_fn(proc, question, final_answer_schema, model, print_bool=False) -> state`,
- `eval_fn(state, proc) -> float` (e.g., EM / numeric match / BLEU).

A ready-to-use `TaskEvalScorer` is included.

## ⚙️ Configuration cheatsheet
```python
GAConfig(
    population_size=6,          # raise for more exploration
    elitism=2,                  # keep best survivors
    crossover_rate=0.7,         # crossover dominates
    mutation_rate=0.3,          # single small edits
    max_generations=3,          # expand once things work
    tournament_k=3,
    random_immigrant_rate=0.10, # 5–20% adds healthy diversity
    seed=42,                    # reproducibility
)
```

Tips
- Increase generations before population.
- Keep elitism ≥ 1 to preserve gains.
- Immigrants help when fitness plateaus.

## 🧩 Extensibility points
- Query backend: any function with signature query(prompt, model, fmt, seed).
- Schema: Pydantic model → .model_json_schema() or your own JSON Schema.
- Validator/repair loop: plug your domain diagnostics and repair strategy.
- Scorer: implement the Scorer protocol (score(ind, **kwargs) -> float).
- Operators: subclass/replace CrossoverOperator / MutationOperator.

## 📚 Documentation

Sphinx + MyST:
- API docs from docstrings (sphinx.ext.autodoc, autosummary).
- User guide: global-state rules, GA tips, examples.
- Example notebooks via myst_nb (configure nb_execution_mode to off or cache).

Minimal docs/conf.py hints:
```python
# Add src/ to path (mono-repo friendly)
from pathlib import Path, sys
CONF_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONF_DIR.parent
for p in (REPO_ROOT/"projects/core/src", REPO_ROOT/"src"):
    if p.exists(): sys.path.insert(0, str(p))

extensions = ["sphinx.ext.autodoc","sphinx.ext.autosummary","myst_parser","myst_nb","sphinx_design"]
autosummary_generate = True
nb_execution_mode = "off"  # or "cache"
```

## 🧰 Troubleshooting
- ModuleNotFoundError during docs build
Ensure the correct src/ paths are on sys.path, or install the package in the docs environment.
- LLM outputs prose instead of JSON
Lower temperature; pass the schema via format= if your backend supports it; include the schema verbatim in prompts.
- “Missing required output …” at runtime
Keep the runner strict to catch it early; optionally retry once with same prompt/seed; refine step outputs.
- Fitness plateau
Increase mutation rate slightly; add random immigrants; switch to task-eval scoring for a more informative gradient.

## 🧑‍💻 Development
```bash
# lint / type-check (if you add configs)
ruff check src
mypy src

# tests (example)
pytest -q
```

Editable install for development:

```bash
pip install -e projects/core
```

## 📝 License

MIT © Malia Barker

## 🔗 Citation

If you use this package in academic work, please cite the repository and (when available) the accompanying paper