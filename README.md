# EvoProc — Evolving Structured Procedures for Reliable LLM Reasoning

EvoProc is a two-package, mono-repo project for **evolving** LLM-generated, structured **procedures** with a lightweight genetic algorithm (GA). It separates the **core GA scaffold** from a **domain plugin** (models, schemas, prompt builders, runners, and backends) so you can reuse the engine across tasks and keep domain specifics modular.

- **Core (GA engine):** [`evoproc` on PyPI](https://pypi.org/project/evoproc/0.1.0/)
- **Procedures plugin:** [`evoproc-procedures` on PyPI](https://pypi.org/project/evoproc-procedures/0.1.0/)

---

## Table of Contents
- [Why EvoProc?](#why-evoproc)
- [Packages](#packages)
- [Install](#install)
- [Quickstart](#quickstart)
- [Repo Structure](#repo-structure)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Why EvoProc?

Large Language Models are great at generating **plans**, but those plans can be messy, inconsistent, or brittle. EvoProc treats a plan as a **structured procedure** (JSON-like steps with inputs/outputs), then **evolves** a population of procedures via crossover, mutation, validation, and scoring—so you end up with cleaner, more reliable procedures that are easier to execute and evaluate.

Key ideas:
- **Global-state procedures:** steps can read any prior variable; step 1 takes only `problem_text`; the final step emits `final_answer`.
- **Validator-driven structure:** tight structural checks + optional **auto-repair** prompts.
- **Pluggable scoring:** structural hygiene (default) or **task-eval** scorers for benchmarks.
- **Backends:** designed to work with local LLMs (e.g., Ollama) or your own query layer.

---

## Packages

### 1) Core GA — `evoproc`
The reusable engine: GA loop, operators (crossover/mutation), scorer adapters, utilities, and validators.

- PyPI: https://pypi.org/project/evoproc/0.1.0/
- Import namespace: `evoproc`

### 2) Procedures Plugin — `evoproc-procedures`
Your domain layer: Pydantic models for procedures, answer schemas, prompt builders, step runners, and optional backends (e.g., Ollama).

- PyPI: https://pypi.org/project/evoproc-procedures/0.1.0/
- Import namespace: `evoproc_procedures`

---

## Install

From PyPI (recommended):

```bash
pip install evoproc==0.1.0
pip install evoproc-procedures==0.1.0
```

Optional extras (example):
```bash
pip install "evoproc-procedures[llm]"
```

Editable/dev install (from the mono-repo root):
```bash
# Core
pip install -e projects/core

# Plugin
pip install -e projects/procedures
```

---

## Quickstart

```python
from evoproc.ga_scaffold_structured import ProcedureGA, GAConfig
from evoproc.validators import validate_procedure_structured
from evoproc.scorers import ProcScorerAdapter, StructuralHygieneScorer

from evoproc_procedures.models import Procedure
from evoproc_procedures.schemas import get_schema
from evoproc_procedures.prompts import create_procedure_prompt
from evoproc_procedures.runners import run_steps_stateful_minimal
from evoproc_procedures.query_backends.ollama import query, hard_query, query_repair_structured

# Configure GA with structural scorer (default)
ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=lambda: Procedure.model_json_schema(),
    validate_fn=validate_procedure_structured,
    repair_fn=query_repair_structured,
    scorer=ProcScorerAdapter(StructuralHygieneScorer(validate_fn=validate_procedure_structured)),
    cfg=GAConfig(population_size=6, max_generations=3, seed=42),
)

question = "Natalia sold clips to 48 friends in April, then half as many in May. How many altogether?"
best, _ = ga.run(
    task_description=question,
    # For task-eval scoring, pass all three:
    # final_answer_schema=get_schema("gsm"),
    # eval_fn=...,
    # run_steps_fn=...,
    print_progress=True,
)

# Execute the evolved procedure end-to-end (example)
final_state = run_steps_stateful_minimal(
    best.proc,
    problem_text=question,
    answer_schema=get_schema("gsm"),
    model=ga.model,
)
print("answer:", final_state.get("final_answer"), "num:", final_state.get("answer_numerical"))
```

---

## Repo Structure

```
.
├── projects/
│   ├── core/
│   │   ├── pyproject.toml            # name = "evoproc"
│   │   └── src/evoproc/              # GA engine, validators, scorers, helpers
│   └── procedures/
│       ├── pyproject.toml            # name = "evoproc-procedures"
│       └── src/evoproc_procedures/   # models, schemas, prompts, runners, backends, examples
├── docs/                             # Sphinx site (API docs + guide + examples)
│   ├── conf.py
│   ├── index.rst
│   ├── notebook_demo.ipynb       # example included in docs
│   └── _build/
├── .readthedocs.yaml                 # RTD build config (installs both packages)
└── README.md                         # this file
```

---

## Documentation

This repo ships a Sphinx documentation site (API + User Guide + examples).  
Build locally:

```bash
pip install -r docs/requirements.txt
cd docs
make clean && make html
# open _build/html/index.html
```

Read the Docs can be enabled via `.readthedocs.yaml` (already provided), which installs both packages before building docs.

---

## Development

Lint & test:
```bash
pip install -e projects/core -e projects/procedures
pip install -r docs/requirements.txt
pip install pytest ruff mypy

ruff check .
mypy projects/core/src/evoproc
mypy projects/procedures/src/evoproc_procedures

pytest -q
```

Build distributions:
```bash
python -m pip install --upgrade build twine

# Core
cd projects/core && rm -rf dist build *.egg-info && python -m build

# Plugin
cd ../procedures && rm -rf dist build *.egg-info && python -m build
```

Release flow (summary):
1. Upload to **TestPyPI** first (`twine upload --repository testpypi dist/*`).
2. Smoke-test installs from TestPyPI.
3. Upload to **PyPI** (`twine upload dist/*`), **core first**, then plugin.

---

## Contributing

Issues and PRs welcome! If you’re adding a new domain, please keep the GA internals in **`evoproc`** and put domain-specific assets in **`evoproc_procedures`** (or your own companion package), with clear boundaries:
- **Core:** GA loop, operators, validators, scorer adapters, utils.
- **Plugin:** Pydantic models, answer schemas, prompt builders, step runners, repairs, query backends.

---

## License

MIT © 2025 Malia Barker

---

## Citation

If you use EvoProc in academic work, please cite the project:

```bibtex
@software{EvoProc2025,
  author  = {Barker, Malia},
  title   = {EvoProc: Evolving Structured Procedures for Reliable LLM Reasoning},
  year    = {2025},
  url     = {https://pypi.org/project/evoproc/0.1.0/}
}
```
