# Quickstart

```python
from evoproc.ga_scaffold_structured import ProcedureGA, GAConfig
from evoproc_procedures.models import Procedure
from evoproc_procedures.prompts import create_procedure_prompt
from evoproc_procedures.validators import validate_procedure_structured
from evoproc_procedures.runners import run_steps_stateful_minimal
from evoproc_procedures.schemas import get_schema
from evoproc_procedures.query_backends.ollama import query, repair_fn_ollama

ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=lambda: Procedure.model_json_schema(),
    validate_fn=validate_procedure_structured,
    repair_fn=repair_fn_ollama,
    cfg=GAConfig(population_size=6, max_generations=3, seed=42),
)

question = "Natalia sold clips to 48 friends in April, then half as many in May. How many altogether?"
best, _ = ga.run(
    task_description=question,
    final_answer_schema=get_schema("gsm"),
    eval_fn=lambda state, proc: 1.0,        # plug your scorer
    run_steps_fn=lambda proc, q, s, m, p=False: run_steps_stateful_minimal(proc, q, s, m, query_fn=query),
)
