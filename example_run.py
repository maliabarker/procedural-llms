from src.scorers import StructuralHygieneScorer, ProcScorerAdapter, TaskEvalScorer
from src.ga_scaffold_structured import ProcedureGA
from src.validators import validate_procedure_structured
from src.ga_scaffold_structured import GAConfig, ProcedureGA
from src.procedure_classes import Procedure
from functions.query import query, query_repair_structured
from functions.run_steps import run_steps
from functions.answer_schemas import GSM_answer_schema
from functions.prompts import create_procedure_prompt


# CREATE SCORER
# For top-level structural hygiene scoring
proc_scorer = StructuralHygieneScorer(validate_fn=validate_procedure_structured)
ga = ProcedureGA(
    ...,
    scorer=ProcScorerAdapter(proc_scorer),
)

# For task-eval scoring (if we wanna do step by step runs of procedures and additionally score for those)
# scorer = TaskEvalScorer(
#     run_steps_fn=run_steps,
#     eval_fn=gsm_eval_fn,
#     question=the_question,
#     final_answer_schema=GSM_answer_schema,
#     model=model_name,
# )
# ga = ProcedureGA(..., scorer=scorer)

# CREATE GA
ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=lambda: Procedure.model_json_schema(),
    validate_fn=validate_procedure_structured,
    repair_fn=query_repair_structured,
    scorer=proc_scorer,
    cfg=GAConfig(population_size=8, max_generations=5, seed=42),
)

best, history = ga.run(
    task_description="Solve: Natalia sold clips to 48 friends in April...",
    final_answer_schema=GSM_answer_schema,    # or ARC_answer_schema, etc.
    eval_fn=None,  # or TaskEvalScorer requires eval_fn if you want accuracy-based
    print_progress=True,
)
