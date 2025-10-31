from src.llm_procedure_generation_ga.scorers import StructuralHygieneScorer, ProcScorerAdapter, TaskEvalScorer
from src.llm_procedure_generation_ga.ga_scaffold_structured import ProcedureGA
from src.llm_procedure_generation_ga.validators import validate_procedure_structured
from src.llm_procedure_generation_ga.ga_scaffold_structured import GAConfig, ProcedureGA
from src.llm_procedure_generation_ga.procedure_classes import Procedure
from functions.query import query, query_repair_structured
from functions.run_steps import run_steps
from functions.answer_schemas import GSM_answer_schema
from functions.prompts import create_procedure_prompt

# ---- Import your schema and helpers from the runtime where this module is used ----
# Expect these names to exist in your environment:
#   - Procedure, Step
#   - create_procedure_prompt(item: str, example_prompt: str | None = None) -> str
#   - query(prompt: str, model: str, fmt: Optional[Dict[str,Any]] = None, seed: Optional[int] = ... ) -> str
#   - validate_procedure_structured(p: Dict[str, Any]) -> List[Diagnostic-like]
#   - query_repair_structured(p: Dict[str, Any], model: str, ...) -> Dict[str, Any]
#   - run_steps(proc_json: Dict[str, Any], question: str, final_answer_schema: Dict[str,Any], model: str, print_bool: bool=False)
#
# We *do not* import them here so this file stays decoupled; pass them as callables.


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
