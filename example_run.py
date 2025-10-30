from src.scorers import StructuralHygieneScorer, ProcScorerAdapter, TaskEvalScorer
from src.ga_scaffold_structured import ProcedureGA
from src.validators import validate_procedure_structured

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

