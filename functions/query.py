import ollama, json
from typing import Dict, Any, Optional
from src.procedure_classes import Procedure
from functions.prompts import create_procedure_prompt
from src.validators import validate_procedure_structured
from src.helpers import pretty_print

# OLLama Queries
def hard_query(prompt: str, model: str, fmt: Dict[str, Any], seed: Optional[int]=1234):
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format=fmt,
        options={ "temperature": 0, "seed": seed }
    )
    return res['response']

def query(prompt: str, model: str, fmt: Optional[Dict[str, Any]] = None, seed: Optional[int] = 1234):
    # This is generalized to use for ANY ollama call
    # Will usually pass in gemma3 as the model
    # Will use Procedure.model_json_schema() for procedure calls
    # Will use answer schema specified for dataset for final answer calls
    # NOTE: Adding seed so answers are re-producible
    res = ollama.generate(
        model=model,
        prompt=prompt,
        format=fmt,
        options={ "temperature": 1, "seed": seed }
    )
    return res['response']

def query_repair_structured(p: Dict[str, Any], model, max_tries=10, print_bool=False) -> Dict[str, Any]:
    for _ in range(max_tries):
        diag_msgs = validate_procedure_structured(p)
        diag_str = [str(i) for i in diag_msgs]
        if print_bool:
            pretty_print(p)
            print(f"Errors:\n- " + "\n- ".join(diag_str))
        if not diag_msgs:
            return p
        repair_prompt = (
            f"This is a procedure with the following format: {Procedure.model_json_schema()} "
            "Make the requested minimal fix(es) and output a correct procedure in JSON format only, no prose."
            f"Instructions:\n- " + "\n- ".join(diag_str) + "\n\nProcedure JSON:\n" + (json.dumps(p))
        )
        p = json.loads(hard_query(repair_prompt, model, Procedure.model_json_schema()))
    raise RuntimeError("Could not satisfy validator after retries.")

def create_and_validate_procedure_structured(i, q: str, model: str, **model_kwargs: Any):
    # Generate a prompt
    p_proc = create_procedure_prompt(q)
    # Generate the procedure
    proc = json.loads(query(p_proc, model, Procedure.model_json_schema(), **model_kwargs,))
    # Validate the procedure
    try:
        reprompted = query_repair_structured(proc, model)
    except Exception as e:
        print(f"[{i}] Unable to get valid reprompt: {e}")
    else:
        proc = reprompted
    return proc