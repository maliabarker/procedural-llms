import json, random, time
from typing import Dict, Any
from datasets import load_dataset
from src.procedure_classes import Procedure
from src.answer_schemas import GSM_answer_schema, ranking_schema
from src.prompts import create_procedure_prompt, create_ranking_prompt
from src.run_steps import run_steps_stateful_minimal as run_steps
from src.query import query_repair_structured, create_and_validate_procedure_structured, query
from src.helpers import extract_final_number


random.seed(42)
JSONDict = Dict[str, Any]
MODEL = "gemma3"
# MODEL = "qwen3-coder"

def run_full_procedure_structured(i, q, model, print_bool=False):
    """
    Variables
    ---------
        i: int
            The original index of the question from the benchmark dataset
        q: str
            The question to be answered
        model: str
            The model to use in the LLM query
    """
    # Generate a prompt
    p_proc = create_procedure_prompt(q)
    # Generate the procedure
    proc = json.loads(query(p_proc, model, Procedure.model_json_schema()))
    # Validate the procedure
    try:
        reprompted = query_repair_structured(proc, model, print_bool=print_bool)
    except Exception as e:
        print(f"[{i}] Unable to get valid reprompt: {e}")
    else:
        proc = reprompted
    # Run the procedure to get the procedural answer
    answer = run_steps(proc, q, GSM_answer_schema, model, print_bool)
    # Return the procedure and the answer
    return (proc, answer)


if __name__ == "__main__":
    # Example code from ollama to test if it is working
    # %pip install -q llama-index-llms-ollama
    # from llama_index.llms.ollama import Ollama
    # from llama_index.llms.ollama import Ollama
    # llm = Ollama(
    #     model="llama3.1:latest",
    #     request_timeout=120.0,
    #     # Manually set the context window to limit memory usage
    #     context_window=8000,
    #     # base_url="http://127.0.0.1:11500"
    # )
    # resp = llm.complete("Who is Paul Graham?")
    # resp

    # Loading in datasets
    gsm_8k_ds = load_dataset("openai/gsm8k", "main")
    SLOW_THRESHOLD = 60.0  # seconds
    incorrect_p = []
    incorrect_d = []
    n = 3
    seeds = [random.randint(1000, 9999) for _ in range(n)]

    t0_all = time.perf_counter()
    all_qs_count = gsm_8k_ds["train"].num_rows
    for i in range(0, all_qs_count):
        t_iter = time.perf_counter()
        try:
            if i < 10:
                these_procedures = []
                # Get the question
                q = gsm_8k_ds["train"][i]["question"]
                a = int(extract_final_number(gsm_8k_ds["train"][i]["answer"]))
                # Get the direct prompt answer
                a_direct = json.loads(query(q, MODEL, GSM_answer_schema))
                if a != a_direct["answer_numerical"]:
                    incorrect_dict_d = {
                            "original_i": i,
                            "actual_answer": a,
                            "given_answer": a_direct["answer_numerical"]
                        }
                    incorrect_d.append(incorrect_dict_d)
                # Generate the original procedure prompt and list of procedures
                prompt = create_procedure_prompt(q)
                procedures = [create_and_validate_procedure_structured(i, q, MODEL, seed=s) for s in seeds]
                # Generate the ranking prompt and get the procedure ranking
                ranking_prompt = create_ranking_prompt(prompt, procedures)
                ranks = json.loads(query(ranking_prompt, MODEL, ranking_schema))
                # Grab the top-ranked procedure and run the steps
                top_index = ranks["ranking"][0]["procedure_index"]
                top_procedure = procedures[top_index]
                ans = run_steps(top_procedure, q, GSM_answer_schema, MODEL)
                # # Check to see if this is correct or not
                if "answer_numerical" not in ans.keys() or a != ans["answer_numerical"]:
                    if "answer_numerical" not in ans.keys():
                        ans = ans
                    else:
                        ans = ans["answer_numerical"] 
                        incorrect_dict_p = {
                            "original_i": i,
                            "actual_answer": a,
                            "given_answer": ans,
                            "procedure": top_procedure
                        }
                    incorrect_p.append(incorrect_dict_p)
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
        finally:
            dt = time.perf_counter() - t_iter
            if dt > SLOW_THRESHOLD:
                print(f"[{i}] {dt:.3f}s")

    total_dt = time.perf_counter() - t0_all
    print(f"Incorrect count direct: {len(incorrect_d)} | Incorrect count procedural: {len(incorrect_p)} | Total time: {total_dt:.3f}s")

