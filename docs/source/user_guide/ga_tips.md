# GA Tips & Recipes

This page distills practical advice for using the GA in {mod}`llm_procedure_generation_ga.ga_scaffold_structured` effectively.

## Scorers: structural vs task-eval

- **Structural hygiene (default):**  
  Uses {mod}`llm_procedure_generation_ga.scorers`’ adapter over the validator suite. Great for bootstrapping before you wire up execution.
- **Task-eval scoring (recommended for benchmarks):**  
  Pass all three to `ga.run(...)`:
  - `final_answer_schema`
  - `run_steps_fn(proc, question, final_answer_schema, model, print_bool=False) -> state`
  - `eval_fn(state, proc) -> float` (e.g., exact-match or numeric closeness)

```python
best, _ = ga.run(
    task_description=question,
    final_answer_schema=get_schema("gsm"),
    run_steps_fn=run_steps_fn,   # calls your runner
    eval_fn=eval_fn,             # returns [0,1]
)
```

## Sensible hyperparameters (start here)

```python
GAConfig(
  population_size=6,         # 6–12 for quick loops
  elitism=2,                 # keep top 2 unchanged
  crossover_rate=0.7,        # favor crossover
  mutation_rate=0.3,         # still allow edits
  max_generations=3,         # 3–6 to start
  tournament_k=3,
  random_immigrant_rate=0.10,# diversity without chaos
  seed=42,
)
```

- Increase generations first, then population.
- Keep elitism ≥ 1 so progress isn’t lost.
- Immigrants help escape local minima; 5–20% is typical.

## Reproducibility 
- Set seed in both GAConfig and your backend query options.
- Log the chosen model and any temperature/top-p settings.
- Persist runs to JSONL: keep best.proc, fitness, and final state.
```python
import json
with open("runs.jsonl","a") as f:
    f.write(json.dumps({
        "qid": qid,
        "fitness": best.fitness,
        "proc": best.proc,
        "state": final_state,
    }) + "\n")
```

## Efficient task-eval loops
- Use a lightweight run_steps_fn: small schemas, minimal I/O, strict JSON parsing.
- Cache step prompts if you’re experimenting; only the best individual per gen must be executed.
- If your runner is expensive, consider generation-end evaluation only (evaluate after reproduction, not for every candidate mid-gen).

## Operators: practical notes
- Crossover tends to clean up naming and structure—keep it dominant (0.6–0.8).
- Mutation is single, small edits by design (split/insert/rename/verify). If progress stalls, slightly raise mutation_rate.

## Diagnosing stagnation
- Flat fitness across generations → Scorer too coarse. Switch to task-eval or add a shaping term (e.g., small penalty for > N steps).
- Validator ping-pong → Tighten repair prompts; prefer minimal fixes and renumber steps after repair.
- Degenerate loops (same child every time) → add random_immigrant_rate and ensure seed is not globally constant if you want exploration.

## GSM8K recipe (mini)
```python
FINAL_SCHEMA = get_schema("gsm")

def run_steps_fn(proc, q, s, m, print_bool=False):
    return run_steps_stateful_minimal(proc, q, s, m, query_fn=query, print_bool=print_bool)

def eval_fn(state, proc):
    import math, re
    nums = re.findall(r"-?\d+(?:\.\d+)?", state.get("answer",""))
    pred = state.get("answer_numerical") or (float(nums[-1]) if nums else None)
    gold = state.get("_gold_num")
    return 1.0 if (pred is not None and gold is not None and math.isclose(pred, gold, abs_tol=1e-6)) else 0.0

# loop
for ex in examples:  # list of {id, question, answer}
    gold_num = extract_number(ex["answer"])
    best, _ = ga.run(
        task_description=ex["question"],
        final_answer_schema=FINAL_SCHEMA,
        run_steps_fn=run_steps_fn,
        eval_fn=lambda st, pr: eval_fn({**st, "_gold_num": gold_num}, pr),
    )
    state = run_steps_fn(best.proc, ex["question"], FINAL_SCHEMA, ga.model)
```

## Troubleshooting
- ModuleNotFoundError during docs build
Make sure both packages are installed in the docs env and conf.py adds their src/ paths.
- LLM returns prose, not JSON
Lower temperature and include the schema verbatim (format= parameter if your backend supports it).
- “Missing required output …” at runtime
Keep strict_missing=True to catch it early; if the model is flaky, allow one retry with the same prompt/seed.

## When to stop evolving
- Fitness plateaus for ≥2 generations.
- Best procedure stabilizes (diff of steps/variables is small).
- Wall-clock budget reached (log total tokens/calls if possible).