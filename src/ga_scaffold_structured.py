"""
ga_scaffold_structured.py — GA scaffold for your existing Procedure schema + Ollama flow.

This module plugs into your current code design (Procedure/Step Pydantic schema,
`query`/`hard_query` with Ollama, your validators & repair functions, and the
stateful step runner). It gives you:
  - GAConfig / Individual / ProcedureGA orchestrator
  - Crossover, Merge, Mutation operators compatible with your schema
  - Two scoring strategies: Structural (validator-driven) and TaskEval (run & grade)
  - Population init using your `create_procedure_prompt` + `query` calls
  - Automatic "child repair" via your `query_repair_structured`

How to use (minimal):
---------------------
from ga_scaffold_structured import *
ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=lambda: Procedure.model_json_schema(),
    validate_fn=validate_procedure_structured,
    repair_fn=query_repair_structured,
    scorer=StructuralScorer(),
    cfg=GAConfig(population_size=8, max_generations=5, seed=42),
)
best, history = ga.run(
    task_description="Solve: Natalia sold clips to 48 friends in April...",
    final_answer_schema=GSM_answer_schema,    # or ARC_answer_schema, etc.
    eval_fn=None,  # or TaskEvalScorer requires eval_fn if you want accuracy-based
    print_progress=True,
)

Notes:
- We keep your Step.id as int and preserve sequential ordering.
- Operators use "repair_fn" to enforce your strict chaining rules after edits.
- If you prefer a different mutation distribution, tweak MutationOperator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Literal
import copy
import json
import math
import random

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

JSONDict = Dict[str, Any]


# ======================
# Config / Individuals
# ======================

@dataclass
class GAConfig:
    population_size: int = 5
    elitism: int = 2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.25
    merge_rate: float = 0.05
    max_generations: int = 10
    tournament_k: int = 3
    seed: Optional[int] = None


@dataclass
class Individual:
    proc: JSONDict
    fitness: Optional[float] = None
    notes: str = ""


# ======================
# Scorers
# ======================

class Scorer:
    def score(self, ind: Individual, **kwargs: Any) -> float:
        raise NotImplementedError


class StructuralScorer(Scorer):
    """
    Penalize diagnostics; reward clean structure. Plug in your own weights if desired.
    Scores for structural diagnostics.
    """
    def __init__(self, validate_fn: Callable[[JSONDict], List[Any]], base: float = 1.0, penalty_fatal: float = 1.0, penalty_repairable: float = 0.25) -> None:
        self.validate_fn = validate_fn
        self.base = base
        self.penalty_fatal = penalty_fatal
        self.penalty_repairable = penalty_repairable

    def score(self, ind: Individual, **kwargs: Any) -> float:
        """Calculates score based on diagnostic messages.

        Score = base - (#fatal * penalty_fatal) - (#repairable * penalty_repairable)
        """
        diags = self.validate_fn(ind.proc)
        # Heuristic: assume presence of the literal strings in your Diagnostic structure
        fatal = sum(1 for d in diags if d.get("severity") == "fatal")
        repair = sum(1 for d in diags if d.get("severity") == "repairable")
        return self.base - fatal * self.penalty_fatal - repair * self.penalty_repairable


class TaskEvalScorer(Scorer):
    """
    Execute procedure and grade with user-provided eval_fn(answer_state)->float.
    Scores for procedure run and final answer.
    """
    def __init__(
        self,
        run_steps_fn: Callable[[JSONDict, str, Dict[str, Any], str], Dict[str, Any]],
        eval_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
        question: str,
        final_answer_schema: Dict[str, Any],
        model: str,
        strict_require_key: Optional[str] = None,  # e.g., "final_answer"
    ) -> None:
        self.run_steps_fn = run_steps_fn
        self.eval_fn = eval_fn
        self.question = question
        self.final_answer_schema = final_answer_schema
        self.model = model
        self.strict_require_key = strict_require_key

    def score(self, ind: Individual, **kwargs: Any) -> float:
        try:
            state = self.run_steps_fn(ind.proc, self.question, self.final_answer_schema, self.model, print_bool=False)
            if self.strict_require_key and self.strict_require_key not in state:
                return -1.0
            return float(self.eval_fn(state, ind.proc))
        except Exception:
            return -1.0


# ======================
# Operators
# ======================

def _copy_proc(p: JSONDict) -> JSONDict:
    return json.loads(json.dumps(p))

def _step_names(step: JSONDict, key: str) -> List[str]:
    # key is "inputs" or "output" (your schema uses list of {name, description})
    return [f["name"] for f in step.get(key, [])]

def _ensure_unique_names(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        if it["name"] not in seen:
            out.append(it)
            seen.add(it["name"])
    return out

def _renumber_steps(proc: JSONDict) -> JSONDict:
    """Ensure step ids are 1..n and consistent with list order."""
    newp = _copy_proc(proc)
    for i, st in enumerate(newp["steps"], start=1):
        st["id"] = i
    return newp

def _append_missing_outputs(prev_step: JSONDict, needed_inputs: List[str]) -> None:
    prev_names = set(_step_names(prev_step, "output"))
    for name in needed_inputs:
        if name not in prev_names:
            prev_step["output"].append({"name": name, "description": f"pass-through for {name}"})
    prev_step["output"] = _ensure_unique_names(prev_step["output"])


class CrossoverOperator:
    """
    LLM crossover that merges Parent A + Parent B into a coherent child that
    uses a GLOBAL STATE of variables (no step-to-step pass-through required).
    It returns ONE JSON object validating the Procedure schema.
    """
    def __init__(
        self,
        model: str,
        query_fn,                 # your query(prompt, model, fmt, seed)
        schema_json_fn,           # lambda: Procedure.model_json_schema()
        validate_fn,              # your validate_procedure_structured
        repair_fn,                # your query_repair_structured
        seed: int = 1234,
    ):
        self.model = model
        self.query_fn = query_fn
        self.schema_json_fn = schema_json_fn
        self.validate_fn = validate_fn
        self.repair_fn = repair_fn
        self.seed = seed

    def _build_prompt(
        self,
        task_description: str,
        parent_a_json: str,
        parent_b_json: str,
        extra_constraints: Optional[str] = None,
        style_hint: Optional[str] = None,
    ) -> str:
        """Construct the crossover prompt."""
        schema_json = json.dumps(self.schema_json_fn(), ensure_ascii=False)
        constraints = extra_constraints or """
            REQUIREMENTS (hard):
            - Output exactly ONE JSON object that validates against the schema.
            - GLOBAL STATE: Each step may READ any variable previously produced by earlier steps.
            - Input(s) declared for a step must be resolvable from variables produced by some earlier step (by name).
            - Output(s) declared for a step must be unique across the procedure (no duplicate names).
            - Variable names must be snake_case and stable across steps.
            - Step 1 must include only 'problem_text' as its input.
            - The final step's outputs include exactly 'final_answer' (a description of the final value, not the computed value).
            - Keep steps single-action and imperative; avoid redundant variables; no unreachable steps.
            - Prefer early extraction: extract primitive facts before any compute/transform steps.
            """
        style = style_hint or "Prefer A's strong extraction and B's clean reasoning; reconcile variable names for consistency."

        return f"""You are a rigorous planner that ONLY outputs a JSON object that validates against the provided schema.

            # TASK
            Synthesize a SINGLE crossover child procedure for the task below using a GLOBAL STATE (no pass-through chaining needed).

            ## Task Description
            {task_description}

            ## Procedure JSON Schema (Pydantic-derived)
            {schema_json}

            ## Parent A (JSON)
            ```json
            {parent_a_json}```

            ## Parent B (JSON)
            ```json
            {parent_b_json}```

            ## Crossover Objective
            - Reuse the best sub-steps, remove duplicates, and align variable names.
            - Steps read from a global state of already-available variables.
            - {style}

            {constraints}

            Return the JSON object only. Do not include markdown, fences, or commentary.
            """
    def __call__(
        self,
        task_description: str,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        n_offspring: int = 1,
    ) -> Dict[str, Any]:
        """Perform crossover between two parent procedures."""
        schema = self.schema_json_fn()
        pa = json.dumps(parent_a, ensure_ascii=False)
        pb = json.dumps(parent_b, ensure_ascii=False)

        children: List[Dict[str, Any]] = []
        for _ in range(max(1, n_offspring)):
            prompt = self._build_prompt(task_description, pa, pb)
            raw = self.query_fn(prompt, self.model, fmt=schema, seed=self.seed)
            child = json.loads(raw) if isinstance(raw, str) else raw
            try:
                child = self.repair_fn(child, self.model)  # still useful to normalize ids/format
            except Exception:
                pass
            children.append(child)

        if len(children) == 1:
            return children[0]
        else:
            # pick the child with fewest fatal/repairable diagnostics (only applicable if multiple children)
            def penalty(proc: Dict[str, Any]) -> tuple[int, int]:
                diags = self.validate_fn(proc)
                fatal = sum(1 for d in diags if d.get("severity") == "fatal")
                repair = sum(1 for d in diags if d.get("severity") == "repairable")
                return (fatal, repair)
            return min(children, key=penalty)




class MergeOperator:
    """
    Concatenate procedures A then B (ensemble-like). We keep Step 1 of A to retain the exact
    constraint 'step 1 inputs == problem_text'. Then we append B's steps, and perform a
    minimal pass-through patch. Final strictness is deferred to repair_fn.
    """
    def __call__(self, a: JSONDict, b: JSONDict) -> JSONDict:
        a_steps = _copy_proc(a)["steps"]
        b_steps = _copy_proc(b)["steps"]
        child_steps = a_steps + b_steps
        child = {"NameDescription": f"merge(A+B)", "steps": child_steps}

        for k in range(len(child_steps) - 1):
            need = _step_names(child_steps[k+1], "inputs")
            _append_missing_outputs(child_steps[k], need)

        return _renumber_steps(child)


class MutationOperator:
    """
    Mutations:
      - rewrite a stepDescription (LLM text op)
      - insert a passthrough step that forwards needed vars
      - drop a redundant output from a random step
    """
    def __init__(
        self,
        query_fn: Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str],
        model: str,
        text_rate: float = 0.6,
        add_step_rate: float = 0.2,
        drop_output_rate: float = 0.2,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.query_fn = query_fn
        self.model = model
        self.text_rate = text_rate
        self.add_step_rate = add_step_rate
        self.drop_output_rate = drop_output_rate
        self.rng = rng or random.Random()

    def __call__(self, proc: JSONDict) -> JSONDict:
        child = _copy_proc(proc)
        if not child["steps"]:
            return child

        choice = self.rng.random()
        if choice < self.text_rate:
            st = self.rng.choice(child["steps"])
            prompt = (
                "Rewrite the following instruction to be more concrete, concise, and single-action. "
                "Return ONLY the rewritten text.\n\n"
                f"{st['stepDescription']}\n"
            )
            try:
                new_text = self.query_fn(prompt, self.model, fmt=None, seed=1234).strip()
                new_text = new_text.strip().strip('"').strip("'")
                if new_text:
                    st["stepDescription"] = new_text
            except Exception:
                pass

        elif choice < self.text_rate + self.add_step_rate:
            # Insert passthrough after random non-final step
            idx = self.rng.randrange(0, len(child["steps"]) - 1) if len(child["steps"]) > 1 else 0
            cur = child["steps"][idx]
            nxt = child["steps"][idx + 1] if idx + 1 < len(child["steps"]) else None
            passthrough_outs = [{"name": n, "description": f"pass-through for {n}"} for n in (_step_names(cur, "output") + (_step_names(nxt, "inputs") if nxt else []))]
            new_step = {
                "id": 0,
                "inputs": [{"name": n, "description": f"forwarded {n}"} for n in _step_names(cur, "output")],
                "stepDescription": "Verify current variables and forward for next operation.",
                "output": _ensure_unique_names(passthrough_outs),
            }
            child["steps"].insert(idx + 1, new_step)

        else:
            # Drop a redundant output (if any)
            idx = self.rng.randrange(0, len(child["steps"]))
            st = child["steps"][idx]
            outs = st.get("output", [])
            if len(outs) > 1:
                drop_idx = self.rng.randrange(0, len(outs))
                outs.pop(drop_idx)

        return _renumber_steps(child)





# ======================
# GA Core
# ======================

class ProcedureGA:
    def __init__(
        self,
        model: str,
        create_proc_fn: Callable[[str], str],
        query_fn: Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str],
        schema_json_fn: Callable[[], Dict[str, Any]],
        validate_fn: Callable[[JSONDict], List[Any]],
        repair_fn: Callable[[JSONDict, str], JSONDict],
        scorer: Optional[Scorer] = None,
        cfg: GAConfig = GAConfig(),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.model = model
        self.create_proc_fn = create_proc_fn
        self.query_fn = query_fn
        self.schema_json_fn = schema_json_fn
        self.validate_fn = validate_fn
        self.repair_fn = repair_fn
        self.cfg = cfg
        self.rng = rng or random.Random(cfg.seed)

        self.crossover = CrossoverOperator(self.rng)
        self.mergeop = MergeOperator()
        self.mutate = MutationOperator(query_fn=self.query_fn, model=self.model, rng=self.rng)
        self.scorer = scorer or StructuralScorer(validate_fn=self.validate_fn)

    # ---- Initialization ----

    def _generate_one(self, task_description: str) -> JSONDict:
        prompt = self.create_proc_fn(task_description)
        raw = self.query_fn(prompt, self.model, fmt=self.schema_json_fn(), seed=1234)
        try:
            proc = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            # Try extracting JSON-ish fallback
            l = raw.find("{")
            r = raw.rfind("}")
            if l != -1 and r != -1 and r > l:
                proc = json.loads(raw[l:r+1])
            else:
                raise
        # attempt structured repair to satisfy strict validators
        try:
            proc = self.repair_fn(proc, self.model)
        except Exception:
            pass
        return proc

    def initialize_population(self, task_description: str) -> List[Individual]:
        pop = [Individual(self._generate_one(task_description)) for _ in range(self.cfg.population_size)]
        return pop

    # ---- Evaluation ----

    def evaluate(self, pop: List[Individual], scorer: Optional[Scorer] = None, **kwargs: Any) -> None:
        scorer = scorer or self.scorer
        for ind in pop:
            ind.fitness = scorer.score(ind, **kwargs)

    # ---- Selection ----

    def _tournament(self, pop: List[Individual]) -> Individual:
        k = min(self.cfg.tournament_k, len(pop))
        group = self.rng.sample(pop, k=k)
        return max(group, key=lambda i: i.fitness if i.fitness is not None else -1e9)

    def _select_parents(self, pop: List[Individual]) -> Tuple[Individual, Individual]:
        return self._tournament(pop), self._tournament(pop)

    # ---- Reproduction ----

    def _reproduce(self, p1: Individual, p2: Individual) -> JSONDict:
        r = self.rng.random()
        if r < self.cfg.crossover_rate:
            child = self.crossover(p1.proc, p2.proc)
        elif r < self.cfg.crossover_rate + self.cfg.mutation_rate:
            child = self.mutate(p1.proc)
        else:
            child = self.mergeop(p1.proc, p2.proc)

        # Always run repair pass to satisfy validators and your strict chaining
        try:
            child = self.repair_fn(child, self.model)
        except Exception:
            pass
        return child

    # ---- Run ----

    def run(
        self,
        task_description: str,
        final_answer_schema: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
        run_steps_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        print_progress: bool = False,
    ) -> Tuple[Individual, List[Individual]]:
        pop = self.initialize_population(task_description)
        history: List[Individual] = []

        for gen in range(self.cfg.max_generations):
            # Evaluate
            if eval_fn and run_steps_fn and final_answer_schema is not None:
                scorer = TaskEvalScorer(
                    run_steps_fn=run_steps_fn,
                    eval_fn=eval_fn,
                    question=task_description,
                    final_answer_schema=final_answer_schema,
                    model=self.model,
                    strict_require_key=None,
                )
            else:
                scorer = self.scorer

            self.evaluate(pop, scorer)
            pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
            best = copy.deepcopy(pop[0])
            history.append(best)
            if print_progress:
                print(f"[gen {gen+1}] best fitness={best.fitness:.3f} steps={len(best.proc.get('steps', []))}")

            # Next generation
            next_pop: List[Individual] = [copy.deepcopy(e) for e in pop[: self.cfg.elitism]]
            while len(next_pop) < self.cfg.population_size:
                p1, p2 = self._select_parents(pop)
                child = self._reproduce(p1, p2)
                next_pop.append(Individual(proc=child))

            pop = next_pop

        # Final evaluate & return best
        self.evaluate(pop, scorer)
        pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
        return pop[0], history


# ======================
# Optional: tiny smoke
# ======================

if __name__ == "__main__":
    print("GA scaffold ready. Import into your environment that defines Procedure, query, etc.")
