"""
ga_scaffold_structured.py

Notes:
- We keep Step.id as int and preserve sequential ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol
from src.scorers import StructuralHygieneScorer
import copy
import json
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

class Scorer(Protocol):
    def score(self, ind: Individual, **kwargs: Any) -> float:
        raise NotImplementedError

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


class MutationOperator:
    """
    LLM-driven mutation for *global-state* procedures.

    Each call asks the model to apply EXACTLY ONE small, coherent mutation and
    return a full, schema-valid Procedure JSON. Examples (LLM chooses one):
      - rewrite a stepDescription to be crisper / single-action
      - insert a missing extraction/verification step
      - split a too-broad step into two clearer steps
      - remove a dead/unused output or tautological step
      - rename a variable consistently (avoid redefinitions; snake_case)
      - consolidate two adjacent trivial steps

    Guardrails:
      - Hard JSON schema in `fmt`
      - Explicit constraints in the prompt (Step 1 rule, final step rule, snake_case, etc.)
      - Run `repair_fn` + validators after mutation
      - Optional acceptance check vs. `proc_scorer.score_proc` (structural or task-based)

    Parameters
    ----------
    model, query_fn, schema_json_fn, validate_fn, repair_fn : callables you already have
    proc_scorer : optional; if provided, used to accept/reject the mutation
    accept_if_not_worse : if True, only accept if score >= original score
                          if False, always accept (pure GA exploration)
    rng : for reproducibility
    seed : forwarded to LLM for reproducibility
    max_llm_tries : retry the LLM a couple times if it fails schema/validation
    """

    def __init__(
        self,
        model: str,
        query_fn: Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str],
        schema_json_fn: Callable[[], Dict[str, Any]],
        validate_fn: Callable[[JSONDict], List[Dict[str, Any]]],
        repair_fn: Callable[[JSONDict, str], JSONDict],
        proc_scorer: Optional[Any] = None,  # object exposing score_proc(proc_json)->float
        *,
        accept_if_not_worse: bool = True,
        rng: Optional[random.Random] = None,
        seed: int = 1234,
        max_llm_tries: int = 2,
    ) -> None:
        self.model = model
        self.query_fn = query_fn
        self.schema_json_fn = schema_json_fn
        self.validate_fn = validate_fn
        self.repair_fn = repair_fn
        self.proc_scorer = proc_scorer
        self.accept_if_not_worse = accept_if_not_worse
        self.rng = rng or random.Random()
        self.seed = seed
        self.max_llm_tries = max_llm_tries

    # ---------- public API ----------
    def __call__(self, proc: JSONDict, task_description: str) -> JSONDict:
        orig = _deepcopy(proc)
        target_score = self._score(orig) if self.proc_scorer else None

        schema = self.schema_json_fn()
        proc_json = json.dumps(orig, ensure_ascii=False)

        # To add light stochasticity without hard-coding “types”, we give a short intent hint.
        intent = self._sample_intent()
        prompt = self._build_prompt(task_description, proc_json, schema, intent=intent)

        # Try LLM, then repair+validate; optionally accept by score
        candidate = None
        for _ in range(max(1, self.max_llm_tries)):
            try:
                raw = self.query_fn(prompt, self.model, fmt=schema, seed=self.seed)
                cand = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue

            try:
                cand = self.repair_fn(cand, self.model)
            except Exception:
                pass  # it's okay; we'll check validators regardless

            # must keep Step IDs 1..n
            cand = _renumber_steps(cand)

            # reject if clearly invalid (fatal diags)
            diags = self.validate_fn(cand)
            if any(d.get("severity") == "fatal" for d in diags):
                continue

            candidate = cand
            break

        if candidate is None:
            # failed to get a valid mutation—return original
            return orig

        if self.proc_scorer and self.accept_if_not_worse:
            new_score = self._score(candidate)
            if new_score < target_score:  # reject if worse
                return orig

        return candidate

    # ---------- helpers ----------
    def _build_prompt(self, task: str, proc_json: str, schema: Dict[str, Any], intent: str) -> str:
        schema_json = json.dumps(schema, ensure_ascii=False)
        # Keep constraints crisp; model returns ONE JSON object only.
        return f"""
        You will perform a SINGLE, SMALL mutation to the Procedure for the task below.
        Return ONLY ONE JSON object that validates against the schema.

        # Task
        {task}

        # Procedure JSON Schema (verbatim)
        {schema_json}

        # Current Procedure (JSON)
        ```json
        {proc_json}```

        # Mutation Goal
        - Apply exactly ONE mutation that improves clarity, correctness likelihood, or structural hygiene.
        - Mutation intent (hint): {intent}

        # Hard Constraints (global-state semantics)
        - Step 1 inputs == ["problem_text"].
        - Later steps may read any variable produced by earlier steps (global state).
        - Final step outputs exactly ["final_answer"] (description only; do not compute numeric value).
        - Variable names must be snake_case; avoid redefining an existing variable name.
        - Remove dead outputs if they become unused; keep each step single-action, imperative.
        - Prefer early extraction: move primitive fact extraction earlier if applicable.

        # Output
        Return the FULL mutated procedure as a SINGLE JSON object valid under the schema.
        Do NOT include markdown, fences, or commentary.
        """.strip()

    def _sample_intent(self) -> str:
        # light diversity; we do not hard-code behavior, only provide a hint
        intents = [
            "rewrite one step to be more concrete/single-action",
            "split one too-broad step into two small steps",
            "insert a missing extraction step for a needed variable",
            "remove one unused output or trivial no-op step",
            "rename an inconsistent variable to a consistent snake_case name",
            "consolidate two adjacent trivial steps without losing information",
            "add one verification/check step to ensure extracted facts are consistent",
        ]
        return self.rng.choice(intents)

    def _score(self, p: JSONDict) -> float:
        try:
            return float(self.proc_scorer.score_proc(p))  # type: ignore[attr-defined]
        except Exception:
            return float("-inf")


# ---- minimal shared helpers (align with your codebase) ----

def _deepcopy(p: JSONDict) -> JSONDict:
    return json.loads(json.dumps(p))

def _renumber_steps(p: JSONDict) -> JSONDict:
    q = _deepcopy(p)
    for i, s in enumerate(q.get("steps", []), start=1):
        s["id"] = i
    return q





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
        self.crossover = CrossoverOperator(
            model=self.model,
            query_fn=self.query_fn,
            schema_json_fn=self.schema_json_fn,
            validate_fn=self.validate_fn,
            repair_fn=self.repair_fn,
            seed=(self.cfg.seed or 1234),
        )
        self.mutate = MutationOperator(query_fn=self.query_fn, model=self.model, rng=self.rng)
        self.scorer = scorer or StructuralHygieneScorer(validate_fn=self.validate_fn)

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

    def _reproduce(self, task_description: str, p1: Individual, p2: Individual) -> JSONDict:
        r = self.rng.random()
        if r < self.cfg.crossover_rate:
            child = self.crossover(task_description, p1.proc, p2.proc)
        elif r < self.cfg.crossover_rate + self.cfg.mutation_rate:
            child = self.mutate(p1.proc)
        else:
            # TODO: need to figure out what to do now that merge is gone
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
                from src.scorers import TaskEvalScorer  # lazy import to avoid cycles
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
