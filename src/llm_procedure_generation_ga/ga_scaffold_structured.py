# src/ga_scaffold_structured.py
"""
GA scaffold for LLM-generated *global-state* procedures.

This module wires up a lightweight genetic algorithm (GA) over your Procedure JSON
schema. It uses LLM-driven **crossover** and **mutation** operators, a structural
hygiene scorer (validator-driven), optional task-eval scoring, and a small dose
of diversity via random immigrants.

Global-state semantics
----------------------
- Step 1 must take exactly one input: ["problem_text"].
- Later steps may read any variable produced by earlier steps (no strict pass-through).
- Final step must output exactly ["final_answer"] (a *description*, not a computed value).

Typical usage
-------------
from src.ga_scaffold_structured import *
from src.scorers import StructuralHygieneScorer, ProcScorerAdapter, TaskEvalScorer
from src.validators import validate_procedure_structured

ga = ProcedureGA(
    model="gemma3:latest",
    create_proc_fn=create_procedure_prompt,
    query_fn=query,
    schema_json_fn=lambda: Procedure.model_json_schema(),
    validate_fn=validate_procedure_structured,
    repair_fn=query_repair_structured,
    scorer=ProcScorerAdapter(StructuralHygieneScorer(validate_fn=validate_procedure_structured)),
    cfg=GAConfig(population_size=8, max_generations=5, seed=42),
)

best, history = ga.run(
    task_description="Solve: Natalia sold clips to 48 friends in April...",
    # Supply these three for TaskEval scoring; otherwise structural scoring is used:
    final_answer_schema=None,
    eval_fn=None,        # (state, proc) -> float
    run_steps_fn=None,   # executes a procedure end-to-end and returns `state`
    print_progress=True,
)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol
import copy
import json
import random

from src.llm_procedure_generation_ga.scorers import StructuralHygieneScorer, ProcScorerAdapter  # default structural scorer

JSONDict = Dict[str, Any]


# ======================
# Config / Individuals
# ======================

@dataclass
class GAConfig:
    """
    Genetic algorithm configuration.

    Parameters
    ----------
    population_size
        Number of individuals per generation.
    elitism
        Number of top individuals copied unchanged to the next generation.
    crossover_rate
        Probability of producing a child via crossover in reproduction.
    mutation_rate
        Probability of producing a child via mutation (fallback is also mutation).
    max_generations
        Total number of evolutionary iterations.
    tournament_k
        Tournament size for parent selection.
    seed
        Random seed for reproducibility (also forwarded to LLM ops where applicable).
    random_immigrant_rate
        Fraction of the remaining slots each generation filled with freshly
        generated procedures (diversity injection).
    """
    population_size: int = 5
    elitism: int = 2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    max_generations: int = 10
    tournament_k: int = 3
    seed: Optional[int] = None
    random_immigrant_rate: float = 0.10


@dataclass
class Individual:
    """
    Population member wrapper.

    Attributes
    ----------
    proc
        Procedure JSON (dict) that validates your Pydantic-derived schema.
    fitness
        Last computed scalar fitness (None until evaluated).
    notes
        Optional debugging/instrumentation notes.
    """
    proc: JSONDict
    fitness: Optional[float] = None
    notes: str = ""


# ======================
# GA-facing Scorer Protocol
# ======================

class Scorer(Protocol):
    """
    GA-facing scorer protocol. Implementations must accept an object with a
    `.proc` JSON field and return a scalar fitness.
    """
    def score(self, ind: Individual, **kwargs: Any) -> float: ...


# ======================
# Utilities
# ======================

def _deepcopy(p: JSONDict) -> JSONDict:
    """
    Return a deep copy of a procedure JSON using JSON round-tripping.
    """
    return json.loads(json.dumps(p))

def _renumber_steps(p: JSONDict) -> JSONDict:
    """
    Renumber step IDs to be contiguous 1..n in list order.

    Returns
    -------
    Dict[str, Any]
        A new procedure JSON with normalized step IDs.
    """
    q = _deepcopy(p)
    for i, s in enumerate(q.get("steps", []), start=1):
        s["id"] = i
    return q


# ======================
# Operators
# ======================

class CrossoverOperator:
    """
    LLM-based crossover for *global-state* procedures.

    Combines two parent procedures (A, B) into a single coherent child by
    prompting the LLM to synthesize an integrated plan that:
      - preserves the Step 1 rule (inputs == ["problem_text"]),
      - adheres to global-state semantics (later steps can read any earlier vars),
      - ends with exactly one output "final_answer",
      - validates against the provided schema.

    The raw child is normalized with `repair_fn` and lightly filtered by
    `validate_fn` before returning.

    Notes
    -----
    This operator does not splice JSON directly; instead it asks the LLM to
    synthesize a crossover child, which tends to yield more coherent procedures
    than mechanical concatenation.
    """
    def __init__(
        self,
        model: str,
        query_fn: Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str],
        schema_json_fn: Callable[[], Dict[str, Any]],
        validate_fn: Callable[[JSONDict], List[Dict[str, Any]]],
        repair_fn: Callable[[JSONDict, str], JSONDict],
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
        """
        Construct the crossover prompt with hard constraints and both parents.

        Returns
        -------
        str
            A single prompt string instructing the LLM to emit ONE JSON object
            that validates the schema and respects global-state constraints.
        """
        schema_json = json.dumps(self.schema_json_fn(), ensure_ascii=False)
        constraints = extra_constraints or """
            REQUIREMENTS (hard):
            - Output exactly ONE JSON object that validates against the schema.
            - GLOBAL STATE: Each step may READ any variable previously produced by earlier steps.
            - Declared inputs must be resolvable from variables produced earlier (by name).
            - Outputs must be unique across the procedure (no duplicate names).
            - Variable names are snake_case and stable across steps.
            - Step 1 must include only 'problem_text' as its input.
            - Final step outputs exactly 'final_answer' (description only; do not compute).
            - Keep steps single-action; avoid redundant variables; remove unreachable steps.
            - Prefer early extraction of primitive facts.
            """
        style = style_hint or "Prefer A's strong extraction and B's clean reasoning; reconcile variable names."

        return f"""You are a rigorous planner that ONLY outputs a JSON object that validates the schema.

                    # TASK
                    Synthesize a SINGLE crossover child procedure for the task below using GLOBAL STATE.

                    ## Task Description
                    {task_description}

                    ## Procedure JSON Schema (verbatim)
                    {schema_json}

                    ## Parent A (JSON)
                    ```json
                    {parent_a_json}```

                    ## Parent B (JSON)
                    ```json
                    {parent_b_json}```

                    ## Crossover Objective
                    - Reuse the best sub-steps, remove duplicates, align variable names.
                    - {style}

                    {constraints}

                    Return the JSON object only. No markdown or commentary.
                    """

    def __call__(
        self,
        task_description: str,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        n_offspring: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform crossover on two parents and return one child (or the best of `n_offspring`).

        Parameters
        ----------
        task_description
            The natural-language problem the procedure will solve.
        parent_a, parent_b
            The two input procedures (as JSON dicts).
        n_offspring
            If >1, generate multiple children and return the one with the fewest
            validator penalties.

        Returns
        -------
        Dict[str, Any]
            Child procedure JSON.
        """
        schema = self.schema_json_fn()
        pa = json.dumps(parent_a, ensure_ascii=False)
        pb = json.dumps(parent_b, ensure_ascii=False)

        children: List[Dict[str, Any]] = []
        for _ in range(max(1, n_offspring)):
            prompt = self._build_prompt(task_description, pa, pb)
            raw = self.query_fn(prompt, self.model, fmt=schema, seed=self.seed)
            child = json.loads(raw) if isinstance(raw, str) else raw
            try:
                child = self.repair_fn(child, self.model)
            except Exception:
                pass
            children.append(child)

        if len(children) == 1:
            return children[0]

        def penalty(proc: Dict[str, Any]) -> tuple[int, int]:
            diags = self.validate_fn(proc)
            fatal = sum(1 for d in diags if d.get("severity") == "fatal")
            repair = sum(1 for d in diags if d.get("severity") == "repairable")
            return (fatal, repair)

        return min(children, key=penalty)


class MutationOperator:
    """
    LLM-driven mutation for *global-state* procedures.

    Applies exactly ONE small edit per call (rewrite/split/insert/remove/rename/verify),
    returning a full, schema-valid procedure JSON. Post-processes with `repair_fn`
    and rejects candidates with fatal validator diagnostics. Optionally uses a
    procedure-level scorer to accept only not-worse mutations.

    Parameters
    ----------
    model
        LLM name to use for mutation prompts.
    query_fn
        Callable `query(prompt, model, fmt, seed) -> str` that returns JSON text.
    schema_json_fn
        Callable returning the Procedure JSON schema (dict).
    validate_fn
        Callable that returns a list of diagnostics for a procedure JSON.
    repair_fn
        Callable that minimally repairs a procedure JSON using the LLM, returning JSON.
    proc_scorer
        Optional object exposing `score_proc(proc_json) -> float` for acceptance check.
    rng
        Optional PRNG for sampling mutation intents.
    seed
        Forwarded to `query_fn` for deterministic results.

    Notes
    -----
    The mutation “type” is *not* hard-coded; instead we sample a short intent hint
    for the LLM (e.g., “split one step”, “remove unused output”), keeping the
    operator expressive but simple.
    """
    def __init__(
        self,
        model: str,
        query_fn: Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str],
        schema_json_fn: Callable[[], Dict[str, Any]],
        validate_fn: Callable[[JSONDict], List[Dict[str, Any]]],
        repair_fn: Callable[[JSONDict, str], JSONDict],
        proc_scorer: Optional[Any],
        rng: Optional[random.Random],
        seed: int,
        *,
        accept_if_not_worse: bool = True,
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

    def __call__(self, proc: JSONDict, task_description: str) -> JSONDict:
        """
        Mutate a single procedure.

        Parameters
        ----------
        proc
            The original procedure JSON.
        task_description
            Natural-language task the procedure addresses.

        Returns
        -------
        Dict[str, Any]
            Mutated (and repaired/validated) procedure JSON. If mutation fails
            validation or acceptance, returns the original.
        """
        orig = _deepcopy(proc)
        target_score = self._score(orig) if self.proc_scorer else None

        schema = self.schema_json_fn()
        proc_json = json.dumps(orig, ensure_ascii=False)
        intent = self._sample_intent()
        prompt = self._build_prompt(task_description, proc_json, schema, intent=intent)

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
                pass
            cand = _renumber_steps(cand)
            diags = self.validate_fn(cand)
            if any(d.get("severity") == "fatal" for d in diags):
                continue
            candidate = cand
            break

        if candidate is None:
            return orig

        if self.proc_scorer and self.accept_if_not_worse:
            new_score = self._score(candidate)
            if new_score < target_score:
                return orig

        return candidate

    # ---- helpers ----

    def _build_prompt(self, task: str, proc_json: str, schema: Dict[str, Any], intent: str) -> str:
        """
        Build an LLM prompt to apply exactly one small mutation under hard constraints.
        """
        schema_json = json.dumps(schema, ensure_ascii=False)
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
        """
        Sample a lightweight mutation intent to diversify edits without hard-coding types.
        """
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
        """
        Score a procedure with the provided `proc_scorer` if available.
        """
        try:
            return float(self.proc_scorer.score_proc(p))  # type: ignore[attr-defined]
        except Exception:
            return float("-inf")


# ======================
# GA Core
# ======================

class ProcedureGA:
    """
    Orchestrates the GA loop over procedures: initialize → evaluate → select →
    reproduce (crossover/mutation) → repair → validate → next generation.

    You provide your model + callable hooks (`query_fn`, `create_proc_fn`, validators,
    repair, and (optionally) a task-eval runner). By default, the GA uses a structural
    hygiene scorer; you can swap in task-eval scoring by supplying
    `final_answer_schema`, `eval_fn`, and `run_steps_fn` to `.run(...)`.
    """
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
        """
        Initialize the GA with model/context functions and configuration.
        """
        self.model = model
        self.create_proc_fn = create_proc_fn
        self.query_fn = query_fn
        self.schema_json_fn = schema_json_fn
        self.validate_fn = validate_fn
        self.repair_fn = repair_fn
        self.cfg = cfg
        self.rng = rng or random.Random(cfg.seed)

        # Default: structural hygiene scorer (adapter because GA calls .score(ind))
        self.scorer: Scorer = scorer or ProcScorerAdapter(
            StructuralHygieneScorer(validate_fn=self.validate_fn)
        )

        # Operators (LLM-based)
        self.crossover = CrossoverOperator(
            model=self.model,
            query_fn=self.query_fn,
            schema_json_fn=self.schema_json_fn,
            validate_fn=self.validate_fn,
            repair_fn=self.repair_fn,
            seed=(self.cfg.seed or 1234),
        )
        self.mutate = MutationOperator(
            model=self.model,
            query_fn=self.query_fn,
            schema_json_fn=self.schema_json_fn,
            validate_fn=self.validate_fn,
            repair_fn=self.repair_fn,
            proc_scorer=getattr(self.scorer, "_proc_scorer", None),  # pass underlying proc scorer if using adapter
            rng=self.rng,
            seed=(self.cfg.seed or 1234),
        )

    # ---- Initialization ----

    def _generate_one(self, task_description: str) -> JSONDict:
        """
        Create a single, valid procedure by:
          1) prompting the LLM with `create_proc_fn(task)`,
          2) parsing JSON (best-effort fallback extraction),
          3) running a repair pass,
          4) renumbering step IDs.

        Returns
        -------
        Dict[str, Any]
            A schema-conforming procedure JSON (best-effort).
        """
        prompt = self.create_proc_fn(task_description)
        raw = self.query_fn(prompt, self.model, fmt=self.schema_json_fn(), seed=1234)
        try:
            proc = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            # fallback: extract best-effort JSON substring
            l = raw.find("{"); r = raw.rfind("}")
            if l != -1 and r != -1 and r > l:
                proc = json.loads(raw[l:r+1])
            else:
                raise
        try:
            proc = self.repair_fn(proc, self.model)
        except Exception:
            pass
        return _renumber_steps(proc)

    def initialize_population(self, task_description: str) -> List[Individual]:
        """
        Generate the initial population by repeatedly calling `_generate_one`.

        Returns
        -------
        List[Individual]
            A list of Individuals with `proc` populated.
        """
        return [Individual(self._generate_one(task_description)) for _ in range(self.cfg.population_size)]

    # ---- Evaluation ----

    def evaluate(self, pop: List[Individual], scorer: Optional[Scorer] = None, **kwargs: Any) -> None:
        """
        Compute fitness for every individual in-place using the provided scorer
        or the GA's default scorer.

        Parameters
        ----------
        pop
            Population to evaluate.
        scorer
            Optional override; must implement `score(individual) -> float`.
        """
        scorer = scorer or self.scorer
        for ind in pop:
            ind.fitness = scorer.score(ind, **kwargs)

    # ---- Selection ----

    def _tournament(self, pop: List[Individual]) -> Individual:
        """
        Tournament selection: pick `k` random individuals and return the best.

        Returns
        -------
        Individual
            Selected parent candidate.
        """
        k = min(self.cfg.tournament_k, len(pop))
        group = self.rng.sample(pop, k=k)
        return max(group, key=lambda i: i.fitness if i.fitness is not None else -1e9)

    def _select_parents(self, pop: List[Individual]) -> Tuple[Individual, Individual]:
        """
        Select two parents independently via tournament selection.
        """
        return self._tournament(pop), self._tournament(pop)

    # ---- Reproduction ----

    def _reproduce(self, task_description: str, p1: Individual, p2: Individual) -> JSONDict:
        """
        Create a child from two parents using crossover or mutation:

        - With probability `crossover_rate`: LLM crossover on (p1, p2).
        - Else: LLM mutation on one randomly chosen parent.

        Returns
        -------
        Dict[str, Any]
            Child procedure JSON, repaired and renumbered.
        """
        r = self.rng.random()
        if r < self.cfg.crossover_rate:
            child = self.crossover(task_description, p1.proc, p2.proc)
        else:
            base = p1 if self.rng.random() < 0.5 else p2
            child = self.mutate(base.proc, task_description)

        try:
            child = self.repair_fn(child, self.model)
        except Exception:
            pass
        return _renumber_steps(child)

    # ---- Run ----

    def run(
        self,
        task_description: str,
        final_answer_schema: Optional[Dict[str, Any]] = None,
        eval_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
        run_steps_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        print_progress: bool = False,
    ) -> Tuple[Individual, List[Individual]]:
        """
        Execute the full GA loop and return the best individual plus a per-generation
        history of elites.

        If `final_answer_schema`, `eval_fn`, and `run_steps_fn` are all provided,
        the GA uses task-eval scoring for that generation; otherwise it uses the
        structural hygiene scorer.

        Parameters
        ----------
        task_description
            Natural-language problem the procedures should solve.
        final_answer_schema
            JSON schema for the final step (required for TaskEvalScorer).
        eval_fn
            Callable `(state, proc) -> float` that grades an executed procedure (TaskEval).
        run_steps_fn
            Callable that executes a procedure and returns the final `state` dict.
        print_progress
            If True, prints generation-level fitness summaries.

        Returns
        -------
        (Individual, List[Individual])
            The final best individual and a list of per-generation elites.
        """
        pop = self.initialize_population(task_description)
        history: List[Individual] = []

        for gen in range(self.cfg.max_generations):
            # Choose scorer (task-eval if fully provided; else structural)
            if eval_fn and run_steps_fn and final_answer_schema is not None:
                from src.llm_procedure_generation_ga.scorers import TaskEvalScorer
                scorer: Scorer = TaskEvalScorer(
                    run_steps_fn=run_steps_fn,
                    eval_fn=eval_fn,
                    question=task_description,
                    final_answer_schema=final_answer_schema,
                    model=self.model,
                    strict_require_key="final_answer",
                )
            else:
                scorer = self.scorer

            self.evaluate(pop, scorer)
            pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
            best = copy.deepcopy(pop[0])
            history.append(best)
            if print_progress:
                print(f"[gen {gen+1}] best={best.fitness:.3f} steps={len(best.proc.get('steps', []))}")

            # Next generation scaffold
            next_pop: List[Individual] = [copy.deepcopy(e) for e in pop[: self.cfg.elitism]]

            # Diversity: random immigrants
            n_slots = self.cfg.population_size - len(next_pop)
            n_imm = int(max(0, n_slots) * self.cfg.random_immigrant_rate)
            for _ in range(n_imm):
                next_pop.append(Individual(self._generate_one(task_description)))

            # Fill remaining slots via reproduction
            while len(next_pop) < self.cfg.population_size:
                p1, p2 = self._select_parents(pop)
                child = self._reproduce(task_description, p1, p2)
                next_pop.append(Individual(proc=child))

            pop = next_pop

        # Final evaluate & return best
        self.evaluate(pop, scorer)
        pop.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
        return pop[0], history


if __name__ == "__main__":
    print("GA scaffold ready. Import into your environment that defines Procedure, query, etc.")