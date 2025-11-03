# scorers.py
"""
Scorers for procedure evolution.

- StructuralHygieneScorer: scores a *procedure JSON* using validators + structure heuristics.
- TaskEvalScorer: runs a procedure end-to-end and scores with a user-provided eval_fn.
- ProcScorerAdapter: adapts any "proc scorer" (that scores JSON) to GA-style "individual" scoring.

Usage with GA
-------------
from src.scoring import StructuralHygieneScorer, ProcScorerAdapter
from src.validators import validate_procedure_structured

proc_scorer = StructuralHygieneScorer(validate_fn=validate_procedure_structured)
scorer = ProcScorerAdapter(proc_scorer)   # GA expects scorer.score(individual)
"""
from __future__ import annotations

from math import exp
from typing import Any, Callable, Dict, List, Optional, Protocol

from llm_procedure_generation_ga.helpers import _names

JSONDict = Dict[str, Any]
Validator = Callable[[JSONDict], List[Dict[str, Any]]]


class HasProc(Protocol):
    """Minimal protocol for GA 'individual'-like objects."""
    proc: JSONDict


class Scorer(Protocol):
    """GA-facing scorer protocol."""
    def score(self, ind: HasProc, **kwargs: Any) -> float: ...


class ProcScorer(Protocol):
    """Scores a raw procedure JSON (not an Individual)."""
    def score_proc(self, p: JSONDict) -> float: ...


class ProcScorerAdapter(Scorer):
    """
    Adapter that lets a procedure-level scorer be used by GA code that calls
    `scorer.score(individual)`.
    """
    def __init__(self, proc_scorer: ProcScorer) -> None:
        self._proc_scorer = proc_scorer

    def score(self, ind: HasProc, **kwargs: Any) -> float:
        return self._proc_scorer.score_proc(ind.proc)


class StructuralHygieneScorer:
    """
    Structural hygiene scorer for global-state procedures (scores *procedure JSON*).

    Components (higher is better; starts at `base`):
      - Validator penalties:
          * fatal diagnostics:  -w_fatal each
          * repairable diags:   -w_repair each
      - Redefinition penalty:   -w_redefine * (# vars redefined)
      - Unused outputs penalty: -w_unused  * (# unused outputs)
      - Soft length cap:        -w_len * sigmoid(max(0, n_steps - target_steps))
      - Extraction-first reward:+w_extract if step 1 looks like an extraction
    """

    def __init__(
        self,
        validate_fn: Validator,
        *,
        base: float = 1.0,
        w_fatal: float = 1.0,
        w_repair: float = 0.2,
        w_redefine: float = 0.25,
        w_unused: float = 0.25,
        w_len: float = 0.3,
        target_steps: int = 6,
        w_extract: float = 0.25,
    ) -> None:
        self.validate_fn = validate_fn
        self.base = base
        self.w_fatal = w_fatal
        self.w_repair = w_repair
        self.w_redefine = w_redefine
        self.w_unused = w_unused
        self.w_len = w_len
        self.target_steps = target_steps
        self.w_extract = w_extract

    # ---- sub-metrics ---------------------------------------------------------

    def _count_redefinitions(self, p: JSONDict) -> int:
        seen = set()
        redefs = 0
        for s in p.get("steps", []):
            for v in _names(s.get("output", [])):
                if v == "final_answer":
                    continue
                if v in seen:
                    redefs += 1
                seen.add(v)
        return redefs

    def _count_unused_outputs(self, p: JSONDict) -> int:
        """Count outputs that never appear in any later step's inputs."""
        steps = p.get("steps", [])
        n = len(steps)
        future_inputs: set[str] = set()
        unused_total = 0
        for i in range(n - 1, -1, -1):
            cur_inputs = set(_names(steps[i].get("inputs", [])))
            # outputs at i that never appear later
            for v in _names(steps[i].get("output", [])):
                if v in {"final_answer", "problem_text"}:
                    continue
                if v not in future_inputs:
                    unused_total += 1
            # add inputs seen at/after this step so earlier outputs see them as "future needs"
            future_inputs |= cur_inputs
        return unused_total

    def _looks_extraction_first(self, p: JSONDict) -> bool:
        if not p.get("steps"):
            return False
        s1 = str(p["steps"][0].get("stepDescription", "")).lower()
        return any(tok in s1 for tok in ("extract", "read", "gather", "identify", "parse"))

    # ---- public API ----------------------------------------------------------

    def score_proc(self, p: JSONDict) -> float:
        """Return a scalar fitness for a procedure JSON."""
        score = self.base

        # 1) validator penalties
        diags = self.validate_fn(p)
        fatal = sum(1 for d in diags if d.get("severity") == "fatal")
        repair = sum(1 for d in diags if d.get("severity") == "repairable")
        score -= self.w_fatal * fatal
        score -= self.w_repair * repair

        # 2) redefinitions
        score -= self.w_redefine * self._count_redefinitions(p)

        # 3) unused outputs
        score -= self.w_unused * self._count_unused_outputs(p)

        # 4) soft length cap
        n = len(p.get("steps", []))
        excess = max(0, n - self.target_steps)
        score -= self.w_len * (1 / (1 + exp(-0.7 * excess)) - 0.5) * 2  # ~[0, w_len]

        # 5) extraction-first reward
        if self._looks_extraction_first(p):
            score += self.w_extract

        return float(score)


class TaskEvalScorer(Scorer):
    """
    Execute a procedure (via `run_steps_fn`) and grade with a user-provided
    `eval_fn(state, proc) -> float`. Expects GA to call `score(individual)`.

    Arguments
    ---------
    run_steps_fn
        Callable that executes the procedure over the question with the given schema/model
        and returns a final `state` dict (e.g., your `run_steps`).
    eval_fn
        Callable `(state, proc) -> float` returning a scalar fitness.
    question
        The task prompt to run.
    final_answer_schema
        JSON schema passed to the last step runner.
    model
        LLM name for execution.
    strict_require_key
        If set, returns -1.0 when the key is missing from `state`.
    """

    def __init__(
        self,
        run_steps_fn: Callable[[JSONDict, str, Dict[str, Any], str], Dict[str, Any]],
        eval_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
        question: str,
        final_answer_schema: Dict[str, Any],
        model: str,
        strict_require_key: Optional[str] = None,
    ) -> None:
        self.run_steps_fn = run_steps_fn
        self.eval_fn = eval_fn
        self.question = question
        self.final_answer_schema = final_answer_schema
        self.model = model
        self.strict_require_key = strict_require_key

    def score(self, ind: HasProc, **kwargs: Any) -> float:
        try:
            state = self.run_steps_fn(
                ind.proc, self.question, self.final_answer_schema, self.model,  # type: ignore[arg-type]
            )
            if self.strict_require_key and self.strict_require_key not in state:
                return -1.0
            return float(self.eval_fn(state, ind.proc))
        except Exception:
            return -1.0


__all__ = [
    "JSONDict",
    "Validator",
    "HasProc",
    "Scorer",
    "ProcScorer",
    "ProcScorerAdapter",
    "StructuralHygieneScorer",
    "TaskEvalScorer",
]