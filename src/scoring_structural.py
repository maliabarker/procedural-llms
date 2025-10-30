# scoring_structural.py
from __future__ import annotations
from typing import Any, Dict, List, Callable
from math import exp

# Expect these helper imports in your env
# from src.old_modules.helpers import _names
# from validators_global import validate_procedure_structured  # your module

JSONDict = Dict[str, Any]
Validator = Callable[[JSONDict], List[Dict[str, Any]]]

class StructuralHygieneScorer:
    """
    Structural hygiene scorer for global-state procedures.

    Components (higher is better; starts at `base`):
      - Validator penalties:
          * fatal diagnostics:  -w_fatal each
          * repairable diags:   -w_repair each
      - Redefinition penalty:   -w_redefine * (# vars redefined)
      - Unused outputs penalty: -w_unused  * (# unused outputs)
      - Soft length cap:        -w_len * sigmoid(max(0, n_steps - target_steps))
      - Extraction-first reward:+w_extract if step 1 looks like an extraction

    Configure weights to taste.
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

    def _count_redefinitions(self, p: JSONDict) -> int:
        seen = set()
        redefs = 0
        for s in p["steps"]:
            for v in _names(s["output"]):
                if v == "final_answer":
                    continue
                if v in seen:
                    redefs += 1
                seen.add(v)
        return redefs

    def _count_unused_outputs(self, p: JSONDict) -> int:
        # Build the set of all inputs used after a step index
        steps = p["steps"]
        n = len(steps)
        future = set()
        unused_total = 0
        # Walk backward so earlier outputs see all later inputs as "future needs"
        for i in range(n - 1, -1, -1):
            cur_inputs = set(_names(steps[i]["inputs"]))
            # outputs at i that never appear in any later inputs
            for v in _names(steps[i]["output"]):
                if v in {"final_answer", "problem_text"}:
                    continue
                if v not in future:
                    unused_total += 1
            # update "future" for next iteration (earlier step)
            future |= cur_inputs
        return unused_total

    def _looks_extraction_first(self, p: JSONDict) -> bool:
        if not p["steps"]:
            return False
        s1 = p["steps"][0]["stepDescription"].lower()
        # lightweight heuristic; adjust to your style (e.g., include "parse", "identify", etc.)
        return any(tok in s1 for tok in ("extract", "read", "gather", "identify"))

    def score(self, p: JSONDict) -> float:
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
        n = len(p["steps"])
        excess = max(0, n - self.target_steps)
        # smooth penalty: saturates as excess grows
        score -= self.w_len * (1 / (1 + exp(-0.7 * excess)) - 0.5) * 2  # range ~[0, w_len]

        # 5) extraction-first reward
        if self._looks_extraction_first(p):
            score += self.w_extract

        return float(score)
    
# Example usage: Wire it in your GA by passing scorer=StructuralHygieneScorer(validate_procedure_structured, ...) and using ind.fitness = scorer.score(ind.proc).