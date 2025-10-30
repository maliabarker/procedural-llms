"""
procedure_ga.py — Scaffolding for a genetic algorithm over LLM-generated procedures.

Overview
--------
This module gives you a production-ready *scaffold* (interfaces + reference ops) for
searching in *procedure space* using a genetic algorithm. Procedures are JSON objects
validated by a Pydantic schema; prompts instruct the LLM to emit only JSON that
validates against that schema. You plug in your own LLM client, validators, and scoring.

Key Pieces
----------
- LLMClient: abstract model interface (swap in OpenAI, Anthropic, local, etc.)
- PromptBuilder: zero-/few-shot builder that injects JSON schema constraints
- Procedure(Pydantic): canonical schema of a procedure graph with steps
- Validators: structural checks (schema, connectivity, final step, dangling refs)
- Scorer: pluggable fitness; CompositeScorer aggregates multiple criteria
- Genetic operators: crossover, merge, mutation (textual step rewrite via LLM)
- ProcedureGA: population init, evaluation, selection, reproduction, run loop

This is intentionally modular and typed so your editor (e.g., VS Code + Pylance) can
reason about it, and the Pydantic model exposes `model_json_schema()` for prompting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
import copy
import json
import random
import re
import uuid

# If you use Pydantic v2 (recommended):
from pydantic import BaseModel, Field, ValidationError


# ==========================
# 1) LLM model abstraction
# ==========================

class LLMClient(ABC):
    """Abstract interface for an LLM. Implement `generate` to return a string response."""

    def __init__(self, model: str, temperature: float = 0.2, **kwargs: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.extra = kwargs

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return raw model text output for `prompt`."""
        raise NotImplementedError


class DummyLLM(LLMClient):
    """A dummy client that returns a minimal, valid procedure for smoke tests."""

    def generate(self, prompt: str) -> str:
        proc = {
            "name": "baseline",
            "description": "Minimal procedure that echoes input and returns final.",
            "entrypoint": "s1",
            "metadata": {},
            "steps": {
                "s1": {
                    "id": "s1",
                    "kind": "action",
                    "instruction": "Read input and prepare summary.",
                    "inputs": {},
                    "outputs": {"summary": "ok"},
                    "next": ["s2"]
                },
                "s2": {
                    "id": "s2",
                    "kind": "final",
                    "instruction": "Return summary as final answer.",
                    "inputs": {"use": "summary"},
                    "outputs": {"final_answer": "ok"},
                    "next": []
                }
            }
        }
        return json.dumps(proc, ensure_ascii=False)


# ==========================
# 2) Procedure schema (Pydantic)
# ==========================

class ProcedureStep(BaseModel):
    id: str = Field(..., description="Unique step id (string).")
    kind: Literal["action", "compute", "reflect", "final"]
    instruction: str = Field(..., description="Natural language instruction for the step.")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Named inputs for this step.")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Named outputs produced by this step.")
    next: List[str] = Field(default_factory=list, description="List of downstream step ids. Empty if final.")

class Procedure(BaseModel):
    name: str
    description: str
    steps: Dict[str, ProcedureStep]
    entrypoint: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def ordered_steps(self) -> List[ProcedureStep]:
        """Topologically traverse from entrypoint; ignores cycles gracefully by visited set."""
        order: List[ProcedureStep] = []
        visited = set()
        stack = [self.entrypoint]
        while stack:
            sid = stack.pop(0)
            if sid in visited or sid not in self.steps:
                continue
            step = self.steps[sid]
            order.append(step)
            visited.add(sid)
            for nxt in step.next:
                if nxt not in visited:
                    stack.append(nxt)
        return order

    def has_final_step(self) -> bool:
        return any(st.kind == "final" for st in self.steps.values())

    def json_schema_for_prompt(self) -> str:
        """Compact JSON schema string for inclusion in prompts."""
        schema = self.model_json_schema()
        return json.dumps(schema, ensure_ascii=False)


# ==========================
# 3) Prompt building
# ==========================

@dataclass
class FewShotExample:
    user: str
    assistant_json_procedure: str  # MUST be JSON matching Procedure schema


class PromptBuilder:
    """
    Builds a strict prompt instructing the LLM to output ONLY JSON that validates
    against the Procedure schema. You can pass few-shot examples (already-valid JSON).
    """

    def __init__(
        self,
        system_instructions: str = (
            "You are a rigorous planner that ONLY outputs JSON objects matching the provided schema. "
            "No markdown, no commentary—return JSON only."
        ),
    ) -> None:
        self.system_instructions = system_instructions

    def build(
        self,
        task_description: str,
        schema_json: str,
        few_shots: Optional[List[FewShotExample]] = None,
        extra_constraints: Optional[str] = None,
    ) -> str:
        shots = ""
        if few_shots:
            for ex in few_shots:
                shots += (
                    "\n### Example\n"
                    f"User:\n{ex.user}\n"
                    "Assistant (JSON only):\n"
                    f"{ex.assistant_json_procedure}\n"
                )

        constraints = extra_constraints or (
            "Requirements:\n"
            "1) Output a single JSON object that validates against the schema.\n"
            "2) Use short, explicit step ids (e.g., s1, s2, s3...).\n"
            "3) Include exactly one 'final' step that returns 'final_answer' in outputs.\n"
            "4) Ensure all `next` references are valid and reachable from the entrypoint.\n"
            "5) Keep instructions concise and executable; avoid vague language.\n"
        )

        prompt = (
            f"{self.system_instructions}\n\n"
            "### Task\n"
            f"{task_description}\n\n"
            "### JSON Schema (Pydantic model)\n"
            f"{schema_json}\n\n"
            f"{constraints}\n"
            f"{shots}\n"
            "Assistant: "
        )
        return prompt


# ==========================
# 4) Validators
# ==========================

class Validator(ABC):
    name: str

    @abstractmethod
    def validate(self, proc: Procedure) -> Tuple[bool, str]:
        ...


class SchemaValidator(Validator):
    name = "schema"

    def validate(self, proc: Procedure) -> Tuple[bool, str]:
        try:
            Procedure.model_validate(proc.model_dump())
            return True, "OK"
        except ValidationError as e:
            return False, f"ValidationError: {e}"


class ConnectivityValidator(Validator):
    name = "connectivity"

    def validate(self, proc: Procedure) -> Tuple[bool, str]:
        seen = {s.id for s in proc.ordered_steps()}
        declared = set(proc.steps.keys())
        unreachable = declared - seen
        if unreachable:
            return False, f"Unreachable steps: {sorted(unreachable)}"
        return True, "OK"


class DanglingRefsValidator(Validator):
    name = "dangling_refs"

    def validate(self, proc: Procedure) -> Tuple[bool, str]:
        all_ids = set(proc.steps.keys())
        bad: List[Tuple[str, str]] = []
        for s in proc.steps.values():
            for nxt in s.next:
                if nxt not in all_ids:
                    bad.append((s.id, nxt))
        if bad:
            return False, f"Dangling references: {bad}"
        return True, "OK"


class FinalStepValidator(Validator):
    name = "final_step"

    def validate(self, proc: Procedure) -> Tuple[bool, str]:
        finals = [s for s in proc.steps.values() if s.kind == "final"]
        if len(finals) != 1:
            return False, f"Expected exactly one final step; found {len(finals)}"
        final = finals[0]
        if "final_answer" not in final.outputs:
            return False, "Final step must set outputs.final_answer"
        return True, "OK"


# ==========================
# 5) Scoring
# ==========================

class Scorer(ABC):
    """Return a *higher-is-better* fitness score."""

    @abstractmethod
    def score(self, proc: Procedure, input_data: Any) -> float:
        ...


class StructuralHeuristicScorer(Scorer):
    """
    Simple default scorer to get you started:
    - +1.0 if valid schema/connectivity/final
    - penalize too many steps
    You should replace with task-specific evaluators (unit tests, accuracy, etc.).
    """

    def __init__(self, validators: List[Validator], target_len: int = 6) -> None:
        self.validators = validators
        self.target_len = target_len

    def score(self, proc: Procedure, input_data: Any) -> float:
        score = 0.0
        # Add points for passing validators
        for v in self.validators:
            ok, _ = v.validate(proc)
            if ok:
                score += 0.25
        # Soft penalty for size
        n = len(proc.steps)
        score += max(0.0, 1.0 - abs(n - self.target_len) * 0.1)
        return score


# ==========================
# 6) Genetic operators
# ==========================

def _new_step_id(existing: Dict[str, ProcedureStep]) -> str:
    i = 1
    while True:
        sid = f"s{i}"
        if sid not in existing:
            return sid
        i += 1


class CrossoverOperator:
    """
    Combine subgraphs from two parents. Strategy:
    - Take prefix from parent A starting at entrypoint (k steps)
    - Append a tail from parent B (m steps), reconnecting last of A to entry of tail
    - Renumber step ids to avoid collisions
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, a: Procedure, b: Procedure) -> Procedure:
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)

        a_order = a_copy.ordered_steps()
        b_order = b_copy.ordered_steps()
        if not a_order or not b_order:
            return a_copy

        k = max(1, min(len(a_order) - 1, self.rng.randint(1, len(a_order) - 1)))
        m = max(1, min(len(b_order), self.rng.randint(1, len(b_order))))

        # Prefix from A
        keep_ids = {st.id for st in a_order[:k]}
        new_steps: Dict[str, ProcedureStep] = {
            sid: copy.deepcopy(st) for sid, st in a_copy.steps.items() if sid in keep_ids
        }

        # Tail from B (renumber and collect)
        id_map: Dict[str, str] = {}
        for st in b_order[:m]:
            nsid = _new_step_id(new_steps)
            id_map[st.id] = nsid
            new_steps[nsid] = ProcedureStep.model_validate(
                {**st.model_dump(), "id": nsid, "next": []}
            )

        # Rewire A's last kept step to start of tail (first in b_order)
        a_tail_attach = next(reversed([s for s in a_order[:k]]), None)
        if a_tail_attach and b_order:
            first_tail_id = id_map[b_order[0].id]
            # ensure no duplicate in next
            nxts = set(a_tail_attach.next)
            nxts.add(first_tail_id)
            new_steps[a_tail_attach.id].next = list(nxts)

        # Rewire `next` inside the copied B tail
        for st in b_order[:m]:
            mapped = new_steps[id_map[st.id]]
            mapped.next = [id_map[n] for n in st.next if n in id_map]

        # Ensure a single final
        # If both sides had finals, let the last tail contain final, or convert earlier finals to 'action'
        finals = [s for s in new_steps.values() if s.kind == "final"]
        if len(finals) == 0:
            # convert the last node into final
            last = list(new_steps.values())[-1]
            last.kind = "final"
            last.outputs["final_answer"] = last.outputs.get("final_answer", "ok")
            last.next = []
        elif len(finals) > 1:
            # keep only the last as final
            keep = finals[-1].id
            for s in finals[:-1]:
                s.kind = "action"
                s.outputs.pop("final_answer", None)

        return Procedure(
            name=f"crossover({a.name},{b.name})",
            description="Crossover child procedure.",
            steps=new_steps,
            entrypoint=a_copy.entrypoint if a_copy.entrypoint in new_steps else next(iter(new_steps)),
            metadata={"parents": [a.name, b.name], "op": "crossover"},
        )


class MergeOperator:
    """
    Merge two procedures into a meta-procedure with a selector front step that can
    choose between subflows (e.g., by heuristic). Useful as an ensemble.
    """

    def __call__(self, a: Procedure, b: Procedure) -> Procedure:
        steps: Dict[str, ProcedureStep] = {}

        # Copy A with new ids
        amap: Dict[str, str] = {}
        for st in a.ordered_steps():
            nid = _new_step_id(steps)
            amap[st.id] = nid
            steps[nid] = ProcedureStep.model_validate({**st.model_dump(), "id": nid, "next": []})
        # Fix next
        for old, new in amap.items():
            steps[new].next = [amap[n] for n in a.steps[old].next if n in amap]

        # Copy B with new ids
        bmap: Dict[str, str] = {}
        for st in b.ordered_steps():
            nid = _new_step_id(steps)
            bmap[st.id] = nid
            steps[nid] = ProcedureStep.model_validate({**st.model_dump(), "id": nid, "next": []})
        for old, new in bmap.items():
            steps[new].next = [bmap[n] for n in b.steps[old].next if n in bmap]

        # Add selector entry
        entry_id = _new_step_id(steps)
        steps[entry_id] = ProcedureStep(
            id=entry_id,
            kind="action",
            instruction="Selector: choose subflow A or B based on input heuristics.",
            inputs={},
            outputs={"route": "A"},  # placeholder
            next=[amap[a.entrypoint], bmap[b.entrypoint]],
        )

        return Procedure(
            name=f"merge({a.name},{b.name})",
            description="Merged ensemble procedure with selector front step.",
            steps=steps,
            entrypoint=entry_id,
            metadata={"parents": [a.name, b.name], "op": "merge"},
        )


class MutationOperator:
    """
    Mutate a single step:
    - With probability, rewrite its instruction via LLM (paraphrase/refine)
    - Or tweak graph structure (add/remove a simple passthrough step)
    """

    def __init__(self, llm: Optional[LLMClient] = None, text_rate: float = 0.7, rng: Optional[random.Random] = None) -> None:
        self.llm = llm
        self.text_rate = text_rate
        self.rng = rng or random.Random()

    def __call__(self, proc: Procedure) -> Procedure:
        child = copy.deepcopy(proc)
        if not child.steps:
            return child

        step = self.rng.choice(list(child.steps.values()))

        if self.llm and self.rng.random() < self.text_rate:
            prompt = (
                "Rewrite the following instruction to be more concrete, concise, and executable. "
                "Return ONLY the rewritten instruction text, no quotes.\n\n"
                f"Instruction:\n{step.instruction}\n\nRewritten:"
            )
            try:
                new_text = self.llm.generate(prompt).strip()
                # Strip potential quotes
                new_text = new_text.strip().strip('"').strip("'")
                if new_text:
                    step.instruction = new_text
            except Exception:
                pass
        else:
            # simple structural tweak: if not final, insert a passthrough after it
            if step.kind != "final":
                new_id = _new_step_id(child.steps)
                passthrough = ProcedureStep(
                    id=new_id,
                    kind="action",
                    instruction="Passthrough: verify inputs and forward.",
                    inputs={},
                    outputs={},
                    next=step.next[:],
                )
                step.next = list({*step.next, new_id})
                child.steps[new_id] = passthrough

        return child


# ==========================
# 7) Parsing helpers
# ==========================

def extract_json(text: str) -> str:
    """
    Try to extract the first JSON object from text. Useful when an LLM
    ignores 'JSON only' and adds prose. Not bulletproof, but pragmatic.
    """
    # Find the first { ... } block using a stack
    start_indices = [i for i, c in enumerate(text) if c == "{"]
    for start in start_indices:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        pass
    # fallback: strip markdown fences
    fenced = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE)
    return fenced


def proc_from_llm_json(raw: str) -> Procedure:
    payload = extract_json(raw)
    data = json.loads(payload)
    return Procedure.model_validate(data)


# ==========================
# 8) GA core
# ==========================

@dataclass
class GAConfig:
    population_size: int = 16
    elitism: int = 2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    max_generations: int = 20
    tournament_k: int = 3
    seed: Optional[int] = None


class Individual(BaseModel):
    proc: Procedure
    fitness: Optional[float] = None
    validation_report: Dict[str, Tuple[bool, str]] = Field(default_factory=dict)


class ProcedureGA:
    def __init__(
        self,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        scorer: Scorer,
        validators: List[Validator],
        cfg: GAConfig = GAConfig(),
        crossover: Optional[CrossoverOperator] = None,
        mergeop: Optional[MergeOperator] = None,
        mutate: Optional[MutationOperator] = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.scorer = scorer
        self.validators = validators
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.crossover = crossover or CrossoverOperator(self.rng)
        self.mergeop = mergeop or MergeOperator()
        self.mutate = mutate or MutationOperator(llm=self.llm, rng=self.rng)

    def _prompt_for_procedure(self, task_description: str, few_shots: Optional[List[FewShotExample]]) -> str:
        # Use an *empty* dummy instance to get schema; better: use Procedure.model_json_schema()
        schema_json = Procedure.model_json_schema_json(indent=None)
        return self.prompt_builder.build(
            task_description=task_description,
            schema_json=schema_json,
            few_shots=few_shots,
        )

    def initialize_population(
        self, task_description: str, n: Optional[int] = None, few_shots: Optional[List[FewShotExample]] = None
    ) -> List[Individual]:
        n = n or self.cfg.population_size
        pop: List[Individual] = []
        prompt = self._prompt_for_procedure(task_description, few_shots)
        for _ in range(n):
            raw = self.llm.generate(prompt)
            try:
                proc = proc_from_llm_json(raw)
            except Exception:
                # fall back to dummy if malformed
                proc = proc_from_llm_json(self.llm.generate(prompt))
            pop.append(Individual(proc=proc))
        return pop

    def _validate(self, proc: Procedure) -> Dict[str, Tuple[bool, str]]:
        return {v.name: v.validate(proc) for v in self.validators}

    def evaluate(self, pop: List[Individual], input_data: Any) -> None:
        for ind in pop:
            ind.validation_report = self._validate(ind.proc)
            # If any hard failure, penalize
            if not all(ok for ok, _ in ind.validation_report.values()):
                # small floor to keep selection pressure but not zero
                ind.fitness = -1.0
            else:
                ind.fitness = self.scorer.score(ind.proc, input_data)

    def _tournament(self, pop: List[Individual]) -> Individual:
        group = self.rng.sample(pop, k=min(self.cfg.tournament_k, len(pop)))
        return max(group, key=lambda i: i.fitness or -1e9)

    def _select_parents(self, pop: List[Individual]) -> Tuple[Individual, Individual]:
        return self._tournament(pop), self._tournament(pop)

    def _reproduce(self, p1: Individual, p2: Individual) -> Procedure:
        r = self.rng.random()
        if r < self.cfg.crossover_rate:
            return self.crossover(p1.proc, p2.proc)
        elif r < self.cfg.crossover_rate + self.cfg.mutation_rate:
            return self.mutate(p1.proc)
        else:
            # fallback: merge as ensemble child
            return self.mergeop(p1.proc, p2.proc)

    def run(
        self,
        task_description: str,
        input_data: Any,
        few_shots: Optional[List[FewShotExample]] = None,
    ) -> Tuple[Individual, List[Individual]]:
        # 1) init
        population = self.initialize_population(task_description, self.cfg.population_size, few_shots)
        history: List[Individual] = []

        for gen in range(self.cfg.max_generations):
            # 2) evaluate
            self.evaluate(population, input_data)
            population.sort(key=lambda i: i.fitness or -1e9, reverse=True)
            best = population[0]
            history.append(copy.deepcopy(best))

            # 3) elitism
            next_gen: List[Individual] = [copy.deepcopy(e) for e in population[: self.cfg.elitism]]

            # 4) breed
            while len(next_gen) < self.cfg.population_size:
                p1, p2 = self._select_parents(population)
                child_proc = self._reproduce(p1, p2)
                next_gen.append(Individual(proc=child_proc))

            population = next_gen

        # final evaluate + return best
        self.evaluate(population, input_data)
        population.sort(key=lambda i: i.fitness or -1e9, reverse=True)
        best = population[0]
        return best, history


# ==========================
# 9) Quick-start (local smoke test)
# ==========================

if __name__ == "__main__":
    # Smoke test with DummyLLM (no external calls)
    llm = DummyLLM(model="dummy")
    pb = PromptBuilder()
    validators = [SchemaValidator(), ConnectivityValidator(), DanglingRefsValidator(), FinalStepValidator()]
    scorer = StructuralHeuristicScorer(validators=validators, target_len=4)
    ga = ProcedureGA(llm, pb, scorer, validators, cfg=GAConfig(population_size=6, max_generations=3, seed=42))

    best, hist = ga.run(task_description="Create a procedure to read a problem and output a final answer.", input_data=None)
    print("Best fitness:", best.fitness)
    print("Best name:", best.proc.name)
    print("Steps:", list(best.proc.steps.keys()))
