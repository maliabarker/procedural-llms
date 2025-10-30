# tests/test_ga_scaffold_structured.py
import json
import types
import random
import pytest

from src.ga_scaffold_structured import (
    ProcedureGA,
    GAConfig,
    Individual,
)

# ---------- helpers: minimal schema + minimal valid procs ----------

def proc_schema():
    # The GA doesn't jsonschema-validate; this is just passed through to your query_fn.
    # Keep it simple; shape matches your Pydantic-derived schema keys.
    return {
        "type": "object",
        "properties": {
            "NameDescription": {"type": "string"},
            "steps": {"type": "array"},
        },
        "required": ["NameDescription", "steps"],
        "additionalProperties": True,
    }

def minimal_proc(name="base", n_steps=3):
    """
    Build a minimal *global-state* valid procedure:
      - step 1: inputs == ["problem_text"], outputs ["facts"]
      - step 2: reads "facts", outputs ["derived"]
      - step 3: reads "derived", outputs ["final_answer"]
    """
    return {
        "NameDescription": name,
        "steps": [
            {
                "id": 1,
                "inputs": [{"name": "problem_text", "description": "raw problem text"}],
                "stepDescription": "Extract primitive facts needed to solve the problem.",
                "output": [{"name": "facts", "description": "structured facts"}],
            },
            {
                "id": 2,
                "inputs": [{"name": "facts", "description": "structured facts"}],
                "stepDescription": "Transform facts to intermediate representation.",
                "output": [{"name": "derived", "description": "intermediate"}],
            },
            {
                "id": 3,
                "inputs": [{"name": "derived", "description": "intermediate"}],
                "stepDescription": "Describe the final answer without computing it.",
                "output": [{"name": "final_answer", "description": "the final problem answer (description only)"}],
            },
        ],
    }

# ---------- fakes / stubs for GA wiring ----------

class CapturingQuery:
    """Fake query_fn that returns a canned valid procedure and captures the last prompt."""
    def __init__(self, template="child"):
        self.last_prompt = None
        self.template = template
        self.calls = 0

    def __call__(self, prompt: str, model: str, fmt=None, seed=None) -> str:
        self.calls += 1
        self.last_prompt = prompt
        # Always return a valid JSON string for a procedure
        return json.dumps(minimal_proc(self.template))

def fake_repair(proc_json, model):
    # Identity "repair": guarantee step ids are 1..n
    proc = json.loads(json.dumps(proc_json))
    for i, s in enumerate(proc.get("steps", []), start=1):
        s["id"] = i
    return proc

def fake_validate_ok(proc_json):
    # No diagnostics (structurally OK)
    return []

def fake_validate_step1_and_final(proc_json):
    diags = []
    steps = proc_json.get("steps", [])
    if not steps:
        diags.append({"severity": "fatal", "message": "no steps"})
        return diags
    # step 1 rule
    s1_inputs = [d["name"] for d in steps[0].get("inputs", [])]
    if s1_inputs != ["problem_text"]:
        diags.append({"severity": "fatal", "message": "bad step1 inputs"})
    # final step rule
    final_outputs = [d["name"] for d in steps[-1].get("output", [])]
    if final_outputs != ["final_answer"]:
        diags.append({"severity": "fatal", "message": "bad final outputs"})
    return diags

def create_proc_fn(task_description: str) -> str:
    # Your generator prompt—unused by our fake query, but GA requires it.
    return f"CREATE_PROCEDURE for: {task_description}"

def run_steps_fn(proc_json, question, final_answer_schema, model, print_bool=False):
    # Return a state that includes final_answer to satisfy strict_require_key
    return {"final_answer": "some description", "other": 123}

def eval_fn(state, proc_json) -> float:
    # Simple grading: reward having final_answer
    return 0.8 if "final_answer" in state else -1.0


# ---------- fixtures ----------

@pytest.fixture
def rng():
    return random.Random(1234)

@pytest.fixture
def schema_json_fn():
    return proc_schema

@pytest.fixture
def capturing_query():
    return CapturingQuery(template="offspring")

# ---------- tests ----------

def test_initialize_population_creates_valid_procs(schema_json_fn, capturing_query, rng):
    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=capturing_query,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_ok,
        repair_fn=fake_repair,
        # default: structural scorer (adapter) is created inside GA
        cfg=GAConfig(population_size=4, seed=42),
        rng=rng,
    )
    pop = ga.initialize_population("Add two numbers")
    assert len(pop) == 4
    for ind in pop:
        assert isinstance(ind.proc, dict)
        steps = ind.proc["steps"]
        # ids are 1..n
        assert [s["id"] for s in steps] == list(range(1, len(steps) + 1))
        # step 1 + final step sanity
        assert [d["name"] for d in steps[0]["inputs"]] == ["problem_text"]
        assert [d["name"] for d in steps[-1]["output"]] == ["final_answer"]

def test_reproduce_uses_crossover_when_prob_triggers(schema_json_fn, rng):
    cq = CapturingQuery(template="xover-child")
    # Force crossover path: crossover_rate=1.0
    cfg = GAConfig(population_size=2, crossover_rate=1.0, mutation_rate=0.0, seed=99)
    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=cq,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_step1_and_final,
        repair_fn=fake_repair,
        cfg=cfg,
        rng=rng,
    )
    p1 = Individual(minimal_proc("A"))
    p2 = Individual(minimal_proc("B"))
    child = ga._reproduce("Task text", p1, p2)
    assert isinstance(child, dict)
    assert "steps" in child
    # Ensure the crossover prompt path was used
    assert "Crossover Objective" in cq.last_prompt or "Synthesize a SINGLE crossover child" in cq.last_prompt

def test_reproduce_uses_mutation_when_no_crossover(schema_json_fn, rng):
    cq = CapturingQuery(template="mut-child")
    # Force mutation path: crossover_rate=0.0
    cfg = GAConfig(population_size=2, crossover_rate=0.0, mutation_rate=1.0, seed=101)
    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=cq,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_step1_and_final,
        repair_fn=fake_repair,
        cfg=cfg,
        rng=rng,
    )
    p1 = Individual(minimal_proc("A"))
    p2 = Individual(minimal_proc("B"))
    child = ga._reproduce("Task text", p1, p2)
    assert isinstance(child, dict)
    # Mutation prompt contains "Mutation Goal"
    assert "Mutation Goal" in cq.last_prompt

def test_run_with_structural_scoring_path(schema_json_fn, rng, capturing_query):
    # Structural path (no task-eval args provided)
    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=capturing_query,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_ok,
        repair_fn=fake_repair,
        cfg=GAConfig(population_size=4, max_generations=3, seed=7),
        rng=rng,
    )
    best, history = ga.run(
        task_description="Sum to 10",
        final_answer_schema=None,
        eval_fn=None,
        run_steps_fn=None,
        print_progress=False,
    )
    assert isinstance(best, Individual)
    assert len(history) == 3
    # Fitness should be a float (adapter/structural scorer is inside GA by default)
    assert isinstance(best.fitness, float)

def test_run_with_task_eval_scorer_path(schema_json_fn, rng, capturing_query):
    # Task-eval path (all three args supplied)
    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=capturing_query,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_ok,
        repair_fn=fake_repair,
        cfg=GAConfig(population_size=3, max_generations=2, seed=11),
        rng=rng,
    )
    best, history = ga.run(
        task_description="Find X",
        final_answer_schema={"type": "object", "properties": {}, "additionalProperties": True},
        eval_fn=eval_fn,
        run_steps_fn=run_steps_fn,
        print_progress=False,
    )
    assert isinstance(best, Individual)
    # Our eval_fn returns 0.8 when final_answer present
    assert best.fitness == pytest.approx(0.8, rel=1e-6)
    assert len(history) == 2

def test_random_immigrants_are_injected(schema_json_fn, rng):
    cq = CapturingQuery(template="immigrant")
    # We'll monkeypatch _generate_one to count immigrant creation calls
    immigrant_counter = {"calls": 0}

    def counting_generate_one(self, task_description: str):
        immigrant_counter["calls"] += 1
        return minimal_proc("immigrant")

    ga = ProcedureGA(
        model="gemma3",
        create_proc_fn=create_proc_fn,
        query_fn=cq,
        schema_json_fn=schema_json_fn,
        validate_fn=fake_validate_ok,
        repair_fn=fake_repair,
        cfg=GAConfig(population_size=6, elitism=2, random_immigrant_rate=0.5, max_generations=2, seed=17),
        rng=rng,
    )

    # Patch method
    ga._generate_one = types.MethodType(counting_generate_one, ga)

    # Initialize once (this will call _generate_one population_size times)
    _ = ga.initialize_population("Immigrant test")
    pre_calls = immigrant_counter["calls"]

    # Run one generation; random_immigrant_rate=0.5 should add immigrants each gen
    _best, _hist = ga.run(
        task_description="Immigrant test",
        final_answer_schema=None, eval_fn=None, run_steps_fn=None, print_progress=False,
    )
    post_calls = immigrant_counter["calls"]

    # We expect _generate_one to be called additional times during run() for immigrants.
    assert post_calls > pre_calls
