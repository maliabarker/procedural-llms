# tests/test_scoring.py
import math
import types
import pytest

from evoproc.scorers import (
    StructuralHygieneScorer,
    ProcScorerAdapter,
    TaskEvalScorer,
)

# -----------------------------
# Helpers to build procedure JSON
# -----------------------------

def mk_step(i, inputs, desc, outputs):
    return {
        "id": i,
        "inputs": [{"name": n, "description": ""} for n in inputs],
        "stepDescription": desc,
        "output": [{"name": n, "description": ""} for n in outputs],
    }

def mk_proc(steps):
    return {"NameDescription": "test proc", "steps": steps}


# -----------------------------
# StructuralHygieneScorer tests
# -----------------------------

def test_structural_base_and_extraction_reward():
    # No diagnostics; extraction keyword in step 1
    fake_validate = lambda p: []
    scorer = StructuralHygieneScorer(
        validate_fn=fake_validate,
        base=1.0, w_fatal=1.0, w_repair=0.2,
        w_redefine=0.25, w_unused=0.25,
        w_len=0.3, target_steps=6, w_extract=0.25,
    )

    proc = mk_proc([
        mk_step(1, ["problem_text"], "Extract primitive facts", ["facts"]),
        mk_step(2, ["facts"], "Summarize", ["final_answer"]),
    ])

    score = scorer.score_proc(proc)
    # No penalties, + extraction reward
    assert score == pytest.approx(1.0 + 0.25, rel=1e-6)


def test_structural_penalties_and_counts():
    # Validator emits one fatal and two repairables
    fake_validate = lambda p: [
        {"severity": "fatal"},
        {"severity": "repairable"},
        {"severity": "repairable"},
    ]
    # Weights chosen to make expected score easy to compute
    scorer = StructuralHygieneScorer(
        validate_fn=fake_validate,
        base=1.0, w_fatal=1.0, w_repair=0.2,
        w_redefine=0.5, w_unused=0.75,
        w_len=0.0,  # ignore length here
        target_steps=6, w_extract=0.0,  # ignore extraction here
    )

    # Redefinition: 'x' produced twice (step 1 and step 2)
    # Unused output: 'dead1' never consumed later
    proc = mk_proc([
        mk_step(1, ["problem_text"], "extract", ["x", "dead1"]),
        mk_step(2, ["x"], "rewrite", ["x"]),             # redefinition of 'x'
        mk_step(3, ["x"], "finalize", ["final_answer"]),
    ])

    score = scorer.score_proc(proc)
    # expected = base - (1*fatal) - (2*repair) - (1*redefine*w) - (1*unused*w)
    expected = 1.0 - 1.0 - (2 * 0.2) - (1 * 0.5) - (1 * 0.75)
    assert score == pytest.approx(expected, rel=1e-6)


def test_structural_soft_length_cap_saturates():
    fake_validate = lambda p: []
    # Make the length penalty "visible"
    scorer = StructuralHygieneScorer(
        validate_fn=fake_validate,
        base=1.0, w_len=1.0, target_steps=2,
        w_fatal=0.0, w_repair=0.0, w_redefine=0.0, w_unused=0.0, w_extract=0.0,
    )

    # n=2 (at target): no penalty
    proc_short = mk_proc([
        mk_step(1, ["problem_text"], "extract", ["a"]),
        mk_step(2, ["a"], "final", ["final_answer"]),
    ])
    s_short = scorer.score_proc(proc_short)
    assert s_short == pytest.approx(1.0, rel=1e-6)

    # n=10 (excess=8): penalty ~ approaches 1.0 with this sigmoid
    proc_long = mk_proc([
        mk_step(1, ["problem_text"], "extract", ["a"]),
        mk_step(2, ["a"], "s", ["b"]),
        mk_step(3, ["b"], "s", ["c"]),
        mk_step(4, ["c"], "s", ["d"]),
        mk_step(5, ["d"], "s", ["e"]),
        mk_step(6, ["e"], "s", ["f"]),
        mk_step(7, ["f"], "s", ["g"]),
        mk_step(8, ["g"], "s", ["h"]),
        mk_step(9, ["h"], "s", ["i"]),
        mk_step(10, ["i"], "final", ["final_answer"]),
    ])
    s_long = scorer.score_proc(proc_long)

    # Should be strictly less than short score and close to base - w_len (~0.0–0.1 margin)
    assert s_long < s_short
    assert s_long < 0.15  # with the given sigmoid it should be near 0


# -----------------------------
# ProcScorerAdapter tests
# -----------------------------

def test_proc_scorer_adapter_returns_underlying_proc_score():
    class DummyProcScorer:
        def __init__(self, val): self.val = val
        def score_proc(self, p): return float(self.val)

    adapter = ProcScorerAdapter(DummyProcScorer(42.0))
    dummy_ind = types.SimpleNamespace(proc={"steps": []})
    assert adapter.score(dummy_ind) == 42.0


# -----------------------------
# TaskEvalScorer tests
# -----------------------------

def test_task_eval_scorer_happy_path_returns_eval_score():
    # run_steps returns a state including final_answer
    def run_steps(proc, question, schema, model, *args, **kwargs):
        return {"final_answer": "42", "other": "ok"}

    # eval_fn reads state and proc and returns a float
    def eval_fn(state, proc):
        return 0.7

    scorer = TaskEvalScorer(
        run_steps_fn=run_steps,
        eval_fn=eval_fn,
        question="Q",
        final_answer_schema={"type": "object"},
        model="gemma3",
        strict_require_key="final_answer",
    )

    ind = types.SimpleNamespace(proc=mk_proc([mk_step(1, ["problem_text"], "extract", ["a"]), mk_step(2, ["a"], "final", ["final_answer"])]))
    assert scorer.score(ind) == pytest.approx(0.7, rel=1e-6)


def test_task_eval_scorer_missing_required_key_returns_minus_one():
    def run_steps(proc, question, schema, model, *args, **kwargs):
        return {"not_final": "x"}  # missing final_answer

    scorer = TaskEvalScorer(
        run_steps_fn=run_steps,
        eval_fn=lambda state, proc: 1.0,
        question="Q",
        final_answer_schema={"type": "object"},
        model="gemma3",
        strict_require_key="final_answer",
    )

    ind = types.SimpleNamespace(proc=mk_proc([mk_step(1, ["problem_text"], "extract", ["a"])]))
    assert scorer.score(ind) == -1.0


def test_task_eval_scorer_catches_exceptions_and_returns_minus_one():
    def run_steps(proc, question, schema, model, *args, **kwargs):
        raise RuntimeError("boom")

    scorer = TaskEvalScorer(
        run_steps_fn=run_steps,
        eval_fn=lambda state, proc: 1.0,
        question="Q",
        final_answer_schema={"type": "object"},
        model="gemma3",
    )

    ind = types.SimpleNamespace(proc=mk_proc([mk_step(1, ["problem_text"], "extract", ["a"])]))
    assert scorer.score(ind) == -1.0
