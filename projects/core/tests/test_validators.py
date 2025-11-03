# test_validators.py
from typing import Dict, Any, List
from evoproc.validators import validate_first_step_inputs, validate_inputs_resolvable_from_prior, validate_unused_outputs

JSONDict = Dict[str, Any]

def mk_step(i: int, inputs: List[str], desc: str, outputs: List[str]) -> Dict[str, Any]:
    return {
        "id": i,
        "inputs": [{"name": n, "description": ""} for n in inputs],
        "stepDescription": desc,
        "output": [{"name": n, "description": ""} for n in outputs],
    }

def test_step1_rule_enforced():
    p: JSONDict = {
        "NameDescription": "bad step1",
        "steps": [
            mk_step(1, ["problem_text", "extra"], "extract facts", ["facts"]),  # illegal extra input
            mk_step(2, ["facts"], "summarize", ["final_answer"]),
        ],
    }
    diags = validate_first_step_inputs(p)
    assert any(d["severity"] == "fatal" and "problem_text" in d["message"] for d in diags)

def test_unresolvable_input_is_fatal():
    p: JSONDict = {
        "NameDescription": "missing producer",
        "steps": [
            mk_step(1, ["problem_text"], "extract facts", ["facts"]),
            mk_step(2, ["nonexistent_var"], "compute something", ["derived"]),
            mk_step(3, ["derived"], "finish", ["final_answer"]),
        ],
    }
    diags = validate_inputs_resolvable_from_prior(p)
    fatal_msgs = [d for d in diags if d["severity"] == "fatal"]
    assert fatal_msgs, "Expected fatal diagnostics for unresolved inputs"
    assert any("nonexistent_var" in d["message"] for d in fatal_msgs)

def test_unused_outputs_flagged():
    p: JSONDict = {
        "NameDescription": "dead outputs",
        "steps": [
            mk_step(1, ["problem_text"], "extract facts", ["facts", "dead1"]),
            mk_step(2, ["facts"], "transform", ["usable"]),
            mk_step(3, ["usable"], "finalize", ["final_answer"]),
        ],
    }
    diags = validate_unused_outputs(p)
    rep = [d for d in diags if d["severity"] == "repairable"]
    assert rep, "Expected repairable diagnostics for dead outputs"
    # ensure it names the right variable and step
    joined = " | ".join(d["message"] for d in rep)
    assert "dead1" in joined
    # step index 1 in message (human 1-based Step 1)
    assert "Step 1" in joined
