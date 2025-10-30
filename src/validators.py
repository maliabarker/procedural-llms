"""
validators.py
====================

Validator helpers for *global-state* procedures.

Design overview
---------------
- **Global state**: Steps may read any variable produced by earlier steps or given in
  the problem. There is no step-to-step pass-through requirement.
- **Step 1 rule**: The first step must take **exactly** one input: ``problem_text``.
  (No other inputs will be provided externally.)
- **Final step rule**: The last step must output exactly ``final_answer`` (a description,
  not the computed value).
- **Safety checks**: Every declared input must be resolvable from prior outputs
  (or ``problem_text``), variables should not be silently redefined, and unused
  outputs are flagged to keep the procedure minimal and readable.

All functions return a list of :class:`Diagnostic` objects that can be fed into a
repair loop. Use :func:`validate_procedure_structured` to run the default suite.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypedDict
from src.helpers import _canon_details, _names
JSONDict = Dict[str, Any]

Action = Literal[
    "PATCH_LOCALLY",            # small JSON edits are enough
    "REWRITE_FIRST_STEP",       # step 1 must be rewritten to only use problem_text
    "ADD_FINAL_STEP",           # final step missing; add step that produces final_answer
    "EXTEND_PROCEDURE_TO_FINAL" # needs more steps to reach final_answer
]
"""Action: machine-usable repair hints for downstream auto-fix prompts."""

Severity = Literal["repairable", "fatal"]
"""Severity: ``fatal`` means the procedure is not runnable without regeneration or
structural rewrite; ``repairable`` means a small, local JSON edit should suffice."""

class Diagnostic(TypedDict):
    """
    Structured validator finding.

    Keys
    ----
    severity:
        Either ``"fatal"`` or ``"repairable"``.
    action:
        A short, machine-usable hint for an auto-repair prompt (see :data:`Action`).
    message:
        Human-readable description of the issue.
    details:
        Machine-targeted payload (e.g., step ids, variable names) that an auto-repair
        routine can use to patch the JSON.
    """
    severity: Severity
    action: Action
    message: str
    details: Dict[str, Any]

def _dedup_diags(diags: List[Diagnostic]) -> List[Diagnostic]:
    """
    Deduplicate diagnostics by a canonical identity.

    Two diagnostics are considered the same if ``(severity, action, message, canon(details))``
    matches. This keeps repair prompts concise when multiple validators surface
    the same underlying issue.

    Parameters
    ----------
    diags
        List of diagnostics produced by one or more validators.

    Returns
    -------
    List[Diagnostic]
        De-duplicated diagnostics (stable order of first occurrence preserved).
    """
    seen = set()
    out: List[Diagnostic] = []
    for d in diags:
        key = (d["severity"], d["action"], d["message"], _canon_details(d.get("details", {})))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# ---- Individual validators ---------------------------------------------------
    
def validate_first_step_inputs(p: JSONDict) -> List[Diagnostic]:
    """
    Enforce the **Step 1** input contract: it must be exactly ``["problem_text"]``.

    Rationale
    ---------
    In the global-state design, the only external input is the problem text.
    All additional information must be produced by earlier steps (which, for Step 1,
    means there are none), so Step 1 cannot declare any other inputs.

    Parameters
    ----------
    p
        Procedure JSON (already parsed to Python dict).

    Returns
    -------
    List[Diagnostic]
        ``fatal`` if Step 1 inputs differ from exactly ``["problem_text"]``, otherwise empty.
    """
    steps = p["steps"]
    step1_inputs = _names(steps[0]["inputs"])
    if step1_inputs != ["problem_text"]:
        return [{
            "severity": "fatal",
            "action": "REWRITE_FIRST_STEP",
            "message": "Step 1 inputs must be exactly ['problem_text'].",
            "details": {"found": step1_inputs}
        }]
    return []

def validate_final_step_output(p: JSONDict) -> List[Diagnostic]:
    """
    Require the **final step** to output exactly ``["final_answer"]``.

    Notes
    -----
    ``final_answer`` is terminal and consumed outside the procedure. The final step
    should describe the final answer (no numeric computation here).

    Parameters
    ----------
    p
        Procedure JSON (already parsed to Python dict).

    Returns
    -------
    List[Diagnostic]
        ``fatal`` if the last step's outputs are not exactly ``["final_answer"]``.
    """
    # TODO: in most cases, this issue occurs when the final answer is derived in the final step but just not assigned the corresponding variable name. Can we somehow check if the answer is given and if so just change the existing variable name to `final_answer`?
    steps = p["steps"]
    final_outputs = _names(steps[-1]["output"])
    if final_outputs != ["final_answer"]:
        return [{
            "severity": "fatal",
            "action": "EXTEND_PROCEDURE_TO_FINAL" if "final_answer" not in final_outputs else "ADD_FINAL_STEP",
            "message": "Final step must produce exactly ['final_answer'].",
            "details": {"found": final_outputs}
        }]
    return []

def validate_inputs_resolvable_from_prior(p: Dict[str, Any]) -> List[Diagnostic]:
    """
    Ensure every step input is **available** from global state when the step runs.

    Rule
    ----
    For each step *i* and input variable *v*:
      - Either ``v == "problem_text"``, or
      - some prior step *k < i* produced ``v`` in its outputs.

    Parameters
    ----------
    p
        Procedure JSON (already parsed to Python dict).

    Returns
    -------
    List[Diagnostic]
        ``fatal`` diagnostics for inputs that cannot be traced to ``problem_text`` or
        some earlier step's outputs.
    """
    diags: List[Diagnostic] = []
    steps = p["steps"]

    # map var -> first producer step index
    producers = {}
    for idx, s in enumerate(steps):
        for v in _names(s["output"]):
            producers.setdefault(v, idx)

    for i, s in enumerate(steps):
        for v in _names(s["inputs"]):
            if v == "problem_text":
                continue
            if v not in producers or producers[v] >= i:
                # TODO: this should probably change in action because it's not always the first step at fault
                diags.append({
                    "severity": "fatal",
                    "action": "REWRITE_FIRST_STEP",
                    "message": f"Input '{v}' of Step {i+1} is not produced by any prior step.",
                    "details": {"step_id": s["id"], "input": v}
                })
    return diags

def validate_no_redefine_existing_vars(p: JSONDict) -> List[Diagnostic]:
    """
    Discourage **shadowing**: warn if a step re-uses an already-produced variable name.

    Rationale
    ---------
    In a global state, silently overwriting prior variables can confuse both the LLM
    and debuggers. Prefer distinct names or explicitly mark transformations
    (e.g., ``normalized_total`` instead of re-using ``total``).

    Parameters
    ----------
    p
        Procedure JSON (already parsed to Python dict).

    Returns
    -------
    List[Diagnostic]
        ``repairable`` diagnostics suggesting renames for redefined variables.
    """
    diags: List[Diagnostic] = []
    seen: Set[str] = set()
    for i, s in enumerate(p["steps"]):
        outs = _names(s["output"])
        redefs = [v for v in outs if v in seen and v != "final_answer"]
        if redefs:
            diags.append({
                "severity": "repairable",
                "action": "PATCH_LOCALLY",
                "message": f"Avoid re-defining existing variables at Step {i+1}: {redefs}",
                "details": {"step_id": s["id"], "consider_renaming": redefs}
            })
        seen.update(outs)
    return diags

def validate_unused_outputs(p: JSONDict) -> List[Diagnostic]:
    """
    Flag **dead variables**: outputs never consumed by any later step.

    Exemptions
    ----------
    - ``final_answer`` (terminal output)
    - ``problem_text`` (not an output; guarded for safety)

    Algorithm
    ---------
    Walk backwards through steps, accumulating the set of inputs that appear
    at or after each position. Any output at index *i* that never appears in a
    later step's inputs is considered unused.

    Parameters
    ----------
    p
        Procedure JSON (already parsed to Python dict).

    Returns
    -------
    List[Diagnostic]
        ``repairable`` diagnostics suggesting removal of unused outputs per step.
    """
    diags: List[Diagnostic] = []
    steps = p["steps"]
    n = len(steps)

    # Build, for each index i, a set of inputs used by steps after i.
    future_inputs_after: List[Set[str]] = [set() for _ in range(n)]
    future: Set[str] = set()
    for i in range(n - 1, -1, -1):
        future_inputs_after[i] = set(future)
        if i + 1 < n:
            future |= set(_names(steps[i + 1]["inputs"]))
        # also add inputs of current step so that earlier outputs see this as "future"
        future |= set(_names(steps[i]["inputs"]))

    for i, s in enumerate(steps):
        outs = [v for v in _names(s["output"]) if v not in {"final_answer", "problem_text"}]
        unused = sorted([v for v in outs if v not in future_inputs_after[i]])
        if unused:
            diags.append({
                "severity": "repairable",
                "action": "PATCH_LOCALLY",
                "message": f"Remove unused outputs at Step {i+1}: {unused}",
                "details": {"step_id": s["id"], "remove_from_outputs": unused}
            })
    return diags

# ---- Master validator (composable) ------------------------------------------

Validator = Callable[[JSONDict], List[Diagnostic]]
"""Callable signature for a validator function."""

DEFAULT_VALIDATORS: List[Validator] = [
    validate_first_step_inputs,
    validate_final_step_output,
    validate_inputs_resolvable_from_prior,
    validate_no_redefine_existing_vars,
    validate_unused_outputs
]
"""Default validator suite for global-state procedures with a strict Step 1 and final step."""

def validate_procedure_structured(p: JSONDict, validators: Optional[List[Validator]] = None) -> List[Diagnostic]:
    """
    Run a composed set of validators and return de-duplicated diagnostics.

    Parameters
    ----------
    p
        Procedure JSON to validate (already parsed to Python dict).
    validators
        Optional custom list of validator callables. If omitted, uses
        :data:`DEFAULT_VALIDATORS`.

    Returns
    -------
    List[Diagnostic]
        De-duplicated diagnostics suitable for a repair loop.
    """
    validators = validators or DEFAULT_VALIDATORS
    all_diags: List[Diagnostic] = []
    for fn in validators:
        all_diags.extend(fn(p))
    return _dedup_diags(all_diags)