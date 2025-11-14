"""Typed models for procedure creation and execution.

These Pydantic models define the structure of a *global-state* procedure:
    - Each `Step` can read variables produced by previous steps.
    - Step 1 consumes only `problem_text`.
    - The final step emits `final_answer` (a description, not a computed value).

The models use snake_case Python attribute names but expose **camelCase**
aliases at the JSON boundary so you can interoperate with existing data.

Example
-------
>>> from .models import Procedure, Step, StepInputField, StepOutputField
>>> proc = Procedure(
...     name_description="Add two numbers",
...     steps=[
...         Step(
...             id=1,
...             inputs=[StepInputField(name="problem_text", description="Task text")],
...             step_description="Extract numbers a and b from the text.",
...             outputs=[StepOutputField(name="a", description="First number"),
...                      StepOutputField(name="b", description="Second number")],
...         ),
...     ],
... )
>>> proc.model_dump(by_alias=True)["nameDescription"]
'Add two numbers'
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class StepInputField(BaseModel):
    """A named input consumed by a step.

    Attributes
    ----------
    name:
        Variable name expected by the step (e.g., ``"problem_text"``).
    description:
        Human-readable description of what this input represents.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Variable name referenced by the step.")
    description: str = Field(..., description="Human-readable description of the input.")


class StepOutputField(BaseModel):
    """A variable produced by a step.

    Attributes
    ----------
    name:
        Variable name produced by the step (snake_case recommended).
    description:
        Human-readable description of the output variable.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Variable name this step emits.")
    description: str = Field(..., description="Human-readable description of the output.")


class Step(BaseModel):
    """One atomic instruction within a global-state procedure.

    Notes
    -----
        - Steps should be **single-action** and declarative.
        - Use prior outputs as inputs by variable **name**.
        - Keep variable names stable and snake_case.

    Attributes
    ----------
    id:
        1-based step identifier (contiguous in execution order).
    inputs:
        List of required input variables for this step.
    step_description:
        Natural-language instruction describing exactly what the step does.
    outputs:
        List of variables produced by this step.
    """
    # Keep Python attrs snake_case; expose camelCase aliases at the boundary.
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., ge=1, description="1-based step identifier.")
    inputs: List[StepInputField] = Field(
        default_factory=list, description="Inputs consumed by this step."
    )
    step_description: str = Field(
        ...,
        alias="stepDescription",
        description="Single-action, imperative instruction for the step.",
    )
    outputs: List[StepOutputField] = Field(
        default_factory=list,
        alias="output",
        description="Variables produced by this step.",
    )


class Procedure(BaseModel):
    """A full global-state procedure composed of ordered steps.

    Attributes
    ----------
    name_description:
        Short summary of the procedure's purpose (task or capability).
    steps:
        Ordered list of steps. Step 1 should consume only ``problem_text``;
        the final step should produce ``final_answer``.
    """
    model_config = ConfigDict(populate_by_name=True)

    name_description: str = Field(
        ...,
        alias="NameDescription",
        description="Human-readable summary of the procedure.",
    )
    steps: List[Step] = Field(
        ...,
        description="Ordered list of steps composing this procedure.",
    )

    @field_validator("steps")
    @classmethod
    def steps_must_be_non_empty(cls, v: List["Step"]) -> List["Step"]:
        if not v:
            raise ValueError("Procedure must contain at least one step.")
        return v