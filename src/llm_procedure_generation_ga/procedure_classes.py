from pydantic import BaseModel
from typing import List

#Procedure Creation
class StepInputField(BaseModel):
    name: str
    description: str
    
class StepOutputField(BaseModel):
    name: str
    description: str

class Step(BaseModel):
    id: int
    inputs: List[StepInputField]
    stepDescription: str
    output: List[StepOutputField]
    
class Procedure(BaseModel):
    NameDescription: str
    #inputs: List[InputField]
    steps: List[Step]
    #output: List[OutputField]