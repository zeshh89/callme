from pydantic import BaseModel
from typing import Dict, Literal

ParamType = Literal["string", "number", "boolean"]
ParameterValue = str | float | bool


class ParameterDefinition(BaseModel):
    type: ParamType


class ReturnDefinition(BaseModel):
    type: ParamType


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, ParameterValue]
