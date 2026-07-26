from pydantic import BaseModel, field_validator
from typing import Dict, Literal

ParamType = Literal["string", "number", "integer", "boolean"]
ParameterValue = str | float | int | bool


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

    @field_validator("prompt")
    @classmethod
    def validate_prompt(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        return value


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, ParameterValue]
