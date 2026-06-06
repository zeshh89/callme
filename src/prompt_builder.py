from src.models import FunctionDefinition


def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:

    return f"""
User request:
{user_prompt}

Function:
{function.name}

Parameters:
{list(function.parameters.keys())}

Return ONLY a JSON object.

Example:
{{"a": 1, "b": 2}}

JSON:
"""
