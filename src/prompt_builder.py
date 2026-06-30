from src.models import FunctionDefinition
import json


def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:

    parameters = "\n".join(
        f"- {name}: {parameter.type}"
        for name, parameter in function.parameters.items()
    )

    example = {}

    for name, parameter in function.parameters.items():

        if parameter.type == "number":
            example[name] = 1

        elif parameter.type == "boolean":
            example[name] = True

        else:
            example[name] = "example"

    example_json = json.dumps(
        example,
        indent=2,
    )

    return f"""User request:
{user_prompt}

Function:
{function.name}

Description:
{function.description}

Parameters:
{parameters}

Return ONLY one valid JSON object.

Rules:
- Use exactly the parameter names listed above.
- Do NOT invent new keys.
- Do NOT include the function name.
- Do NOT include the result.
- Do NOT include explanations.
- Do NOT wrap the JSON inside another object.

Example:
{example_json}

JSON:
"""
