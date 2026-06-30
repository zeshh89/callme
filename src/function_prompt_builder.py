from src.models import FunctionDefinition


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:

    names = "\n".join(
        function.name
        for function in functions
    )

    return f"""
You must choose exactly one function.

Available functions:

{names}

User request:
{user_prompt}

Function:
"""
