from src.models import FunctionDefinition


def find_function_by_name(
    functions: list[FunctionDefinition],
    name: str,
) -> FunctionDefinition | None:

    for function in functions:

        if function.name == name:
            return function

    return None
