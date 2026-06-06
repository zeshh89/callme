from src.models import FunctionDefinition


class FunctionRegistry:

    def __init__(
        self,
        functions: list[FunctionDefinition],
    ) -> None:

        self._functions = {
            function.name: function
            for function in functions
        }

    def get(
        self,
        name: str,
    ) -> FunctionDefinition | None:

        return self._functions.get(name)

    def names(
        self,
    ) -> list[str]:

        return list(self._functions.keys())
