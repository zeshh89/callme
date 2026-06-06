from src.io import load_function_definitions
from src.registry import FunctionRegistry
from src.llm import LLM


def main() -> None:

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    registry = FunctionRegistry(functions)

    llm = LLM()

    for name in registry.names():

        ids = llm.encode(name)

        print(f"\n{name}")
        print(ids)

        for token_id in ids:
            print(
                token_id,
                "->",
                repr(llm.decode([token_id]))
            )


if __name__ == "__main__":
    main()
