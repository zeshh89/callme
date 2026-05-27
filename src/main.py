from src.io import (
    load_function_definitions,
    load_prompts,
)


def main() -> None:
    """
    Main entry point.
    """

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    prompts = load_prompts(
        "data/input/function_calling_tests.json"
    )

    print("Functions loaded:")
    for function in functions:
        print(function.name)

    print("\nPrompts loaded:")
    for prompt in prompts:
        print(prompt.prompt)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
