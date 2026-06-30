from src.llm import LLM
from src.io import (
    load_function_definitions,
    load_prompts,
    save_results,
)
from src.function_trie_builder import (
    FunctionTrieBuilder,
)
from src.registry import FunctionRegistry
from src.pipeline import run_pipeline
from src.models import FunctionCallResult


def main() -> None:

    print("Loading model...")
    llm = LLM()
    print("Model loaded")

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    registry = FunctionRegistry(
        functions
    )

    prompts = load_prompts(
        "data/input/function_calling_tests.json"
    )

    trie = FunctionTrieBuilder(
        llm
    ).build(
        functions
    )

    results: list[FunctionCallResult] = []

    for index, prompt_obj in enumerate(
        prompts,
        start=1,
    ):

        print(
            f"\n--- Processing {index}/{len(prompts)} ---"
        )

        try:

            result = run_pipeline(
                prompt_obj,
                llm,
                trie,
                registry,
            )

            results.append(result)

            print("OK:", result.name)

        except Exception as exc:

            print("ERROR:", exc)

            results.append(
                FunctionCallResult(
                    prompt=prompt_obj.prompt,
                    name="ERROR",
                    parameters={},
                )
            )

    save_results(
        "data/output/results.json",
        results,
    )

    print("\nDONE. Results saved.")


if __name__ == "__main__":
    main()
