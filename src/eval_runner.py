import argparse

from src.llm import LLM
from src.io import (
    load_function_definitions,
    load_prompts,
    save_results,
)
from src.function_trie_builder import FunctionTrieBuilder
from src.registry import FunctionRegistry
from src.pipeline import run_pipeline
from src.models import FunctionCallResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts"
            "into structured function calls."
        )
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions_definition.json file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the function_calling_tests.json file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading model...")
    llm = LLM()
    print("Model loaded")

    functions = load_function_definitions(args.functions_definition)
    registry = FunctionRegistry(functions)
    prompts = load_prompts(args.input)
    trie = FunctionTrieBuilder(llm).build(functions)

    results: list[FunctionCallResult] = []
    for index, prompt_obj in enumerate(prompts, start=1):
        print(f"\n--- Processing {index}/{len(prompts)} ---")
        try:
            result = run_pipeline(prompt_obj, llm, trie, registry)
            results.append(result)
            print("OK:", result.name, "->", result.parameters)
        except Exception as exc:
            print("ERROR:", exc)
            results.append(
                FunctionCallResult(
                    prompt=prompt_obj.prompt,
                    name="ERROR",
                    parameters={},
                )
            )

    save_results(args.output, results)
    print("\nDONE. Results saved.")


if __name__ == "__main__":
    main()
