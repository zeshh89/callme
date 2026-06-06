from src.llm import LLM
from src.io import load_function_definitions, load_prompts, save_results
from src.function_trie_builder import FunctionTrieBuilder
from src.function_resolver import find_function_by_name
from src.greedy_decoder import greedy_decode
from src.prompt_builder import build_parameter_prompt
from src.models import FunctionCallResult
from src.json_decoder import generate_json


def run_pipeline(prompt_obj, llm, trie, functions):
    prompt = prompt_obj.prompt

    # 1. seleccionar función
    function_tokens = greedy_decode(prompt, llm, trie)
    function_name = llm.decode(function_tokens)
    function = find_function_by_name(functions, function_name)

    if function is None:
        return FunctionCallResult(
            prompt=prompt,
            name="UNKNOWN",
            parameters={}
        )

    # 2. construir prompt de parámetros
    parameter_prompt = build_parameter_prompt(prompt, function)

    # 3. generar JSON (usa tu generate_json)
    parameters = generate_json(llm, parameter_prompt)

    # 4. devolver resultado final
    return FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=parameters,
    )


def main():

    print("Loading model...")
    llm = LLM()
    print("Model loaded")

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    prompts = load_prompts(
        "data/input/function_calling_tests.json"
    )

    trie = FunctionTrieBuilder(llm).build(functions)

    results = []

    for i, prompt_obj in enumerate(prompts):

        print(f"\n--- Processing {i+1}/{len(prompts)} ---")

        try:
            result = run_pipeline(
                prompt_obj,
                llm,
                trie,
                functions,
            )

            results.append(result)

            print("OK:", result.name)

        except Exception as e:
            print("ERROR:", e)

            results.append(FunctionCallResult(
                prompt=prompt_obj.prompt,
                name="ERROR",
                parameters={}
            ))

    save_results(
        "data/output/results.json",
        results,
    )

    print("\nDONE. Results saved.")


if __name__ == "__main__":
    main()