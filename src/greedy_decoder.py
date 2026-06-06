from src.llm import LLM
from src.io import load_function_definitions
from src.function_trie_builder import FunctionTrieBuilder
from src.masking import mask_logits
from src.prompt_builder import build_parameter_prompt
from src.json_decoder import generate_json


def greedy_decode(
    prompt: str,
    llm: LLM,
    trie,
) -> list[int]:

    prompt_ids = llm.encode(prompt)

    generated_function_tokens: list[int] = []

    max_steps = 20

    for _ in range(max_steps):

        current_ids = (
            prompt_ids +
            generated_function_tokens
        )

        logits = llm.logits(current_ids)

        allowed_tokens = trie.get_allowed_tokens(
            generated_function_tokens
        )

        if not allowed_tokens:
            break

        masked_logits = mask_logits(
            logits,
            allowed_tokens,
        )

        next_token = int(
            masked_logits.argmax().item()
        )

        generated_function_tokens.append(
            next_token
        )

        if trie.is_complete(
            generated_function_tokens
        ):
            break

        print(
            llm.decode([next_token]),
            end="",
            flush=True,
        )

    print()

    return generated_function_tokens


def main() -> None:

    print("Loading model...")
    llm = LLM()
    print("Model loaded")

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    trie = FunctionTrieBuilder(llm).build(
        functions
    )

    print("Trie built")

    prompt = (
        "What is the sum of 2 and 3?"
    )

    print("\nGENERATING:\n")

    generated_tokens = greedy_decode(
        prompt,
        llm,
        trie,
    )

    from src.function_resolver import (
        find_function_by_name,
    )

    function_name = llm.decode(
        generated_tokens
    )

    print("\nFUNCTION NAME:")
    print(function_name)

    function = find_function_by_name(
        functions,
        function_name,
    )

    print("\nFUNCTION:")
    print(function)

    parameter_prompt = build_parameter_prompt(
        prompt,
        function
    )

    print("\nPARAMETER PROMPT:")
    print(parameter_prompt)

    parameter_ids = llm.encode(
        parameter_prompt
    )

    logits = llm.logits(
        parameter_ids
    )

    parameters = generate_json(
        llm,
        parameter_prompt,
    )

    next_token = int(
        logits.argmax().item()
    )

    print("\nFIRST PARAMETER TOKEN:")
    print(next_token)

    print(
        llm.decode([next_token])
    )

    print("\nGENERATED TOKEN IDS:")
    print(generated_tokens)

    print("\nGENERATED TEXT:")
    print(
        llm.decode(
            generated_tokens
        )
    )

    print("\nPARAMETERS:")
    print(parameters)


if __name__ == "__main__":
    main()
