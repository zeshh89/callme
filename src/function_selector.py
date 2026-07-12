from src.decoding import greedy_decode
from src.function_resolver import find_function_by_name


def select_function(
    prompt: str,
    llm,
    trie,
    functions,
):
    tokens = greedy_decode(
        prompt,
        llm,
        trie,
    )

    name = llm.decode(tokens)

    function = find_function_by_name(
        functions,
        name,
    )

    return function
