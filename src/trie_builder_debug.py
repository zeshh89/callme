from src.llm import LLM
from src.io import load_function_definitions
from src.function_trie_builder import FunctionTrieBuilder
from src.masking import mask_logits


def main():

    print("Loading model...")
    llm = LLM()
    print("Model loaded")

    functions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    trie = FunctionTrieBuilder(llm).build(functions)
    print("Trie built")

    prompt = "What is the sum of 2 and 3?"

    print("\nINPUT IDS:")
    ids = llm.encode(prompt)
    print(ids)   # <-- NO tolist()

    logits = llm.logits(ids)

    allowed = trie.get_allowed_tokens([])

    print("\nALLOWED TOKENS:", allowed)

    print("\nTOP LOGITS BEFORE MASKING:")
    top = sorted(
        enumerate(logits.tolist()),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for t, v in top:
        print(t, llm.decode([t]), v)

    masked = mask_logits(logits, allowed)

    print("\nTOP LOGITS AFTER MASKING:")
    top_masked = sorted(
        enumerate(masked.tolist()),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for t, v in top_masked:
        print(t, llm.decode([t]), v)


if __name__ == "__main__":
    main()