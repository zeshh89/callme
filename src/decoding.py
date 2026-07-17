from src.masking import mask_logits
from src.llm import LLM
from src.function_trie import FunctionTrie


def greedy_decode(prompt: str, llm: LLM, trie: FunctionTrie) -> list[int]:
    prompt_ids = llm.encode(prompt)
    generated: list[int] = []

    for _ in range(20):
        logits = llm.logits(prompt_ids + generated)
        allowed = trie.get_allowed_tokens(generated)
        if not allowed:
            break
        masked = mask_logits(logits, allowed)
        token = int(masked.argmax().item())
        generated.append(token)
        if trie.is_complete(generated):
            break

    return generated
