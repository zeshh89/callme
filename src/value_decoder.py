from src.function_trie import FunctionTrie
from src.decoding import greedy_decode

NUMBER_CHARS = set("0123456789.-")


def generate_number(llm, prompt: str, as_integer: bool = False, max_tokens: int = 10, max_lookahead: int = 5) -> float | int:
    ids = llm.encode(prompt)
    digits = ""
    started = False
    skipped = 0

    for _ in range(max_tokens + max_lookahead):
        logits = llm.logits(ids)
        token = int(logits.argmax().item())
        text = llm.decode([token])
        stripped = text.strip()

        if not started:
            if stripped == "":
                # espacio/salto de línea antes de empezar: lo saltamos
                skipped += 1
                ids = ids + [token]
                if skipped > max_lookahead:
                    break
                continue
            if all(c in NUMBER_CHARS for c in stripped):
                started = True
                digits += stripped
                ids = ids + [token]
                continue
            # el primer token con contenido no es numérico -> abortar
            break
        else:
            if text and all(c in NUMBER_CHARS for c in text):
                digits += text
                ids = ids + [token]
            else:
                break

    digits = digits.strip()
    if not digits:
        raise ValueError(f"Could not extract a number from prompt: {prompt!r}")
    value = float(digits)
    return int(value) if as_integer else value


def generate_boolean(llm, prompt: str) -> bool:
    trie = FunctionTrie()
    trie.insert(llm.encode("true"))
    trie.insert(llm.encode("false"))
    tokens = greedy_decode(prompt, llm, trie)
    return llm.decode(tokens).strip().lower() == "true"


def generate_string(llm, prompt: str, max_tokens: int = 30) -> str:
    ids = llm.encode(prompt)
    text = ""
    started = False

    for _ in range(max_tokens):
        logits = llm.logits(ids)
        token = int(logits.argmax().item())
        token_text = llm.decode([token])
        ids = ids + [token]

        if "\n" in token_text:
            text += token_text.split("\n")[0]
            break

        if not started and token_text.strip() == "":
            continue

        started = True
        text += token_text

        # si el modelo abrió comillas y ya las cerró, cortamos ahí
        if text.count('"') >= 2:
            break

    text = text.strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return text.strip()
