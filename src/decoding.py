from src.masking import mask_logits


def greedy_decode(prompt, llm, trie):

    prompt_ids = llm.encode(prompt)

    generated = []

    for _ in range(20):

        logits = llm.logits(
            prompt_ids + generated
        )

        allowed = trie.get_allowed_tokens(
            generated
        )

        if not allowed:
            break

        masked = mask_logits(
            logits,
            allowed,
        )

        token = int(
            masked.argmax().item()
        )

        generated.append(token)

        if trie.is_complete(generated):
            break

    return generated
