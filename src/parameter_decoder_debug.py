def generate_text(
    llm,
    prompt,
    max_tokens=50,
):
    ids = llm.encode(prompt)

    generated = []

    for _ in range(max_tokens):

        logits = llm.logits(
            ids + generated
        )

        next_token = int(
            logits.argmax().item()
        )

        generated.append(
            next_token
        )

    return llm.decode(generated)
