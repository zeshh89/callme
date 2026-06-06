import json


def generate_json(
    llm,
    prompt: str,
    max_tokens: int = 100,
) -> dict:

    prompt_ids = llm.encode(prompt)

    generated_tokens: list[int] = []

    started = False
    brace_depth = 0

    for _ in range(max_tokens):

        current_ids = (
            prompt_ids +
            generated_tokens
        )

        logits = llm.logits(current_ids)

        next_token = int(
            logits.argmax().item()
        )

        generated_tokens.append(
            next_token
        )

        token_text = llm.decode(
            [next_token]
        )

        print(
            token_text,
            end="",
            flush=True,
        )

        for char in token_text:

            if char == "{":
                started = True
                brace_depth += 1

            elif char == "}":
                brace_depth -= 1

                if (
                    started
                    and brace_depth == 0
                ):
                    text = llm.decode(
                        generated_tokens
                    )

                    print()

                    return json.loads(text)

    raise ValueError(
        "Could not generate valid JSON"
    )
