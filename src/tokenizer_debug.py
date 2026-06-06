from src.llm import LLM


def main() -> None:
    llm = LLM()

    text = '{"name":"fn_add_numbers"}'

    ids = llm.encode(text)

    print("TEXT:")
    print(text)

    print("\nIDS:")
    print(ids)

    print("\nTOKEN BY TOKEN:")

    for token_id in ids:
        print(
            token_id,
            "->",
            repr(llm.decode([token_id]))
        )


if __name__ == "__main__":
    main()
