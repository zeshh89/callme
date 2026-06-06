from src.function_trie import FunctionTrie


def main() -> None:

    trie = FunctionTrie()

    trie.insert([8822, 2891, 32964])
    trie.insert([8822, 1889, 3744])

    print(
        trie.get_allowed_tokens([8822])
    )


if __name__ == "__main__":
    main()
