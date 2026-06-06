from src.function_trie import FunctionTrie
from src.models import FunctionDefinition
from src.llm import LLM


class FunctionTrieBuilder:

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def build(
        self,
        functions: list[FunctionDefinition],
    ) -> FunctionTrie:

        trie = FunctionTrie()

        for function in functions:

            token_ids = self.llm.encode(
                function.name
            )

            trie.insert(token_ids)

        return trie
