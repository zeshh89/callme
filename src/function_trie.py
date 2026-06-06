from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class FunctionTrie:

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, tokens: list[int]) -> None:

        node = self.root

        for token in tokens:

            if token not in node.children:
                node.children[token] = TrieNode()

            node = node.children[token]

        node.is_end = True

    def get_allowed_tokens(
        self,
        prefix: list[int],
    ) -> list[int]:

        node = self.root

        for token in prefix:

            if token not in node.children:
                return []

            node = node.children[token]

        return list(node.children.keys())

    def is_complete(
        self,
        prefix: list[int],
    ) -> bool:

        node = self.root

        for token in prefix:

            if token not in node.children:
                return False

            node = node.children[token]

        return node.is_end
