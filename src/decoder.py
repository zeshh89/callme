import torch


class ConstrainedDecoder:

    def __init__(self, trie, llm):
        self.trie = trie
        self.llm = llm

    def mask_logits(
        self,
        logits: list[float],
        allowed_tokens: list[int],
    ) -> list[float]:

        masked = []

        for i, logit in enumerate(logits):

            if i in allowed_tokens:
                masked.append(logit)
            else:
                masked.append(float("-inf"))

        return masked