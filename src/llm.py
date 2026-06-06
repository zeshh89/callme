from llm_sdk import Small_LLM_Model


class LLM:

    def __init__(self) -> None:
        self.model = Small_LLM_Model()

    def encode(self, text: str) -> list[int]:
        tensor = self.model.encode(text)
        return tensor[0].tolist()

    def decode(self, ids: list[int]) -> str:
        return self.model.decode(ids)

    def logits(self, ids: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(ids)
