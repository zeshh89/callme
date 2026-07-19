from llm_sdk import Small_LLM_Model
import torch


class LLM:

    def __init__(self) -> None:
        self.model = Small_LLM_Model()

    def encode(self, text: str) -> list[int]:
        tensor = self.model.encode(text)
        return tensor[0].tolist()

    def decode(self, ids: list[int]) -> str:
        return self.model.decode(ids)

    def logits(self, ids: list[int]) -> torch.Tensor:
        return self.model.get_logits_from_input_ids(ids)
