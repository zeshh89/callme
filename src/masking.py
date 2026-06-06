import torch


def mask_logits(
    logits: torch.Tensor,
    allowed_tokens: list[int]
) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))

    allowed = torch.tensor(allowed_tokens, device=logits.device)

    mask[allowed] = logits[allowed]

    return mask
