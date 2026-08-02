import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        cosine_steps: int,
        max_lr: float,
        min_lr: float,
        start_factor: float = 1e-5,
) -> LambdaLR:
    min_factor = min_lr / max_lr

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * step / max(1, warmup_steps)

        cosine_step = step - warmup_steps
        if cosine_step < cosine_steps:
            progress = cosine_step / max(1, cosine_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return min_factor

    return LambdaLR(optimizer, lr_lambda=lr_factor)
