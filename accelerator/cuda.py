import torch


class Accelerator:
    def __init__(self, device_index: int):
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA accelerator requested, but CUDA is unavailable")
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index out of range: {device_index}; "
                f"found {torch.cuda.device_count()} device(s)"
            )

        self.device_index = device_index
        self.device = torch.device("cuda", device_index)

    def name(self) -> str:
        return str(self.device)

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def autocast(
            self,
            *,
            dtype: torch.dtype,
            enabled: bool = True,
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=enabled,
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)
