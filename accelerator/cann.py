import torch
import torch_npu


class Accelerator:
    def __init__(self, device_index: int):
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if not torch_npu.npu.is_available():
            raise RuntimeError("Ascend accelerator requested, but NPU is unavailable")
        if device_index >= torch_npu.npu.device_count():
            raise ValueError(
                f"Ascend device index out of range: {device_index}; "
                f"found {torch_npu.npu.device_count()} device(s)"
            )

        self.device_index = device_index
        self.device = torch.device("npu", device_index)
        self.name = f"npu:{device_index}"
        self.distributed_backend = "hccl"

    def manual_seed(self, seed: int) -> None:
        torch_npu.npu.manual_seed(seed)
        torch_npu.npu.manual_seed_all(seed)

    def autocast(
            self,
            *,
            dtype: torch.dtype,
            enabled: bool = True,
    ):
        return torch.autocast(
            device_type="npu",
            dtype=dtype,
            enabled=enabled,
        )

    def sync(self) -> None:
        torch_npu.npu.synchronize(self.device)

    def set_device(self) -> None:
        torch_npu.npu.set_device(self.device_index)
