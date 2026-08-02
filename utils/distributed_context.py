import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def from_env(cls) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            return cls(rank=0, local_rank=0, world_size=1)
        return cls(
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=world_size,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0
