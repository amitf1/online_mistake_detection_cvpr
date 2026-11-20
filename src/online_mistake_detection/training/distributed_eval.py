from __future__ import annotations

import os
from typing import Any


def distributed_rank_and_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def shard_examples_for_rank(
    examples: list[Any],
    *,
    rank: int,
    world_size: int,
) -> list[Any]:
    if world_size <= 1:
        return list(examples)
    return list(examples[rank::world_size])


def merge_round_robin_shards(shards: list[list[Any] | None]) -> list[Any]:
    clean = [list(shard or []) for shard in shards]
    merged: list[Any] = []
    max_len = max((len(shard) for shard in clean), default=0)
    for index in range(max_len):
        for shard in clean:
            if index < len(shard):
                merged.append(shard[index])
    return merged


def gather_round_robin_shards(local_rows: list[Any]) -> list[Any]:
    rank, world_size = distributed_rank_and_world()
    if world_size <= 1:
        return list(local_rows)
    import torch.distributed as dist

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    merged = merge_round_robin_shards(gathered)
    if rank == 0:
        print(
            f"Gathered {len(merged)} eval rows from {world_size} ranks "
            f"(local={len(local_rows)})."
        )
    return merged


def unwrap_trainer_model(trainer: Any, model: Any) -> Any:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return getattr(model, "module", model)
