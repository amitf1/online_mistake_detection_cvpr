"""Point Triton / Inductor kernel caches at local, per-rank directories."""

from __future__ import annotations

import os
from pathlib import Path


def _local_rank() -> str:
    return str(os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0")


def _job_id() -> str:
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("WANDB_NAME")
        or os.environ.get("RUN_NAME")
        or "local"
    )


def _rank_dir(root: str, rank: str) -> str:
    suffix = f"/rank{rank}"
    normalized = root.rstrip("/")
    if normalized.endswith(suffix):
        return normalized
    return f"{normalized}{suffix}"


_CONFIGURED = False


def configure_kernel_caches(*, verbose: bool = True) -> str:
    global _CONFIGURED
    rank = _local_rank()
    job = _job_id()
    triton_root = os.environ.get("TRITON_CACHE_DIR") or f"/tmp/triton-cache-{job}"
    inductor_root = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or f"/tmp/torchinductor-{job}"
    cuda_root = os.environ.get("CUDA_CACHE_PATH") or f"/tmp/nv-cache-{job}"
    unsloth_root = (
        os.environ.get("UNSLOTH_COMPILE_LOCATION")
        or f"/tmp/unsloth-compile-{job}"
    )

    triton_dir = _rank_dir(triton_root, rank)
    inductor_dir = _rank_dir(inductor_root, rank)
    unsloth_dir = _rank_dir(unsloth_root, rank)
    Path(triton_dir).mkdir(parents=True, exist_ok=True)
    Path(inductor_dir).mkdir(parents=True, exist_ok=True)
    Path(cuda_root).mkdir(parents=True, exist_ok=True)
    Path(unsloth_dir).mkdir(parents=True, exist_ok=True)

    os.environ["TRITON_CACHE_DIR"] = triton_dir
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
    os.environ["CUDA_CACHE_PATH"] = cuda_root
    os.environ["UNSLOTH_COMPILE_LOCATION"] = unsloth_dir
    if verbose and not _CONFIGURED:
        print(
            f"Kernel caches rank={rank}: triton={triton_dir} "
            f"inductor={inductor_dir} cuda={cuda_root} unsloth={unsloth_dir}"
        )
    _CONFIGURED = True
    return triton_dir
