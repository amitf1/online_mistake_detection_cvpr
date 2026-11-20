from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def validate_adapter_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid adapter checkpoint {checkpoint}; missing {missing}"
        )
    return checkpoint


def write_trainer_checkpoint(trainer: Any, path: str | Path) -> Path:
    """Write a full HF-style checkpoint: weights, optimizer, scheduler, RNG, step."""
    import torch

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(destination))
    torch.save(trainer.args, destination / "training_args.bin")
    trainer.state.save_to_json(str(destination / "trainer_state.json"))
    if getattr(trainer, "optimizer", None) is not None:
        torch.save(trainer.optimizer.state_dict(), destination / "optimizer.pt")
    if getattr(trainer, "lr_scheduler", None) is not None:
        torch.save(trainer.lr_scheduler.state_dict(), destination / "scheduler.pt")
    scaler = getattr(trainer, "scaler", None)
    if scaler is not None:
        torch.save(scaler.state_dict(), destination / "scaler.pt")
    rng_states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.random.get_rng_state()
    torch.save(rng_states, destination / "rng_state.pth")
    return destination
