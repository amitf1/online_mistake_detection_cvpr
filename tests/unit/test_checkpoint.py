from __future__ import annotations

import sys
from types import SimpleNamespace

from online_mistake_detection.models.checkpoint import write_trainer_checkpoint


def test_distributed_checkpoint_reads_only_current_cuda_rng(monkeypatch, tmp_path) -> None:
    saved: dict[str, object] = {}
    cuda_rng_state = object()

    class FakeCudaRandom:
        @staticmethod
        def get_rng_state() -> object:
            return cuda_rng_state

        @staticmethod
        def get_rng_state_all() -> object:
            raise AssertionError("must not initialize every visible CUDA device")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            random=FakeCudaRandom(),
        ),
        random=SimpleNamespace(get_rng_state=lambda: object()),
        save=lambda value, path: saved.__setitem__(str(path), value),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    trainer = SimpleNamespace(
        args=SimpleNamespace(world_size=8),
        state=SimpleNamespace(save_to_json=lambda path: None),
        optimizer=None,
        lr_scheduler=None,
        scaler=None,
        save_model=lambda path: None,
    )
    destination = write_trainer_checkpoint(trainer, tmp_path / "checkpoint")

    rng_states = saved[str(destination / "rng_state.pth")]
    assert isinstance(rng_states, dict)
    assert rng_states["cuda"] is cuda_rng_state
