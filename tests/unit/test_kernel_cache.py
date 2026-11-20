from __future__ import annotations

from pathlib import Path

from online_mistake_detection.training.kernel_cache import configure_kernel_caches


def test_configure_kernel_caches_is_per_rank_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    import online_mistake_detection.training.kernel_cache as kernel_cache

    monkeypatch.setattr(kernel_cache, "_CONFIGURED", False)
    root = tmp_path / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(root))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "nv"))
    monkeypatch.setenv(
        "UNSLOTH_COMPILE_LOCATION",
        str(tmp_path / "unsloth"),
    )
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.delenv("RANK", raising=False)

    first = configure_kernel_caches(verbose=False)
    second = configure_kernel_caches(verbose=False)

    assert first == second
    assert first.endswith("/rank2")
    assert Path(first).is_dir()
    assert Path(first).resolve().is_relative_to(root.resolve())
    unsloth = Path(kernel_cache.os.environ["UNSLOTH_COMPILE_LOCATION"])
    assert unsloth.name == "rank2"
    assert unsloth.is_dir()
    assert unsloth.resolve().is_relative_to((tmp_path / "unsloth").resolve())
