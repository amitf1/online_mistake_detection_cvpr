from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Frame sampling settings shared by datasets and collators."""

    sample_fps: float = 2.0
    max_frames: int = 64
    resize_short_side: int | None = None
    strategy: Literal["uniform", "mixed_uniform_recent"] = "uniform"
    recent_fraction: float = 0.20


ModuleALabelMode = Literal["legacy", "step_id", "completion_set"]


@dataclass(frozen=True)
class ModuleAConfig:
    stride_seconds: float = 5.0
    completion_margin: float = 0.10
    negative_to_positive_ratio: int = 2
    keep_last_wait_windows: int = 2
    max_negative_windows_per_episode: int = 3
    reset_jitter_seconds: float = 0.0
    seed: int = 13
    label_mode: ModuleALabelMode = "step_id"
    context_seconds: float = 1200.0


@dataclass(frozen=True)
class ModuleBConfig:
    late_trigger_probability: float = 0.10
    pre_pad_ratio_min: float = 0.10
    pre_pad_ratio_max: float = 0.50
    post_pad_ratio_min: float = 0.10
    post_pad_ratio_max: float = 0.50
    include_incomplete_negatives: bool = True
    negative_ratio: float = 0.10
    seed: int = 17


@dataclass(frozen=True)
class ModuleCConfig:
    min_duration_seconds: float = 0.5
    jitter_ratio_min: float = 0.05
    jitter_ratio_max: float = 0.10
    include_not_completed_examples: bool = True
    not_completed_ratio: float = 0.10
    history_max_events: int = 12
    seed: int = 19


@dataclass(frozen=True)
class TimeTokenConfig:
    num_bins: int = 101
    token_template: str = "<time_{index:03d}>"
    no_action_token: str = "<no_action>"


def module_a_config_from_args(
    args: Any,
    *,
    label_mode: ModuleALabelMode | None = None,
    reset_jitter_seconds: float | None = None,
) -> ModuleAConfig:
    resolved_mode = label_mode or getattr(args, "module_a_label_mode", "step_id")
    return ModuleAConfig(
        stride_seconds=getattr(args, "stride_seconds", 5.0),
        completion_margin=getattr(args, "completion_margin", 0.10),
        negative_to_positive_ratio=getattr(args, "negative_to_positive_ratio", 2),
        keep_last_wait_windows=getattr(args, "keep_last_wait_windows", 2),
        max_negative_windows_per_episode=getattr(args, "max_negative_windows_per_episode", 3),
        reset_jitter_seconds=(
            float(reset_jitter_seconds)
            if reset_jitter_seconds is not None
            else float(getattr(args, "reset_jitter_seconds", 0.0))
        ),
        seed=getattr(args, "seed", 13),
        label_mode=resolved_mode,
    )


def warn_deprecated_module_a_sampling(args: Any) -> None:
    if getattr(args, "negative_to_positive_ratio", 2) != 2:
        warnings.warn(
            "--negative-to-positive-ratio is deprecated; Module A training now caps "
            "negatives with --max-negative-windows-per-episode.",
            DeprecationWarning,
            stacklevel=2,
        )
    if getattr(args, "keep_last_wait_windows", 2) != 2:
        warnings.warn(
            "--keep-last-wait-windows is deprecated; Module A training now keeps the "
            "first, midpoint, and final negatives per episode.",
            DeprecationWarning,
            stacklevel=2,
        )
