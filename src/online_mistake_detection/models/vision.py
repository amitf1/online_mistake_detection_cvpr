from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]


QWEN_SPATIAL_PATCH = 32
QWEN_TEMPORAL_PATCH = 2


def parse_vision_resize(value: object) -> int:
    """Parse ``VISION_RESIZE``: positive long-edge cap, or 0 for native padding."""

    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("vision resize cannot be a boolean")
    if isinstance(value, (int, float)):
        parsed = int(value)
        if parsed < 0:
            raise ValueError("vision resize cannot be negative")
        return parsed
    cleaned = str(value).strip().lower()
    if cleaned in {"", "0", "native", "off", "none", "false"}:
        return 0
    parsed = int(cleaned)
    if parsed < 0:
        raise ValueError("vision resize cannot be negative")
    return parsed


def vision_resize_arg(value: str) -> int:
    return parse_vision_resize(value)


def ceil_to_grid(value: int, grid: int = QWEN_SPATIAL_PATCH) -> int:
    return max(grid, int(math.ceil(value / grid) * grid))


def floor_to_grid(value: int, grid: int = QWEN_SPATIAL_PATCH) -> int:
    return max(grid, (int(value) // grid) * grid)


@dataclass(frozen=True)
class SpatialResizePlan:
    source_width: int
    source_height: int
    width: int
    height: int
    mode: str
    interpolated: bool
    long_edge: int

    @property
    def padded_native(self) -> bool:
        return self.mode == "native" and not self.interpolated


def resolve_spatial_size(
    source_width: int,
    source_height: int,
    long_edge: int,
    *,
    grid: int = QWEN_SPATIAL_PATCH,
) -> SpatialResizePlan:
    width = max(1, int(source_width))
    height = max(1, int(source_height))
    cap = int(long_edge)
    source_long = max(width, height)
    if cap <= 0 or source_long <= cap:
        out_w = ceil_to_grid(width, grid)
        out_h = ceil_to_grid(height, grid)
        return SpatialResizePlan(
            source_width=width,
            source_height=height,
            width=out_w,
            height=out_h,
            mode="native",
            interpolated=False,
            long_edge=cap,
        )
    scale = cap / source_long
    scaled_w = width * scale
    scaled_h = height * scale
    out_w = floor_to_grid(math.floor(scaled_w), grid)
    out_h = floor_to_grid(math.floor(scaled_h), grid)
    return SpatialResizePlan(
        source_width=width,
        source_height=height,
        width=out_w,
        height=out_h,
        mode="resize",
        interpolated=True,
        long_edge=cap,
    )


def probe_video_size(path: str | Path) -> tuple[int, int] | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None
        return width, height
    finally:
        cap.release()


def estimate_vision_tokens(
    width: int,
    height: int,
    frames: int,
    *,
    spatial_patch: int = QWEN_SPATIAL_PATCH,
    temporal_patch: int = QWEN_TEMPORAL_PATCH,
) -> int:
    temporal = max(temporal_patch, (max(0, int(frames)) // temporal_patch) * temporal_patch)
    patches_w = max(1, int(width) // spatial_patch)
    patches_h = max(1, int(height) // spatial_patch)
    return patches_w * patches_h * (temporal // temporal_patch)


def even_frame_count(frames: int, *, temporal_patch: int = QWEN_TEMPORAL_PATCH) -> int:
    return max(temporal_patch, (int(frames) // temporal_patch) * temporal_patch)


def pad_spatial_array(array: Any, width: int, height: int) -> Any:
    """Pad a video or image array to ``height`` x ``width`` without interpolation."""

    import numpy as np

    data = np.asarray(array)
    if data.ndim == 3:
        current_h, current_w = data.shape[:2]
        pad_h = max(0, height - current_h)
        pad_w = max(0, width - current_w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return np.pad(data, ((top, bottom), (left, right), (0, 0)))
    if data.ndim == 4:
        if data.shape[-1] in {1, 3, 4}:
            current_h, current_w = data.shape[1], data.shape[2]
            pad_h = max(0, height - current_h)
            pad_w = max(0, width - current_w)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            return np.pad(data, ((0, 0), (top, bottom), (left, right), (0, 0)))
        current_h, current_w = data.shape[-2], data.shape[-1]
        pad_h = max(0, height - current_h)
        pad_w = max(0, width - current_w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return np.pad(data, ((0, 0), (0, 0), (top, bottom), (left, right)))
    raise ValueError(f"Unsupported video array rank: {data.ndim}")


def pad_video_to_grid(video: Any, *, grid: int = QWEN_SPATIAL_PATCH) -> Any:
    torch = None
    try:
        import torch as torch_mod
    except ModuleNotFoundError:
        torch_mod = None  # type: ignore[assignment]
    if torch_mod is not None and torch_mod.is_tensor(video):
        torch = torch_mod
        data = video
        if data.ndim == 4 and data.shape[1] in {1, 3, 4}:
            height, width = int(data.shape[2]), int(data.shape[3])
            target_h = ceil_to_grid(height, grid)
            target_w = ceil_to_grid(width, grid)
            pad_h = target_h - height
            pad_w = target_w - width
            if pad_h == 0 and pad_w == 0:
                return data
            return torch.nn.functional.pad(
                data,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        if data.ndim == 4:
            height, width = int(data.shape[1]), int(data.shape[2])
            target_h = ceil_to_grid(height, grid)
            target_w = ceil_to_grid(width, grid)
            pad_h = target_h - height
            pad_w = target_w - width
            if pad_h == 0 and pad_w == 0:
                return data
            return torch.nn.functional.pad(
                data,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return data

    import numpy as np

    data = np.asarray(video)
    if data.ndim == 3:
        height, width = data.shape[:2]
        return pad_spatial_array(data, ceil_to_grid(width, grid), ceil_to_grid(height, grid))
    if data.ndim == 4 and data.shape[-1] in {1, 3, 4}:
        height, width = data.shape[1], data.shape[2]
        return pad_spatial_array(data, ceil_to_grid(width, grid), ceil_to_grid(height, grid))
    if data.ndim == 4:
        height, width = data.shape[-2], data.shape[-1]
        return pad_spatial_array(data, ceil_to_grid(width, grid), ceil_to_grid(height, grid))
    return video


def _size_from_array(video: Any) -> tuple[int, int] | None:
    shape = getattr(video, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    if len(shape) == 2:
        return int(shape[1]), int(shape[0])
    if shape[-1] in {1, 3, 4}:
        return int(shape[-2]), int(shape[1] if len(shape) == 4 else shape[0])
    return int(shape[-1]), int(shape[-2])


def raise_qwen_video_frame_pixel_limit(min_pixels: int, *, image_factor: int = QWEN_SPATIAL_PATCH) -> None:
    """Raise qwen-vl-utils' per-frame cap so native 720p is not clamped."""

    try:
        import qwen_vl_utils.vision_process as vision_process
    except ImportError:
        return
    factor = max(1, int(image_factor))
    pixels = max(1, int(min_pixels))
    needed = max(1, math.ceil(pixels / (factor * factor)))
    current = int(getattr(vision_process, "VIDEO_MAX_TOKEN_NUM", 0) or 0)
    if current < needed:
        vision_process.VIDEO_MAX_TOKEN_NUM = needed
    current_pixels = int(getattr(vision_process, "VIDEO_MAX_PIXELS", 0) or 0)
    if hasattr(vision_process, "VIDEO_MAX_PIXELS") and current_pixels < pixels:
        vision_process.VIDEO_MAX_PIXELS = pixels


def apply_resize_to_video_part(part: dict[str, Any], long_edge: int) -> SpatialResizePlan | None:
    video = part.get("video")
    if isinstance(video, (str, Path)):
        size = probe_video_size(video)
    else:
        size = _size_from_array(video)
    if size is None:
        return None
    plan = resolve_spatial_size(size[0], size[1], long_edge)
    if plan.mode == "native":
        # Ask fetch_video for the padded 32-grid size (1280×736 for 720p).
        # Passing source 720 lets smart_resize round to 704 and interpolate.
        part["resized_width"] = plan.width
        part["resized_height"] = plan.height
        part["min_pixels"] = 1
        part.pop("max_pixels", None)
        raise_qwen_video_frame_pixel_limit(plan.width * plan.height)
    else:
        part["resized_width"] = plan.width
        part["resized_height"] = plan.height
    part["vision_source_width"] = plan.source_width
    part["vision_source_height"] = plan.source_height
    part["vision_effective_width"] = plan.width
    part["vision_effective_height"] = plan.height
    part["vision_resize_mode"] = plan.mode
    part["vision_interpolated"] = plan.interpolated
    return plan


def apply_resize_to_messages(messages: list[dict[str, Any]], long_edge: int) -> list[SpatialResizePlan]:
    plans: list[SpatialResizePlan] = []
    for message in messages:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                plan = apply_resize_to_video_part(part, long_edge)
                if plan is not None:
                    plans.append(plan)
    return plans


def _pad_tchw(video: Any, target_h: int, target_w: int) -> Any:
    import torch

    height, width = int(video.shape[-2]), int(video.shape[-1])
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h == 0 and pad_w == 0:
        return video
    return torch.nn.functional.pad(
        video,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
    )


@contextlib.contextmanager
def pad_instead_of_interpolate() -> Iterator[None]:
    """Replace spatial upscaling with padding so native 1280x720 becomes 1280x736."""

    import torchvision.transforms.functional as functional

    original = functional.resize

    def wrapped(video: Any, size: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(video, "ndim") and int(getattr(video, "ndim", 0)) == 4:
            target_h, target_w = int(size[0]), int(size[1])
            height, width = int(video.shape[-2]), int(video.shape[-1])
            if height <= target_h and width <= target_w:
                return _pad_tchw(video, target_h, target_w)
        return original(video, size, *args, **kwargs)

    functional.resize = wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        functional.resize = original  # type: ignore[method-assign]
