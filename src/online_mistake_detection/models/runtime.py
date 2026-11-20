"""Shared model-loading and generation surface for training and inference."""

from ..training.module_a import (
    ModuleADetectionMetricsCallback,
    Qwen3VLMetadataVisionDataCollator,
    binary_detection_metrics,
    extract_module_a_label,
    json_safe,
    load_generation_eval_model,
    load_or_create_video_split,
    multiclass_letter_metrics,
    release_cuda_memory,
    training_example_to_conversation as module_a_to_conversation,
)
from ..training.module_b import (
    ModuleBTemporalMetricsCallback,
    parse_temporal_prediction,
    temporal_iou,
    temporal_metrics,
    training_example_to_conversation as module_b_to_conversation,
)
from ..training.module_c import (
    ModuleCMistakeMetricsCallback,
    lexical_overlap,
    parse_module_c_prediction,
    training_example_to_conversation as module_c_to_conversation,
)

__all__ = [
    "ModuleADetectionMetricsCallback",
    "ModuleBTemporalMetricsCallback",
    "ModuleCMistakeMetricsCallback",
    "Qwen3VLMetadataVisionDataCollator",
    "binary_detection_metrics",
    "extract_module_a_label",
    "json_safe",
    "lexical_overlap",
    "load_generation_eval_model",
    "load_or_create_video_split",
    "module_a_to_conversation",
    "module_b_to_conversation",
    "module_c_to_conversation",
    "multiclass_letter_metrics",
    "parse_module_c_prediction",
    "parse_temporal_prediction",
    "release_cuda_memory",
    "temporal_iou",
    "temporal_metrics",
]
