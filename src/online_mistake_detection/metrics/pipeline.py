from __future__ import annotations

from typing import Any

from ..pipeline.events import GroundTruthEvent, PipelineEvent
from ..schemas import WindowSpec


def evaluate_pipeline_events(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] | None = None,
    label_mode: str = "step_id",
    dataset: str = "egoper",
) -> dict[str, Any]:
    """Pure evaluation facade for event JSON produced by inference."""

    # Lazy import keeps event loading and schema inspection independent of model
    # dependencies while the legacy metric implementation is migrated.
    from ..pipeline.runner import compute_metrics, match_events

    match_events(events, ground_truth)
    return compute_metrics(
        events,
        ground_truth,
        module_a_predictions=module_a_predictions,
        label_mode=label_mode,
        dataset=dataset,
    )
