from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..schemas import AggregateClass, VideoRecord


@runtime_checkable
class DatasetProvider(Protocol):
    """Dataset-neutral source of normalized procedural video records."""

    dataset_name: str

    @property
    def fine_taxonomy(self) -> Sequence[str]:
        ...

    @property
    def aggregate_mapping(self) -> Mapping[str, AggregateClass]:
        ...

    def iter_video_records(self) -> Iterable[VideoRecord]:
        ...

    def official_split(self, video_id: str) -> str | None:
        ...


def validate_provider(provider: DatasetProvider) -> None:
    fine = set(provider.fine_taxonomy)
    missing = fine.difference(provider.aggregate_mapping)
    if missing:
        raise ValueError(
            f"{provider.dataset_name} aggregate mapping is missing classes: "
            f"{sorted(missing)}"
        )
    for record in provider.iter_video_records():
        if record.dataset != provider.dataset_name:
            raise ValueError(
                f"Provider {provider.dataset_name!r} returned record for "
                f"{record.dataset!r}"
            )
        previous_start = -1.0
        for segment in record.segments:
            if segment.is_missing:
                raise ValueError(
                    f"{record.video_id}: missing-step entries cannot be visual segments"
                )
            if not 0.0 <= segment.start < segment.end:
                raise ValueError(
                    f"{record.video_id}: invalid segment [{segment.start}, {segment.end}]"
                )
            if segment.start < previous_start:
                raise ValueError(f"{record.video_id}: segments are not start-sorted")
            if segment.fine_class not in fine:
                raise ValueError(
                    f"{record.video_id}: unknown fine class {segment.fine_class!r}"
                )
            expected = provider.aggregate_mapping[segment.fine_class]
            if segment.aggregate_class != expected:
                raise ValueError(
                    f"{record.video_id}: {segment.fine_class} maps to {expected}, "
                    f"not {segment.aggregate_class}"
                )
            previous_start = segment.start
