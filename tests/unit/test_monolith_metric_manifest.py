import pytest

from online_mistake_detection.metrics.abc_monolith import validate_metric_manifest


def test_manifest_reports_and_raises_for_missing_families() -> None:
    metrics = {"egoper_standard": {"tas": {}}}
    missing = validate_metric_manifest(metrics, raise_on_missing=False)
    assert "module_a" in missing
    assert "egoper_standard.tas" not in missing
    with pytest.raises(ValueError, match="Incomplete monolith metric artifact"):
        validate_metric_manifest(metrics)

