from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from online_mistake_detection.pipeline.adapters import (
    LoraAdapterBank,
    check_adapter_configs,
    lora_parameters,
)


class _Lora(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(size, size, bias=False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(size, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, size, bias=False)})


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language = _Lora(4)
        self.visual = _Lora(3)


KEYS = {
    "language": ("language.lora_A.weight", "language.lora_B.weight"),
    "visual": ("visual.lora_A.weight", "visual.lora_B.weight"),
}


def _language_only(model: _Model, value: float) -> dict[str, torch.Tensor]:
    return {
        "language.lora_A.weight": torch.full_like(model.language.lora_A["default"].weight, value),
        "language.lora_B.weight": torch.full_like(model.language.lora_B["default"].weight, value),
    }


def test_lora_parameters_use_saved_names() -> None:
    assert set(lora_parameters(_Model())) == {*KEYS["language"], *KEYS["visual"]}


def test_activate_swaps_weights_and_zeroes_untargeted_modules() -> None:
    model = _Model()
    original = {key: value.detach().clone() for key, value in lora_parameters(model).items()}
    bank = LoraAdapterBank(model, primary="A", adapters={"B": _language_only(model, 0.5)})

    bank.activate("B")
    params = lora_parameters(model)
    assert torch.all(params["language.lora_A.weight"] == 0.5)
    assert torch.all(params["visual.lora_A.weight"] == 0)
    assert torch.all(params["visual.lora_B.weight"] == 0)

    bank.activate("A")
    for key, value in lora_parameters(model).items():
        assert torch.equal(value, original[key])


def test_weights_take_the_model_dtype() -> None:
    model = _Model().to(torch.bfloat16)
    bank = LoraAdapterBank(model, primary="A", adapters={"B": _language_only(_Model(), 0.1)})
    bank.activate("B")
    weight = lora_parameters(model)["language.lora_A.weight"]
    assert weight.dtype == torch.bfloat16
    assert torch.all(weight == torch.tensor(0.1, dtype=torch.bfloat16))


def test_rejects_tensors_the_primary_lacks_or_wrong_shapes() -> None:
    model = _Model()
    with pytest.raises(ValueError, match="lacks"):
        LoraAdapterBank(model, primary="A", adapters={"B": {"other.lora_A.weight": torch.zeros(2, 4)}})
    with pytest.raises(ValueError, match="shape"):
        LoraAdapterBank(model, primary="A", adapters={"B": {"language.lora_A.weight": torch.zeros(3, 4)}})


def _write_config(path: Path, **overrides) -> Path:
    path.mkdir()
    config = {
        "base_model_name_or_path": "base",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "use_rslora": False,
        "use_dora": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "target_modules": str(path),
    }
    config.update(overrides)
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def test_adapter_configs_must_match_except_targets(tmp_path: Path) -> None:
    a = _write_config(tmp_path / "a")
    b = _write_config(tmp_path / "b")
    check_adapter_configs({"A": a, "B": b})
    c = _write_config(tmp_path / "c", lora_alpha=16)
    with pytest.raises(ValueError, match="lora_alpha"):
        check_adapter_configs({"A": a, "C": c})
    d = _write_config(tmp_path / "d", modules_to_save=["lm_head"])
    with pytest.raises(ValueError, match="full modules"):
        check_adapter_configs({"A": a, "D": d})
