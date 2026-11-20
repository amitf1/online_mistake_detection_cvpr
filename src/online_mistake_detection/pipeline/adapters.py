"""Several LoRA adapters on one resident base model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

ADAPTER_CONFIG_KEYS = (
    "base_model_name_or_path",
    "peft_type",
    "r",
    "lora_alpha",
    "use_rslora",
    "use_dora",
    "lora_dropout",
    "modules_to_save",
)


def _saved_key(parameter_name: str, adapter_name: str) -> str | None:
    match = re.search(rf"\.(lora_[AB])\.{re.escape(adapter_name)}\.", parameter_name)
    if match is None:
        return None
    return parameter_name[: match.start()] + f".{match.group(1)}." + parameter_name[match.end():]


def lora_parameters(model: Any, adapter_name: str = "default") -> dict[str, Any]:
    """LoRA parameters keyed by their name in a saved adapter file."""

    parameters = {}
    for name, parameter in model.named_parameters():
        key = _saved_key(name, adapter_name)
        if key is not None:
            parameters[key] = parameter
    if not parameters:
        raise ValueError(f"Model has no LoRA parameters for adapter {adapter_name!r}")
    return parameters


def adapter_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    return {key: payload.get(key) for key in ADAPTER_CONFIG_KEYS}


def check_adapter_configs(paths: Mapping[str, str | Path]) -> None:
    configs = {name: adapter_config(path) for name, path in paths.items()}
    first_name, first = next(iter(configs.items()))
    for name, config in configs.items():
        if config.get("modules_to_save"):
            raise ValueError(f"Adapter {name} saves full modules; weight swapping cannot share them")
        differing = sorted(key for key in ADAPTER_CONFIG_KEYS if config.get(key) != first.get(key))
        if differing:
            raise ValueError(f"Adapter {name} differs from {first_name} in {differing}")


def load_adapter_weights(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")


class LoraAdapterBank:
    def __init__(
        self,
        model: Any,
        *,
        primary: str,
        adapters: Mapping[str, Mapping[str, Any]],
        adapter_name: str = "default",
    ) -> None:
        import torch

        self.model = model
        self.parameters = lora_parameters(model, adapter_name)
        self.states: dict[str, dict[str, Any]] = {
            primary: {key: parameter.detach().clone() for key, parameter in self.parameters.items()}
        }
        for name, weights in adapters.items():
            unknown = sorted(set(weights) - set(self.parameters))
            if unknown:
                raise ValueError(
                    f"Adapter {name} targets {len(unknown)} tensors the {primary} model lacks, "
                    f"e.g. {unknown[0]}"
                )
            state = {}
            for key, parameter in self.parameters.items():
                if key in weights:
                    if tuple(weights[key].shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"Adapter {name} tensor {key} has shape {tuple(weights[key].shape)}, "
                            f"expected {tuple(parameter.shape)}"
                        )
                    state[key] = weights[key].to(device=parameter.device, dtype=parameter.dtype)
                else:
                    state[key] = torch.zeros_like(parameter)
            self.states[name] = state
        self.active = primary

    def activate(self, name: str) -> None:
        if name == self.active:
            return
        import torch

        state = self.states[name]
        with torch.no_grad():
            for key, parameter in self.parameters.items():
                parameter.copy_(state[key])
        self.active = name
