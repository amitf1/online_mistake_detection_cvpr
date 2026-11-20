from __future__ import annotations

from typing import Any

from .config import TimeTokenConfig
from .datasets import EgoPERProvider
from .paths import egoper_root


def build_processor(
    model_id: str = "Qwen/Qwen3.5-4B",
    *,
    trust_remote_code: bool = True,
    add_time_tokens: bool = True,
    time_config: TimeTokenConfig | None = None,
) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if add_time_tokens:
        add_special_training_tokens(processor, time_config=time_config)
    return processor


def add_special_training_tokens(
    processor: Any,
    *,
    time_config: TimeTokenConfig | None = None,
) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return 0
    cfg = time_config or TimeTokenConfig()
    tokens = [
        cfg.token_template.format(index=index)
        for index in range(cfg.num_bins)
    ]
    tokens.append(cfg.no_action_token)
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [token for token in tokens if token not in existing_vocab]
    if not new_tokens:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": new_tokens}))


def build_provider(dataset: str = "egoper", **kwargs: Any) -> EgoPERProvider:
    if dataset.strip().lower() != "egoper":
        raise ValueError(f"Unsupported dataset: {dataset!r}; only 'egoper' is supported.")
    root = kwargs.pop("root", egoper_root())
    return EgoPERProvider(root, **kwargs)
