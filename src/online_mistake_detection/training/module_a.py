from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import gc
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..config import ModuleAConfig, warn_deprecated_module_a_sampling
from ..models.vision import vision_resize_arg
from ..prompts.module_a import (
    MODULE_A_COMPLETION_SET_SYSTEM_PROMPT,
    MODULE_A_STEP_ID_SYSTEM_PROMPT,
    MODULE_A_SYSTEM_PROMPT,
)
from ..windowing.module_a import cap_negative_conversations
from ..datasets import (
    EgoPERProvider,
    ModuleTrainingDataset,
    build_module_a_training_dataset,
)
from ..models.checkpoint import write_trainer_checkpoint
from ..paths import egoper_root
from ..datasets.labels import (
    MODULE_A_STEP_ID_EXTRA_LETTER,
    MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER,
    MODULE_A_STEP_ID_NONE_LETTER,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
)
from ..schemas import TrainingExample
from .distributed_eval import (
    distributed_rank_and_world,
    gather_round_robin_shards,
    shard_examples_for_rank,
    unwrap_trainer_model,
)
from .kernel_cache import configure_kernel_caches
from .preemption import add_preemption_args, apply_resume_checkpoint, attach_cluster_checkpointing

DEFAULT_OUTPUT_DIR = str(Path(os.environ.get("OUTPUT_ROOT", "outputs")) / "module_a")
MODULE_A_SPLIT_VERSION = 1
MODULE_A_LABELS = ("WAIT", "COMPLETE")
MODULE_A_POSITIVE_LABEL = "COMPLETE"
MODULE_A_STEP_ID_POSITIVE_NOT = MODULE_A_STEP_ID_NONE_LETTER
MODULE_A_STEP_ID_CANDIDATE_LABELS = tuple(chr(ord("A") + index) for index in range(26))
SEMANTIC_NONE = "NONE"
SEMANTIC_EXTRA = "EXTRA"


def rendered_answer_tail_token_ids(
    tokenizer_or_processor: Any,
    rendered_text: str,
    answer: str,
) -> list[int]:
    """Return tokens from the token containing the final answer through chat end."""
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    answer_start = rendered_text.rfind(answer)
    if answer_start < 0:
        raise RuntimeError(f"Could not find answer {answer!r} in rendered chat template.")
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    for token_index, (start, end) in enumerate(offsets):
        if int(start) <= answer_start < int(end):
            return [int(token_id) for token_id in token_ids[token_index:]]
    raise RuntimeError(
        f"Could not map answer {answer!r} to a rendered chat-template token."
    )


def answer_letter_index_from_tail(
    input_ids_row: Any,
    attention_mask_row: Any,
    answer_tail: list[int],
    *,
    max_seq_length: int,
    video_grid_thw: Any = None,
    row_id: Any = None,
) -> int:
    """Return the sequence index of the first answer-tail token (the class letter)."""
    import torch

    valid_positions = (attention_mask_row != 0).nonzero(as_tuple=False).reshape(-1)
    valid_ids = input_ids_row[valid_positions]
    expected_tail = torch.tensor(
        answer_tail,
        dtype=valid_ids.dtype,
        device=valid_ids.device,
    )
    if (
        valid_ids.shape[0] < expected_tail.shape[0]
        or not torch.equal(valid_ids[-expected_tail.shape[0] :], expected_tail)
    ):
        grid_value = None
        if torch.is_tensor(video_grid_thw):
            grid_value = video_grid_thw.detach().cpu().tolist()
        raise RuntimeError(
            "Assistant answer tail was right-truncated"
            f"{'' if row_id is None else f' for row {row_id}'} at sequence_length="
            f"{int(valid_ids.shape[0])}, max_seq_length={max_seq_length}, "
            f"video_grid_thw={grid_value}."
        )
    return int(valid_positions[-expected_tail.shape[0]].item())


def apply_letter_only_labels(
    labels: Any,
    letter_indices: Any,
    *,
    ignore_index: int = -100,
) -> Any:
    """Keep CE on the answer letter; mask every other labeled response token."""
    import torch

    if labels.shape[0] != letter_indices.shape[0]:
        raise RuntimeError(
            "Letter-only label mask size mismatch: "
            f"labels={tuple(labels.shape)} indices={tuple(letter_indices.shape)}."
        )
    masked = torch.full_like(labels, ignore_index)
    batch = torch.arange(labels.shape[0], device=labels.device)
    selected = labels[batch, letter_indices]
    if bool((selected == ignore_index).any()):
        bad = (selected == ignore_index).nonzero(as_tuple=False).reshape(-1).tolist()
        raise RuntimeError(
            "Letter-only loss requires the answer token to be a labeled response "
            f"token; fully masked rows={bad}."
        )
    masked[batch, letter_indices] = selected
    return masked


def contextual_step_id_candidate_token_ids(
    processor: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """Token IDs for A-Z at the exact assistant-answer boundary."""
    candidate_ids: list[int] = []
    for candidate in MODULE_A_STEP_ID_CANDIDATE_LABELS:
        candidate_messages = copy.deepcopy(messages)
        assistant = next(
            message
            for message in candidate_messages
            if message.get("role") == "assistant"
        )
        content = assistant.get("content")
        if isinstance(content, list):
            assistant["content"] = [{"type": "text", "text": candidate}]
        else:
            assistant["content"] = candidate
        rendered = processor.apply_chat_template(
            candidate_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        candidate_ids.append(
            rendered_answer_tail_token_ids(processor, rendered, candidate)[0]
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError(
            "Contextual Module A candidate tokens are not unique; "
            f"ids={candidate_ids}."
        )
    return candidate_ids


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def disable_dynamo_lru_cache() -> bool:
    """Keep checkpoint recompute on the same compiled graph as forward."""
    try:
        import torch
    except ModuleNotFoundError:
        return False
    eval_frame = getattr(getattr(getattr(torch, "_C", None), "_dynamo", None), "eval_frame", None)
    set_lru = getattr(eval_frame, "_set_lru_cache", None)
    if not callable(set_lru):
        return False
    set_lru(False)
    return True


def is_eval_only_run(args: argparse.Namespace) -> bool:
    """True when the run scores validation once and exits, base or adapter."""
    return bool(
        getattr(args, "zero_shot_eval", False) or getattr(args, "eval_only", False)
    )


def inference_runtime_metrics(
    response_seconds: list[float],
    *,
    amortized_seconds: list[float] | None = None,
) -> dict[str, float]:
    """Summarize request latency and per-rank inference throughput."""

    if not response_seconds:
        return {}
    amortized = amortized_seconds or response_seconds
    mean_response = sum(response_seconds) / len(response_seconds)
    mean_amortized = sum(amortized) / len(amortized)
    ordered_response = sorted(response_seconds)
    p95_position = 0.95 * (len(ordered_response) - 1)
    p95_lower = math.floor(p95_position)
    p95_upper = math.ceil(p95_position)
    p95_weight = p95_position - p95_lower
    p95_response = (
        ordered_response[p95_lower] * (1.0 - p95_weight)
        + ordered_response[p95_upper] * p95_weight
    )
    return {
        "eval_inference/mean_response_seconds": mean_response,
        "eval_inference/p95_response_seconds": p95_response,
        "eval_inference/mean_seconds_per_example": mean_amortized,
        "eval_inference/examples_per_second_per_rank": (
            1.0 / mean_amortized if mean_amortized > 0.0 else 0.0
        ),
    }


def trainer_inference_runtime_metrics(
    metrics: dict[str, Any],
    *,
    num_examples: int,
    world_size: int,
    per_device_batch_size: int,
) -> dict[str, float]:
    """Derive eval-only timings from Trainer's wall-clock runtime."""

    runtime = float(metrics.get("eval_runtime") or 0.0)
    if runtime <= 0.0 or num_examples <= 0:
        return {}
    ranks = max(int(world_size), 1)
    batch_size = max(int(per_device_batch_size), 1)
    examples_per_rank = math.ceil(num_examples / ranks)
    batches_per_rank = math.ceil(examples_per_rank / batch_size)
    mean_amortized = runtime * ranks / num_examples
    return {
        "eval_inference/mean_response_seconds": runtime / batches_per_rank,
        "eval_inference/mean_seconds_per_example": mean_amortized,
        "eval_inference/examples_per_second_per_rank": 1.0 / mean_amortized,
    }


def configure_unsloth_logits(args: argparse.Namespace) -> None:
    if disable_dynamo_lru_cache():
        print(
            "Disabled torch dynamo LRU cache so gradient-checkpoint recompute "
            "uses the same compiled graph as forward."
        )
    if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1":
        print("UNSLOTH_COMPILE_DISABLE=1: Unsloth torch.compile is off.")
    if (
        args.positive_loss_weight != 1.0
        or args.label_score_eval
        or args.letter_only_loss
        or is_eval_only_run(args)
    ):
        # Hidden states only. Returning full [seq, vocab] logits OOMs native video
        # (25k x 248k x fp32 ≈ 24 GiB) on the LM-head projection.
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "/tmp/unsloth_compiled_cache_module_a_logits")
        print(
            "Unsloth hidden states enabled for weighted loss / label scoring "
            f"(compile cache: {os.environ['UNSLOTH_COMPILE_LOCATION']})."
        )


def unwrap_output_embeddings(model: Any, trainer: Any | None = None) -> Any:
    model = unwrap_trainer_model(trainer, model)
    while not hasattr(model, "get_output_embeddings") and hasattr(model, "module"):
        model = model.module
    return model.get_output_embeddings()


def last_hidden_states(outputs: Any) -> Any:
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    last = None
    if isinstance(hidden_states, list | tuple) and hidden_states:
        last = hidden_states[-1]
    elif torch.is_tensor(hidden_states) and hidden_states.ndim >= 3:
        last = hidden_states
    if last is None:
        logits = getattr(outputs, "logits", None)
        if torch.is_tensor(logits) and logits.ndim >= 3:
            last = logits
    if last is None:
        raise RuntimeError(
            "Could not obtain hidden states from Unsloth output. "
            "Weighted positive loss requires output_hidden_states=True."
        )
    # output_hidden_states keeps every layer. Gradient checkpointing cannot
    # free those tensors while this tuple/list still holds them.
    if isinstance(hidden_states, list):
        hidden_states.clear()
        hidden_states.append(last)
    elif isinstance(hidden_states, tuple):
        outputs.hidden_states = (last,)
    return last


def logits_for_loss(model: Any, outputs: Any, trainer: Any | None = None) -> Any:
    import torch

    output_embeddings = unwrap_output_embeddings(model, trainer)
    hidden = last_hidden_states(outputs)
    vocab_size = getattr(output_embeddings, "out_features", None)
    if vocab_size is not None and hidden.shape[-1] == vocab_size:
        return hidden
    weight = getattr(output_embeddings, "weight", None)
    if torch.is_tensor(weight):
        hidden = hidden.to(dtype=weight.dtype)
    return output_embeddings(hidden)


def response_token_loss(
    model: Any,
    outputs: Any,
    labels: Any,
    *,
    token_weights: Any | None = None,
    trainer: Any | None = None,
    ignore_index: int = -100,
    reduction: str = "weighted_mean",
) -> Any:
    """Cross-entropy on labeled response tokens only."""
    import torch
    import torch.nn.functional as F

    output_embeddings = unwrap_output_embeddings(model, trainer)
    hidden = last_hidden_states(outputs)
    vocab_size = getattr(output_embeddings, "out_features", None)
    shift_labels = labels[:, 1:]
    shift_hidden = hidden[:, :-1]
    if vocab_size is not None and shift_hidden.shape[-1] == vocab_size:
        token_logits = shift_hidden
    else:
        token_logits = None
    if token_weights is None:
        shift_weights = torch.ones_like(shift_labels, dtype=torch.float32)
    else:
        shift_weights = token_weights[:, 1:].to(device=shift_labels.device)
    token_mask = shift_labels != ignore_index
    if not bool(token_mask.any()):
        raise RuntimeError(
            "Module A batch has no labeled response tokens. The assistant response "
            "was likely right-truncated; reduce MAX_FRAMES/VISION_RESIZE or increase "
            "MAX_SEQ_LENGTH."
        )

    selected_labels = shift_labels[token_mask]
    selected_weights = shift_weights[token_mask].to(dtype=torch.float32)
    if token_logits is None:
        selected_hidden = shift_hidden[token_mask]
        weight = getattr(output_embeddings, "weight", None)
        if torch.is_tensor(weight):
            selected_hidden = selected_hidden.to(dtype=weight.dtype)
        selected_logits = output_embeddings(selected_hidden).float()
    else:
        selected_logits = token_logits[token_mask].float()
    token_nll = F.cross_entropy(selected_logits, selected_labels, reduction="none")
    if reduction == "sum":
        return token_nll.sum()
    if reduction == "mean":
        return token_nll.mean()
    weighted = token_nll * selected_weights.to(dtype=token_nll.dtype)
    return weighted.sum() / selected_weights.sum().clamp_min(1.0)


def step_id_letter_logits_from_outputs(
    model: Any,
    outputs: Any,
    answer_positions: Any,
    candidate_token_ids: Any,
    *,
    trainer: Any | None = None,
) -> Any:
    """Reuse an eval forward to score A-Z at each response-letter position."""
    import torch

    hidden = last_hidden_states(outputs)
    positions = answer_positions.reshape(-1).to(device=hidden.device, dtype=torch.long)
    if positions.shape[0] != hidden.shape[0] or bool((positions <= 0).any()):
        raise RuntimeError(
            f"Invalid Module A answer positions: {positions.detach().cpu().tolist()}."
        )
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    answer_hidden = hidden[batch_indices, positions - 1]

    output_embeddings = unwrap_output_embeddings(model, trainer)
    vocab_size = getattr(output_embeddings, "out_features", None)
    candidate_ids = candidate_token_ids.to(
        device=answer_hidden.device,
        dtype=torch.long,
    )
    if vocab_size is not None and answer_hidden.shape[-1] == vocab_size:
        return answer_hidden.gather(1, candidate_ids)
    weight = getattr(output_embeddings, "weight", None)
    if torch.is_tensor(weight):
        answer_hidden = answer_hidden.to(device=weight.device, dtype=weight.dtype)
    full_logits = output_embeddings(answer_hidden)
    return full_logits.gather(1, candidate_ids.to(full_logits.device))


def generation_eval_examples(
    examples: list[dict[str, Any]],
    max_samples: int,
) -> list[dict[str, Any]]:
    if max_samples < 0:
        return list(examples)
    return examples[:max_samples]


def checkpoint_needs_resume_evaluation(checkpoint: str | Path) -> bool:
    """True only for a completed epoch that has no eval logged at its step."""
    state_path = Path(checkpoint) / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot determine whether resumed checkpoint needs evaluation: {state_path}"
        ) from exc

    epoch = state.get("epoch")
    if not isinstance(epoch, int | float) or epoch <= 0:
        return False
    completed_epoch = round(float(epoch))
    if not math.isclose(float(epoch), completed_epoch, rel_tol=0.0, abs_tol=1e-6):
        return False

    global_step = int(state.get("global_step", -1))
    for entry in state.get("log_history") or []:
        if not isinstance(entry, dict):
            continue
        try:
            entry_step = int(entry.get("step", -2))
        except (TypeError, ValueError):
            continue
        if entry_step != global_step:
            continue
        if any(str(key).startswith(("eval_", "eval/")) for key in entry):
            return False
    return True


def cap_conversations(
    conversations: list[dict[str, Any]],
    max_examples: int,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    if max_examples < 0:
        return list(conversations)
    capped = conversations[:max_examples]
    print(
        f"Capped Module A {split_name} examples from {len(conversations)} to "
        f"{len(capped)} (--max-{split_name}-examples={max_examples})."
    )
    if not capped:
        raise ValueError(f"Capping {split_name} examples produced an empty split.")
    return capped


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def offload_trainer_cuda_state(trainer: Any) -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    trainer.model.to("cpu")
    move_optimizer_state_to_device(getattr(trainer, "optimizer", None), torch.device("cpu"))
    release_cuda_memory()
    return torch.device("cuda")


def restore_trainer_cuda_state(trainer: Any, device: Any | None) -> None:
    if device is None:
        return
    trainer.model.to(device)
    move_optimizer_state_to_device(getattr(trainer, "optimizer", None), device)
    release_cuda_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 for Module A online step detection with Unsloth."
    )

    # Data
    parser.add_argument("--dataset", choices=("egoper",), default="egoper")
    parser.add_argument("--egoper-root", default=str(egoper_root()))
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=-1,
        help="Keep only the first N train conversations after sampling. -1 keeps all.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=-1,
        help="Keep only the first N validation conversations. -1 keeps all.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="FPS value written into video messages. Keep low for 16GB GPUs.",
    )
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)

    # Module A simulation
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--completion-margin", type=float, default=0.10)
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=int,
        default=2,
        help="Deprecated. Use --max-negative-windows-per-episode.",
    )
    parser.add_argument(
        "--keep-last-wait-windows",
        type=int,
        default=2,
        help="Deprecated. Use --max-negative-windows-per-episode.",
    )
    parser.add_argument(
        "--max-negative-windows-per-episode",
        type=int,
        default=3,
        help="Train-only cap on NONE windows kept per completion episode.",
    )
    parser.add_argument(
        "--reset-jitter-seconds",
        type=float,
        default=0.5,
        help="Train-only backward reset jitter. Inference should pass 0.",
    )
    parser.add_argument(
        "--module-a-label-mode",
        choices=("legacy", "step_id", "completion_set"),
        default="step_id",
        help=(
            "legacy=WAIT/COMPLETE; step_id=A/B/C+ letters; "
            "completion_set=JSON completed_steps for overlapping datasets."
        ),
    )
    parser.add_argument(
        "--positive-oversample-factor",
        type=float,
        default=1.0,
        help="Train-only positive oversampling factor. 1.0 keeps the natural split.",
    )
    parser.add_argument(
        "--positive-loss-weight",
        type=float,
        default=2.0,
        help="Train-time loss multiplier for positive assistant response tokens.",
    )
    parser.add_argument(
        "--letter-only-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train CE only on the step-ID answer letter. Chat-template tokens "
            "stay masked. Eval still scores the full response for eval_loss."
        ),
    )
    parser.add_argument(
        "--load-adapter-from",
        default=(os.environ.get("LOAD_ADAPTER_FROM") or "").strip() or None,
        help=(
            "Load LoRA weights only and start a fresh Adam/LR schedule. Use this "
            "when continuing from ckpt_ep1 with a changed loss. Cluster requeues "
            "still use --resume / $CKPT_DIR/latest.json."
        ),
    )
    parser.add_argument(
        "--zero-shot-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the same label-score validation as training eval on the base "
            "model (identity LoRA, no optimizer steps), then exit."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Score the adapter named by --load-adapter-from with the training "
            "eval path, then exit without taking an optimizer step. Same "
            "metrics as --zero-shot-eval, so the two are directly comparable."
        ),
    )

    # Model / LoRA
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lora-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Storage and compute dtype for trainable LoRA parameters; base quantization is separate.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        choices=("auto", "all-linear"),
        default="auto",
        help=(
            "LoRA target selection. 'auto' lets Unsloth derive targets from the finetune_* flags. "
            "'all-linear' explicitly targets all linear layers and therefore requires all finetune_* "
            "flags to be true."
        ),
    )
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-resize", type=vision_resize_arg, default=512)

    # Training
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--train-mode",
        choices=("steps", "epochs"),
        default="steps",
        help="Cap training by optimizer steps or by epochs.",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=2,
        help="Save rolling ckpt_epN checkpoints every N completed epochs.",
    )
    parser.add_argument(
        "--save-strategy",
        choices=("steps", "epoch"),
        default="steps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    add_preemption_args(parser)
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="HF Trainer optimizer. Default fused AdamW; use adamw_8bit only if optimizer memory is tight.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=1,
        help=(
            "Call torch.cuda.empty_cache() every N optimizer steps; 0 disables "
            "it. Every step was the default when the 35B MoE ran at the memory "
            "ceiling, but it synchronizes and drops the caching allocator's "
            "blocks, so it costs real throughput once memory is not scarce."
        ),
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Batch windows of similar duration together. At per-device batch 1 "
            "there is no padding to save; the win is that all DDP ranks get "
            "similar-length windows, so a step is no longer paced by whichever "
            "rank drew the longest clip."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help=(
            "Gradient clipping threshold. Matches the Trainer default, but is "
            "exposed because a run can sit at 100%% clipped for its whole "
            "duration without anything in the config revealing it."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        help="Comma-separated Trainer loggers, for example: tensorboard,wandb.",
    )
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Batches prefetched per worker. None uses the HuggingFace/PyTorch default.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep enabled for routed MoE experts that may be unused in a local batch.",
    )
    parser.add_argument("--ddp-timeout", type=int, default=7200)

    # Validation / execution mode
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON file with train/validation video IDs. Created if missing.",
    )
    parser.add_argument(
        "--regenerate-split",
        action="store_true",
        help="Overwrite an existing split file using the configured split policy.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--train-videos-per-task",
        type=int,
        default=None,
        help="Optional deterministic cap per task from the training split; useful for smoke runs.",
    )
    parser.add_argument(
        "--train-validation-erroneous-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For capped EgoPER smoke runs, choose train/validation videos only from erroneous videos.",
    )
    parser.add_argument(
        "--val-videos-per-task",
        type=int,
        default=2,
        help="Validation videos per task for the default 50-video corpus.",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=2,
        help="Run validation every N completed epochs.",
    )
    parser.add_argument(
        "--eval-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run validation before training only when the restored checkpoint is "
            "at an epoch boundary with no evaluation logged at that step."
        ),
    )
    parser.add_argument(
        "--eval-strategy",
        choices=("steps", "epoch"),
        default="steps",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-generation-max-samples",
        type=int,
        default=-1,
        help="Generated-label metric sample cap. Use -1 for the full validation split, 0 to skip.",
    )
    parser.add_argument(
        "--generation-eval-mode",
        choices=("subprocess", "inline", "off"),
        default="inline",
        help="How to run generated-label validation metrics. inline shards across DDP ranks.",
    )
    parser.add_argument("--generation-eval-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--eval-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generation-eval-output-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--write-confusion-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --generation-eval-only, write per-task confusion-matrix heatmap PNGs "
            "next to the output JSON (offline / post-training). Off during train-time eval."
        ),
    )
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--label-score-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate Module A by min-NLL over candidate labels (default on). "
            "Use --no-label-score-eval for free generate+parse (debug)."
        ),
    )
    parser.add_argument(
        "--write-calibration-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write per-window candidate-letter probabilities as JSONL at every "
            "label-score eval, so a decision threshold can be swept offline."
        ),
    )
    parser.add_argument(
        "--calibration-output-dir",
        default=None,
        help="Where to write calibration JSONL. Defaults to --output-dir.",
    )
    parser.add_argument("--fbeta-beta", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--wandb-log-best-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload every newly best checkpoint as a W&B model artifact.",
    )
    parser.add_argument(
        "--wandb-artifact-prefix",
        default="module-a",
        help="Prefix for W&B checkpoint artifact names.",
    )
    parser.add_argument("--keep-last-checkpoints", type=int, default=4)
    parser.add_argument("--keep-best-checkpoints", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and print examples only.")
    parser.add_argument("--print-examples", type=int, default=1)
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.train_mode == "steps" and args.max_steps <= 0:
        parser.error("--max-steps must be positive when --train-mode=steps.")
    if args.train_mode == "epochs" and args.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be positive when --train-mode=epochs.")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1.")
    if args.val_videos_per_task < 1:
        parser.error("--val-videos-per-task must be positive.")
    if args.train_videos_per_task is not None and args.train_videos_per_task < 1:
        parser.error("--train-videos-per-task must be positive when provided.")
    if args.eval_steps < 1:
        parser.error("--eval-steps must be positive.")
    if args.eval_epochs < 1:
        parser.error("--eval-epochs must be positive.")
    if args.max_train_examples < -1 or args.max_train_examples == 0:
        parser.error("--max-train-examples must be -1 or a positive count.")
    if args.max_eval_examples < -1 or args.max_eval_examples == 0:
        parser.error("--max-eval-examples must be -1 or a positive count.")
    if args.eval_generation_max_samples < -1:
        parser.error("--eval-generation-max-samples must be -1, 0, or positive.")
    if args.letter_only_loss and args.module_a_label_mode != "step_id":
        parser.error("--letter-only-loss requires --module-a-label-mode=step_id.")
    if args.zero_shot_eval and args.generation_eval_only:
        parser.error("--zero-shot-eval cannot be combined with --generation-eval-only.")
    if args.zero_shot_eval and args.dry_run:
        parser.error("--zero-shot-eval cannot be combined with --dry-run.")
    if args.zero_shot_eval and args.load_adapter_from:
        parser.error("--zero-shot-eval is the base model; do not pass --load-adapter-from.")
    if args.eval_only and args.zero_shot_eval:
        parser.error("--eval-only scores an adapter; --zero-shot-eval scores the base model.")
    if args.eval_only and not args.load_adapter_from:
        parser.error("--eval-only requires --load-adapter-from (use --zero-shot-eval for base).")
    if args.eval_only and args.generation_eval_only:
        parser.error("--eval-only cannot be combined with --generation-eval-only.")
    if args.eval_only and args.dry_run:
        parser.error("--eval-only cannot be combined with --dry-run.")
    if args.eval_only and args.eval_generation_max_samples == 0:
        parser.error("--eval-only requires --eval-generation-max-samples != 0.")
    if args.load_adapter_from and getattr(args, "initial_resume_from_checkpoint", None):
        parser.error(
            "Use --load-adapter-from (weights only, fresh optimizer) or "
            "--initial-resume-from-checkpoint (full trainer resume), not both."
        )
    if args.generation_eval_only and not args.eval_checkpoint:
        parser.error("--generation-eval-only requires --eval-checkpoint.")
    if args.checkpoint_epochs < 1:
        parser.error("--checkpoint-epochs must be positive.")
    if args.keep_last_checkpoints < 0:
        parser.error("--keep-last-checkpoints cannot be negative.")
    if args.keep_best_checkpoints < 0:
        parser.error("--keep-best-checkpoints cannot be negative.")
    if args.positive_oversample_factor < 1.0:
        parser.error("--positive-oversample-factor must be >= 1.0.")
    if args.positive_loss_weight <= 0:
        parser.error("--positive-loss-weight must be positive.")
    if args.fbeta_beta <= 0:
        parser.error("--fbeta-beta must be positive.")
    validate_model_precision(args, parser.error)
    validate_lora_target_configuration(args, parser.error)
    if args.greater_is_better is None:
        args.greater_is_better = not args.metric_for_best_model.endswith("loss")
    warn_deprecated_module_a_sampling(args)
    return args


def is_bf16_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def validate_model_precision(args: argparse.Namespace, error: Any) -> None:
    selected = sum(
        bool(value)
        for value in (
            args.load_in_4bit,
            args.load_in_8bit,
            args.load_in_16bit,
        )
    )
    if selected != 1:
        error("Exactly one of --load-in-4bit, --load-in-8bit, or --load-in-16bit must be enabled.")


def distributed_device_map() -> str | dict[str, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    return {"": local_rank} if local_rank >= 0 else "sequential"


def is_distributed_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def generation_eval_subprocess_env() -> dict[str, str]:
    """Run epoch eval as a single process, not as a torchrun worker."""
    env = os.environ.copy()
    local_rank = env.get("LOCAL_RANK", "0")
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "WORLD_SIZE_JOB",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "PET_NPROC_PER_NODE",
    ):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = local_rank
    return env


@contextlib.contextmanager
def split_file_lock(split_path: Path) -> Any:
    import fcntl

    lock_path = split_path.with_suffix(f"{split_path.suffix}.lock")
    try:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w", encoding="utf-8")
    except OSError:
        # Read-only repo: an existing split can still be read without a lock.
        if not split_path.is_file():
            raise
        yield
        return
    with lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def cast_trainable_parameters(model: Any, dtype_name: str) -> dict[str, int]:
    import torch

    if dtype_name == "auto":
        dtype = torch.bfloat16 if is_bf16_available() else torch.float16
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype_name]
    summary: dict[str, int] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        parameter.data = parameter.data.to(dtype=dtype)
        key = str(parameter.dtype).removeprefix("torch.")
        summary[key] = summary.get(key, 0) + parameter.numel()
    return summary


def summarize_trainable_parameters(model: Any) -> dict[str, int]:
    summary = {
        "total": 0,
        "visual": 0,
        "language": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        summary["total"] += count
        if ".visual" in name or ".vision" in name:
            summary["visual"] += count
        elif "language_model" in name or "lm_head" in name or "embed_tokens" in name:
            summary["language"] += count
        else:
            summary["other"] += count
    return summary


def validate_lora_target_configuration(args: argparse.Namespace, error: Any) -> None:
    if not any((
        args.finetune_vision_layers,
        args.finetune_language_layers,
        args.finetune_attention_modules,
        args.finetune_mlp_modules,
    )):
        error("At least one finetune_* LoRA target flag must be true.")

    if args.lora_target_modules == "all-linear" and not all((
        args.finetune_vision_layers,
        args.finetune_language_layers,
        args.finetune_attention_modules,
        args.finetune_mlp_modules,
    )):
        error(
            "--lora-target-modules=all-linear conflicts with disabled finetune_* flags. "
            "Use --lora-target-modules=auto to respect the flags, or enable all finetune_* flags."
        )


def resolve_lora_target_modules(args: argparse.Namespace) -> str | None:
    target_modules = getattr(args, "lora_target_modules", "auto")
    if target_modules == "auto":
        return None
    if target_modules == "all-linear":
        return "all-linear"
    raise ValueError(f"Unsupported lora_target_modules: {target_modules}")


def build_module_a_dataset(args: argparse.Namespace) -> ModuleTrainingDataset:
    config = ModuleAConfig(
        stride_seconds=args.stride_seconds,
        completion_margin=args.completion_margin,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        keep_last_wait_windows=args.keep_last_wait_windows,
        max_negative_windows_per_episode=args.max_negative_windows_per_episode,
        reset_jitter_seconds=args.reset_jitter_seconds,
        seed=args.seed,
        label_mode=args.module_a_label_mode,
    )
    provider = EgoPERProvider(
        args.egoper_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        require_existing_videos=True,
    )
    return build_module_a_training_dataset(
        provider,
        config=config,
        max_videos=args.max_videos,
    )


def default_split_file(args: argparse.Namespace) -> Path:
    return PROJECT_ROOT / "configs" / "splits" / f"egoper_video_split_seed{args.seed}.json"


def grouped_video_ids(dataset: ModuleTrainingDataset) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in dataset.records:
        grouped[record.task_id].add(record.video_id)
    return {
        task_id: sorted(video_ids)
        for task_id, video_ids in sorted(grouped.items())
    }


def split_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "per_task_video_holdout",
        "val_fraction": getattr(args, "val_fraction", 0.2),
        "val_videos_per_task": getattr(args, "val_videos_per_task", 2),
        "train_videos_per_task": getattr(args, "train_videos_per_task", None),
        "train_validation_erroneous_only": getattr(
            args, "train_validation_erroneous_only", False
        ),
    }


def data_config_for_split(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_videos": getattr(args, "max_videos", None),
        "task_ids": sorted(getattr(args, "task_ids", None) or []),
        "video_ids": sorted(getattr(args, "video_ids", None) or []),
        "train_videos_per_task": getattr(args, "train_videos_per_task", None),
        "val_videos_per_task": getattr(args, "val_videos_per_task", 2),
        "train_validation_erroneous_only": getattr(
            args, "train_validation_erroneous_only", False
        ),
        "stride_seconds": getattr(args, "stride_seconds", 5.0),
        "completion_margin": getattr(args, "completion_margin", 0.10),
        "negative_to_positive_ratio": getattr(args, "negative_to_positive_ratio", 2),
        "keep_last_wait_windows": getattr(args, "keep_last_wait_windows", 2),
        "max_negative_windows_per_episode": getattr(
            args, "max_negative_windows_per_episode", 3
        ),
        "reset_jitter_seconds": getattr(args, "reset_jitter_seconds", 0.5),
        "label_mode": getattr(args, "module_a_label_mode", "step_id"),
    }


def egoper_video_stratum(record: Any) -> str:
    """Use each video's observed error taxonomy as a split stratum."""
    errors = sorted({
        str(segment.fine_class)
        for segment in record.segments
        if str(segment.fine_class) not in {"CORRECT", "NO_VISUAL_ERROR"}
    })
    if record.dataset_metadata.get("missing_steps"):
        errors.append("MISSING_STEP")
    errors.extend(
        label
        for segment in record.segments
        for label in segment.structural_labels
        if label not in errors
    )
    return "NORMAL" if not errors else "ERROR_" + "__".join(sorted(set(errors)))


def make_egoper_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> dict[str, Any]:
    """Create a deterministic 70/15/15 task-and-error-stratified video split."""
    rng = random.Random(args.seed)
    train_videos_per_task = getattr(args, "train_videos_per_task", None)
    val_videos_per_task = getattr(args, "val_videos_per_task", 2)
    erroneous_only = getattr(args, "train_validation_erroneous_only", False)
    by_task_stratum: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    video_to_task: dict[str, str] = {}
    video_is_error: dict[str, bool] = {}
    for record in dataset.records:
        stratum = egoper_video_stratum(record)
        by_task_stratum[record.task_id][stratum].append(record.video_id)
        video_to_task[record.video_id] = record.task_id
        video_is_error[record.video_id] = stratum != "NORMAL"

    tasks: dict[str, dict[str, list[str]]] = {}
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for task_id, strata in sorted(by_task_stratum.items()):
        task_split = {"train": [], "val": [], "test": []}
        for video_ids in strata.values():
            shuffled = sorted(video_ids)
            rng.shuffle(shuffled)
            count = len(shuffled)
            train_count = round(count * 0.70)
            val_count = round(count * 0.15)
            if count >= 3:
                train_count = max(1, train_count)
                val_count = max(1, val_count)
                train_count = min(train_count, count - 2)
                val_count = min(val_count, count - train_count - 1)
            else:
                # A category with fewer than three videos cannot appear in all
                # splits; retain it for training rather than discard it.
                train_count, val_count = count, 0
            task_split["train"].extend(shuffled[:train_count])
            task_split["val"].extend(shuffled[train_count:train_count + val_count])
            task_split["test"].extend(shuffled[train_count + val_count:])

        # Per-stratum rounding can drift from the requested task-level ratio.
        # Rebalance minimally after stratification to make each task 70/15/15.
        total = sum(len(video_ids) for video_ids in task_split.values())
        targets = {
            "train": round(total * 0.70),
            "val": round(total * 0.15),
        }
        targets["test"] = total - targets["train"] - targets["val"]
        for destination in ("train", "val", "test"):
            while len(task_split[destination]) < targets[destination]:
                source = max(
                    (
                        split_name for split_name in ("train", "val", "test")
                        if len(task_split[split_name]) > targets[split_name]
                    ),
                    key=lambda split_name: len(task_split[split_name]) - targets[split_name],
                    default=None,
                )
                if source is None:
                    break
                rng.shuffle(task_split[source])
                task_split[destination].append(task_split[source].pop())

        if train_videos_per_task is not None and erroneous_only:
            erroneous_ids = [
                video_id
                for split_name in ("train", "val", "test")
                for video_id in task_split[split_name]
                if video_is_error[video_id]
            ]
            required = train_videos_per_task + val_videos_per_task
            if len(erroneous_ids) < required:
                raise ValueError(
                    f"{task_id}: only {len(erroneous_ids)} erroneous videos available; "
                    f"need {required} for the requested subset."
                )
            rng.shuffle(erroneous_ids)
            all_video_ids = [
                video_id
                for split_name in ("train", "val", "test")
                for video_id in task_split[split_name]
            ]
            selected = set(erroneous_ids[:required])
            task_split["train"] = erroneous_ids[:train_videos_per_task]
            task_split["val"] = erroneous_ids[train_videos_per_task:required]
            task_split["test"] = [
                video_id
                for video_id in all_video_ids
                if video_id not in selected
            ]
        elif train_videos_per_task is not None:
            if len(task_split["train"]) < train_videos_per_task:
                raise ValueError(f"{task_id}: only {len(task_split['train'])} training videos available.")
            rng.shuffle(task_split["train"])
            retained_train = task_split["train"][:train_videos_per_task]
            task_split["test"].extend(task_split["train"][train_videos_per_task:])
            task_split["train"] = retained_train
            if len(task_split["val"]) < val_videos_per_task:
                raise ValueError(f"{task_id}: only {len(task_split['val'])} validation videos available.")
            rng.shuffle(task_split["val"])
            retained_val = task_split["val"][:val_videos_per_task]
            task_split["test"].extend(task_split["val"][val_videos_per_task:])
            task_split["val"] = retained_val

        for split_name, destination in (
            ("train", train_ids), ("val", val_ids), ("test", test_ids)
        ):
            task_split[split_name] = sorted(task_split[split_name])
            destination.extend(task_split[split_name])
        tasks[task_id] = task_split

    for is_error in ((False, True) if train_videos_per_task is None else ()):
        video_ids = [
            video_id
            for task_split in tasks.values()
            for split_name in ("train", "val", "test")
            for video_id in task_split[split_name]
            if video_is_error[video_id] == is_error
        ]
        target_counts = {
            "train": round(len(video_ids) * 0.70),
            "val": round(len(video_ids) * 0.15),
        }
        target_counts["test"] = len(video_ids) - target_counts["train"] - target_counts["val"]
        for destination in ("train", "val", "test"):
            current_count = lambda split_name: sum(
                video_is_error[video_id] == is_error
                for task_split in tasks.values()
                for video_id in task_split[split_name]
            )
            while current_count(destination) < target_counts[destination]:
                source = max(
                    (
                        split_name for split_name in ("train", "val", "test")
                        if current_count(split_name) > target_counts[split_name]
                    ),
                    key=lambda split_name: current_count(split_name) - target_counts[split_name],
                    default=None,
                )
                if source is None:
                    break
                candidate = next(
                    video_id
                    for task_id in sorted(tasks)
                    for video_id in sorted(tasks[task_id][source])
                    if video_is_error[video_id] == is_error
                )
                source_task = video_to_task[candidate]
                tasks[source_task][source].remove(candidate)
                tasks[source_task][destination].append(candidate)

    train_ids = [video_id for split in tasks.values() for video_id in split["train"]]
    val_ids = [video_id for split in tasks.values() for video_id in split["val"]]
    test_ids = [video_id for split in tasks.values() for video_id in split["test"]]
    return {
        "version": MODULE_A_SPLIT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "split_policy": {
            "type": "egoper_task_error_stratified",
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "stratum": "task_id + observed fine error classes",
            "train_videos_per_task": train_videos_per_task,
            "val_videos_per_task": val_videos_per_task,
            "train_validation_erroneous_only": erroneous_only,
        },
        "data_config": data_config_for_split(args),
        "train_video_ids": sorted(train_ids),
        "val_video_ids": sorted(val_ids),
        "test_video_ids": sorted(test_ids),
        "tasks": tasks,
    }


def make_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> dict[str, Any]:
    grouped = grouped_video_ids(dataset)
    if not grouped:
        raise ValueError("Cannot create a split from an empty dataset.")
    return make_egoper_video_split(dataset, args)


def validate_video_split(split: dict[str, Any], dataset: ModuleTrainingDataset) -> None:
    train_ids = set(split.get("train_video_ids", []))
    val_ids = set(split.get("val_video_ids", []))
    test_ids = set(split.get("test_video_ids", []))
    if not train_ids or not val_ids:
        raise ValueError("Split file must contain non-empty train_video_ids and val_video_ids.")
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Split file has overlapping video IDs: {sorted(overlap)}")

    available = {record.video_id for record in dataset.records}
    missing = (train_ids | val_ids | test_ids) - available
    if missing:
        raise ValueError(f"Split file references videos not in the selected dataset: {sorted(missing)}")

    grouped = grouped_video_ids(dataset)
    for task_id, task_split in split.get("tasks", {}).items():
        if task_id not in grouped:
            raise ValueError(f"Split file references missing task: {task_id}")
        task_ids = (
            set(task_split.get("train", []))
            | set(task_split.get("val", []))
            | set(task_split.get("test", []))
        )
        outside_task = task_ids - set(grouped[task_id])
        if outside_task:
            raise ValueError(
                f"Split file places videos under the wrong task {task_id!r}: {sorted(outside_task)}"
            )


def apply_video_split_subset(
    split: dict[str, Any],
    dataset: ModuleTrainingDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return a deterministic run-local subset without modifying the split file."""
    train_cap = getattr(args, "train_videos_per_task", None)
    if train_cap is None:
        return split
    val_cap = getattr(args, "val_videos_per_task", 1)
    erroneous_only = getattr(args, "train_validation_erroneous_only", False)
    error_by_video = {
        record.video_id: egoper_video_stratum(record) != "NORMAL"
        for record in dataset.records
    }
    rng = random.Random(args.seed)
    subset = copy.deepcopy(split)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for task_id, task_split in sorted(subset.get("tasks", {}).items()):
        selected: dict[str, list[str]] = {}
        for split_name, cap in (("train", train_cap), ("val", val_cap)):
            candidates = sorted(task_split.get(split_name, []))
            if erroneous_only:
                candidates = [
                    video_id
                    for video_id in candidates
                    if error_by_video.get(video_id, False)
                ]
            if len(candidates) < cap:
                qualifier = " erroneous" if erroneous_only else ""
                raise ValueError(
                    f"{task_id}: canonical {split_name} split has only "
                    f"{len(candidates)}{qualifier} videos; requested {cap}."
                )
            rng.shuffle(candidates)
            selected[split_name] = sorted(candidates[:cap])
        task_split["train"] = selected["train"]
        task_split["val"] = selected["val"]
        train_ids.extend(selected["train"])
        val_ids.extend(selected["val"])
    subset["train_video_ids"] = sorted(train_ids)
    subset["val_video_ids"] = sorted(val_ids)
    subset["run_subset_policy"] = {
        "source_train_count": len(split.get("train_video_ids", [])),
        "source_val_count": len(split.get("val_video_ids", [])),
        "train_videos_per_task": train_cap,
        "val_videos_per_task": val_cap,
        "train_validation_erroneous_only": erroneous_only,
        "seed": args.seed,
    }
    validate_video_split(subset, dataset)
    return subset


def load_or_create_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    with split_file_lock(split_path):
        if split_path.exists() and not args.regenerate_split:
            with open(split_path, "r", encoding="utf-8") as file:
                split = json.load(file)
            validate_video_split(split, dataset)
            print(f"Loaded Module A split from: {split_path}")
            return apply_video_split_subset(split, dataset, args), split_path

        split = make_video_split(dataset, args)
        validate_video_split(split, dataset)
        with open(split_path, "w", encoding="utf-8") as file:
            json.dump(split, file, indent=2, sort_keys=True)
            file.write("\n")
    print(f"Wrote Module A split to: {split_path}")
    return apply_video_split_subset(split, dataset, args), split_path


def split_conversations(
    conversations: list[dict[str, Any]],
    split: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = set(split["train_video_ids"])
    val_ids = set(split["val_video_ids"])
    train = [item for item in conversations if item["video_id"] in train_ids]
    val = [item for item in conversations if item["video_id"] in val_ids]
    if not train or not val:
        raise ValueError("Train/validation split produced an empty conversation split.")
    return train, val


def oversample_positive_conversations(
    conversations: list[dict[str, Any]],
    *,
    factor: float,
    seed: int,
) -> list[dict[str, Any]]:
    if factor <= 1.0:
        return list(conversations)
    positives = [
        item for item in conversations
        if label_to_positive(
            str(item.get("target", "")),
            label_mode=str(item.get("label_mode") or "legacy"),
        )
    ]
    if not positives:
        return list(conversations)

    extra_count = int(math.floor((factor - 1.0) * len(positives)))
    fractional = (factor - 1.0) * len(positives) - extra_count
    rng = random.Random(seed)
    if fractional > 0 and rng.random() < fractional:
        extra_count += 1
    extras = [copy.deepcopy(rng.choice(positives)) for _ in range(extra_count)]
    expanded = list(conversations) + extras
    rng.shuffle(expanded)
    return expanded


def apply_positive_loss_weights(
    conversations: list[dict[str, Any]],
    *,
    positive_weight: float,
) -> None:
    for item in conversations:
        item["loss_weight"] = (
            positive_weight
            if label_to_positive(
                str(item.get("target", "")),
                label_mode=str(item.get("label_mode") or "legacy"),
            )
            else 1.0
        )


def to_unsloth_conversations(
    dataset: ModuleTrainingDataset,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(
            example,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        for example in dataset
    ]


def training_example_to_conversation(
    example: TrainingExample,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    label_mode = str((example.metadata or {}).get("label_mode") or "legacy")
    if label_mode == "completion_set":
        system_prompt = MODULE_A_COMPLETION_SET_SYSTEM_PROMPT
    elif label_mode == "step_id":
        system_prompt = MODULE_A_STEP_ID_SYSTEM_PROMPT
    else:
        system_prompt = MODULE_A_SYSTEM_PROMPT
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(example.video_path),
                        "video_start": example.window_start,
                        "video_end": example.window_end,
                        "fps": fps,
                        "min_frames": min_frames,
                        "max_frames": max_frames,
                    },
                    {
                        "type": "text",
                        "text": example.prompt_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": example.target_text,
                    }
                ],
            },
        ],
        "video_id": example.video_id,
        "task_id": example.task_id,
        "step_index": example.step_index,
        "window_start": example.window_start,
        "window_end": example.window_end,
        "target": example.target_text,
        "label_mode": label_mode,
        "all_instructions": list((example.metadata or {}).get("all_instructions") or []),
        "loss_weight": 1.0,
    }


def estimated_window_frames(
    window_start: float,
    window_end: float,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """Frames qwen-vl-utils will sample for this window."""
    span = max(0.0, float(window_end) - float(window_start))
    return int(min(max_frames, max(min_frames, math.floor(span * fps))))


def summarize_module_a(dataset: ModuleTrainingDataset) -> dict[str, Any]:
    counts = Counter(example.target_text for example in dataset)
    videos = {example.video_id for example in dataset}
    return {
        "num_examples": len(dataset),
        "num_videos": len(videos),
        "label_counts": dict(sorted(counts.items())),
    }


def summarize_conversations(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    from ..windowing.module_a import is_negative_label

    label_counts = Counter(str(item["target"]) for item in conversations)
    videos_by_task: dict[str, set[str]] = defaultdict(set)
    examples_by_task = Counter()
    negative_windows = 0
    for item in conversations:
        task_id = str(item["task_id"])
        videos_by_task[task_id].add(str(item["video_id"]))
        examples_by_task[task_id] += 1
        if is_negative_label(str(item.get("target") or "")):
            negative_windows += 1
    return {
        "num_examples": len(conversations),
        "num_videos": len({item["video_id"] for item in conversations}),
        "label_counts": dict(sorted(label_counts.items())),
        "negative_windows": negative_windows,
        "positive_windows": len(conversations) - negative_windows,
        "tasks": {
            task_id: {
                "num_videos": len(video_ids),
                "num_examples": examples_by_task[task_id],
            }
            for task_id, video_ids in sorted(videos_by_task.items())
        },
    }


def print_dry_run(
    dataset: ModuleTrainingDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module A full dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))
    print(f"Module A split file: {split_path}")
    print("Module A split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "test_video_ids": split.get("test_video_ids", []),
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module A train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module A validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def save_run_config(
    *,
    args: argparse.Namespace,
    dataset: ModuleTrainingDataset,
    conversations: list[dict[str, Any]],
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": json_safe(vars(args)),
        "dataset_summary": summarize_module_a(dataset),
        "num_conversations": len(conversations),
        "split_file": str(split_path),
        "split": json_safe(split),
        "train_summary": summarize_conversations(train_conversations),
        "validation_summary": summarize_conversations(val_conversations),
        "negative_sampling": json_safe(getattr(args, "negative_sampling_stats", {})),
    }
    config_path = output_path / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    print(f"Saved run config to: {config_path}")


def label_to_positive(label: str, *, label_mode: str = "legacy") -> bool:
    return is_module_a_completion_label(label, label_mode=label_mode)


def binary_detection_metrics(
    targets: list[str],
    predictions: list[str],
    *,
    prefix: str,
    beta: float = 2.0,
    label_mode: str = "legacy",
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for target, prediction in zip(targets, predictions):
        target_positive = label_to_positive(target, label_mode=label_mode)
        predicted_positive = label_to_positive(prediction, label_mode=label_mode)
        if target_positive and predicted_positive:
            tp += 1
        elif not target_positive and predicted_positive:
            fp += 1
        elif not target_positive and not predicted_positive:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_sq = beta * beta
    fbeta = (
        (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)
        if precision + recall else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    return {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": f1,
        f"{prefix}/f_beta": fbeta,
        f"{prefix}/specificity": specificity,
        f"{prefix}/balanced_accuracy": balanced_accuracy,
        f"{prefix}/mcc": mcc,
        f"{prefix}/tp": float(tp),
        f"{prefix}/fp": float(fp),
        f"{prefix}/tn": float(tn),
        f"{prefix}/fn": float(fn),
        f"{prefix}/num_samples": float(total),
    }


def step_id_letter_to_semantic_class(
    letter: str,
    *,
    task_id: str,
    all_instructions: list[str] | tuple[str, ...],
) -> str:
    """Map a prompt letter to a task-local semantic class id."""
    normalized = extract_module_a_label(letter, label_mode="step_id")
    if normalized == MODULE_A_STEP_ID_NONE_LETTER:
        return SEMANTIC_NONE
    if normalized == MODULE_A_STEP_ID_EXTRA_LETTER:
        return SEMANTIC_EXTRA
    index = ord(normalized) - ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER)
    if 0 <= index < len(all_instructions):
        return f"STEP_{index}"
    instruction = module_a_instruction_for_step_id_letter(normalized, all_instructions)
    if instruction is not None and normalized != MODULE_A_STEP_ID_EXTRA_LETTER:
        return f"STEP_{normalized}"
    return f"UNKNOWN_{normalized}"


def step_id_task_class_catalog(
    task_id: str,
    all_instructions: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Return (class_keys, display_labels) for NONE/EXTRA/steps in a task."""
    del task_id  # class keys are task-local; task id is carried by metric prefix
    keys = [SEMANTIC_NONE, SEMANTIC_EXTRA]
    labels = [SEMANTIC_NONE, SEMANTIC_EXTRA]
    for index, instruction in enumerate(all_instructions):
        keys.append(f"STEP_{index}")
        labels.append(str(instruction))
    return keys, labels


def _one_vs_rest_counts(
    targets: list[str],
    predictions: list[str],
    *,
    positive_class: str,
) -> tuple[int, int, int, int, int]:
    tp = fp = tn = fn = 0
    for target, prediction in zip(targets, predictions):
        target_pos = target == positive_class
        pred_pos = prediction == positive_class
        if target_pos and pred_pos:
            tp += 1
        elif not target_pos and pred_pos:
            fp += 1
        elif not target_pos and not pred_pos:
            tn += 1
        else:
            fn += 1
    support = tp + fn
    return tp, fp, tn, fn, support


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def step_id_semantic_metrics(
    *,
    task_ids: list[str],
    all_instructions: list[list[str] | tuple[str, ...]],
    targets: list[str],
    predictions: list[str],
    prefix: str = "eval_step_id",
    beta: float = 2.0,
    include_class_count_scalars: bool = False,
) -> dict[str, Any]:
    """Per-task supported-class OVR F1 (NONE/EXTRA/steps) → mean task F1."""
    if not (
        len(task_ids) == len(all_instructions) == len(targets) == len(predictions)
    ):
        raise ValueError("step_id_semantic_metrics requires aligned input lists.")

    semantic_targets = [
        step_id_letter_to_semantic_class(
            target,
            task_id=task_id,
            all_instructions=instructions,
        )
        for target, task_id, instructions in zip(targets, task_ids, all_instructions)
    ]
    semantic_preds = [
        step_id_letter_to_semantic_class(
            prediction,
            task_id=task_id,
            all_instructions=instructions,
        )
        for prediction, task_id, instructions in zip(predictions, task_ids, all_instructions)
    ]

    by_task_indices: dict[str, list[int]] = defaultdict(list)
    task_instructions: dict[str, list[str]] = {}
    for index, task_id in enumerate(task_ids):
        by_task_indices[task_id].append(index)
        if task_id not in task_instructions:
            task_instructions[task_id] = [str(item) for item in all_instructions[index]]

    metrics: dict[str, Any] = {
        f"{prefix}/num_samples": float(len(targets)),
        f"{prefix}/num_tasks": float(len(by_task_indices)),
    }
    task_macro_f1s: list[float] = []
    task_confusion: dict[str, Any] = {}

    for task_id in sorted(by_task_indices):
        indices = by_task_indices[task_id]
        task_targets = [semantic_targets[i] for i in indices]
        task_preds = [semantic_preds[i] for i in indices]
        class_keys, display_labels = step_id_task_class_catalog(
            task_id,
            task_instructions[task_id],
        )
        # Include unexpected predicted classes so CM stays complete.
        extras = sorted(
            (set(task_targets) | set(task_preds)) - set(class_keys)
        )
        class_keys = class_keys + extras
        display_labels = display_labels + extras

        task_prefix = f"{prefix}/task_{task_id}"
        supported_class_f1: list[float] = []
        num_supported_classes = 0
        for class_key in class_keys:
            tp, fp, tn, fn, support = _one_vs_rest_counts(
                task_targets,
                task_preds,
                positive_class=class_key,
            )
            precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
            if support > 0:
                supported_class_f1.append(f1)
                num_supported_classes += 1
            if include_class_count_scalars:
                metrics[f"{task_prefix}/support_{class_key}"] = float(support)
                metrics[f"{task_prefix}/pred_{class_key}"] = float(
                    sum(1 for pred in task_preds if pred == class_key)
                )
                metrics[f"{task_prefix}/tp_{class_key}"] = float(tp)
                metrics[f"{task_prefix}/fp_{class_key}"] = float(fp)
                metrics[f"{task_prefix}/fn_{class_key}"] = float(fn)
                metrics[f"{task_prefix}/tn_{class_key}"] = float(tn)
            metrics[f"{task_prefix}/precision_{class_key}"] = precision
            metrics[f"{task_prefix}/recall_{class_key}"] = recall
            metrics[f"{task_prefix}/f1_{class_key}"] = f1

        task_macro = (
            sum(supported_class_f1) / len(supported_class_f1)
            if supported_class_f1
            else 0.0
        )
        task_macro_f1s.append(task_macro)
        metrics[f"{task_prefix}/macro_f1"] = task_macro
        metrics[f"{task_prefix}/num_samples"] = float(len(indices))
        metrics[f"{task_prefix}/num_classes"] = float(len(class_keys))
        metrics[f"{task_prefix}/num_supported_classes"] = float(
            num_supported_classes
        )

        matrix = [
            [
                sum(
                    1
                    for target, pred in zip(task_targets, task_preds)
                    if target == true_label and pred == pred_label
                )
                for pred_label in class_keys
            ]
            for true_label in class_keys
        ]
        task_confusion[task_id] = {
            "labels": display_labels,
            "label_keys": class_keys,
            "matrix": matrix,
        }

    total_macro = sum(task_macro_f1s) / len(task_macro_f1s) if task_macro_f1s else 0.0
    metrics[f"{prefix}/total_macro_f1"] = total_macro
    # Keep legacy key pointing at the checkpoint metric for older dashboards.
    metrics[f"{prefix}/macro_f1"] = total_macro
    metrics["task_confusion_matrices"] = task_confusion

    trigger = binary_detection_metrics(
        [extract_module_a_label(item, label_mode="step_id") for item in targets],
        [extract_module_a_label(item, label_mode="step_id") for item in predictions],
        prefix=f"{prefix}/trigger",
        beta=beta,
        label_mode="step_id",
    )
    metrics.update(trigger)
    # Alias gate keys used by older runs / env docs.
    for key, value in list(trigger.items()):
        metrics[key.replace(f"{prefix}/trigger", f"{prefix}/gate")] = value
    return metrics


def multiclass_letter_metrics(
    targets: list[str],
    predictions: list[str],
    *,
    prefix: str = "eval_step_id",
    include_class_count_scalars: bool = False,
) -> dict[str, Any]:
    normalized_targets = [extract_module_a_label(target, label_mode="step_id") for target in targets]
    normalized_preds = [extract_module_a_label(prediction, label_mode="step_id") for prediction in predictions]
    total = len(normalized_targets)
    correct = sum(1 for target, prediction in zip(normalized_targets, normalized_preds) if target == prediction)
    labels = sorted(set(normalized_targets) | set(normalized_preds))
    metrics: dict[str, Any] = {
        f"{prefix}/accuracy": (correct / total) if total else 0.0,
        f"{prefix}/num_samples": float(total),
        f"{prefix}/num_classes": float(len(labels)),
    }

    per_class_f1: list[float] = []
    weighted_f1_sum = 0.0
    support_sum = 0
    micro_tp = micro_fp = micro_fn = 0

    for label in labels:
        tp = sum(1 for t, p in zip(normalized_targets, normalized_preds) if t == label and p == label)
        fp = sum(1 for t, p in zip(normalized_targets, normalized_preds) if t != label and p == label)
        fn = sum(1 for t, p in zip(normalized_targets, normalized_preds) if t == label and p != label)
        support = sum(1 for t in normalized_targets if t == label)
        pred_count = sum(1 for p in normalized_preds if p == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if include_class_count_scalars:
            metrics[f"{prefix}/support_{label}"] = float(support)
            metrics[f"{prefix}/pred_{label}"] = float(pred_count)
            metrics[f"{prefix}/tp_{label}"] = float(tp)
        metrics[f"{prefix}/precision_{label}"] = precision
        metrics[f"{prefix}/recall_{label}"] = recall
        metrics[f"{prefix}/f1_{label}"] = f1
        if support > 0:
            per_class_f1.append(f1)
            weighted_f1_sum += f1 * support
            support_sum += support
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    metrics[f"{prefix}/macro_f1"] = (
        sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0
    )
    metrics[f"{prefix}/weighted_f1"] = (
        weighted_f1_sum / support_sum if support_sum else 0.0
    )
    metrics[f"{prefix}/micro_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    metrics[f"{prefix}/micro_precision"] = micro_precision
    metrics[f"{prefix}/micro_recall"] = micro_recall

    matrix = [
        [
            sum(
                1
                for t, p in zip(normalized_targets, normalized_preds)
                if t == true_label and p == pred_label
            )
            for pred_label in labels
        ]
        for true_label in labels
    ]
    metrics["confusion_matrix"] = {"labels": labels, "matrix": matrix}

    gate = binary_detection_metrics(
        normalized_targets,
        normalized_preds,
        prefix=f"{prefix}/gate",
        label_mode="step_id",
    )
    metrics.update(gate)
    return metrics


def confusion_matrix_from_metrics(
    metrics: dict[str, Any],
    *,
    prefix: str = "eval_step_id",
) -> dict[str, Any]:
    nested = metrics.get("confusion_matrix")
    if isinstance(nested, dict) and "matrix" in nested and "labels" in nested:
        return {
            "labels": list(nested["labels"]),
            "matrix": [list(row) for row in nested["matrix"]],
        }
    labels: set[str] = set()
    token = f"{prefix}/cm_"
    for key in metrics:
        if not key.startswith(token):
            continue
        remainder = key[len(token) :]
        if "_" not in remainder:
            continue
        true_label, pred_label = remainder.split("_", 1)
        labels.add(true_label)
        labels.add(pred_label)
    ordered = sorted(labels)
    matrix = [
        [int(metrics.get(f"{prefix}/cm_{true_label}_{pred_label}", 0.0)) for pred_label in ordered]
        for true_label in ordered
    ]
    return {"labels": ordered, "matrix": matrix}


# Keep training callbacks on the same pure calculators used by pipeline/evaluation.
# The legacy local definitions above remain temporarily for checkpoint compatibility.
from ..metrics.module_a import (  # noqa: E402
    binary_detection_metrics,
    multiclass_letter_metrics,
    negative_sampling_metrics,
    step_id_semantic_metrics,
)
from ..metrics.calibration import (  # noqa: E402
    binary_roc_auc,
    calibration_output_path,
    calibration_record,
    label_score_diagnostics,
    logits_to_probabilities,
    scores_to_probabilities,
    write_calibration_records,
)


def _write_png_rgba(path: Path, width: int, height: int, rgba: bytes) -> None:
    """Minimal unfiltered RGBA PNG writer (stdlib only)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + rgba[row * width * 4 : (row + 1) * width * 4] for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")
        file.write(chunk(b"IHDR", ihdr))
        file.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        file.write(chunk(b"IEND", b""))


def save_confusion_heatmap_png(
    *,
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Write a confusion-matrix heatmap PNG (matplotlib if present, else simple grid)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    if plt is not None:
        values = [[float(cell) for cell in row] for row in matrix]
        fig, ax = plt.subplots(
            figsize=(max(5.5, 1.2 * len(col_labels) + 2), max(4.5, 1.0 * len(row_labels) + 2))
        )
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(title)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                ax.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    # Stdlib fallback: colored cell grid (labels stored beside the PNG).
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    if rows == 0 or cols == 0:
        raise ValueError("Cannot write an empty confusion matrix heatmap.")
    max_value = max((max(row) for row in matrix), default=0) or 1
    cell = 48
    width = cols * cell
    height = rows * cell
    pixels = bytearray(width * height * 4)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(40 + 200 * (float(value) / float(max_value)))
            for y in range(row_index * cell, (row_index + 1) * cell):
                for x in range(col_index * cell, (col_index + 1) * cell):
                    offset = (y * width + x) * 4
                    pixels[offset] = 255 - intensity
                    pixels[offset + 1] = 255 - intensity // 3
                    pixels[offset + 2] = 255
                    pixels[offset + 3] = 255
    _write_png_rgba(output_path, width, height, bytes(pixels))
    sidecar = output_path.with_suffix(".json")
    with open(sidecar, "w", encoding="utf-8") as file:
        json.dump(
            {
                "title": title,
                "row_labels": list(row_labels),
                "col_labels": list(col_labels),
                "matrix": matrix,
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def write_step_id_confusion_heatmaps(
    task_confusion: dict[str, Any],
    figures_dir: Path,
    *,
    title_prefix: str = "Module A step_id",
) -> list[Path]:
    """Write one confusion-matrix heatmap PNG per task (offline / post-eval artifact)."""
    written: list[Path] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for task_id, payload in sorted(task_confusion.items()):
        if not isinstance(payload, dict):
            continue
        matrix = payload.get("matrix")
        labels = payload.get("labels") or payload.get("label_keys")
        if not matrix or not labels:
            continue
        path = figures_dir / f"confusion_task_{task_id}.png"
        save_confusion_heatmap_png(
            matrix=matrix,
            row_labels=[str(item) for item in labels],
            col_labels=[str(item) for item in labels],
            title=f"{title_prefix}: {task_id}",
            output_path=path,
        )
        written.append(path)
    return written


def scalar_metrics_for_logging(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten only W&B-safe scalars; drop CM cells and nested structures."""
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        if "/cm_" in key:
            continue
        if not isinstance(value, (int, float)):
            continue
        result[str(key)] = float(value)
    return result


def persist_calibration_records(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tag: str | None = None,
    global_step: int | None = None,
) -> Path | None:
    """Write per-window scores so a threshold can be swept without the model."""
    if not records or not getattr(args, "write_calibration_records", True):
        return None
    if not is_distributed_rank_zero():
        return None
    output_dir = getattr(args, "calibration_output_dir", None) or args.output_dir
    path = calibration_output_path(output_dir, global_step=global_step, tag=tag)
    write_calibration_records(records, path)
    print(f"[calibration] wrote {len(records)} scored windows to: {path}", flush=True)
    return path


def module_a_metrics_from_predictions(
    examples: list[dict[str, Any]],
    predictions: list[str],
    *,
    beta: float,
    sampling_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    if len(examples) != len(predictions):
        raise ValueError("Module A metric examples and predictions must be aligned.")
    if not examples:
        return {}

    targets = [str(example["target"]) for example in examples]
    task_ids = [str(example["task_id"]) for example in examples]
    instructions_list = [
        [str(item) for item in (example.get("all_instructions") or [])]
        for example in examples
    ]
    label_mode = str(examples[0].get("label_mode") or "legacy")
    task_targets: dict[str, list[str]] = defaultdict(list)
    task_predictions: dict[str, list[str]] = defaultdict(list)
    for task_id, target, prediction in zip(task_ids, targets, predictions):
        task_targets[task_id].append(target)
        task_predictions[task_id].append(prediction)

    if label_mode == "step_id":
        metrics = step_id_semantic_metrics(
            task_ids=task_ids,
            all_instructions=instructions_list,
            targets=targets,
            predictions=predictions,
            prefix="eval_step_id",
            beta=beta,
        )
        metrics.update(
            binary_detection_metrics(
                targets,
                predictions,
                prefix="eval_detection",
                beta=beta,
                label_mode="step_id",
            )
        )
    else:
        metrics = binary_detection_metrics(
            targets,
            predictions,
            prefix="eval_detection",
            beta=beta,
            label_mode="legacy",
        )
    for task_id in sorted(task_targets):
        task_metrics = binary_detection_metrics(
            task_targets[task_id],
            task_predictions[task_id],
            prefix=f"eval_detection/task_{task_id}",
            beta=beta,
            label_mode=label_mode,
        )
        metrics.update({
            key: value
            for key, value in task_metrics.items()
            if key.endswith(("/accuracy", "/f1", "/recall", "/num_samples"))
        })
    metrics.update(negative_sampling_metrics(sampling_stats or {}, prefix="eval"))
    return scalar_metrics_for_logging(metrics)


def extract_module_a_label(text: str, *, label_mode: str = "legacy") -> str:
    if label_mode == "completion_set":
        from ..prompts.module_a import format_completion_set_target, parse_completion_set

        return format_completion_set_target(parse_completion_set(text))
    if label_mode == "step_id":
        normalized = text.strip().upper()
        for char in normalized:
            if "A" <= char <= "Z":
                return char
        return MODULE_A_STEP_ID_NONE_LETTER
    for label in MODULE_A_LABELS:
        if label in text.strip().strip("[]").upper():
            return label
    normalized = text.strip().upper()
    if "COMPLETE" in normalized:
        return "COMPLETE"
    if "WAIT" in normalized:
        return "WAIT"
    return "WAIT"


def load_adapter_weights(model: Any, checkpoint_dir: str | Path) -> None:
    """Load LoRA tensors without optimizer / trainer state."""
    path = Path(checkpoint_dir).expanduser()
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.is_file():
        import torch

        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise SystemExit(
            f"--load-adapter-from={path} has no adapter_model.safetensors or adapter_model.bin"
        )
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        from peft.utils.save_and_load import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, state_dict)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    adapter_missing = [
        key for key in missing
        if "lora_" in key or key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")
    ]
    if adapter_missing:
        raise RuntimeError(
            f"Failed to load LoRA adapter from {path}; missing={adapter_missing[:8]}"
        )
    print(
        f"Loaded LoRA weights from {path} "
        f"(keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}). "
        "Adam and LR schedule start fresh.",
        flush=True,
    )


def load_unsloth_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
        device_map=distributed_device_map(),
        use_gradient_checkpointing="unsloth",
    )
    if getattr(args, "zero_shot_eval", False):
        print(
            "Zero-shot eval: attaching untrained LoRA (B=0, identity) so the "
            "training eval trainer can run; the forward pass matches the base model.",
            flush=True,
        )
    elif getattr(args, "eval_only", False):
        print(
            f"Eval-only: scoring adapter {args.load_adapter_from} with no optimizer steps.",
            flush=True,
        )
    peft_kwargs = dict(
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules=resolve_lora_target_modules(args),
        ensure_weight_tying=True,
    )
    try:
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    except TypeError as error:
        if "ensure_weight_tying" not in str(error):
            raise
        peft_kwargs.pop("ensure_weight_tying")
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    adapter_source = (
        getattr(args, "load_adapter_from", None) if not args.resume_from_checkpoint else None
    )
    if adapter_source:
        load_adapter_weights(model, adapter_source)
    trainable_dtypes = cast_trainable_parameters(model, args.lora_dtype)
    trainable = summarize_trainable_parameters(model)
    print("Trainable parameter summary:")
    print(json.dumps(trainable, indent=2, sort_keys=True))
    print("Trainable parameter dtypes:")
    print(json.dumps(trainable_dtypes, indent=2, sort_keys=True))
    if not args.finetune_vision_layers and trainable["visual"] != 0:
        raise RuntimeError(
            "finetune_vision_layers=False, but visual LoRA parameters are trainable. "
            "Check --lora-target-modules for conflicting target selection."
        )
    if args.finetune_vision_layers and trainable["visual"] == 0:
        raise RuntimeError(
            "finetune_vision_layers=True, but no trainable visual parameters were found."
        )
    return model, tokenizer


def load_generation_eval_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.eval_checkpoint or args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
        device_map=distributed_device_map(),
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def collapse_fps(fps_values: list[float], tol: float = 1e-4) -> float | list[float] | None:
    if not fps_values:
        return None
    first = float(fps_values[0])
    if all(math.isclose(float(value), first, rel_tol=tol, abs_tol=tol) for value in fps_values[1:]):
        return first
    return [float(value) for value in fps_values]


class Qwen3VLMetadataVisionDataCollator:
    """Unsloth vision collator wrapper that preserves Qwen3-VL video metadata."""

    def __init__(
        self,
        *args: Any,
        include_response_loss_weight: bool = False,
        letter_only_loss: bool = False,
        **kwargs: Any,
    ) -> None:
        from unsloth.trainer import UnslothVisionDataCollator

        from ..models.vision import parse_vision_resize, raise_qwen_video_frame_pixel_limit

        self.vision_resize = parse_vision_resize(kwargs.get("resize", 512))
        if self.vision_resize <= 0:
            kwargs["resize"] = 1280
            # 1280×736 padded native; default qwen-vl-utils cap is 786432 px.
            raise_qwen_video_frame_pixel_limit(1280 * 736)
        self._base = UnslothVisionDataCollator(*args, **kwargs)
        self.include_response_loss_weight = include_response_loss_weight
        self.letter_only_loss = letter_only_loss
        self._step_id_candidate_token_ids: list[int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _extract_images_videos_metadata(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any], list[Any], dict[str, Any], list[dict[str, Any]] | None]:
        from qwen_vl_utils import process_vision_info

        from ..models.vision import (
            apply_resize_to_messages,
            pad_instead_of_interpolate,
            pad_video_to_grid,
        )

        messages = copy.deepcopy(messages)
        apply_resize_to_messages(messages, self.vision_resize)

        signature = inspect.signature(process_vision_info)
        kwargs: dict[str, Any] = {
            "return_video_kwargs": True,
            "return_video_metadata": True,
        }
        if "image_patch_size" in signature.parameters:
            image_processor = getattr(self._base.processor, "image_processor", None)
            kwargs["image_patch_size"] = getattr(image_processor, "patch_size", self._base.patch_size)
        elif "size_factor" in signature.parameters:
            kwargs["size_factor"] = self._base.patch_size * 2
        else:
            raise RuntimeError("Unsupported qwen_vl_utils.process_vision_info signature.")

        with pad_instead_of_interpolate():
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, **kwargs)
        images = [] if image_inputs is None else list(image_inputs)
        videos: list[Any] = []
        video_metadata: list[dict[str, Any]] = []

        if video_inputs is not None:
            for item in video_inputs:
                if isinstance(item, tuple) and len(item) == 2:
                    video, metadata = item
                    videos.append(pad_video_to_grid(video))
                    video_metadata.append(metadata)
                else:
                    videos.append(pad_video_to_grid(item))

        return images, videos, video_kwargs, video_metadata or None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "prompt" in examples[0] and "completion" in examples[0]:
            return self._base(examples)

        texts = []
        normalized_messages = []
        images = []
        videos = []
        video_metadatas = []
        video_processor_kwargs: dict[str, Any] = {}
        fps_values: list[float] = []

        for example in examples:
            messages = self._base._select_messages_or_raw(example)
            if len(messages) != 0:
                messages = self._base._validate_and_normalize_first_message(messages)
                if self._base.assistant_single_content:
                    messages = self._base._collapse_assistant_content(messages)
                messages = self._base._clean_none_keys(messages)

            texts.append(self._base.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ))
            normalized_messages.append(messages)
            image, video, video_kwargs, video_metadata = self._extract_images_videos_metadata(messages)
            if image:
                images.append(image)
            if video:
                videos.append(video)
                for key, value in video_kwargs.items():
                    if key == "fps":
                        fps_values.extend(value)
                    else:
                        video_processor_kwargs[key] = value
                if video_metadata:
                    video_metadatas.extend(video_metadata)

        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "truncation": self._base.truncation,
            "max_length": self._base.max_seq_length,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
            proc_kwargs["do_resize"] = False
            proc_kwargs.update(video_processor_kwargs)
            if video_metadatas:
                proc_kwargs["video_metadata"] = video_metadatas
            else:
                collapsed_fps = collapse_fps(fps_values)
                if collapsed_fps is not None:
                    proc_kwargs["fps"] = collapsed_fps
        if self._base.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self._base.pad_to_multiple_of

        batch = self._base.processor(**proc_kwargs)
        if "pixel_values" in batch:
            batch = self._base._cast_pixel_values_dtype_inplace(batch)
        if "pixel_values_videos" in batch:
            batch = self._base._cast_pixel_values_dtype_inplace(batch, "pixel_values_videos")

        import torch

        labels = batch["input_ids"].clone()
        padding_ids = self._base._get_padding_token_ids_on_device(labels.device)
        labels[torch.isin(labels, padding_ids)] = self._base.ignore_index
        batch["labels"] = labels
        if self._base.train_on_responses_only:
            batch["labels"] = self._base.train_on_responses_only(batch)["labels"]
        is_eval_batch = all("eval_row_index" in example for example in examples)
        letter_indices: list[int] = []
        if self.letter_only_loss or is_eval_batch:
            for index, (example, text) in enumerate(zip(examples, texts)):
                target = str(example["target"]).strip().upper()
                answer_tail = rendered_answer_tail_token_ids(
                    self._base.processor,
                    text,
                    target,
                )
                grid = batch.get("video_grid_thw")
                grid_row = (
                    grid[index]
                    if torch.is_tensor(grid) and index < grid.shape[0]
                    else None
                )
                letter_indices.append(
                    answer_letter_index_from_tail(
                        batch["input_ids"][index],
                        batch["attention_mask"][index],
                        answer_tail,
                        max_seq_length=self._base.max_seq_length,
                        video_grid_thw=grid_row,
                        row_id=example.get("eval_row_index", index),
                    )
                )
        if self.letter_only_loss and not is_eval_batch:
            batch["labels"] = apply_letter_only_labels(
                batch["labels"],
                torch.tensor(letter_indices, dtype=torch.long, device=batch["labels"].device),
                ignore_index=self._base.ignore_index,
            )
        if self.include_response_loss_weight:
            weights = torch.tensor(
                [float(example.get("loss_weight", 1.0)) for example in examples],
                dtype=torch.float32,
                device=batch["labels"].device,
            )
            response_mask = (batch["labels"] != self._base.ignore_index).to(torch.float32)
            batch["response_loss_weight"] = response_mask * weights[:, None]
        if is_eval_batch:
            batch["eval_row_index"] = torch.tensor(
                [int(example["eval_row_index"]) for example in examples],
                dtype=torch.long,
                device=batch["labels"].device,
            ).unsqueeze(-1)
            candidate_rows: list[list[int]] = []
            for index, (example, messages, letter_index) in enumerate(
                zip(examples, normalized_messages, letter_indices)
            ):
                target = str(example["target"]).strip().upper()
                candidate_ids = self._step_id_candidate_token_ids
                if candidate_ids is None:
                    candidate_ids = contextual_step_id_candidate_token_ids(
                        self._base.processor,
                        messages,
                    )
                    self._step_id_candidate_token_ids = candidate_ids
                target_index = ord(target) - ord("A")
                letter_id = int(batch["input_ids"][index, letter_index].item())
                if (
                    target_index < 0
                    or target_index >= len(candidate_ids)
                    or candidate_ids[target_index] != letter_id
                ):
                    candidate_ids = contextual_step_id_candidate_token_ids(
                        self._base.processor,
                        messages,
                    )
                candidate_rows.append(candidate_ids)
            batch["eval_answer_positions"] = torch.tensor(
                letter_indices,
                dtype=torch.long,
                device=batch["labels"].device,
            ).unsqueeze(-1)
            batch["eval_candidate_token_ids"] = torch.tensor(
                candidate_rows,
                dtype=torch.long,
                device=batch["labels"].device,
            )
        missing_response = ~(batch["labels"] != self._base.ignore_index).any(dim=1)
        if bool(missing_response.any()):
            missing_indices = missing_response.nonzero(as_tuple=False).reshape(-1).tolist()
            row_ids = [
                examples[index].get("eval_row_index", index)
                for index in missing_indices
            ]
            raise RuntimeError(
                "Assistant response was fully masked for batch "
                f"row(s) {row_ids} at max_seq_length={self._base.max_seq_length}."
            )
        return batch


class ModuleADetectionMetricsCallback:
    def __init__(
        self,
        *,
        eval_examples: list[dict[str, Any]],
        data_collator: Qwen3VLMetadataVisionDataCollator,
        tokenizer: Any,
        max_samples: int,
        max_new_tokens: int,
        beta: float,
        label_score_eval: bool,
        sampling_stats: dict[str, Any] | None = None,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.eval_examples = generation_eval_examples(eval_examples, max_samples)
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.beta = beta
        self.label_score_eval = label_score_eval
        self.sampling_stats = sampling_stats or {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None or not self.eval_examples:
            return control

        rank, world_size = distributed_rank_and_world()
        local_examples = shard_examples_for_rank(
            self.eval_examples,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(
                "Module A inline generation eval: "
                f"{len(self.eval_examples)} examples sharded across {world_size} GPU(s)."
            )

        model = unwrap_trainer_model(self.trainer, kwargs["model"])
        was_training = model.training
        local_rows: list[dict[str, Any]] = []

        import torch
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for example in local_examples:
                    label_mode = str(example.get("label_mode") or "legacy")
                    if self.label_score_eval:
                        prediction = self._score_label(model, example)
                    else:
                        generated_text = self._generate_label(model, example)
                        prediction = extract_module_a_label(generated_text, label_mode=label_mode)
                    local_rows.append({
                        "prediction": prediction,
                        "target": str(example["target"]),
                        "task_id": str(example["task_id"]),
                        "instructions": [
                            str(item) for item in (example.get("all_instructions") or [])
                        ],
                        "label_mode": label_mode,
                    })
        finally:
            if was_training:
                FastVisionModel.for_training(model)
            release_cuda_memory()

        rows = gather_round_robin_shards(local_rows)
        if not rows:
            return control

        predictions = [str(row["prediction"]) for row in rows]
        metric_examples = [
            {
                "target": row["target"],
                "task_id": row["task_id"],
                "all_instructions": row["instructions"],
                "label_mode": row["label_mode"],
            }
            for row in rows
        ]
        log_metrics = module_a_metrics_from_predictions(
            metric_examples,
            predictions,
            beta=self.beta,
            sampling_stats=self.sampling_stats,
        )
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(log_metrics)
        if rank == 0:
            self.trainer.log(log_metrics)
        release_cuda_memory()
        return control

    def _generation_inputs(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = [
            copy.deepcopy(message)
            for message in example["messages"]
            if message.get("role") != "assistant"
        ]
        text = self.data_collator._base.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image, video, video_kwargs, video_metadata = self.data_collator._extract_images_videos_metadata(messages)
        proc_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if image:
            proc_kwargs["images"] = image
        if video:
            proc_kwargs["videos"] = video
            proc_kwargs["do_resize"] = False
            proc_kwargs.update(video_kwargs)
            if video_metadata:
                proc_kwargs["video_metadata"] = video_metadata
        return self.data_collator._base.processor(**proc_kwargs)

    def _generate_label(self, model: Any, example: dict[str, Any]) -> str:
        inputs = self._generation_inputs(example)
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": False,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        try:
            output_ids = model.generate(**inputs, **generate_kwargs)
            generated_ids = output_ids[:, input_length:]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        finally:
            del inputs
            if "output_ids" in locals():
                del output_ids
            if "generated_ids" in locals():
                del generated_ids
            release_cuda_memory()

    def _score_label(self, model: Any, example: dict[str, Any]) -> str:
        scores = self._score_label_nlls(model, example)
        return min(scores, key=scores.get)

    def _candidate_labels_for_example(self, example: dict[str, Any]) -> list[str]:
        label_mode = str(example.get("label_mode") or "legacy")
        if label_mode != "step_id":
            return list(MODULE_A_LABELS)
        candidate_labels = [MODULE_A_STEP_ID_NONE_LETTER, "B"]
        num_instructions = len(example.get("all_instructions") or [])
        for i in range(num_instructions):
            letter = chr(ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER) + i)
            if letter not in candidate_labels:
                candidate_labels.append(letter)
        return candidate_labels

    def _score_label_nlls(self, model: Any, example: dict[str, Any]) -> dict[str, float]:
        import torch

        candidate_labels = self._candidate_labels_for_example(example)
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        candidate_token_ids: list[int] = []
        for label in candidate_labels:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(
                    f"Module A label {label!r} must be exactly one token, got {token_ids}."
                )
            candidate_token_ids.append(int(token_ids[0]))

        # Collate one valid candidate so the common assistant prefix
        # (<think>...</think>) is represented exactly as it was during training.
        anchor_label = candidate_labels[0]
        device = next(model.parameters()).device
        scored_example = copy.deepcopy(example)
        for message in scored_example["messages"]:
            if message.get("role") == "assistant":
                message["content"] = [{"type": "text", "text": anchor_label}]
                break
        batch = self.data_collator([scored_example])
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        inputs.pop("response_loss_weight", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise RuntimeError("Module A label scoring requires response-only labels.")

        outputs = None
        try:
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden = last_hidden_states(outputs)
                shift_labels = labels[:, 1:]
                anchor_token_id = candidate_token_ids[0]
                anchor_positions = (shift_labels == anchor_token_id).nonzero(as_tuple=False)
                if anchor_positions.shape[0] != 1:
                    raise RuntimeError(
                        "Could not uniquely locate the assistant answer token for "
                        f"{anchor_label!r}; matches={anchor_positions.shape[0]}."
                    )
                batch_index, shift_index = anchor_positions[0].tolist()
                answer_hidden = hidden[batch_index, shift_index]

                output_embeddings = unwrap_output_embeddings(model, self.trainer)
                candidate_ids = torch.tensor(
                    candidate_token_ids,
                    dtype=torch.long,
                    device=answer_hidden.device,
                )
                vocab_size = getattr(output_embeddings, "out_features", None)
                if vocab_size is not None and answer_hidden.shape[-1] == vocab_size:
                    candidate_logits = answer_hidden.index_select(0, candidate_ids)
                else:
                    # Project one token, rather than every sequence token. Calling
                    # the module preserves any adapters/hooks on the LM head.
                    full_logits = output_embeddings(answer_hidden.unsqueeze(0))[0]
                    candidate_logits = full_logits.index_select(
                        0, candidate_ids.to(full_logits.device)
                    )
                candidate_nlls = -torch.log_softmax(candidate_logits.float(), dim=-1)
            return {
                label: float(candidate_nlls[index].detach().cpu())
                for index, label in enumerate(candidate_labels)
            }
        finally:
            del inputs
            if outputs is not None:
                del outputs
            release_cuda_memory()


def compute_generation_detection_metrics(
    *,
    model: Any,
    tokenizer: Any,
    eval_examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    data_collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )
    generator = ModuleADetectionMetricsCallback(
        eval_examples=eval_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
        beta=args.fbeta_beta,
        label_score_eval=args.label_score_eval,
        sampling_stats=getattr(args, "negative_sampling_stats", {}),
    )
    predictions: list[str] = []
    targets: list[str] = []
    task_ids: list[str] = []
    instructions_list: list[list[str]] = []
    task_targets: dict[str, list[str]] = defaultdict(list)
    task_predictions: dict[str, list[str]] = defaultdict(list)
    score_probabilities: list[dict[str, float]] = []
    score_target_letters: list[str] = []
    calibration_records: list[dict[str, Any]] = []
    response_seconds: list[float] = []

    for row_index, example in enumerate(generator.eval_examples):
        label_mode = str(example.get("label_mode") or args.module_a_label_mode)
        score_map: dict[str, float] | None = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        if args.label_score_eval:
            score_map = generator._score_label_nlls(model, example)
            prediction = min(score_map, key=score_map.get)
        else:
            generated_text = generator._generate_label(model, example)
            prediction = extract_module_a_label(generated_text, label_mode=label_mode)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        response_seconds.append(time.perf_counter() - inference_started)
        target = str(example["target"])
        task_id = str(example["task_id"])
        predictions.append(prediction)
        targets.append(target)
        task_ids.append(task_id)
        instructions_list.append([str(item) for item in (example.get("all_instructions") or [])])
        task_predictions[task_id].append(prediction)
        task_targets[task_id].append(target)

        if score_map is not None and label_mode == "step_id":
            probs = scores_to_probabilities(score_map)
            target_letter = extract_module_a_label(target, label_mode="step_id")
            score_probabilities.append(probs)
            score_target_letters.append(target_letter)
            calibration_records.append(
                calibration_record(
                    example,
                    probabilities=probs,
                    target_letter=target_letter,
                    prediction=prediction,
                    row_index=row_index,
                )
            )

    label_mode = str(args.module_a_label_mode)
    if label_mode == "step_id":
        metrics: dict[str, Any] = step_id_semantic_metrics(
            task_ids=task_ids,
            all_instructions=instructions_list,
            targets=targets,
            predictions=predictions,
            prefix="eval_step_id",
            beta=args.fbeta_beta,
        )
        metrics.update(
            binary_detection_metrics(
                targets,
                predictions,
                prefix="eval_detection",
                beta=args.fbeta_beta,
                label_mode="step_id",
            )
        )
        metrics["confusion_matrix"] = metrics.get("task_confusion_matrices", {})
    else:
        metrics = binary_detection_metrics(
            targets,
            predictions,
            prefix="eval_detection",
            beta=args.fbeta_beta,
            label_mode="legacy",
        )
        metrics["confusion_matrix"] = {
            "labels": ["WAIT", "COMPLETE"],
            "matrix": [
                [
                    sum(
                        1
                        for t, p in zip(targets, predictions)
                        if extract_module_a_label(t, label_mode="legacy") == row
                        and extract_module_a_label(p, label_mode="legacy") == col
                    )
                    for col in ("WAIT", "COMPLETE")
                ]
                for row in ("WAIT", "COMPLETE")
            ],
        }

    if score_probabilities:
        metrics.update(label_score_diagnostics(score_probabilities, score_target_letters))
        calibration_path = persist_calibration_records(
            calibration_records,
            args=args,
            tag="generation_eval",
        )
        if calibration_path is not None:
            metrics["calibration_records_path"] = str(calibration_path)
    else:
        metrics["eval_detection/auc"] = None
        metrics["eval_step_id/macro_auc"] = None
        metrics["auc_note"] = (
            "AUC requires --label-score-eval so soft label NLLs are available; "
            "hard generation-only predictions cannot define a ranking score."
        )
    metrics.update(negative_sampling_metrics(getattr(args, "negative_sampling_stats", {}), prefix="eval"))

    for task_id in sorted(task_targets):
        task_metrics = binary_detection_metrics(
            task_targets[task_id],
            task_predictions[task_id],
            prefix=f"eval_detection/task_{task_id}",
            beta=args.fbeta_beta,
            label_mode=label_mode,
        )
        metrics.update({
            key: value
            for key, value in task_metrics.items()
            if key.endswith(("/accuracy", "/f1", "/recall", "/precision", "/num_samples"))
        })
    metrics.update(inference_runtime_metrics(response_seconds))
    return metrics


def build_generation_eval_command(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "online_mistake_detection.training.module_a",
        "--generation-eval-only",
        "--eval-checkpoint",
        str(checkpoint_path),
        "--generation-eval-output-json",
        str(output_json),
        "--dataset",
        str(args.dataset),
        "--egoper-root",
        str(args.egoper_root),
        "--max-videos",
        str(args.max_videos),
        "--fps",
        str(args.fps),
        "--min-frames",
        str(args.min_frames),
        "--max-frames",
        str(args.max_frames),
        "--stride-seconds",
        str(args.stride_seconds),
        "--completion-margin",
        str(args.completion_margin),
        "--negative-to-positive-ratio",
        str(args.negative_to_positive_ratio),
        "--keep-last-wait-windows",
        str(args.keep_last_wait_windows),
        "--max-negative-windows-per-episode",
        str(args.max_negative_windows_per_episode),
        "--reset-jitter-seconds",
        "0",
        "--max-seq-length",
        str(args.max_seq_length),
        "--vision-resize",
        str(args.vision_resize),
        "--eval-generation-max-samples",
        str(args.eval_generation_max_samples),
        "--eval-generation-max-new-tokens",
        str(args.eval_generation_max_new_tokens),
        "--fbeta-beta",
        str(args.fbeta_beta),
        "--module-a-label-mode",
        str(args.module_a_label_mode),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
    ]
    if args.split_file:
        command.extend(["--split-file", str(args.split_file)])
    if args.video_ids:
        command.append("--video-ids")
        command.extend(str(item) for item in args.video_ids)
    if args.task_ids:
        command.append("--task-ids")
        command.extend(str(item) for item in args.task_ids)
    command.append("--load-in-4bit" if args.load_in_4bit else "--no-load-in-4bit")
    command.append("--load-in-8bit" if args.load_in_8bit else "--no-load-in-8bit")
    command.append("--load-in-16bit" if args.load_in_16bit else "--no-load-in-16bit")
    command.append("--label-score-eval" if args.label_score_eval else "--no-label-score-eval")
    command.append(
        "--write-calibration-records"
        if getattr(args, "write_calibration_records", True)
        else "--no-write-calibration-records"
    )
    command.extend(["--calibration-output-dir", str(Path(output_json).parent)])
    return command


def sanitize_wandb_artifact_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(char if char in allowed else "-" for char in name)
    return sanitized.strip(".-") or "checkpoint"


class EpochCheckpointCallback:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        eval_epochs: int,
        checkpoint_epochs: int,
        metric_for_best_model: str,
        greater_is_better: bool,
        keep_last_checkpoints: int,
        keep_best_checkpoints: int,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        wandb_log_best_checkpoints: bool,
        wandb_artifact_prefix: str,
        generation_eval_args: argparse.Namespace,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.output_dir = Path(output_dir)
        self.eval_epochs = eval_epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.keep_last_checkpoints = keep_last_checkpoints
        self.keep_best_checkpoints = keep_best_checkpoints
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.wandb_log_best_checkpoints = wandb_log_best_checkpoints
        self.wandb_artifact_prefix = wandb_artifact_prefix
        self.generation_eval_args = generation_eval_args
        self.best_metric: float | None = None
        self.no_improvement_count = 0
        self.best_records: list[dict[str, Any]] = []
        self._last_latest_epoch: int | None = None
        self._restore_best_manifest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        epoch = self._completed_epoch(state)
        if epoch is None:
            return control
        if epoch % self.eval_epochs == 0:
            control.should_evaluate = True
        if not state.is_world_process_zero:
            return control

        self._save_named_checkpoint("latest")
        self._last_latest_epoch = epoch

        if epoch % self.checkpoint_epochs == 0:
            self._save_named_checkpoint(f"ckpt_ep{epoch}")
            self._prune_epoch_checkpoints("ckpt_ep", self.keep_last_checkpoints)

        release_cuda_memory()
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if not state.is_world_process_zero or not metrics:
            return control
        epoch = self._epoch_label(state)
        if self.generation_eval_args.generation_eval_mode == "subprocess":
            generation_metrics = self._run_subprocess_generation_eval(epoch=epoch)
            if generation_metrics:
                metrics.update(generation_metrics)
                if self.trainer is not None:
                    self.trainer.log(generation_metrics)
        metric = self._metric_value(metrics)
        if metric is None:
            print(f"Metric {self.metric_for_best_model!r} was not found; skipping best checkpoint update.")
            return control

        if self._is_improvement(metric):
            self.best_metric = metric
            self.no_improvement_count = 0
            path = self._save_named_checkpoint(f"best_ep{epoch}")
            self.best_records.append({
                "epoch": epoch,
                "metric": metric,
                "path": str(path),
                "global_step": state.global_step,
            })
            self._prune_best_checkpoints()
            self._write_best_manifest()
            if self.wandb_log_best_checkpoints and path.exists():
                self._log_wandb_checkpoint(path=path, epoch=epoch, metric=metric, state=state)
        else:
            self.no_improvement_count += 1
            self._write_best_manifest()
            if (
                self.early_stopping_patience > 0
                and self.no_improvement_count >= self.early_stopping_patience
            ):
                print(
                    "Early stopping: "
                    f"{self.no_improvement_count} validation checks without improvement in "
                    f"{self.metric_for_best_model}."
                )
                control.should_training_stop = True
        release_cuda_memory()
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not state.is_world_process_zero:
            return control
        epoch = self._completed_epoch(state)
        if epoch is None or epoch != self._last_latest_epoch:
            self._save_named_checkpoint("latest")
        return control

    def _completed_epoch(self, state: Any) -> int | None:
        if state.epoch is None:
            return None
        rounded = round(float(state.epoch))
        if rounded < 1 or not math.isclose(float(state.epoch), rounded, abs_tol=1e-3):
            return None
        return int(rounded)

    def _epoch_label(self, state: Any) -> str:
        completed = self._completed_epoch(state)
        if completed is not None:
            return str(completed)
        if state.epoch is None:
            return f"step{state.global_step}"
        return str(float(state.epoch)).replace(".", "_")

    def _metric_value(self, metrics: dict[str, Any]) -> float | None:
        keys = [
            self.metric_for_best_model,
            f"eval_{self.metric_for_best_model}",
        ]
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric > self.best_metric + self.early_stopping_threshold
        return metric < self.best_metric - self.early_stopping_threshold

    def _save_named_checkpoint(self, name: str) -> Path:
        """Save a full HF-style checkpoint (weights + optimizer + scheduler + RNG)."""
        if self.trainer is None:
            raise RuntimeError("EpochCheckpointCallback.trainer must be assigned before training.")

        path = self.output_dir / name
        if path.exists():
            shutil.rmtree(path)
        write_trainer_checkpoint(self.trainer, path)
        release_cuda_memory()
        return path

    def _prune_epoch_checkpoints(self, prefix: str, keep: int) -> None:
        if keep == 0:
            candidates = list(self.output_dir.glob(f"{prefix}*"))
        else:
            candidates = sorted(
                self.output_dir.glob(f"{prefix}*"),
                key=lambda path: self._epoch_from_name(path.name, prefix),
            )[:-keep]
        for path in candidates:
            if path.is_dir():
                shutil.rmtree(path)

    def _prune_best_checkpoints(self) -> None:
        if self.keep_best_checkpoints == 0:
            records_to_remove = list(self.best_records)
            self.best_records = []
        else:
            reverse = self.greater_is_better
            self.best_records.sort(key=lambda item: item["metric"], reverse=reverse)
            records_to_remove = self.best_records[self.keep_best_checkpoints :]
            self.best_records = self.best_records[: self.keep_best_checkpoints]
        for record in records_to_remove:
            path = Path(record["path"])
            if path.is_dir():
                shutil.rmtree(path)

    def _write_best_manifest(self) -> None:
        manifest = {
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "best_metric": self.best_metric,
            "no_improvement_count": self.no_improvement_count,
            "best_checkpoints": self.best_records,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "best_checkpoints.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

    def _restore_best_manifest(self) -> None:
        manifest_path = self.output_dir / "best_checkpoints.json"
        if not manifest_path.is_file():
            return
        try:
            with open(manifest_path, encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        if (
            manifest.get("metric_for_best_model") != self.metric_for_best_model
            or bool(manifest.get("greater_is_better")) != self.greater_is_better
        ):
            return
        try:
            best_metric = float(manifest["best_metric"])
        except (KeyError, TypeError, ValueError):
            return
        records = manifest.get("best_checkpoints")
        if not isinstance(records, list):
            return
        self.best_metric = best_metric
        self.no_improvement_count = max(
            0,
            int(manifest.get("no_improvement_count", 0)),
        )
        self.best_records = [
            dict(record)
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and Path(record["path"]).is_dir()
        ]

    def _log_wandb_checkpoint(self, *, path: Path, epoch: str, metric: float, state: Any) -> None:
        try:
            import wandb
        except ModuleNotFoundError:
            print("W&B is not installed; skipping best checkpoint artifact upload.")
            return
        if wandb.run is None:
            print("W&B run is not active; skipping best checkpoint artifact upload.")
            return

        run_id = wandb.run.id or "run"
        artifact_name = sanitize_wandb_artifact_name(f"{self.wandb_artifact_prefix}-{run_id}-best")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "checkpoint": path.name,
                "epoch": epoch,
                "global_step": state.global_step,
                "metric_for_best_model": self.metric_for_best_model,
                "metric": metric,
                "greater_is_better": self.greater_is_better,
            },
        )
        artifact.add_dir(str(path), name=path.name)
        run_config = self.output_dir / "run_config.json"
        if run_config.exists():
            artifact.add_file(str(run_config), name="run_config.json")
        best_manifest = self.output_dir / "best_checkpoints.json"
        if best_manifest.exists():
            artifact.add_file(str(best_manifest), name="best_checkpoints.json")
        wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
        print(f"Logged W&B best checkpoint artifact: {artifact_name} ({path})")

    def _run_subprocess_generation_eval(self, *, epoch: str) -> dict[str, float]:
        if self.trainer is None:
            return {}
        if self.generation_eval_args.eval_generation_max_samples == 0:
            return {}

        checkpoint_path = self.output_dir / "latest"
        if not checkpoint_path.exists():
            return {}

        output_json = self.output_dir / f"generation_eval_ep{epoch}.json"
        command = build_generation_eval_command(
            args=self.generation_eval_args,
            checkpoint_path=checkpoint_path,
            output_json=output_json,
        )
        restore_device = offload_trainer_cuda_state(self.trainer)
        print("Module A generation eval command:")
        print(" ".join(command))
        try:
            completed = subprocess.run(
                command,
                check=False,
                env=generation_eval_subprocess_env(),
                cwd=str(PROJECT_ROOT),
            )
            if completed.returncode != 0:
                print(
                    "Module A generation eval failed with "
                    f"exit {completed.returncode}; continuing training without "
                    "eval_step_id metrics. See the subprocess traceback above."
                )
                return {}
        finally:
            restore_trainer_cuda_state(self.trainer, restore_device)

        if not output_json.exists():
            return {}
        with open(output_json, "r", encoding="utf-8") as file:
            payload = json.load(file)
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            return {}
        return scalar_metrics_for_logging(metrics)

    @staticmethod
    def _epoch_from_name(name: str, prefix: str) -> int:
        try:
            return int(name.removeprefix(prefix))
        except ValueError:
            return -1


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    class ModuleASFTTrainer(SFTTrainer):
        def evaluate(self, *positional: Any, **keyword: Any) -> Any:
            eval_count = int(getattr(self, "_module_a_eval_count", 0))
            reason = (
                "resume-start"
                if eval_count == 0 and bool(self.args.eval_on_start)
                else "scheduled"
            )
            if self.is_world_process_zero():
                try:
                    num_examples: int | str = len(self.eval_dataset)
                except TypeError:
                    num_examples = "unknown"
                print(
                    "[eval] START "
                    f"reason={reason} epoch={self.state.epoch} "
                    f"global_step={self.state.global_step} examples={num_examples} "
                    f"reuse_loss_forward={bool(getattr(self, 'module_a_reuse_eval_forward', False))}",
                    flush=True,
                )
            self._module_a_eval_count = eval_count + 1
            return super().evaluate(*positional, **keyword)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            inputs.pop("eval_row_index", None)
            inputs.pop("eval_answer_positions", None)
            inputs.pop("eval_candidate_token_ids", None)
            response_loss_weight = inputs.pop("response_loss_weight", None)
            if response_loss_weight is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs.pop("labels", None)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            loss = response_token_loss(
                model,
                outputs,
                labels,
                token_weights=response_loss_weight,
                trainer=self,
            )
            return (loss, outputs) if return_outputs else loss

        def prediction_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: list[str] | None = None,
        ) -> tuple[Any | None, Any | None, Any | None]:
            if "eval_row_index" not in inputs:
                return super().prediction_step(
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )

            import torch

            prepared = self._prepare_inputs(inputs)
            row_indices = prepared.pop("eval_row_index")
            answer_positions = prepared.pop("eval_answer_positions")
            candidate_token_ids = prepared.pop("eval_candidate_token_ids")
            labels = prepared.get("labels")
            if labels is None:
                raise RuntimeError("Reused Module A label scoring requires labels.")
            model_inputs = dict(prepared)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model,
                        model_inputs,
                        return_outputs=True,
                    )
                letter_logits = step_id_letter_logits_from_outputs(
                    model,
                    outputs,
                    answer_positions,
                    candidate_token_ids,
                    trainer=self,
                )
            return (
                loss.detach().mean(),
                letter_logits.detach(),
                row_indices.detach(),
            )

    FastVisionModel.for_training(model)
    bf16 = is_bf16_available()
    reuse_eval_forward = bool(
        args.generation_eval_mode == "inline"
        and args.label_score_eval
        and args.module_a_label_mode == "step_id"
        and args.eval_generation_max_samples != 0
    )
    eval_on_start = bool(
        args.eval_on_resume
        and (
            (
                args.resume_from_checkpoint
                and checkpoint_needs_resume_evaluation(args.resume_from_checkpoint)
            )
            or (args.load_adapter_from and not args.resume_from_checkpoint)
        )
    )
    if args.eval_on_resume and int(os.environ.get("RANK", "0")) == 0:
        if args.resume_from_checkpoint:
            status = "enabled" if eval_on_start else "skipped"
            detail = (
                "unevaluated epoch boundary"
                if eval_on_start
                else "checkpoint is mid-epoch or already has eval metrics"
            )
            print(f"[eval] resume-start {status}: {detail}", flush=True)
        elif args.load_adapter_from:
            print(
                f"[eval] adapter-start enabled: scoring loaded weights from {args.load_adapter_from}",
                flush=True,
            )
    if args.group_by_length:
        train_rows = [
            {
                **row,
                "length": estimated_window_frames(
                    row["window_start"],
                    row["window_end"],
                    fps=args.fps,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                ),
            }
            for row in train_dataset
        ]
    else:
        train_rows = train_dataset
    hf_train_dataset = Dataset.from_list(train_rows)
    hf_eval_dataset = Dataset.from_list([
        {
            **example,
            **(
                {
                    "length": estimated_window_frames(
                        example["window_start"],
                        example["window_end"],
                        fps=args.fps,
                        min_frames=args.min_frames,
                        max_frames=args.max_frames,
                    )
                }
                if args.group_by_length
                else {}
            ),
            **({"eval_row_index": index} if reuse_eval_forward else {}),
        }
        for index, example in enumerate(eval_dataset)
    ])
    callbacks: list[TrainerCallback] = []
    trainer_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        report_to=args.report_to,
        dataset_num_proc=args.dataset_num_proc,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        max_seq_length=args.max_seq_length,
        torch_empty_cache_steps=args.torch_empty_cache_steps or None,
        train_sampling_strategy=(
            "group_by_length" if args.group_by_length else "random"
        ),
        length_column_name="length",
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_broadcast_buffers=False,
        ddp_timeout=args.ddp_timeout,
        fp16=not bf16,
        bf16=bf16,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_on_start=eval_on_start,
    )
    if args.train_mode == "epochs":
        trainer_args["num_train_epochs"] = args.num_train_epochs
    else:
        trainer_args["max_steps"] = args.max_steps

    data_collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
        include_response_loss_weight=args.positive_loss_weight != 1.0,
        letter_only_loss=bool(args.letter_only_loss),
    )
    detection_callback: ModuleADetectionMetricsCallback | None = None
    if (
        args.generation_eval_mode == "inline"
        and args.eval_generation_max_samples != 0
        and not reuse_eval_forward
    ):
        detection_callback = ModuleADetectionMetricsCallback(
            eval_examples=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            max_samples=args.eval_generation_max_samples,
            max_new_tokens=args.eval_generation_max_new_tokens,
            beta=args.fbeta_beta,
            label_score_eval=args.label_score_eval,
            sampling_stats=getattr(args, "negative_sampling_stats", {}),
        )
        callbacks.append(detection_callback)
    checkpoint_callback = EpochCheckpointCallback(
        output_dir=args.output_dir,
        eval_epochs=args.eval_epochs,
        checkpoint_epochs=args.checkpoint_epochs,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        keep_last_checkpoints=args.keep_last_checkpoints,
        keep_best_checkpoints=args.keep_best_checkpoints,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        wandb_log_best_checkpoints=args.wandb_log_best_checkpoints,
        wandb_artifact_prefix=args.wandb_artifact_prefix,
        generation_eval_args=args,
    )
    if not is_eval_only_run(args):
        callbacks.append(checkpoint_callback)

    def compute_reused_label_score_metrics(eval_prediction: Any) -> dict[str, float]:
        import numpy as np

        logits = np.asarray(eval_prediction.predictions)
        row_indices = np.asarray(eval_prediction.label_ids).reshape(-1)
        metric_examples: list[dict[str, Any]] = []
        metric_predictions: list[str] = []
        score_probabilities: list[dict[str, float]] = []
        score_target_letters: list[str] = []
        calibration_records: list[dict[str, Any]] = []
        max_samples = args.eval_generation_max_samples
        for row_logits, row_index_value in zip(logits, row_indices):
            row_index = int(row_index_value)
            if max_samples >= 0 and row_index >= max_samples:
                continue
            example = eval_dataset[row_index]
            candidate_count = 2 + len(example.get("all_instructions") or [])
            if candidate_count > len(MODULE_A_STEP_ID_CANDIDATE_LABELS):
                raise RuntimeError(
                    f"Module A eval row {row_index} requires {candidate_count} labels; "
                    "only A-Z are supported."
                )
            candidate_labels = MODULE_A_STEP_ID_CANDIDATE_LABELS[:candidate_count]
            prediction_index = int(np.argmax(row_logits[:candidate_count]))
            prediction = candidate_labels[prediction_index]
            metric_examples.append(example)
            metric_predictions.append(prediction)

            # Keep the full letter distribution: precision/recall below collapse
            # it to one argmax decision, which fixes the operating point.
            probabilities = logits_to_probabilities(
                [float(value) for value in row_logits[:candidate_count]],
                candidate_labels,
            )
            target_letter = extract_module_a_label(
                str(example["target"]), label_mode="step_id"
            )
            score_probabilities.append(probabilities)
            score_target_letters.append(target_letter)
            calibration_records.append(
                calibration_record(
                    example,
                    probabilities=probabilities,
                    target_letter=target_letter,
                    prediction=prediction,
                    row_index=row_index,
                )
            )

        metrics = module_a_metrics_from_predictions(
            metric_examples,
            metric_predictions,
            beta=args.fbeta_beta,
            sampling_stats=getattr(args, "negative_sampling_stats", {}),
        )
        metrics.update(label_score_diagnostics(score_probabilities, score_target_letters))
        # DDP pads the eval sampler so every rank gets equal batches, which can
        # repeat rows in the gathered predictions. One line per window.
        unique_records = {record["row_index"]: record for record in calibration_records}
        persist_calibration_records(
            [unique_records[key] for key in sorted(unique_records)],
            args=args,
            global_step=getattr(getattr(trainer, "state", None), "global_step", None),
        )
        return metrics

    trainer = ModuleASFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
        compute_metrics=compute_reused_label_score_metrics if reuse_eval_forward else None,
    )
    trainer.module_a_eval_examples = eval_dataset
    trainer.module_a_reuse_eval_forward = reuse_eval_forward
    if detection_callback is not None:
        detection_callback.trainer = trainer
    if not is_eval_only_run(args):
        checkpoint_callback.trainer = trainer
        attach_cluster_checkpointing(trainer, args)
    return trainer


def save_outputs(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved LoRA adapter and tokenizer to: {output_path}")


def run_generation_eval_only(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    if args.eval_generation_max_samples == 0:
        metrics: dict[str, Any] = {}
    else:
        model, tokenizer = load_generation_eval_model(args)
        metrics = compute_generation_detection_metrics(
            model=model,
            tokenizer=tokenizer,
            eval_examples=val_conversations,
            args=args,
        )
        del model
        release_cuda_memory()

    confusion_matrix = metrics.pop("confusion_matrix", None) if isinstance(metrics, dict) else None
    task_confusion = (
        metrics.pop("task_confusion_matrices", None) if isinstance(metrics, dict) else None
    )
    if confusion_matrix is None and isinstance(task_confusion, dict):
        confusion_matrix = task_confusion
    auc_note = metrics.pop("auc_note", None) if isinstance(metrics, dict) else None
    # Scalar-only metrics for W&B / trainer; nested CMs live beside them in the JSON.
    scalar_metrics = scalar_metrics_for_logging(metrics) if isinstance(metrics, dict) else {}
    payload = {
        "checkpoint": args.eval_checkpoint,
        "module_a_label_mode": args.module_a_label_mode,
        "label_score_eval": bool(args.label_score_eval),
        "num_validation_examples": len(val_conversations),
        "num_generation_examples": (
            len(val_conversations)
            if args.eval_generation_max_samples < 0
            else min(args.eval_generation_max_samples, len(val_conversations))
        ),
        "metrics": scalar_metrics,
        "confusion_matrix": confusion_matrix,
        "auc_note": auc_note,
    }
    heatmap_paths: list[str] = []
    if args.generation_eval_output_json:
        output_path = Path(args.generation_eval_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Heatmaps only for standalone / post-training eval (not epoch subprocess logs).
        write_heatmaps = bool(getattr(args, "write_confusion_heatmaps", False))
        if write_heatmaps and isinstance(confusion_matrix, dict) and confusion_matrix:
            figures_dir = output_path.parent / f"{output_path.stem}_confusion"
            if all(isinstance(v, dict) and "matrix" in v for v in confusion_matrix.values()):
                written = write_step_id_confusion_heatmaps(confusion_matrix, figures_dir)
            else:
                labels = confusion_matrix.get("labels") or []
                matrix = confusion_matrix.get("matrix")
                written = []
                if labels and matrix:
                    path = figures_dir / "confusion.png"
                    save_confusion_heatmap_png(
                        matrix=matrix,
                        row_labels=[str(item) for item in labels],
                        col_labels=[str(item) for item in labels],
                        title="Module A confusion",
                        output_path=path,
                    )
                    written = [path]
            heatmap_paths = [str(path) for path in written]
            payload["confusion_heatmap_dir"] = str(figures_dir)
            payload["confusion_heatmaps"] = heatmap_paths
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        if heatmap_paths:
            print(f"Wrote {len(heatmap_paths)} confusion heatmaps under: {figures_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def dummy_train_rows_for_eval_only(
    eval_rows: list[dict[str, Any]],
    world_size: int,
) -> list[dict[str, Any]]:
    """HF Trainer still builds a train loader; size it for DDP without training."""
    if not eval_rows:
        raise RuntimeError("Zero-shot eval requires a non-empty validation set.")
    return eval_rows[:1] * max(int(world_size), 1)


def run_eval_only(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    adapter = getattr(args, "load_adapter_from", None) or None
    print(
        f"[eval] EVAL-ONLY START base_model={args.model_name} "
        f"adapter={adapter or 'none (zero-shot)'} "
        f"examples={len(val_conversations)} reuse_loss_forward=True",
        flush=True,
    )
    args.resume_from_checkpoint = None
    model, tokenizer = load_unsloth_model(args)
    _, world_size = distributed_rank_and_world()
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dummy_train_rows_for_eval_only(val_conversations, world_size),
        eval_dataset=val_conversations,
        args=args,
    )
    metrics = trainer.evaluate()
    runtime_metrics = trainer_inference_runtime_metrics(
        metrics,
        num_examples=len(val_conversations),
        world_size=world_size,
        per_device_batch_size=args.per_device_eval_batch_size,
    )
    metrics.update(runtime_metrics)
    if runtime_metrics:
        trainer.log(runtime_metrics)
    payload = {
        "checkpoint": adapter,
        "zero_shot_eval": adapter is None,
        "model_name": args.model_name,
        "module_a_label_mode": args.module_a_label_mode,
        "label_score_eval": bool(args.label_score_eval),
        "max_frames": args.max_frames,
        "vision_resize": args.vision_resize,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "num_validation_examples": len(val_conversations),
        "metrics": scalar_metrics_for_logging(metrics) if isinstance(metrics, dict) else {},
    }
    output_json = args.generation_eval_output_json
    if not output_json:
        stem = "zero_shot_eval" if adapter is None else f"eval_{Path(adapter).name}"
        output_json = str(Path(args.output_dir) / f"{stem}.json")
    if trainer.is_world_process_zero():
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"Wrote zero-shot eval metrics to: {output_path}", flush=True)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    configure_kernel_caches()
    args = parse_args()
    configure_unsloth_logits(args)
    dataset = build_module_a_dataset(args)
    conversations = to_unsloth_conversations(
        dataset,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    if not conversations:
        raise SystemExit("Module A dataset produced no conversations.")
    split, split_path = load_or_create_video_split(dataset, args)
    train_conversations, val_conversations = split_conversations(conversations, split)
    train_conversations, sampling_stats = cap_negative_conversations(
        train_conversations,
        max_negatives_per_episode=args.max_negative_windows_per_episode,
    )
    sampling_stats = {
        **sampling_stats,
        "validation_windows_sampled": False,
        "validation_policy": "full_accumulated_windows",
        "train_positive_loss_weight": float(args.positive_loss_weight),
        "train_positive_oversample_factor": float(args.positive_oversample_factor),
        "letter_only_loss": bool(args.letter_only_loss),
        "load_adapter_from": args.load_adapter_from,
    }
    args.negative_sampling_stats = sampling_stats
    print("Module A train negative sampling:")
    print(json.dumps(sampling_stats, indent=2, sort_keys=True))
    print("Module A validation retains every accumulated window.")
    train_conversations = oversample_positive_conversations(
        train_conversations,
        factor=args.positive_oversample_factor,
        seed=args.seed,
    )
    apply_positive_loss_weights(
        train_conversations,
        positive_weight=args.positive_loss_weight,
    )
    apply_positive_loss_weights(val_conversations, positive_weight=1.0)
    train_conversations = cap_conversations(
        train_conversations,
        args.max_train_examples,
        split_name="train",
    )
    val_conversations = cap_conversations(
        val_conversations,
        args.max_eval_examples,
        split_name="eval",
    )
    if args.generation_eval_only:
        run_generation_eval_only(args=args, val_conversations=val_conversations)
        return
    if is_eval_only_run(args):
        if is_distributed_rank_zero():
            save_run_config(
                args=args,
                dataset=dataset,
                conversations=conversations,
                train_conversations=train_conversations,
                val_conversations=val_conversations,
                split=split,
                split_path=split_path,
            )
        print("Module A validation summary:")
        print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
        run_eval_only(args=args, val_conversations=val_conversations)
        return
    if args.dry_run:
        print_dry_run(
            dataset,
            train_conversations,
            val_conversations,
            split,
            split_path,
            args.print_examples,
        )
        return
    if is_distributed_rank_zero():
        save_run_config(
            args=args,
            dataset=dataset,
            conversations=conversations,
            train_conversations=train_conversations,
            val_conversations=val_conversations,
            split=split,
            split_path=split_path,
        )

    print("Module A full dataset summary:")
    print(json.dumps(summarize_module_a(dataset), indent=2, ensure_ascii=False))
    print(f"Module A split file: {split_path}")
    print("Module A train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module A validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))

    apply_resume_checkpoint(args)

    model, tokenizer = load_unsloth_model(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_conversations,
        eval_dataset=val_conversations,
        args=args,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        save_outputs(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
