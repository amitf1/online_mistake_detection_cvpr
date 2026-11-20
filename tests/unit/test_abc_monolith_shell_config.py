from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / "configs/env/abc_monolith.env"
BUILDER = ROOT / "train/abc_monolith/build_train_args.sh"
TEST_ENV_FILE = ROOT / "configs/env/monolith_test.env"
TEST_RUNNER = ROOT / "evaluate/abc_monolith/run_test.sh"
SHELL_FILES = (
    BUILDER,
    TEST_RUNNER,
    ROOT / "scripts/run.sh",
    ROOT / "scripts/slurm/train.sbatch",
    ROOT / "scripts/slurm/test.sbatch",
    ROOT / "train/module_a/build_train_args.sh",
    ROOT / "train/module_b/build_train_args.sh",
    ROOT / "train/module_c/build_train_args.sh",
    ROOT / "train/pipeline/build_pipeline_args.sh",
    ROOT / "train/pipeline/run_sharded_pipeline.sh",
    ROOT / "train/lib/cluster_checkpoint.sh",
)


def _values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_monolith_profile_is_isolated_and_fixed() -> None:
    values = _values(ENV_FILE)
    assert values["MODEL_NAME"] == "unsloth/Qwen3.8-27B"
    assert values["SPLIT_FILE"] == "configs/splits/egoper_video_split_seed3407.json"
    assert values["MAX_FRAMES"] == values["EVAL_MAX_FRAMES"] == "54"
    assert values["VISION_RESIZE"] == "1024"
    assert values["MAX_SEQ_LENGTH"] == "20480"
    assert values["EVAL_GENERATION_MAX_NEW_TOKENS"] == "192"
    assert values["METRIC_FOR_BEST_MODEL"] == "eval_monolith/stage_composite"
    assert values["EVAL_ON_RESUME"] == "true"
    assert values["FINETUNE_VISION_LAYERS"] == "true"
    assert "/abc_monolith/" in values["OUTPUT_DIR"]
    assert "/abc_monolith/" in values["CKPT_DIR"]


def test_builder_uses_latest_pointer_only_on_requeue(tmp_path: Path) -> None:
    egoper = tmp_path / "EgoPER"
    egoper.mkdir()
    (egoper / "annotation.json").write_text("{}\n", encoding="utf-8")
    ckpt_dir = tmp_path / "cluster"
    ckpt_dir.mkdir()
    pointer = ckpt_dir / "latest.json"
    pointer.write_text('{"checkpoint_dir":"step_3"}\n', encoding="utf-8")
    command = f'source "{BUILDER}"; printf \'%s\\n\' "${{TRAIN_ARGS[@]}}"'
    env = {
        **os.environ,
        "PYTHON_BIN": "python3",
        "EGOPER_ROOT": str(egoper),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "HF_CACHE": str(tmp_path / "hf"),
        "WANDB_DIR": str(tmp_path / "wandb"),
        "CKPT_DIR": str(ckpt_dir),
        "SLURM_RESTART_COUNT": "0",
    }
    fresh = subprocess.run(
        ["bash", "-c", command], cwd=ROOT, env=env, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert "--resume" not in fresh
    assert fresh[fresh.index("--egoper-root") + 1] == str(egoper)

    env["SLURM_RESTART_COUNT"] = "1"
    resumed = subprocess.run(
        ["bash", "-c", command], cwd=ROOT, env=env, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert resumed[resumed.index("--resume") + 1] == str(pointer)


def test_shell_files_parse() -> None:
    for path in SHELL_FILES:
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_test_runner_uses_independent_workers() -> None:
    text = TEST_RUNNER.read_text(encoding="utf-8")
    assert "torch.distributed.run" not in text
    assert "--video-shard-rank" in text
    assert "CUDA_VISIBLE_DEVICES" in text


def test_full_test_profile_enables_all_final_metrics() -> None:
    values = _values(TEST_ENV_FILE)
    assert values["EVAL_SPLIT"] == "test"
    assert values["MAX_FRAMES"] == "54"
    assert values["VISION_RESIZE"] == "1024"
    assert values["MONOLITH_CANDIDATE_SCORING"] == "true"
    assert values["DEFER_METRICS"] == "true"
    assert values["CHECKPOINT"].endswith("/best_ep6")
