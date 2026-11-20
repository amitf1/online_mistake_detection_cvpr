# Online Mistake Detection in Instruction-Conditioned Procedural Video

This repository contains the implementation of our paper: Online Mistake Detection in Instruction-Conditioned Procedural Video.

Code, configurations, final predictions and result tables for our two online mistake-detection systems on **EgoPER**:

- **Modular.** Three LoRA adapters on one video-language model, run causally on a video stream:
  - **Module A** watches a growing clip and decides when a step has finished and which step it was (or an off-script `EXTRA` step).
  - **Module B** places the finished step in time (start and end).
  - **Module C** classifies how the step was executed: `CORRECT`, `CORRECTION`, `MODIFICATION` or `SLIP`. `EXTRA` steps become `ADDITION`.

  Module C runs inside the stream: it classifies each step as soon as Module B places it. When Module B or Module C judges that the step is not finished yet, the stream keeps its position, the clip grows, and the step is tried again at the next poll.
- **Joint.** One LoRA adapter answers all three questions in a single generation per poll.

Both systems use `unsloth/Qwen3.8-27B` (bf16, LoRA r=16, α=32) and the fixed split `configs/splits/egoper_video_split_seed3407.json` (70/15/15 by video, stratified by recipe and error class, 58 test videos).

## Table of contents

1. [Repository contents](#1-repository-contents)
2. [Environment](#2-environment)
3. [Data](#3-data)
4. [Checkpoints](#4-checkpoints)
5. [How runs are configured](#5-how-runs-are-configured)
6. [Training (with validation)](#6-training-with-validation)
7. [Continue from a checkpoint](#7-continue-from-a-checkpoint)
8. [Testing on the EgoPER test split](#8-testing-on-the-egoper-test-split)
9. [Reproducing all results from the checkpoints](#9-reproducing-all-results-from-the-checkpoints)
10. [Evaluation and metric definitions](#10-evaluation-and-metric-definitions)
11. [Released results](#11-released-results)
12. [Running with or without Slurm](#12-running-with-or-without-slurm)
13. [Unit tests](#13-unit-tests)

## 1. Repository contents

| Path | What it is |
|---|---|
| `src/online_mistake_detection/` | The library. |
| `src/online_mistake_detection/datasets/` | EgoPER loader, labels and training-example builders. |
| `src/online_mistake_detection/windowing/` | Builds training and validation clips for Modules A, B and C and for the Joint model. |
| `src/online_mistake_detection/prompts/` | The prompts and output parsers for each module and for the Joint model. |
| `src/online_mistake_detection/training/` | Trainers for Modules A, B and C and for the Joint model (Unsloth + TRL), plus preemption-safe checkpointing. |
| `src/online_mistake_detection/pipeline/` | The causal online runner of the Modular system, the adapter bank that serves A, B and C from one base model, event types, video sharding and a work queue. |
| `src/online_mistake_detection/metrics/` | Validation metrics for each module and the EgoPER test metrics (TAS F1, binary and fine-grained mistake F1, EDA, omission). |
| `src/online_mistake_detection/models/` | Model loading, video decoding and collators. |
| `train/module_{a,b,c}.py`, `train/abc_monolith.py` | Training entry points. |
| `train/*/build_*_args.sh` | Turn an env file into the training or test command. |
| `train/pipeline/` | Launchers for the Modular test, one process per GPU. |
| `inference/online_pipeline.py` | Modular test entry point. |
| `evaluate/abc_monolith/` | Joint test inference (`run_test.sh`, `run.py`) and shard merging (`merge.py`). |
| `evaluate/online_pipeline/` | Offline rescoring of a Modular test artifact. |
| `evaluate/egoper/final_predictions.py` | Turns the two test artifacts into the final per-video segment files. |
| `evaluate/egoper/evaluate_final.py` | System-level metrics from those files (Section 10). |
| `evaluate/egoper/component_metrics.py` | Component-level metrics of Modules A, B and C from the Modular test artifact. |
| `evaluate/egoper/explanations.py` | BERTScore of the mistake explanations, with two controls. |
| `evaluate/egoper/timing.py` | Inference time of both systems from the test artifacts. |
| `evaluate/egoper/tables.py` | Renders the result files as tables and compares two result folders. |
| `scripts/run.sh` | The single launcher for every stage, run locally or inside Slurm. |
| `scripts/reproduce.sh` | Runs both tests from the checkpoints and computes every result table. |
| `scripts/slurm/{train,test}.sbatch` | Slurm templates. |
| `configs/env/` | The six run configurations used for the reported numbers (see below). |
| `configs/splits/egoper_video_split_seed3407.json` | The train, validation and test split. |
| `configs/{datasets,module_a,module_b,module_c}/` | Static dataset and module settings. |
| `results/final_predictions/` | Final test predictions of both systems and the matching ground truth. |
| `results/*.json`, `results/tables.md` | The released result tables (Section 11). |
| `docker/` | Two Dockerfiles: generic CUDA 12.8, and the NGC CUDA 13 image used for the reported runs. |
| `tests/` | CPU unit tests. |
| `checkpoints/` | Empty. Put the released adapters here (see Section 4). |

These are the configurations used for the reported numbers:

| Env file | Stage | Released checkpoint |
|---|---|---|
| `configs/env/module_a.env` | Train Module A | `best_ep6` |
| `configs/env/module_b.env` | Train Module B | `best_ep4` |
| `configs/env/module_c.env` | Train Module C | `best_ep4` |
| `configs/env/abc_monolith.env` | Train the Joint model | `best_ep6` |
| `configs/env/pipeline_test.env` | Test the Modular system (causal stream) | uses the three module checkpoints |
| `configs/env/monolith_test.env` | Test the Joint system | uses the Joint checkpoint |

## 2. Environment

The reported runs used NVIDIA B200 GPUs (192 GB): 8 GPUs per training job and 2 GPUs per test job. During the Modular test each GPU worker holds one copy of the base model in bf16 and swaps the A, B and C adapters in place before each call. On smaller GPUs, set `LOAD_IN_8BIT=true` or `LOAD_IN_4BIT=true` in the env file. Those settings were not used for the reported numbers.

`evaluate/egoper/explanations.py` uses `bert-score` (in `requirements.txt`) and downloads `microsoft/deberta-xlarge-mnli` on first use.

### Option A: Python virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "torch>=2.10.0,<2.11.0" torchvision --index-url https://download.pytorch.org/whl/cu128
pip install unsloth==2026.6.7 unsloth_zoo==2026.6.5 --no-deps
pip install -r requirements.txt
pip install -e . --no-deps
```

Install PyTorch before Unsloth, and install Unsloth with `--no-deps`, so it doesn't replace the CUDA build of PyTorch. `ffmpeg` must be available on the system.

### Option B: Docker

```bash
# NGC PyTorch (CUDA 13); the environment used for the reported runs
docker build -f docker/Dockerfile.ngc -t omd:ngc .
# or a generic CUDA 12.8 image
docker build -f docker/Dockerfile -t omd:cuda12 .

docker run --gpus all --ipc=host -it \
  -v "$PWD":/workspace -v /path/to/EgoPER:/workspace/data/EgoPER \
  -v /path/to/hf_cache:/cache/huggingface omd:ngc bash
```

The NGC image also installs the fast linear-attention kernels (`causal-conv1d`, `flash-linear-attention`) used by the Qwen3.5-family model. Without them, long clips run out of memory.

The model weights are downloaded from the Hugging Face Hub on first use. Set `HF_HOME` (or `HF_CACHE`) to choose where they are cached.

## 3. Data

Download EgoPER from its official release and arrange it as follows. By default the code looks in `data/EgoPER`; to use another location, set `EGOPER_ROOT=/path/to/EgoPER`.

```
data/EgoPER/
├── annotation.json
├── task_graph.txt
├── coffee/
│   ├── training.txt  validation.txt  test.txt
│   ├── trim_start_end.txt
│   ├── coffee_normal_actions.txt
│   └── videos/trim_videos/<video_id>.mp4
├── oatmeal/  pinwheels/  quesadilla/  tea/   (same layout)
```

Our split is `configs/splits/egoper_video_split_seed3407.json`. All runs read it, and nothing regenerates it (`REGENERATE_SPLIT=false`). EgoPER's own split files are not used.

## 4. Checkpoints

The trained adapters are distributed separately. Put them here:

```
checkpoints/
├── module_a/best_ep6/
├── module_b/best_ep4/
├── module_c/best_ep4/
└── abc_monolith/best_ep6/
```

Each folder is a full trainer checkpoint: the LoRA adapter, plus optimizer, scheduler and RNG state. It can be used for inference (Section 8) or to continue training (Section 7). The test env files already point to these paths. To use checkpoints stored somewhere else, change `MODULE_{A,B,C}_CHECKPOINT` in `configs/env/pipeline_test.env` and `CHECKPOINT` in `configs/env/monolith_test.env`, or set them in the environment.

## 5. How runs are configured

Every stage is started with the same command:

```bash
bash scripts/run.sh <stage> <env file>
#   <stage> = module_a | module_b | module_c | abc_monolith | pipeline_test | monolith_test
```

`scripts/run.sh` does the following:

1. Sets `PYTHONPATH`, the Hugging Face cache and per-job Triton/Inductor caches under `/tmp`.
2. Detects how many GPUs are visible (from Slurm or `CUDA_VISIBLE_DEVICES`).
3. Builds the command from the env file and runs it, passing `SIGTERM` on to it so preemption is handled cleanly.

The env file holds the run settings. **Variables already set in your shell override it**, because the file only fills in values that aren't already set. You can also put machine-specific settings in an optional `.env` at the repository root; it is ignored by git. For example:

```bash
EGOPER_ROOT=/datasets/EgoPER OUTPUT_ROOT=/scratch/omd NUM_GPUS=4 \
  bash scripts/run.sh module_a configs/env/module_a.env
```

Commonly overridden variables:

| Variable | Default | Meaning |
|---|---|---|
| `EGOPER_ROOT` | `data/EgoPER` | Dataset root. |
| `OUTPUT_ROOT` | `outputs` | Where runs, checkpoints and test artifacts are written. |
| `HF_CACHE` / `HF_HOME` | `~/.cache/huggingface` | Model cache. |
| `NUM_GPUS` | from the env file, or detected | Number of GPUs (torchrun processes for training, workers for testing). |
| `PYTHON_BIN` | `python` | Python interpreter. |
| `REPORT_TO` | `tensorboard` | Trainer logger. Set `wandb` (with `WANDB_PROJECT` and `WANDB_ENTITY`) to log to Weights & Biases. |
| `RUN_NAME` | from the env file | Name of the output folder. |

Training keeps the global batch size at 16 (`NUM_GPUS × PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`). If you change `NUM_GPUS`, change `GRADIENT_ACCUMULATION_STEPS` with it. For the Joint model, also set `EXPECTED_WORLD_SIZE` to the new GPU count. For example, with 4 GPUs: `NUM_GPUS=4 GRADIENT_ACCUMULATION_STEPS=4 EXPECTED_WORLD_SIZE=4`.

## 6. Training (with validation)

Each module and the Joint model is trained from the base model with one command:

```bash
bash scripts/run.sh module_a     configs/env/module_a.env
bash scripts/run.sh module_b     configs/env/module_b.env
bash scripts/run.sh module_c     configs/env/module_c.env
bash scripts/run.sh abc_monolith configs/env/abc_monolith.env
```

The three modules are trained independently; they don't depend on each other.

**Validation** runs on the validation split at the end of every epoch (`EVAL_EPOCHS=1`). It scores the metric named by `METRIC_FOR_BEST_MODEL` and keeps the best checkpoints:

| Run | Validation metric used for model selection |
|---|---|
| Module A | `eval_step_id/total_macro_f1`: macro F1 of step recognition. |
| Module B | `eval_temporal/mixed/raw/macro_score`: temporal localization score of the generated spans. |
| Module C | `eval_all/macro_f1`: macro F1 over the execution classes. |
| Joint | `eval_monolith/stage_composite`: geometric mean of the Joint model's A, B and C validation scores (the three metrics above). |

Each run writes to `outputs/<module>/runs/<RUN_NAME>/`:

- `ckpt_ep*`, `best_ep*` and `latest`: full trainer checkpoints.
- `best_checkpoints.json`: the best checkpoints ranked by validation metric.
- validation metrics for each epoch, plus TensorBoard logs.

The released checkpoints are the best epochs of these runs: Module A epoch 6, Module B epoch 4, Module C epoch 4, and Joint epoch 6. All runs use 6 epochs, early-stopping patience 5, learning rate 2e-4 with linear decay and 10 warm-up steps, and global batch 16.

To check a run quickly before a long job, set `DRY_RUN=true`. It builds and prints a few training examples without loading the model.

## 7. Continue from a checkpoint

There are three ways to continue training:

1. **Resume the full training state.** This includes optimizer, scheduler, RNG and epoch counter. Point `RESUME_FROM_CHECKPOINT` at any full checkpoint folder (`latest`, `ckpt_epN`, `best_epN`, or a released checkpoint):

   ```bash
   RESUME_FROM_CHECKPOINT=outputs/module_b/runs/module_b_egoper_27b/latest \
     bash scripts/run.sh module_b configs/env/module_b.env
   ```

   With `EVAL_ON_RESUME=true`, the restored model is validated before training continues.

2. **Automatic resume after preemption.** Every `CLUSTER_SAVE_INTERVAL_SECONDS` (1 hour by default), and on `SIGTERM`, the trainer saves a checkpoint to `CKPT_DIR`. The default is `outputs/<module>/cluster_checkpoints/<RUN_NAME>`, and the trainer also writes a `latest.json` pointer there. When a Slurm job is requeued, training resumes from that pointer automatically. Outside Slurm, resume by hand with `RESUME=<CKPT_DIR>/latest.json`.

3. **Fresh first submission, automatic requeues.** `INITIAL_RESUME_FROM_CHECKPOINT=<checkpoint>` is used only for the first submission of a job. Later Slurm requeues ignore it and resume from `CKPT_DIR/latest.json`.

Module A and the Joint model can also start a new training run from an existing adapter, using the adapter weights only: `LOAD_ADAPTER_FROM=<checkpoint>`.

## 8. Testing on the EgoPER test split

Both tests run the system **causally**, as it would run online:

- The video is polled every 5 s.
- Each clip starts where the system itself placed the end of the previous step.
- Each prompt contains only the system's own earlier outputs.

The ground truth is used only for scoring. Each GPU runs an independent worker that takes videos from a shared queue and saves each finished video. If a run is interrupted, run the same command again and it continues where it stopped. When all videos are done, the per-video outputs are merged into one artifact.

```bash
# Modular: uses checkpoints/module_{a,b,c}
bash scripts/run.sh pipeline_test configs/env/pipeline_test.env
# -> outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json

# Joint: uses checkpoints/abc_monolith/best_ep6
bash scripts/run.sh monolith_test configs/env/monolith_test.env
# -> outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json
```

Run the same command again until `work_queue/progress.json` in the output folder reports `"complete": true`.

**Testing your own checkpoints.** To test checkpoints from your own training runs instead of the released ones, set the checkpoint variables to `auto`. The best checkpoint is then read from each run's `best_checkpoints.json`:

```bash
MODULE_A_CHECKPOINT=auto MODULE_A_RUN_DIR=outputs/module_a/runs/module_a_egoper_27b \
MODULE_B_CHECKPOINT=auto MODULE_B_RUN_DIR=outputs/module_b/runs/module_b_egoper_27b \
MODULE_C_CHECKPOINT=auto MODULE_C_RUN_DIR=outputs/module_c/runs/module_c_egoper_27b \
  bash scripts/run.sh pipeline_test configs/env/pipeline_test.env

CHECKPOINT=auto RUN_DIR=outputs/abc_monolith/runs/abc_monolith_egoper_27b \
  bash scripts/run.sh monolith_test configs/env/monolith_test.env
```

**End-to-end validation.** Add `EVAL_SPLIT=val` to either command to run the whole system on the validation split instead of the test split. Use a different `RUN_NAME` so the test outputs are not overwritten.

## 9. Reproducing all results from the checkpoints

One script runs both tests from the checkpoints and then computes every result table:

```bash
bash scripts/reproduce.sh all
```

It has two stages, which can also be run separately:

- `bash scripts/reproduce.sh test` runs the Modular test and then the Joint test, exactly as in Section 8. A test whose artifact already exists is skipped. If a test stops before all videos are done (time limit or preemption), the script exits with code 3. Run the same command again and it continues where it stopped.
- `bash scripts/reproduce.sh evaluate` reads the two test artifacts and writes to `outputs/results/`:
  - `final_predictions/`: the final per-video segments of both systems and the ground truth;
  - `metrics.json`: system-level metrics;
  - `component_metrics.json`: component-level metrics of Modules A, B and C;
  - `explanations.json`: BERTScore of the explanations and its two controls;
  - `timing.json`: inference time;
  - `tables.md`: all of the above as tables.

  It then compares the new files with the released ones in `results/` and prints every value that differs by more than 0.005 (timing is not compared, since it depends on the hardware).

The script reads the same env files and variables as `scripts/run.sh` (Section 5). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `EGOPER_ROOT` | `data/EgoPER` | Dataset root. |
| `OUTPUT_ROOT` | `outputs` | Test artifacts are written under it; results go to `<OUTPUT_ROOT>/results`. |
| `MODULE_{A,B,C}_CHECKPOINT`, `CHECKPOINT` | `checkpoints/...` | Modular and Joint checkpoints (Section 4). |
| `PIPELINE_RUN_NAME`, `MONOLITH_RUN_NAME` | `RUN_NAME` in each env file | Output folder names of the two tests. |
| `RESULTS_DIR` | `<OUTPUT_ROOT>/results` | Where the evaluate stage writes. |
| `PYTHON_BIN` | `python` | Python interpreter. |
| `BERTSCORE_DEVICE` | `cuda` when available | Device for BERTScore. |
| `SKIP_BERTSCORE` | `false` | Skip the explanation scores (for example, when the scorer model cannot be downloaded). |

The Modular test took about 2 h 15 min on two B200 GPUs. Decoding is greedy, but GPU kernels are not bitwise deterministic, so a rerun on other hardware can differ slightly from the released files.

## 10. Evaluation and metric definitions

`scripts/reproduce.sh evaluate` runs the following commands; each can also be run on its own.

**Final predictions.** This turns the two test artifacts into one file per system, plus the ground truth in the same format:

```bash
PYTHONPATH=src:. python evaluate/egoper/final_predictions.py \
  --pipeline-artifact outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json \
  --monolith-artifact outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json \
  --output-dir outputs/results/final_predictions
```

| File | Contents |
|---|---|
| `pipeline_test.json` | Modular: one segment per final decision, with `start`, `end`, `step_id`, `step`, `classification`, `is_error`, `reasoning`, decision time and clip. |
| `monolith_test.json` | Joint: the same fields, plus the model's answer as emitted. |
| `ground_truth_test.json` | The annotated test segments in the same format. |

A final decision is a step the system placed and classified: for Modular, a step Module C classified, or an off-script step labelled `ADDITION`; for Joint, a completed answer. Every system-level metric is computed from these files only.

**Metrics.**

```bash
PYTHONPATH=src:. python evaluate/egoper/evaluate_final.py \
  --predictions outputs/results/final_predictions --output outputs/results/metrics.json
PYTHONPATH=src:. python evaluate/egoper/component_metrics.py \
  --pipeline-artifact outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json \
  --output outputs/results/component_metrics.json
PYTHONPATH=src:. python evaluate/egoper/explanations.py \
  --predictions outputs/results/final_predictions --output outputs/results/explanations.json
PYTHONPATH=src:. python evaluate/egoper/timing.py \
  --pipeline-artifact outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json \
  --monolith-artifact outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json \
  --output outputs/results/timing.json
PYTHONPATH=src:. python evaluate/egoper/tables.py --results outputs/results --compare results
```

Results are reported in percent for two sets of videos:

- **all 58 test videos** (694 annotated segments, 73 mistakes);
- **the 25 erroneous test videos**, those with at least one annotated mistake (280 segments, 73 mistakes). These are also exactly the test videos that appear in EgoPER's official test split.

System-level metrics (`evaluate_final.py`). Predictions and annotated segments are matched one to one at temporal IoU ≥ 0.5.

| Metric | Definition |
|---|---|
| **Predicted segments** | Number of final segments, in total, with an error class, and per class. |
| **EDA** | Error detection accuracy: an annotated segment is detected as an error when predicted error segments cover more than half of it. EDA is the fraction of annotated segments labelled correctly, averaged over the five recipes. |
| **Binary F1@0.5** | Each segment is labelled *error* (correction, modification, slip or addition) or *normal*. Matches need the same label. Macro F1 over the two labels per recipe, averaged over recipes. |
| **Fine P / R / F1@0.5** | Only mistake segments are scored, and matches need the same class (correction, modification, slip or addition). Counts are pooled over all videos. Per-class precision, recall and F1; the macro values are the unweighted means over the four classes. |
| **TAS F1@0.5** | Segmental step F1: matches need the same step. An `ADDITION` segment accepts an `EXTRA` prediction or the step EgoPER's annotation gives it. F1 per recipe, averaged over recipes. |
| **O-Acc, O-IoU** | EgoPER's omission metrics: a step of the recipe's task graph (`<EGOPER_ROOT>/task_graph.txt`) is omitted when no segment of that step exists in the video. Per video, O-Acc is the fraction of annotated omissions that are predicted and O-IoU the IoU of the predicted and annotated omission sets; both are averaged over videos and then over recipes. |

Component-level metrics (`component_metrics.py`, Modular only). Every model call is scored, including "not completed" answers and calls repeated on a longer clip after Module C judged the step not finished. A poll's target is the first annotated segment that ends inside its clip and is not already covered (IoU ≥ 0.5) by a span the stream accepted at an earlier poll; without one, the target is "no step completed".

| Module | Metric |
|---|---|
| **A** | Step-ID macro F1 and accuracy over all polls; "no step completed" is a class of its own. |
| **B** | A span is a hit at IoU ≥ 0.5 with the poll's target. Precision is over returned spans, recall over calls with a target. Also mean IoU and mean boundary error of the scored spans. |
| **C** | The gold class is the class of the annotated segment the span overlaps most, or `NO_MATCH` below IoU 0.5. A "not completed" answer counts as `NO_MATCH`. Accuracy, per-class F1, macro F1 over the four execution classes, and macro F1 including `NO_MATCH`. |

Each block is reported on every call and on **upstream-correct** calls: for Module B, Module A named the target step; for Module C, also Module B's span has IoU ≥ 0.5 with the target.

Explanations (`explanations.py`). Pairs are final segments matched one to one to an annotated segment of the same step at IoU ≥ 0.5, where the annotation is a mistake and the prediction is any mistake class; pairs with an empty explanation are left out. Raw BERTScore F1 (`microsoft/deberta-xlarge-mnli`, layer 40, no IDF, no baseline rescaling) of the explanation against the annotated description (**Gold**), and two controls: the step instruction scored against the annotated description (**Instruction restatement**), and the explanation scored against another annotated description of the same recipe (**Mismatched reference**; fixed derangement with seed 3407; recipes where one description makes up more than half of the pairs are left out).

Inference time (`timing.py`). Measured generation time of every model call (mean, median, 95th percentile), per final output, and summed per video. The candidate-scoring passes are not included.

## 11. Released results

`results/` holds the outputs of `scripts/reproduce.sh evaluate` for the released checkpoints: `final_predictions/` (both systems and the ground truth), `metrics.json`, `component_metrics.json`, `explanations.json`, `timing.json` and `tables.md`. To recompute the system-level metrics from the released predictions without running any model:

```bash
PYTHONPATH=src:. python evaluate/egoper/evaluate_final.py \
  --predictions results/final_predictions --output /tmp/metrics.json
```

### System level (IoU 0.5)

#### All test videos (58 videos)

| System | EDA | Binary F1 | Fine P | Fine R | Fine F1 | TAS F1 | O-Acc | O-IoU |
|---|---|---|---|---|---|---|---|---|
| Modular | 97.71 | 80.66 | 60.38 | 71.03 | 62.77 | 92.05 | 99.58 | 79.13 |
| Joint | 97.84 | 84.08 | 59.38 | 71.00 | 63.70 | 93.63 | 97.78 | 95.40 |

#### Erroneous videos (25 videos)

| System | EDA | Binary F1 | Fine P | Fine R | Fine F1 | TAS F1 | O-Acc | O-IoU |
|---|---|---|---|---|---|---|---|---|
| Modular | 93.93 | 81.27 | 64.46 | 71.03 | 65.16 | 91.45 | 99.17 | 82.17 |
| Joint | 94.81 | 86.22 | 67.87 | 71.00 | 68.07 | 94.57 | 94.89 | 93.78 |

#### Fine-grained mistakes per class (precision / recall / F1)

All test videos:

| System | Correction | Modification | Slip | Addition |
|---|---|---|---|---|
| Modular | 40.00 / 80.00 / 53.33 | 92.86 / 78.79 / 85.25 | 46.15 / 31.58 / 37.50 | 62.50 / 93.75 / 75.00 |
| Joint | 30.00 / 60.00 / 40.00 | 87.50 / 84.85 / 86.15 | 55.00 / 57.89 / 56.41 | 65.00 / 81.25 / 72.22 |

Erroneous videos:

| System | Correction | Modification | Slip | Addition |
|---|---|---|---|---|
| Modular | 40.00 / 80.00 / 53.33 | 92.86 / 78.79 / 85.25 | 50.00 / 31.58 / 38.71 | 75.00 / 93.75 / 83.33 |
| Joint | 30.00 / 60.00 / 40.00 | 87.50 / 84.85 / 86.15 | 61.11 / 57.89 / 59.46 | 92.86 / 81.25 / 86.67 |

#### Predicted segments

| Scope | Source | Segments | Errors | Correct | Correction | Modification | Slip | Addition |
|---|---|---|---|---|---|---|---|---|
| All test videos | Annotation | 694 | 73 | 621 | 5 | 33 | 19 | 16 |
| All test videos | Modular | 696 | 75 | 621 | 10 | 28 | 13 | 24 |
| All test videos | Joint | 724 | 82 | 642 | 10 | 32 | 20 | 20 |
| Erroneous videos | Annotation | 280 | 73 | 207 | 5 | 33 | 19 | 16 |
| Erroneous videos | Modular | 283 | 70 | 213 | 10 | 28 | 12 | 20 |
| Erroneous videos | Joint | 284 | 74 | 210 | 10 | 32 | 18 | 14 |

### Component level (Modular)

Scored calls, "not completed" answers and repeated calls on a longer clip:

| Scope | Module | Calls | Not completed | Repeated |
|---|---|---|---|---|
| All test videos | A | 2679 | 1808 | 144 |
| All test videos | B | 871 | 53 | 120 |
| All test videos | C | 794 | 122 | 118 |
| Erroneous videos | A | 965 | 632 | 44 |
| Erroneous videos | B | 333 | 11 | 40 |
| Erroneous videos | C | 302 | 39 | 39 |

#### Module A

| Scope | Polls | Step-ID macro-F1 | Accuracy |
|---|---|---|---|
| All test videos | 2679 | 80.27 | 88.88 |
| Erroneous videos | 965 | 83.77 | 90.05 |

#### Module B

| Scope | Calls | N | P@0.5 | R@0.5 | F1@0.5 | Mean IoU | Boundary MAE (s) |
|---|---|---|---|---|---|---|---|
| All test videos | every call | 871 | 78.48 | 92.64 | 84.98 | 81.51 | 2.26 |
| All test videos | upstream correct | 656 | 97.09 | 96.80 | 96.95 | 84.54 | 1.42 |
| Erroneous videos | every call | 333 | 80.12 | 93.48 | 86.29 | 82.14 | 2.10 |
| Erroneous videos | upstream correct | 266 | 96.98 | 96.62 | 96.80 | 84.67 | 1.47 |

#### Module C

| Scope | Calls | N | Accuracy | Fine macro-F1 | Macro-F1 with NO_MATCH | Correct F1 | Correction F1 | Modification F1 | Slip F1 |
|---|---|---|---|---|---|---|---|---|---|
| All test videos | every call | 794 | 77.71 | 66.59 | 56.06 | 87.86 | 57.14 | 83.87 | 37.50 |
| All test videos | upstream correct | 623 | 84.75 | 73.32 | 58.66 | 92.01 | 61.54 | 85.19 | 54.55 |
| Erroneous videos | every call | 302 | 77.48 | 67.11 | 56.92 | 88.74 | 57.14 | 83.87 | 38.71 |
| Erroneous videos | upstream correct | 245 | 81.63 | 72.88 | 58.31 | 90.27 | 61.54 | 85.19 | 54.55 |

### Explanations (raw BERTScore F1)

| System | Pairs | Gold | Instruction restatement | Mismatched reference |
|---|---|---|---|---|
| Modular | 41 | 89.94 | 73.04 | 63.12 |
| Joint | 59 | 94.43 | 68.51 | 64.30 |

### Inference time (seconds)

| System | Measurement | N | Mean | Median | P95 |
|---|---|---|---|---|---|
| Modular | Module A, per call | 2679 | 1.75 | 1.49 | 3.06 |
| Modular | Module B, per call | 871 | 3.09 | 2.78 | 5.17 |
| Modular | Module C, per call | 794 | 3.10 | 2.95 | 4.53 |
| Modular | A + B + C, per final output | 672 | 7.63 | 7.18 | 11.81 |
| Modular | Summed per video | 58 | 169.48 | 150.45 | 349.14 |
| Joint | Per call | 2679 | 3.18 | 2.22 | 6.42 |
| Joint | Per final output | 724 | 5.58 | 5.36 | 7.89 |
| Joint | Summed per video | 58 | 146.90 | 120.55 | 424.46 |

The test set has only 5 correction segments, so the correction scores are noisy.

## 12. Running with or without Slurm

**Without Slurm.** Run `scripts/run.sh` directly. It uses the GPUs listed in `CUDA_VISIBLE_DEVICES` (or `NUM_GPUS` from the env file) and stops cleanly on Ctrl-C or `SIGTERM`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run.sh abc_monolith configs/env/abc_monolith.env
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run.sh monolith_test configs/env/monolith_test.env
```

**With Slurm.** First fill in the partition and account lines in `scripts/slurm/*.sbatch`. If you use a container plugin, add its options there too. Then submit:

```bash
sbatch scripts/slurm/train.sbatch module_a      configs/env/module_a.env
sbatch scripts/slurm/train.sbatch module_b      configs/env/module_b.env
sbatch scripts/slurm/train.sbatch module_c      configs/env/module_c.env
sbatch scripts/slurm/train.sbatch abc_monolith  configs/env/abc_monolith.env
sbatch scripts/slurm/test.sbatch  pipeline_test configs/env/pipeline_test.env
sbatch scripts/slurm/test.sbatch  monolith_test configs/env/monolith_test.env
```

The templates request `--requeue` and `--signal=B:TERM@300`. Five minutes before the time limit, the job saves a checkpoint, and a requeued job resumes from it automatically (Section 7). Test jobs keep every finished video, so resubmitting the same command continues the queue.

`scripts/reproduce.sh` can run inside the test template in the same way: `sbatch scripts/slurm/test.sbatch` runs `scripts/run.sh`, so for the full reproduction replace its last line with `exec bash scripts/reproduce.sh all` and resubmit until the job prints the tables.

## 13. Unit tests

The tests run on CPU and need neither the dataset nor a GPU:

```bash
PYTHONPATH=src:. python -m pytest -q tests
```

`tests/unit/test_released_results.py` recomputes the system-level metrics from the files in `results/final_predictions/` and checks them against `results/metrics.json`. O-Acc and O-IoU are checked only when `<EGOPER_ROOT>/task_graph.txt` is available.
