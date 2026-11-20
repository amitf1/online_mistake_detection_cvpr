## System level (IoU 0.5)

### All test videos (58 videos)

| System | EDA | Binary F1 | Fine P | Fine R | Fine F1 | TAS F1 | O-Acc | O-IoU |
|---|---|---|---|---|---|---|---|---|
| Modular | 97.71 | 80.66 | 60.38 | 71.03 | 62.77 | 92.05 | 99.58 | 79.13 |
| Joint | 97.84 | 84.08 | 59.38 | 71.00 | 63.70 | 93.63 | 97.78 | 95.40 |

### Erroneous videos (25 videos)

| System | EDA | Binary F1 | Fine P | Fine R | Fine F1 | TAS F1 | O-Acc | O-IoU |
|---|---|---|---|---|---|---|---|---|
| Modular | 93.93 | 81.27 | 64.46 | 71.03 | 65.16 | 91.45 | 99.17 | 82.17 |
| Joint | 94.81 | 86.22 | 67.87 | 71.00 | 68.07 | 94.57 | 94.89 | 93.78 |

### Fine-grained mistakes per class (precision / recall / F1)

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

### Predicted segments

| Scope | Source | Segments | Errors | Correct | Correction | Modification | Slip | Addition |
|---|---|---|---|---|---|---|---|---|
| All test videos | Annotation | 694 | 73 | 621 | 5 | 33 | 19 | 16 |
| All test videos | Modular | 696 | 75 | 621 | 10 | 28 | 13 | 24 |
| All test videos | Joint | 724 | 82 | 642 | 10 | 32 | 20 | 20 |
| Erroneous videos | Annotation | 280 | 73 | 207 | 5 | 33 | 19 | 16 |
| Erroneous videos | Modular | 283 | 70 | 213 | 10 | 28 | 12 | 20 |
| Erroneous videos | Joint | 284 | 74 | 210 | 10 | 32 | 18 | 14 |

## Component level (Modular)

Scored calls, "not completed" answers and repeated calls on a longer clip:

| Scope | Module | Calls | Not completed | Repeated |
|---|---|---|---|---|
| All test videos | A | 2679 | 1808 | 144 |
| All test videos | B | 871 | 53 | 120 |
| All test videos | C | 794 | 122 | 118 |
| Erroneous videos | A | 965 | 632 | 44 |
| Erroneous videos | B | 333 | 11 | 40 |
| Erroneous videos | C | 302 | 39 | 39 |

### Module A

| Scope | Polls | Step-ID macro-F1 | Accuracy |
|---|---|---|---|
| All test videos | 2679 | 80.27 | 88.88 |
| Erroneous videos | 965 | 83.77 | 90.05 |

### Module B

| Scope | Calls | N | P@0.5 | R@0.5 | F1@0.5 | Mean IoU | Boundary MAE (s) |
|---|---|---|---|---|---|---|---|
| All test videos | every call | 871 | 78.48 | 92.64 | 84.98 | 81.51 | 2.26 |
| All test videos | upstream correct | 656 | 97.09 | 96.80 | 96.95 | 84.54 | 1.42 |
| Erroneous videos | every call | 333 | 80.12 | 93.48 | 86.29 | 82.14 | 2.10 |
| Erroneous videos | upstream correct | 266 | 96.98 | 96.62 | 96.80 | 84.67 | 1.47 |

### Module C

| Scope | Calls | N | Accuracy | Fine macro-F1 | Macro-F1 with NO_MATCH | Correct F1 | Correction F1 | Modification F1 | Slip F1 |
|---|---|---|---|---|---|---|---|---|---|
| All test videos | every call | 794 | 77.71 | 66.59 | 56.06 | 87.86 | 57.14 | 83.87 | 37.50 |
| All test videos | upstream correct | 623 | 84.75 | 73.32 | 58.66 | 92.01 | 61.54 | 85.19 | 54.55 |
| Erroneous videos | every call | 302 | 77.48 | 67.11 | 56.92 | 88.74 | 57.14 | 83.87 | 38.71 |
| Erroneous videos | upstream correct | 245 | 81.63 | 72.88 | 58.31 | 90.27 | 61.54 | 85.19 | 54.55 |

## Explanations (raw BERTScore F1)

| System | Pairs | Gold | Instruction restatement | Mismatched reference |
|---|---|---|---|---|
| Modular | 41 | 89.94 | 73.04 | 63.12 |
| Joint | 59 | 94.43 | 68.51 | 64.30 |

## Inference time (seconds)

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

