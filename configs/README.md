# Configuration guide

The repository keeps configuration separate from implementation code. Paths
are intentionally stable because the training entry point and cloud scripts
refer to them directly.

| Directory | Contents |
| --- | --- |
| `configs/models/` | YOLOv5-OBB model definitions (`yolov5n` through `yolov5x`) |
| `configs/train/` | Baseline, degradation, clear-branch, teacher, and full experiments |

The DOTA dataset schema lives in [`msdyolo/data/dota.yaml`](../msdyolo/data/dota.yaml),
while reusable hyperparameters live under `msdyolo/data/hyps/`. Keep dataset
locations, model architecture, and training policy in YAML rather than
hard-coding them in Python.
