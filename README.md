# MSDYOLO

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

MSDYOLO is a YOLOv5-OBB based rotated-object detector for aerial imagery. The
supported cloud workflow prepares DOTA v1.5 patches, trains a clean DOTA
teacher, then launches the full MSD distillation stage.

## Quick start (cloud GPU)

```bash
git clone https://github.com/DexZane/MSDYOLO.git
cd MSDYOLO
bash scripts/setup.sh --config configs/train/teacher.yaml
```

The setup script installs the Python package, pins `setuptools==69.5.1` for
Python 3.12 compatibility, downloads DOTA v1.5 through the OpenDataLab SDK,
normalizes its directory layout, validates and atomically builds 1024-pixel
patches (200-pixel gap), downloads `yolov5s.pt`, and starts the requested
configuration in the background.

Useful modes:

```bash
bash scripts/setup.sh --prepare-only                 # prepare data only
bash scripts/setup.sh --force-resplit                # rebuild a stale/current split
bash scripts/setup.sh --foreground                   # keep training in this terminal
bash scripts/setup.sh --config configs/train/degradation.yaml
```

Full MSD training has an explicit two-stage hand-off. `yolov5s.pt` is a COCO
checkpoint: its 80-class detection head cannot be used as a DOTA v1.5 teacher
because the 16-class head is newly initialized. Train the clean teacher first,
then use its checkpoint for the student/distillation stage:

```bash
bash scripts/setup.sh --config configs/train/teacher.yaml
# wait for runs/train/dota_teacher/weights/last.pt
bash scripts/setup.sh --config configs/train/full.yaml
```

The full configuration refuses to start if that DOTA-adapted teacher checkpoint
does not exist. The trainer checkpoints after every epoch, so
`runs/train/dota_teacher/weights/last.pt` remains available if the cloud job is
interrupted.

Background output is written to `training.log`. The exact process id is kept
in `runs/setup/training.pid`; stop only that process with
`kill $(cat runs/setup/training.pid)`. A second setup invocation refuses to
start while the recorded process is alive.

To resume after a stopped cloud job, prepare the same checkout again and start
from the latest student weights explicitly. This restores model weights but
does not restore the optimizer or epoch counter:

```bash
bash scripts/setup.sh --prepare-only
python -m msdyolo.train \
  --config configs/train/full.yaml \
  --weights runs/train/msdyolofull/weights/last.pt \
  --device 0
```

The teacher checkpoint remains `runs/train/dota_teacher/weights/last.pt` and
is required by the full configuration.

## Dataset and label contract

Raw data is downloaded to `dataset/DOTA`. The preparer produces
`dataset/DOTA/split/{train,val}/{images,labelTxt}` and records the source
snapshot in `.msdyolo-split.json`. DOTA v1.5 validation images are official
unlabelled data, so an empty `val/labelTxt` directory is valid; training labels
must be present.

Every DOTA label line is:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class difficult
```

Coordinates are pixel coordinates in the patch (`0..1024`), not normalized
`[0,1]` values. The splitter clips polygon vertices to patch boundaries and
rejects malformed or degenerate objects. A prepared split is reused only when
its source snapshot, geometry, and validation marker still match.

## Training and diagnostics

The five current experiment configurations are:

| Configuration | Purpose |
| --- | --- |
| `configs/train/baseline.yaml` | small CPU/fixture smoke test; distillation is off |
| `configs/train/degradation.yaml` | degradation branch |
| `configs/train/clearbranch.yaml` | degradation plus clear branch |
| `configs/train/teacher.yaml` | clean DOTA v1.5 baseline; produces the frozen teacher |
| `configs/train/full.yaml` | degradation, clear branch, distillation from the teacher checkpoint |

For a direct run after data preparation:

```bash
python -m msdyolo.train --config configs/train/full.yaml --device 0
```

Cloud configs use four dataloader workers to avoid the multiprocess deadlock
seen with eight workers. In a real full-mode run, the first epoch is healthy
when its log contains `match > 0`. A zero-match teacher-stage smoke test is
intentional because distillation is disabled there; it is not a valid
full-training acceptance signal. Matching makes at most three CPU snapshots
per batch, instead of synchronizing CUDA once per candidate/target pair.

For a deterministic local CPU check:

```bash
python -m msdyolo.train \
  --config configs/train/baseline.yaml \
  --data tests/fixtures/dota.yaml \
  --cfg configs/models/yolov5n.yaml \
  --hyp msdyolo/data/hyps/obb/hyp.finetune_dota.yaml \
  --weights "" --device cpu --batch-size 1 --img-size 320 --single-batch
```

## Installed commands

After `pip install -e .` (or after `setup.sh`), the package exposes:

```bash
msdyolo-train --help
msdyolo-val --help
msdyolo-detect --help
msdyolo-export --help
```

The root `train.py`, `val.py`, `detect.py`, and `export.py` files remain thin
compatibility wrappers; implementation imports live under `msdyolo/`.

## Project layout

```text
configs/models/       YOLOv5-OBB model definitions
configs/train/        baseline, ablation, teacher, and full configurations
configs/README.md     configuration ownership and path conventions
data/README.md        local raw/processed data policy (artifacts ignored)
notebooks/README.md   exploratory-analysis policy
msdyolo/data/          DOTA YAML, hyperparameters, and preparation tools
msdyolo/models/        model implementation
msdyolo/utils/         training, data, loss, and rotated-NMS utilities
scripts/setup.sh       cloud preparation and launch orchestrator
tests/fixtures/        small synthetic DOTA data for local verification
docs/archive/          historical upstream/restructure documents
```

Cloud-only runtime data is deliberately kept outside the source tree tracked
by Git:

```text
dataset/DOTA/           raw DOTA files and prepared patches
runs/setup/             launch lock, PID, and setup state
runs/train/             checkpoints and trainer outputs
training.log            background setup log
yolov5s.pt              downloaded pretrained checkpoint
```

These paths are ignored by `.gitignore`; a fresh clone therefore contains only
reproducible source, configuration, tests, and documentation.

## Requirements

- Python 3.12 is supported; Python 3.8+ is accepted by the package metadata.
- PyTorch 2.5 with CUDA 12.4 works for the cloud setup; install the matching
  PyTorch build for the host GPU.
- A V100/A100-class GPU is recommended for full DOTA training. CPU execution
  is intended for the single-batch smoke test only.

See [docs/install.md](docs/install.md) for package installation and the
optional rotated-NMS extension. [docs/GetStart.md](docs/GetStart.md) contains
the short operational checklist.

## Citation

The repository citation metadata is in [`CITATION.cff`](CITATION.cff), which
GitHub can use to generate a citation in its standard formats.

## License

MSDYOLO is distributed under the [GNU General Public License v3](LICENSE).
