# Getting started

The repository has one supported implementation tree (`msdyolo/`) and one
cloud entry point (`scripts/setup.sh`). For the complete walkthrough, start
with the [README](../README.md).

## Cloud preparation and training

From a fresh machine:

```bash
git clone https://github.com/DexZane/MSDYOLO.git
cd MSDYOLO
bash scripts/setup.sh
```

The default is background training with `configs/train/full.yaml`. Use
`--prepare-only` to stop after dataset preparation, `--force-resplit` to
rebuild patches, `--foreground` to keep the trainer attached to the terminal,
or `--config PATH` to select one of the five files under `configs/train/`.

Monitor a background run with:

```bash
tail -f training.log
cat runs/setup/training.pid
kill "$(cat runs/setup/training.pid)"  # only the recorded training process
```

The setup lock and PID checks make repeated invocations fail safely while a
job is running. It never uses a global process-name kill.

The cloud workspace keeps generated files out of the Git checkout: raw and
split data live under `dataset/DOTA`, launch state under `runs/setup`, trainer
outputs under `runs/train`, and the downloaded `yolov5s.pt` checkpoint at the
project root. These paths are ignored and can be removed or archived
independently of source code.

## Data checks

The raw dataset lives at `dataset/DOTA`; patches are written to
`dataset/DOTA/split`. Train labels use DOTA polygon lines with pixel
coordinates in `0..1024` after clipping. They must not be normalized to
`[0,1]`. DOTA v1.5 validation images are officially unlabelled, so an empty
`val/labelTxt` is expected and accepted.

To inspect the preparation without starting training:

```bash
bash scripts/setup.sh --prepare-only
```

## Local CPU smoke test

The checked-in fixture is deliberately small and is not a cloud benchmark:

```bash
python -m msdyolo.train \
  --config configs/train/baseline.yaml \
  --data tests/fixtures/dota.yaml \
  --cfg configs/models/yolov5n.yaml \
  --hyp msdyolo/data/hyps/obb/hyp.finetune_dota.yaml \
  --weights "" --device cpu --batch-size 1 --img-size 320 --single-batch
```

Baseline may report `match=0` because distillation is disabled. For the cloud
full configuration, the first epoch must report `match > 0`; otherwise stop
and inspect the prepared pixel-coordinate labels before continuing.

## Validation and inference

Use the canonical modules (or their installed console-script equivalents):

```bash
python -m msdyolo.val --help
python -m msdyolo.detect --help
python -m msdyolo.export --help
```

The root command files are compatibility wrappers only. Generated runs are
kept below `runs/` and are not part of the source dataset.
