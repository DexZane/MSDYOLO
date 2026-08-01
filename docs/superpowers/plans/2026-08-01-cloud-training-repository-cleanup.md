# MSDYOLO Cloud Training Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bash scripts/setup.sh` reliably prepare DOTA v1.5 and launch full MSDYOLO training while converging the repository on one importable `msdyolo` implementation.

**Architecture:** Put DOTA label rules in a small domain module, keep splitting/downloading/preparation as separate command modules, and make `setup.sh` a thin orchestrator. Migrate all Python callers to `msdyolo.*`, retain root CLI wrappers for compatibility, then remove duplicate root packages and obsolete scripts only after regression tests pass.

**Tech Stack:** Python 3.12, PyTorch 2.5, OpenCV, NumPy, PyYAML, pytest, Bash, OpenDataLab Python SDK.

## Global Constraints

- DOTA v1.5 training labels use 10 columns: eight pixel coordinates, class name, and difficult flag.
- Split coordinates remain in `[0, 1024]`; they are never normalized to `[0, 1]`.
- The official unlabelled DOTA v1.5 val split is valid; train labels must be present and non-empty.
- The cloud training dataloader uses exactly 4 workers by default.
- The one-command workflow remains `bash scripts/setup.sh` and defaults to background full-MSD training.
- Baseline mode intentionally reports `match=0`; `match > 0` is a full-MSD cloud acceptance criterion.
- Do not modify, delete, stage, or commit the user's untracked `scripts/verify.sh`.
- Do not redesign the model, distillation algorithm, or DOTA evaluation protocol.

---

### Task 1: Establish the DOTA label contract

**Files:**
- Create: `msdyolo/data/dota.py`
- Modify: `msdyolo/data/scripts/split_dota.py`
- Create: `tests/checkdotalabels.py`

**Interfaces:**
- Produces: `DOTA15_CLASSES: tuple[str, ...]`
- Produces: `DotaObject` with `coordinates`, `classname`, and `difficult`
- Produces: `parse_dota_label(path: Path) -> list[DotaObject]`
- Produces: `clip_object_to_patch(obj, xstart, ystart, subsize) -> DotaObject`
- Produces: `format_dota_object(obj) -> str`
- Preserves: `split_single_image(args) -> int` and `split_dataset(...) -> int`

- [ ] **Step 1: Write failing parser and validation tests**

Add tests using the repository's `check*` pytest naming convention:

```python
from pathlib import Path

import pytest

from msdyolo.data.dota import parse_dota_label


class CheckDotaLabels:
    def checkheaderandheaderlesslabelsparse(self, tmp_path: Path):
        headed = tmp_path / "headed.txt"
        headed.write_text(
            "imagesource:GoogleEarth\ngsd:0.5\n"
            "100 200 300 200 300 400 100 400 ship 0\n",
            encoding="utf-8",
        )
        plain = tmp_path / "plain.txt"
        plain.write_text(
            "100 200 300 200 300 400 100 400 ship 0\n",
            encoding="utf-8",
        )
        assert parse_dota_label(headed) == parse_dota_label(plain)
        assert parse_dota_label(plain)[0].coordinates == (
            100.0, 200.0, 300.0, 200.0, 300.0, 400.0, 100.0, 400.0
        )

    @pytest.mark.parametrize(
        "line, message",
        [
            ("0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 ship 0", "normalized"),
            ("nan 2 3 4 5 6 7 8 ship 0", "finite"),
            ("1 2 3 4 5 6 7 8 unknown 0", "unknown class"),
            ("1 2 3 4 5 6 7 8 ship 9", "difficult"),
        ],
    )
    def checkinvalidlabelsfail(self, tmp_path: Path, line: str, message: str):
        label = tmp_path / "bad.txt"
        label.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            parse_dota_label(label)
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run: `pytest -q tests/checkdotalabels.py`

Expected: collection fails because `msdyolo.data.dota` does not exist.

- [ ] **Step 3: Implement the domain model and strict parser**

Create a frozen dataclass and content-aware parser. Metadata lines are ignored only when they cannot be label records and begin with `imagesource:` or `gsd:`. Raise `ValueError` with the file and line number for malformed target-like lines. Validate finite coordinates, known class, difficult in `{0, 1, 2}`, and reject records whose entire coordinate range is within `[0, 1]` with a `normalized coordinates` message.

```python
@dataclass(frozen=True)
class DotaObject:
    coordinates: tuple[float, float, float, float, float, float, float, float]
    classname: str
    difficult: int


def format_dota_object(obj: DotaObject) -> str:
    coordinates = " ".join(f"{value:.1f}" for value in obj.coordinates)
    return f"{coordinates} {obj.classname} {obj.difficult}"
```

- [ ] **Step 4: Write failing split geometry tests**

Extend `tests/checkdotalabels.py` to create a 1200×1200 synthetic PNG and one target crossing the 1024 boundary. Assert `split_single_image()` writes `labelTxt/image_0_0.txt`, does not create `labels/`, and all coordinates are between 0 and 1024 with at least one coordinate greater than 1. Add parameter tests asserting `split_dataset()` rejects `gap >= subsize`, negative gap, and `num_process <= 0`.

- [ ] **Step 5: Run the geometry tests and verify RED**

Run: `pytest -q tests/checkdotalabels.py`

Expected: failures show the current output directory is `labels/` and invalid split arguments are accepted.

- [ ] **Step 6: Update the splitter minimally**

Import the new domain functions, replace dictionary objects with `DotaObject`, output directly to `labelTxt/`, and validate arguments before collecting images:

```python
def validate_split_arguments(subsize: int, gap: int, num_process: int) -> None:
    if subsize <= 0:
        raise ValueError("subsize must be positive")
    if gap < 0 or gap >= subsize:
        raise ValueError("gap must satisfy 0 <= gap < subsize")
    if num_process <= 0:
        raise ValueError("num_process must be positive")
```

Use `clip_object_to_patch()` before `format_dota_object()` and reject a clipped polygon when all four points are identical. Keep the center-in-patch assignment rule unchanged.

- [ ] **Step 7: Verify GREEN and regression safety**

Run: `pytest -q tests/checkdotalabels.py`

Expected: all label and geometry tests pass.

Run: `python -m msdyolo.data.scripts.split_dota --help`

Expected: exit 0 and the three split parameters are documented.

- [ ] **Step 8: Commit Task 1**

```bash
git add msdyolo/data/dota.py msdyolo/data/scripts/split_dota.py tests/checkdotalabels.py
git commit -m "fix: enforce pixel DOTA label contract"
```

---

### Task 2: Fix the downloader and normalize OpenDataLab layout

**Files:**
- Modify: `msdyolo/data/scripts/download_dota.py`
- Create: `tests/checkdotadownload.py`

**Interfaces:**
- Produces: `verify_dataset(dataset_dir: Path) -> DatasetStatus`
- Produces: `normalize_download_layout(dataset_dir: Path) -> DatasetStatus`
- Preserves: `download_dota_sdk(target_dir, download_fn=None) -> bool`
- Consumes: `DOTA15_CLASSES` only for later split validation, not download success.

- [ ] **Step 1: Write failing directory-contract tests**

Create temporary `train/images`, nested `train/labelTxt/DOTA-v1.5_train`, and `val/images` trees. Assert normalization moves label files into `train/labelTxt`, removes the empty nested directory, creates empty `val/labelTxt`, and reports ready. Add a case with missing train labels that reports not ready. Add a download test with an injected fake SDK callable and assert it receives exactly `dataset_repo`, `source_path`, and `target_path`, with no `timeout`.

```python
def fakedownload(**kwargs):
    captured.update(kwargs)


assert captured == {
    "dataset_repo": "OpenDataLab/DOTA_V1_dot_5",
    "source_path": "",
    "target_path": str(target),
}
```

- [ ] **Step 2: Run the downloader tests and verify RED**

Run: `pytest -q tests/checkdotadownload.py`

Expected: import failures for `normalize_download_layout` and failure because empty val labels are currently treated as incomplete.

- [ ] **Step 3: Implement explicit download status**

Use a frozen `DatasetStatus` dataclass containing `ready`, `trainimages`, `trainlabels`, `valimages`, and `errors`. The readiness rule is:

```python
ready = trainimages > 0 and trainlabels > 0 and valimages > 0
```

Do not require a non-empty val label directory. Do not run pip from this module. If `openxlab` is unavailable, raise a message directing the caller to install the cloud extra. Keep dependency injection internal to tests by resolving the SDK only when `download_fn is None`.

- [ ] **Step 4: Implement safe nested-layout normalization**

Move only `.txt` files from recognized `DOTA-v1.5_train` and `DOTA-v1.5_val` child directories. Refuse to overwrite a different existing label with the same name. Remove only an empty recognized child directory. Create `val/labelTxt` with `parents=True, exist_ok=True`, then return `verify_dataset()`.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/checkdotadownload.py`

Expected: all downloader and layout tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add msdyolo/data/scripts/download_dota.py tests/checkdotadownload.py
git commit -m "fix: accept unlabelled DOTA validation data"
```

---

### Task 3: Add validated, atomic, idempotent dataset preparation

**Files:**
- Create: `msdyolo/data/scripts/prepare_dota.py`
- Create: `tests/checkdotaprepare.py`

**Interfaces:**
- Produces: `source_snapshot(dataset_dir: Path) -> dict[str, object]`
- Produces: `validate_split_tree(split_dir: Path, subsize: int) -> SplitStatus`
- Produces: `prepare_dataset(dataset_dir, split_dir, subsize, gap, num_process, force=False, splitter=split_dataset) -> SplitStatus`
- Produces CLI arguments: `--dataset`, `--output`, `--subsize`, `--gap`, `--num-process`, `--force-resplit`.
- Consumes: `normalize_download_layout()`, `split_dataset()`, and the Task 1 label parser.

- [ ] **Step 1: Write failing completion-marker tests**

Use a fake splitter that writes one image and one valid `labelTxt` file per split. Assert the first prepare call invokes it and writes `.msdyolo-split.json`; a second call with unchanged sources does not invoke it; changing `gap`, touching a source label, removing an output image, or setting `force=True` invokes it again.

```python
calls = []


def fakesplit(image_dir, label_dir, output_dir, subsize, gap, num_process):
    calls.append((Path(image_dir), Path(label_dir), Path(output_dir), subsize, gap, num_process))
    write_valid_split(Path(output_dir))
    return 1
```

- [ ] **Step 2: Write a failing rollback test**

Prepare a valid existing split, then use a failing splitter that writes an invalid normalized label and raises. Assert `prepare_dataset()` raises and the previous split image, label, and completion marker remain byte-for-byte unchanged.

- [ ] **Step 3: Run preparation tests and verify RED**

Run: `pytest -q tests/checkdotaprepare.py`

Expected: collection fails because `prepare_dota.py` does not exist.

- [ ] **Step 4: Implement split validation and state**

`validate_split_tree()` must require non-empty `train/images`, non-empty `train/labelTxt`, and existing `val/images` when the source val exists. Parse every train label; require at least one object across the tree and `max(coordinates) > 1`. State JSON contains `format_version: "dota-pixel-v1"`, source relative paths with size and `mtime_ns`, `subsize`, `gap`, and source counts. Serialize with sorted keys for deterministic tests.

- [ ] **Step 5: Implement atomic preparation**

Create the candidate with `tempfile.mkdtemp(prefix=".split-candidate-", dir=split_dir.parent)`. Split train and val into the candidate, validate it, write its marker, then exchange trees on the same filesystem:

```python
backup = split_dir.with_name(f".{split_dir.name}-backup")
if split_dir.exists():
    split_dir.rename(backup)
try:
    candidate.rename(split_dir)
except Exception:
    if backup.exists():
        backup.rename(split_dir)
    raise
else:
    if backup.exists():
        shutil.rmtree(backup)
```

Clean a failed candidate in `finally`. Refuse to begin if a stale backup exists and explain how to recover it; do not guess which copy is authoritative.

- [ ] **Step 6: Implement the CLI**

The CLI calls `prepare_dataset()` and prints source/split counts plus whether the result was reused or rebuilt. It returns nonzero for invalid raw data, bad labels, stale backup, or split failure. It never downloads data and never starts training.

- [ ] **Step 7: Verify GREEN and end-to-end synthetic preparation**

Run: `pytest -q tests/checkdotaprepare.py tests/checkdotalabels.py tests/checkdotadownload.py`

Expected: all data-pipeline tests pass.

Run the module against a pytest-created or `mktemp -d` synthetic DOTA tree with `--num-process 1`, then run it again.

Expected: first output says rebuilt, second output says reused, and both exit 0.

- [ ] **Step 8: Commit Task 3**

```bash
git add msdyolo/data/scripts/prepare_dota.py tests/checkdotaprepare.py
git commit -m "feat: add atomic DOTA dataset preparation"
```

---

### Task 4: Consolidate training configurations and health semantics

**Files:**
- Modify: `configs/train/baseline.yaml`
- Create: `configs/train/degradation.yaml`
- Create: `configs/train/clearbranch.yaml`
- Create: `configs/train/full.yaml`
- Delete after migration: `configs/train/msdyolo-baseline-p2.yaml`
- Delete after migration: `configs/train/msdyolo-baseline.yaml`
- Delete after migration: `configs/train/msdyolo-clearbranch.yaml`
- Delete after migration: `configs/train/msdyolo-degradation.yaml`
- Delete after migration: `configs/train/msdyolo-full.yaml`
- Modify: `msdyolo/train.py`
- Create: `tests/checktraininghealth.py`
- Modify: `tests/checkall.py`
- Modify: `tests/checkconfig.py`

**Interfaces:**
- Produces exactly four current experiment configs in `configs/train/`.
- Produces: `training_health_message(distillation_enabled: bool, epoch_matches: int, target_count: int) -> str | None`
- Default cloud full config uses `msdyolo/data/dota.yaml`, `configs/models/yolov5s.yaml`, `yolov5s.pt`, 200 epochs, batch 16, image 1024, device `0`, workers 4.

- [ ] **Step 1: Write failing configuration tests**

Assert the config filenames are exactly `baseline.yaml`, `degradation.yaml`, `clearbranch.yaml`, and `full.yaml`. Assert `full.yaml` enables all three MSD components and has workers 4. Assert baseline disables them and `training_health_message(False, 0, 10)` returns `None`; assert full mode with targets and zero matches returns a warning containing `targets=10` and `match=0`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/checktraininghealth.py tests/checkall.py tests/checkconfig.py`

Expected: failures identify obsolete filenames and the missing health helper.

- [ ] **Step 3: Create the four canonical configs**

Use continuous lowercase YAML keys already required by the project tests. Keep baseline/degradation/clearbranch/full behavior distinct. All cloud-capable configs reference canonical package data and config paths; full uses the exact default values in the interface block.

- [ ] **Step 4: Add epoch health accounting**

Accumulate `epochtargetcount += len(targets)` and `epochmatchcount += lastresult["matchcount"]`. At epoch end call the pure helper. Print a warning only when distillation is enabled, targets were present, and matches stayed zero. Baseline must remain silent. Preserve existing per-batch logging fields.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/checktraininghealth.py tests/checkall.py tests/checkconfig.py`

Expected: all selected tests pass and exactly four train configs are discovered.

- [ ] **Step 6: Commit Task 4**

```bash
git add configs/train msdyolo/train.py tests/checktraininghealth.py tests/checkall.py tests/checkconfig.py
git commit -m "fix: align cloud config with full MSD training"
```

---

### Task 5: Make setup a safe, testable orchestrator

**Files:**
- Modify: `scripts/setup.sh`
- Create: `tests/checksetup.py`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `scripts/setup.sh [--prepare-only] [--force-resplit] [--foreground] [--config PATH]`
- Runtime files: `runs/setup/training.pid` and `training.log`.
- Consumes: `download_dota`, `prepare_dota`, `configs/train/full.yaml`, and `python -m msdyolo.train`.

- [ ] **Step 1: Write failing setup contract tests**

Run `bash scripts/setup.sh --help` in a subprocess and assert exit 0 with all four flags. Read the script and assert it contains `python3 -m pip` or a computed Python command followed by `-m pip`, invokes both package data modules, defaults to `configs/train/full.yaml`, and does not contain `pkill`. Run `bash -n scripts/setup.sh` from the test and require exit 0.

- [ ] **Step 2: Run setup tests and verify RED**

Run: `pytest -q tests/checksetup.py`

Expected: failures show missing flags, baseline default, and the global `pkill`.

- [ ] **Step 3: Implement strict argument parsing and installation order**

Use `set -euo pipefail`. Resolve `PROJECT_DIR` once and use `PYTHON_BIN="${PYTHON_BIN:-python3}"`. Install in this order:

```bash
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"
"$PYTHON_BIN" -m pip install -q -e .
"$PYTHON_BIN" -m pip install -q openxlab
```

Unknown flags and missing `--config` values exit 2. `--help` performs no installation.

- [ ] **Step 4: Replace inline data mutation with module calls**

Call the downloader only when `train/images`, `train/labelTxt`, or `val/images` is missing. Always call the prepare module so it can validate state. Forward `--force-resplit` only when selected. Do not use `mv ... || true`, `sed -i`, or label symlink creation.

- [ ] **Step 5: Implement PID-safe training launch**

Before launch, if the PID file contains a numeric PID and `kill -0 "$pid"` succeeds, exit 1 with the PID and log path. Remove only a stale PID file. In foreground mode use `exec "$PYTHON_BIN" -m msdyolo.train --config "$CONFIG"`; in background mode redirect stdout/stderr to `training.log`, write `$!` atomically to the PID file, and print monitoring and stop commands. Never kill an existing process.

- [ ] **Step 6: Verify GREEN**

Run: `pytest -q tests/checksetup.py`

Expected: all setup contract tests pass.

Run: `bash -n scripts/setup.sh`

Expected: exit 0 with no output.

- [ ] **Step 7: Commit Task 5**

```bash
git add scripts/setup.sh tests/checksetup.py .gitignore
git commit -m "fix: make cloud setup safe and idempotent"
```

---

### Task 6: Converge imports on the `msdyolo` package

**Files:**
- Create from migrated implementation: `msdyolo/val.py`
- Modify: `msdyolo/models/*.py`
- Modify: `msdyolo/utils/**/*.py`
- Modify: `msdyolo/detect.py`
- Modify: `msdyolo/export.py`
- Replace: `train.py`
- Replace: `val.py`
- Replace: `detect.py`
- Replace: `export.py`
- Modify: `tests/checkall.py`
- Modify: `tests/checkbaseline.py`
- Modify: `tests/checkconfig.py`
- Modify: `tests/checknaming.py`
- Modify: `tests/checkp0.py`
- Modify: `tests/checkp0a1.py`
- Modify: `tests/checkp0a2.py`
- Modify: `tests/checkrotatediou.py`
- Create: `tests/checkcanonicalimports.py`
- Delete after GREEN: `models/`
- Delete after GREEN: `utils/`

**Interfaces:**
- Root wrapper contract: each root CLI delegates to `msdyolo.<command>`.
- Package command contract: `python -m msdyolo.train|val|detect|export --help` exits 0.
- All project-owned Python imports resolve through `msdyolo.models` or `msdyolo.utils`.

- [ ] **Step 1: Write failing canonical import tests**

Assert all eight module and root CLI help commands exit 0. Walk project-owned Python files outside `docs/archive` and fail on `from models`, `from utils`, `import models`, or `import utils`. Assert root `models` and `utils` paths do not exist. These assertions intentionally fail before migration.

- [ ] **Step 2: Run canonical tests and verify RED**

Run: `pytest -q tests/checkcanonicalimports.py`

Expected: failures list root imports and duplicate directories.

- [ ] **Step 3: Migrate validation and package-relative imports**

Copy the current val implementation into `msdyolo/val.py`, change imports to `msdyolo.*`, and change repository-relative defaults to canonical config paths. Update any remaining package file that computes the wrong root or refers to a removed root path. Avoid `sys.path` mutation when ordinary absolute package imports work.

- [ ] **Step 4: Replace root commands with compatibility wrappers**

Use this exact shape for train:

```python
#!/usr/bin/env python3
from msdyolo.train import main


if __name__ == "__main__":
    main()
```

For val/detect/export, import `parse_opt` and `main`, then call `main(parse_opt())`. Keep no model or utility implementation in root wrappers.

- [ ] **Step 5: Migrate tests and fixtures to package imports**

Replace all root imports with `msdyolo.*`. Update model/hyp/config paths to `configs/models/` and `msdyolo/data/`. Adjust naming tests to inspect package-owned Python files instead of the deleted root utils tree.

- [ ] **Step 6: Run the focused suite before deletion**

Run: `pytest -q tests/checkbaseline.py tests/checkconfig.py tests/checkp0.py tests/checkp0a1.py tests/checkp0a2.py tests/checkrotatediou.py`

Expected: all focused behavioral tests pass while both trees still exist.

- [ ] **Step 7: Delete the duplicate packages and verify GREEN**

Delete tracked `models/` and `utils/` only after Step 6 passes. Then run:

`pytest -q tests/checkcanonicalimports.py tests/checkbaseline.py tests/checkp0.py tests/checkp0a1.py tests/checkp0a2.py tests/checkrotatediou.py`

Expected: all tests pass without fallback to root imports.

- [ ] **Step 8: Commit Task 6**

```bash
git add msdyolo train.py val.py detect.py export.py tests models utils
git commit -m "refactor: use one canonical msdyolo package"
```

---

### Task 7: Remove obsolete configuration and migration paths

**Files:**
- Move: `.old_structure/trainmsd.py` to `docs/archive/code/trainmsd.py`
- Move: `README_old.md` to `docs/archive/README-pre-restructure.md`
- Move: `README_original.md` to `docs/archive/README-upstream.md`
- Move: `RESTRUCTURE_PLAN.md` to `docs/archive/RESTRUCTURE_PLAN.md`
- Move: `data/dota-test/` to `tests/fixtures/dota/`
- Move: `msdyolo/data/dota-test.yaml` to `tests/fixtures/dota.yaml`
- Move: `data/examples/` to `msdyolo/data/examples/`
- Delete duplicate: root `configs/*.yaml`
- Delete duplicate: root `data/*.yaml` and `data/hyps/`
- Delete obsolete: `scripts/cleanup.sh`
- Delete obsolete: `scripts/download_dota.py`
- Delete obsolete: `scripts/fix_imports.py`
- Delete obsolete: `scripts/fixpy312.sh`
- Delete obsolete: `scripts/restructure.sh`
- Delete obsolete: `scripts/setup_and_train.sh`
- Delete obsolete: `scripts/setupcloud.sh`
- Delete obsolete: `scripts/train_baseline_p2.sh`
- Modify: `scripts/ddp_train.sh`
- Modify: `tests/checkall.py`
- Modify: `tests/checknaming.py`
- Create: `tests/checkrepositorylayout.py`

**Interfaces:**
- Exactly one current setup, downloader, splitter, package tree, dataset YAML set, and training config set remain.
- Historical material remains readable under `docs/archive/` and is excluded from current-path tests.

- [ ] **Step 1: Write failing layout assertions**

Assert obsolete scripts and duplicate YAML paths do not exist, archive files do exist, and the fixture data config resolves to `tests/fixtures/dota`. Assert `scripts/verify.sh` is not mentioned in the deletion list and, if present, remains readable.

- [ ] **Step 2: Run layout tests and verify RED**

Run: `pytest -q tests/checkrepositorylayout.py`

Expected: failures enumerate the legacy paths still present.

- [ ] **Step 3: Move historical and fixture files**

Use `git mv` for tracked history files and fixtures. Remove the tracked `data/dota-test/labelTxt.cache`; caches are generated artifacts. Update test fixture YAML and every test reference before removing root data.

- [ ] **Step 4: Delete only confirmed duplicate or obsolete tracked files**

Compare each duplicate YAML before deletion. Preserve unique current data under `msdyolo/data/`. Do not touch the ignored `dataset/` download tree or untracked `scripts/verify.sh`.

- [ ] **Step 5: Update DDP and naming checks**

Change DDP commands to package entry points and canonical config/data paths. Make naming checks scan current source/config/test directories while excluding `docs/archive`.

- [ ] **Step 6: Verify GREEN**

Run: `pytest -q tests/checkrepositorylayout.py tests/checkall.py tests/checknaming.py`

Expected: all repository structure tests pass.

- [ ] **Step 7: Commit Task 7**

```bash
git add -A .old_structure README_old.md README_original.md RESTRUCTURE_PLAN.md data configs docs/archive tests msdyolo/data
git add -u scripts/cleanup.sh scripts/download_dota.py scripts/fix_imports.py scripts/fixpy312.sh scripts/restructure.sh scripts/setup_and_train.sh scripts/setupcloud.sh scripts/train_baseline_p2.sh scripts/ddp_train.sh
git commit -m "chore: remove obsolete repository duplicates"
```

Before committing, inspect `git status --short` and confirm `scripts/verify.sh` remains `??` and unstaged.

---

### Task 8: Fix packaging and current documentation

**Files:**
- Modify: `setup.py`
- Modify: `setup.cfg`
- Modify: `README.md`
- Modify: `docs/GetStart.md`
- Modify: `docs/install.md`
- Create: `tests/checkpackaging.py`

**Interfaces:**
- Console scripts: `msdyolo-train`, `msdyolo-val`, `msdyolo-detect`, `msdyolo-export`.
- Installed distribution includes YAML configs and NMS extension source assets required from `msdyolo/`.
- Metadata license matches the repository GPL-3.0 license.

- [ ] **Step 1: Write failing metadata and documentation tests**

Assert setup metadata contains all four console scripts, GPL license metadata, and package data rules for YAML. Scan current README/GetStart/install docs and fail on removed paths such as `utils/imgsplit.py`, `data/dotav15_poly.yaml`, `configs/msdyolo-`, `setupcloud.sh`, and direct root implementation claims.

- [ ] **Step 2: Run packaging tests and verify RED**

Run: `pytest -q tests/checkpackaging.py`

Expected: failures show missing console scripts, incorrect MIT classifier, and stale documentation paths.

- [ ] **Step 3: Update packaging metadata**

Add the four entry points, `include_package_data=True`, and explicit package-data globs for YAML and NMS extension sources. Change the license classifier to GPLv3. Keep Python compatibility consistent between README and `python_requires`.

- [ ] **Step 4: Rewrite the operational documentation**

README becomes the single current quick start. Document default full training, `--prepare-only`, `--force-resplit`, `--foreground`, `--config`, PID/log paths, empty official val labels, baseline zero-match semantics, and full-mode cloud acceptance. Update GetStart/install only where they describe current commands; label archived files as historical.

- [ ] **Step 5: Verify GREEN and install smoke**

Run: `pytest -q tests/checkpackaging.py`

Expected: all packaging/docs assertions pass.

Run: `python -m pip install -e .`

Run: `msdyolo-train --help && msdyolo-val --help && msdyolo-detect --help && msdyolo-export --help`

Expected: installation and all four help commands exit 0.

- [ ] **Step 6: Commit Task 8**

```bash
git add setup.py setup.cfg README.md docs/GetStart.md docs/install.md tests/checkpackaging.py
git commit -m "docs: document canonical cloud workflow"
```

---

### Task 9: Run full verification and produce cloud acceptance instructions

**Files:**
- Modify only if a verification failure exposes a requirement gap: files already owned by Tasks 1–8
- Do not modify: `scripts/verify.sh`

**Interfaces:**
- Final local evidence: full test suite, shell syntax, clean imports, synthetic preparation, editable install, CLI help, and CPU single-batch result.
- Final cloud evidence still required: DOTA v1.5 full dataset, V100/CUDA environment, and first full-MSD epoch with `match > 0`.

- [ ] **Step 1: Run static and repository checks**

Run: `git diff --check`

Run: `bash -n scripts/setup.sh scripts/ddp_train.sh`

Run: `rg -n "from (models|utils)|import (models|utils)|train_split_1024_gap200|setupcloud\.sh|setup_and_train\.sh" --glob '*.py' --glob '*.sh' --glob '*.yaml' --glob '!docs/archive/**' .`

Expected: diff check and Bash syntax exit 0; ripgrep finds no live stale imports or paths.

- [ ] **Step 2: Run the complete test suite**

Run: `pytest -q`

Expected: all existing and new tests pass with zero failures. Record the exact count and warnings.

- [ ] **Step 3: Run synthetic data preparation twice**

Use a temporary directory outside the repository, generate one labelled train image and one unlabelled val image, then run `prepare_dota` twice with `--subsize 1024 --gap 200 --num-process 1`.

Expected: first run rebuilds, second run reuses, output train labels contain pixel coordinates and val label directory is empty but valid.

- [ ] **Step 4: Run CPU single-batch training**

Run:

```bash
python -m msdyolo.train \
  --config configs/train/baseline.yaml \
  --data tests/fixtures/dota.yaml \
  --cfg configs/models/yolov5n.yaml \
  --hyp msdyolo/data/hyps/obb/hyp.finetune_dota.yaml \
  --weights "" \
  --device cpu \
  --batch-size 1 \
  --img-size 320 \
  --single-batch
```

Expected: non-empty targets, finite nonzero detection loss, successful backward pass, and exit 0. Do not require `match > 0` in baseline mode.

- [ ] **Step 5: Verify Git scope**

Run: `git status --short`

Expected: only intended task changes are committed or staged as appropriate; `scripts/verify.sh` remains untracked and unchanged. Inspect `git diff --stat HEAD~8..HEAD` and the deletion list against the approved design.

- [ ] **Step 6: Provide cloud acceptance command**

Document this final external check without claiming it ran locally:

```bash
bash scripts/setup.sh --foreground
```

Acceptance: setup completes, train labels are reported as pixel-format and non-empty, full MSD mode is enabled, detection loss is finite/nonzero, and the first epoch reports `match > 0`. If it reports zero, use the emitted matching diagnostics rather than changing coordinates blindly.

- [ ] **Step 7: Commit any verification-only corrections**

If Step 1–5 exposes a correction, return to the owning task, repeat that task's RED/GREEN verification, and use its explicit file-scoped commit command. Then rerun all of Task 9. If no correction is required, do not create an empty commit.
