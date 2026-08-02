# Cloud Training Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DOTA training use a trained baseline checkpoint as its teacher and remove matching hot-path GPU synchronizations.

**Architecture:** Phase 1 fine-tunes a detector on DOTA and saves a checkpoint. Phase 2 loads that checkpoint into a separate frozen teacher while a student learns from degraded images. Matching transfers candidate data once per batch, prefilters candidates, then preserves exact Shapely IoU for final pairs.

**Tech Stack:** Python 3.12, PyTorch 2.5 CUDA 12.4, YOLOv5-OBB, Shapely 2, pytest.

## Global Constraints

- Preserve pixel-coordinate DOTA labels and `workers: 4`.
- Never use the unmatched COCO 80-class detection head as the DOTA teacher.
- Retain exact Shapely rotated IoU for accepted pairs.
- Do not stage user-owned `scripts/verify.sh`.

---

### Task 1: Add a frozen DOTA teacher input

**Files:**

- Modify: `msdyolo/train.py`, `msdyolo/utils/trainer.py`, `configs/train/full.yaml`
- Test: `tests/checkp0a2.py`

**Interfaces:**

- Consumes: `training.teacherweights: str`.
- Produces: `MSDYOLOTrainer(..., teachermodel=...)` with an eval-mode model whose parameters have `requires_grad=False`.

- [ ] **Step 1: Write the failing test**

```python
def checkfullmodeusesaseparatefrozenteacher(...):
    trainer = MSDYOLOTrainer(student, config, device, teachermodel=teacher)
    assert trainer.teachermodel is teacher
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/checkp0a2.py -k separatefrozenteacher`

Expected: FAIL because the trainer does not accept `teachermodel`.

- [ ] **Step 3: Write minimal implementation**

```python
def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
```

Load `training.teacherweights` into a second model and reject full mode when the value is empty.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/checkp0a2.py -k separatefrozenteacher`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add msdyolo/train.py msdyolo/utils/trainer.py configs/train/full.yaml tests/checkp0a2.py
git commit -m "feat: load frozen DOTA teacher for full training"
```

### Task 2: Make exact matching batch-efficient

**Files:**

- Modify: `msdyolo/utils/matching.py`
- Test: `tests/checkp0a1.py`

**Interfaces:**

- Consumes: existing `matchpredictions(student, teacher, targets, ...)`.
- Produces: unchanged one-to-one `DistillationMatch`.

- [ ] **Step 1: Write the failing test**

```python
def checkmatchingusesonecpusnapshotperbatch(monkeypatch):
    # Count Tensor.cpu calls while matching one batch.
    assert snapshots <= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/checkp0a1.py -k cpusnapshot`

Expected: FAIL because each inner-loop IoU presently calls `.cpu()`.

- [ ] **Step 3: Write minimal implementation**

Transfer teacher values, decoded angles, and targets once per batch. Filter confidence/class vectorially, conservatively prefilter by center distance, apply `rotatediou` only to survivors, and calculate student distances as a matrix.

- [ ] **Step 4: Run behavior tests**

Run: `python -m pytest -q tests/checkp0a1.py tests/checkp0a2.py -k matching`

Expected: PASS; order and uniqueness contracts remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add msdyolo/utils/matching.py tests/checkp0a1.py
git commit -m "perf: batch matching transfers"
```

### Task 3: Run cloud stage transition

**Files:**

- Modify: `scripts/setup.sh`, `README.md`
- Test: cloud commands

**Interfaces:**

- Consumes: prepared `dataset/DOTA/split`, `yolov5s.pt`, baseline checkpoint.
- Produces: full-mode run with a teacher path and finite, nonzero matching.

- [ ] **Step 1: Add setup validation for missing teacher checkpoint**

Run: `bash scripts/setup.sh --full --prepare-only`

Expected: explicit missing-DOTA-teacher failure.

- [ ] **Step 2: Implement stage commands and documentation**

Expose a baseline command and a full command accepting an explicit teacher checkpoint. Document the hand-off path and `match > 0` criterion.

- [ ] **Step 3: Verify on cloud**

Run baseline single-batch training, save a checkpoint, then run full-mode single-batch training from that teacher. Confirm finite loss, positive `matchcount`, and no matching stall.

- [ ] **Step 4: Commit**

```bash
git add scripts/setup.sh README.md
git commit -m "docs: document staged cloud training"
```

