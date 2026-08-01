#!/usr/bin/env bash
set -euo pipefail

# Launch the canonical package entry point.  Override these for a multi-GPU
# cloud job, for example: NPROC_PER_NODE=4 CONFIG=configs/train/full.yaml.
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
CONFIG="${CONFIG:-configs/train/full.yaml}"

python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node "$NPROC_PER_NODE" \
    -m msdyolo.train \
    --config "$CONFIG"
