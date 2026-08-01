#!/usr/bin/env bash
# Prepare DOTA v1.5 and start the canonical full-MSD cloud training job.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Prepare DOTA v1.5 patches and launch full MSDYOLO training.

Options:
  --prepare-only       Install and prepare data, but do not start training.
  --force-resplit      Rebuild the prepared DOTA split even when it is current.
  --foreground         Run training in this terminal instead of the background.
  --config PATH        Training config (default: configs/train/full.yaml).
  -h, --help           Show this help and exit.
EOF
}

PREPARE_ONLY=false
FORCE_RESPLIT=false
FOREGROUND=false
CONFIG="configs/train/full.yaml"

while (($#)); do
    case "$1" in
        --prepare-only)
            PREPARE_ONLY=true
            ;;
        --force-resplit)
            FORCE_RESPLIT=true
            ;;
        --foreground)
            FOREGROUND=true
            ;;
        --config)
            if (($# < 2)); then
                echo "error: --config requires a path" >&2
                exit 2
            fi
            CONFIG="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="$PROJECT_DIR/dataset/DOTA"
SPLIT_DIR="$DATASET_DIR/split"
RUN_DIR="$PROJECT_DIR/runs/setup"
PID_FILE="$RUN_DIR/training.pid"
LOG_FILE="$RUN_DIR/training.log"

cd "$PROJECT_DIR"

echo "Installing cloud dependencies..."
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"
"$PYTHON_BIN" -m pip install -q -e .
"$PYTHON_BIN" -m pip install -q openxlab

if [[ ! -d "$DATASET_DIR/train/images" \
    || ! -d "$DATASET_DIR/train/labelTxt" \
    || ! -d "$DATASET_DIR/val/images" ]]; then
    echo "Downloading DOTA v1.5..."
    "$PYTHON_BIN" -m msdyolo.data.scripts.download_dota "$DATASET_DIR"
fi

PREPARE_ARGS=(
    -m msdyolo.data.scripts.prepare_dota
    --dataset "$DATASET_DIR"
    --output "$SPLIT_DIR"
    --subsize 1024
    --gap 200
    --num-process 4
)
if [[ "$FORCE_RESPLIT" == true ]]; then
    PREPARE_ARGS+=(--force-resplit)
fi

echo "Preparing validated DOTA patches..."
"$PYTHON_BIN" "${PREPARE_ARGS[@]}"

if [[ "$PREPARE_ONLY" == true ]]; then
    echo "DOTA preparation complete. Training was not started (--prepare-only)."
    exit 0
fi

mkdir -p "$RUN_DIR"
if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "error: training is already running (PID: $EXISTING_PID; log: $LOG_FILE)" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if [[ "$FOREGROUND" == true ]]; then
    exec "$PYTHON_BIN" -m msdyolo.train --config "$CONFIG"
fi

"$PYTHON_BIN" -m msdyolo.train --config "$CONFIG" >"$LOG_FILE" 2>&1 &
TRAIN_PID=$!
PID_TMP="$(mktemp "$RUN_DIR/.training.pid.XXXXXX")"
printf '%s\n' "$TRAIN_PID" >"$PID_TMP"
mv -f "$PID_TMP" "$PID_FILE"

echo "Training started in the background (PID: $TRAIN_PID)."
echo "Monitor: tail -f $LOG_FILE"
echo "Stop: kill $TRAIN_PID"
