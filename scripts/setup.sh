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
            if (($# < 2)) || [[ "$2" == -* ]]; then
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
LOCK_DIR="$RUN_DIR/.launch.lock"
LOG_FILE="$PROJECT_DIR/training.log"
WEIGHTS_FILE="${WEIGHTS_FILE:-$PROJECT_DIR/yolov5s.pt}"
WEIGHTS_URL="${WEIGHTS_URL:-https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt}"

if [[ "$CONFIG" == /* ]]; then
    CONFIG_PATH="$CONFIG"
else
    CONFIG_PATH="$PROJECT_DIR/$CONFIG"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "error: config file not found: $CONFIG_PATH" >&2
    exit 2
fi

releaselaunchlock() {
    rm -f "$LOCK_DIR/owner.pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

acquirelaunchlock() {
    mkdir -p "$RUN_DIR"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCK_OWNER=""
        if [[ -f "$LOCK_DIR/owner.pid" ]]; then
            LOCK_OWNER="$(tr -d '[:space:]' < "$LOCK_DIR/owner.pid")"
        fi
        if [[ "$LOCK_OWNER" =~ ^[0-9]+$ ]] && kill -0 "$LOCK_OWNER" 2>/dev/null; then
            echo "error: setup is already preparing or launching training (PID: $LOCK_OWNER)" >&2
        else
            echo "error: setup launch lock exists at $LOCK_DIR; inspect and remove it after confirming no setup is active" >&2
        fi
        exit 1
    fi
    printf '%s\n' "$$" >"$LOCK_DIR/owner.pid"
    trap releaselaunchlock EXIT
}

checkexistingtraining() {
    if [[ ! -f "$PID_FILE" ]]; then
        return
    fi
    EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "error: training is already running (PID: $EXISTING_PID; log: $LOG_FILE)" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
}

writepid() {
    PID_TMP="$(mktemp "$RUN_DIR/.training.pid.XXXXXX")"
    printf '%s\n' "$1" >"$PID_TMP"
    mv -f "$PID_TMP" "$PID_FILE"
}

downloadweights() {
    if [[ -s "$WEIGHTS_FILE" ]]; then
        if validateweights "$WEIGHTS_FILE"; then
            return
        fi
        echo "error: invalid pretrained weights: $WEIGHTS_FILE" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$WEIGHTS_FILE")"
    WEIGHTS_TMP="$(mktemp "${WEIGHTS_FILE}.download.XXXXXX")"
    echo "Downloading pretrained yolov5s weights..."
    if ! "$PYTHON_BIN" -c 'from urllib.request import urlretrieve; import sys; urlretrieve(sys.argv[1], sys.argv[2])' "$WEIGHTS_URL" "$WEIGHTS_TMP"; then
        rm -f "$WEIGHTS_TMP"
        echo "error: failed to download pretrained weights from $WEIGHTS_URL" >&2
        exit 1
    fi
    if [[ ! -s "$WEIGHTS_TMP" ]] || ! validateweights "$WEIGHTS_TMP"; then
        rm -f "$WEIGHTS_TMP"
        echo "error: invalid pretrained weights from $WEIGHTS_URL" >&2
        exit 1
    fi
    mv -f "$WEIGHTS_TMP" "$WEIGHTS_FILE"
}

validateweights() {
    "$PYTHON_BIN" -c 'from collections.abc import Mapping; import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); wrapped = isinstance(checkpoint, Mapping) and "model" in checkpoint; candidate = checkpoint["model"] if wrapped else checkpoint; state = candidate.float().state_dict() if wrapped and hasattr(candidate, "float") else candidate; assert (not wrapped or hasattr(candidate, "float") or isinstance(candidate, Mapping)) and isinstance(state, Mapping) and state and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())' "$1"
}

cd "$PROJECT_DIR"

acquirelaunchlock
checkexistingtraining

echo "Installing cloud dependencies..."
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"
"$PYTHON_BIN" -m pip install -q -e .
"$PYTHON_BIN" -m pip install -q openxlab
# Dependency resolution may upgrade setuptools while installing the editable
# package or OpenDataLab. Restore the Python 3.12 compatibility pin last.
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"

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

downloadweights

if [[ "$FOREGROUND" == true ]]; then
    writepid "$$"
    releaselaunchlock
    trap - EXIT
    exec "$PYTHON_BIN" -m msdyolo.train --config "$CONFIG"
fi

"$PYTHON_BIN" -m msdyolo.train --config "$CONFIG" >"$LOG_FILE" 2>&1 &
TRAIN_PID=$!
writepid "$TRAIN_PID"

echo "Training started in the background (PID: $TRAIN_PID)."
echo "Monitor: tail -f $LOG_FILE"
echo "Stop: kill $TRAIN_PID"
