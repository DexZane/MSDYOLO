#!/usr/bin/env bash
# Prepare DIOR dataset and start the canonical full-MSD cloud training job.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Prepare DIOR dataset and launch full MSDYOLO training.

Options:
  --prepare-only       Install and prepare data, but do not start training.
  --foreground         Run training in this terminal instead of the background.
  --config PATH        Training config (default: configs/train/full.yaml).
  -h, --help           Show this help and exit.
EOF
}

PREPARE_ONLY=false
FOREGROUND=false
CONFIG="configs/train/full.yaml"

while (($#)); do
    case "$1" in
        --prepare-only)
            PREPARE_ONLY=true
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
DATASET_DIR="$PROJECT_DIR/dataset/DIOR"
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
    if ! "$PYTHON_BIN" -c 'from urllib.request import urlretrieve; import sys, ssl; ssl._create_default_https_context = ssl._create_unverified_context; urlretrieve(sys.argv[1], sys.argv[2])' "$WEIGHTS_URL" "$WEIGHTS_TMP"; then
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
    "$PYTHON_BIN" -c 'from collections.abc import Mapping; import sys, torch, msdyolo; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); wrapped = isinstance(checkpoint, Mapping) and "model" in checkpoint; candidate = checkpoint["model"] if wrapped else checkpoint; state = candidate.float().state_dict() if wrapped and hasattr(candidate, "float") else candidate; assert (not wrapped or hasattr(candidate, "float") or isinstance(candidate, Mapping)) and isinstance(state, Mapping) and state and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())' "$1"
}

cd "$PROJECT_DIR"

acquirelaunchlock
checkexistingtraining

echo "Installing cloud dependencies..."
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"
"$PYTHON_BIN" -m pip install -q "numpy<2.0"
"$PYTHON_BIN" -m pip install -q -e .
"$PYTHON_BIN" -m pip install -q kaggle
# Dependency resolution may upgrade setuptools. Restore the pin last.
"$PYTHON_BIN" -m pip install -q "setuptools==69.5.1"

# Download DIOR-R from Kaggle if not already present
if [[ ! -d "$DATASET_DIR/train" \
    || ! -d "$DATASET_DIR/val" \
    || ! -d "$DATASET_DIR/test" ]]; then
    echo "Downloading DIOR-R dataset from Kaggle..."
    "$PYTHON_BIN" - <<'PYEOF'
import os
import sys
import zipfile
from pathlib import Path

# Set Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = "KGAT_be2904c745791157e2023e2a117180f2"
os.environ["KAGGLE_KEY"] = "KGAT_be2904c745791157e2023e2a117180f2"

# Import after setting environment
from kaggle.api.kaggle_api_extended import KaggleApi

target = Path("dataset/DIOR")
temp_zip = target.parent / "dior-r-dataset-yolov11-obb-format.zip"

target.mkdir(parents=True, exist_ok=True)

# Download dataset
api = KaggleApi()
api.authenticate()

print("Downloading DIOR-R from Kaggle (redzapdos123/dior-r-dataset-yolov11-obb-format)...")
api.dataset_download_files(
    "redzapdos123/dior-r-dataset-yolov11-obb-format",
    path=str(target.parent),
    unzip=False
)

# Extract
print(f"Extracting {temp_zip.name}...")
with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
    zip_ref.extractall(target)

# Cleanup
temp_zip.unlink()

# Check if extracted into a subdirectory (YOLODIOR-R/)
subdir = target / "YOLODIOR-R"
if subdir.exists():
    print("Moving files from YOLODIOR-R/ to dataset/DIOR/...")
    for item in subdir.iterdir():
        if item.name not in ['.DS_Store']:
            item.rename(target / item.name)
    subdir.rmdir()

print("DIOR-R dataset ready.")
print(f"  Train images: {len(list((target / 'train' / 'images').glob('*.jpg')))}")
print(f"  Val images: {len(list((target / 'val' / 'images').glob('*.jpg')))}")
print(f"  Test images: {len(list((target / 'test' / 'images').glob('*.jpg')))}")
PYEOF
fi

if [[ "$PREPARE_ONLY" == true ]]; then
    echo "DIOR preparation complete. Training was not started (--prepare-only)."
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
