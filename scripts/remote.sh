#!/usr/bin/env bash
set -euo pipefail


# =========================================================
# REMOTE CONFIG
# =========================================================
REMOTE_HOST="mutolo"
REMOTE_DIR="~/nospy"

# =========================================================
# ARGUMENT PARSING
# =========================================================
REBUILD_VENV=1
if [[ "${1:-}" == "--no-venv" ]]; then
  REBUILD_VENV=0
fi

# =========================================================
# LOCAL PROJECT ROOT
# =========================================================
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =========================================================
# SYNC PROJECT TO REMOTE
# =========================================================
echo "Syncing project to remote..."

rsync -avz \
  --delete \
  --exclude ".venv" \
  --exclude "out" \
  --exclude "__pycache__" \
  --exclude "lightning_logs" \
  --exclude ".git" \
  "$LOCAL_DIR/" \
  "$REMOTE_HOST:$REMOTE_DIR/"

# =========================================================
# REMOTE EXECUTION
# =========================================================
if [[ $REBUILD_VENV -eq 1 ]]; then
  echo "Rebuilding venv and running experiment remotely..."
  ssh "$REMOTE_HOST" "
    cd $REMOTE_DIR &&
    echo 'Removing old venv...' &&
    rm -rf .venv &&
    echo 'Creating new venv...' &&
    make venv &&
    echo 'Running experiment...' &&
    make run
  "
else
  echo "Running experiment remotely without rebuilding venv..."
  ssh "$REMOTE_HOST" "
    cd $REMOTE_DIR &&
    echo 'Running experiment...' &&
    make run
  "
fi

# =========================================================
# COPY RESULTS BACK
# =========================================================
echo "Copying results back locally..."

mkdir -p "$LOCAL_DIR/out"

rsync -avz \
  "$REMOTE_HOST:$REMOTE_DIR/out/" \
  "$LOCAL_DIR/out/"

# Clean remote out directory after copying
ssh "$REMOTE_HOST" "rm -rf $REMOTE_DIR/out/*"

echo "Done."
echo "Results available in ./out"
