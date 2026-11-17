#!/usr/bin/env bash
# Build the pinned environment of the path model (PathCRF) in one command, then fetch its checkpoint.
#   1. CPU-only venv from pathcrf/uv.lock (hash-locked; Python 3.12)
#   2. the authors' repository (MPL-2.0) at a fixed commit, plus a one-line pandas 3 fix
#   3. the checkpoint: scripts/pathcrf_checkpoint.sh fetch (sha256-verified)
# Env: PATHCRF_HOME  target dir (default work/pathcrf); contains .venv/ and repo/.
# Idempotent: an existing repo/ is checked, not re-cloned; the venv is re-synced to the lock (--exact).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${PATHCRF_HOME:-$ROOT/work/pathcrf}
SRC=${PATHCRF_SRC:-https://github.com/hyunsungkim-ds/pathcrf}
COMMIT=1b7c6f85a6b78c92e4543fde72ca2b86a933f997
PATCH=$ROOT/pathcrf/pandas3.patch
mkdir -p "$HOME_DIR"

UV_PROJECT_ENVIRONMENT=$HOME_DIR/.venv uv sync --project "$ROOT/pathcrf" --locked --exact --no-install-project

REPO=$HOME_DIR/repo
if [ ! -d "$REPO/.git" ]; then
  git clone --quiet "$SRC" "$REPO"
fi
git -C "$REPO" checkout --quiet "$COMMIT"
# Reset the file the patch touches to the pinned commit, then apply it.
for path in $(sed -n 's|^diff --git a/\([^ ]*\) b/.*|\1|p' "$PATCH" | sort -u); do
  git -C "$REPO" checkout --quiet "$COMMIT" -- "$path"
done
git -C "$REPO" apply "$PATCH" || { echo "cannot apply $(basename "$PATCH") to $REPO" >&2; exit 1; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT" ] || { echo "wrong PathCRF commit" >&2; exit 1; }

"$ROOT/scripts/pathcrf_checkpoint.sh" fetch
"$HOME_DIR/.venv/bin/python" -c "import torch, torch_geometric; print('pathcrf env ok: torch', torch.__version__, 'cpu-only' if not torch.version.cuda else 'CUDA')"
