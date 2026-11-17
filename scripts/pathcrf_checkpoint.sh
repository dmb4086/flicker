#!/usr/bin/env bash
# The path model's checkpoint: a GitHub Release asset, sha256-verified against pathcrf/checkpoints.sha256.
# The pack holds what inference reads: saved/<id>/args.json and saved/<id>/model/state_dict_best_acc.pt (~18 MB).
#
#   scripts/pathcrf_checkpoint.sh fetch    # download (or reuse the local pack), verify, unpack into $PATHCRF_HOME/repo/saved/<id>
#   scripts/pathcrf_checkpoint.sh verify   # check the unpacked checkpoint against the pack (offline)
# Env: PATHCRF_HOME (default work/pathcrf); PATHCRF_CHECKPOINT_URL (overrides the download URL, file:// works).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${PATHCRF_HOME:-$ROOT/work/pathcrf}
CACHE=$ROOT/work/checkpoints
MANIFEST=$ROOT/pathcrf/checkpoints.sha256
CMD=${1:?usage: pathcrf_checkpoint.sh fetch|verify}
ID=760
PACK=pathcrf-trial-$ID.tar.gz
FILES=(args.json model/state_dict_best_acc.pt)
URL=${PATHCRF_CHECKPOINT_URL:-https://github.com/dmb4086/flicker/releases/download/pathcrf-$ID/$PACK}
mkdir -p "$CACHE"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
exp=$(awk -v f="$PACK" '$2==f {print $1}' "$MANIFEST")
[ -n "$exp" ] || { echo "no line for $PACK in $MANIFEST" >&2; exit 1; }

case $CMD in
  fetch)
    if [ ! -f "$CACHE/$PACK" ] || [ "$(sha "$CACHE/$PACK")" != "$exp" ]; then
      curl -fsSL "$URL" -o "$CACHE/$PACK.part"
      mv "$CACHE/$PACK.part" "$CACHE/$PACK"
    fi
    got=$(sha "$CACHE/$PACK")
    [ "$got" = "$exp" ] || { echo "sha256 mismatch for $PACK: got $got, expected $exp" >&2; rm -f "$CACHE/$PACK"; exit 1; }
    mkdir -p "$HOME_DIR/repo"
    tar -xzf "$CACHE/$PACK" -C "$HOME_DIR/repo"
    echo "checkpoint ok: $HOME_DIR/repo/saved/$ID (pack sha256 $got)"
    ;;
  verify)
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    tar -xzf "$CACHE/$PACK" -C "$tmp"
    for f in "${FILES[@]}"; do cmp -s "$tmp/saved/$ID/$f" "$HOME_DIR/repo/saved/$ID/$f" || { echo "$f differs from the pack" >&2; exit 1; }; done
    echo "checkpoint matches its pack"
    ;;
  *) echo "unknown command $CMD" >&2; exit 2;;
esac
