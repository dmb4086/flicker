#!/usr/bin/env bash
# Fetch the official More Than a Hack 2027 repository at a pinned commit.
# The validator and the game constants are used unchanged from here.
set -euo pipefail

REPO_URL="https://github.com/metrica-sports/more-than-a-hack-2027"
PINNED_COMMIT="f8f2629ff1e0985f81d7e59d8413678514ee1530"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/more-than-a-hack-2027"

if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$REPO_URL" "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin "$PINNED_COMMIT"
git -C "$DEST" checkout --quiet "$PINNED_COMMIT"
echo "official repo at $(git -C "$DEST" rev-parse HEAD)"
