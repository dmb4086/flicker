#!/usr/bin/env bash
# Reproduce the submitted file: two clean runs, a byte comparison, and the sha256 of the result.
#
#   scripts/reproduce.sh <data-dir>      # the competition data/ folder, as delivered
#
# Prints key=value lines; csv_sha256 must be the submitted file's and identical_runs must be yes.
# Exit 0 only if both hold. Env: PATHCRF_HOME (default work/pathcrf), REPRODUCE_OUT (default work/reproduce).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WANT=1fbc1db7ada3f7f2e54c2b28c962aabaf8485edb86467364a77961460499b1b0
DATA=${1:?usage: scripts/reproduce.sh <data-dir>}
DATA=$(cd "$DATA" && pwd)
OUT=${REPRODUCE_OUT:-$ROOT/work/reproduce}
PCRF=${PATHCRF_HOME:-$ROOT/work/pathcrf}
rm -rf "${OUT:?}"; mkdir -p "$OUT"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$ROOT/$1")" = "$2" ] || { echo "$1: sha256 differs from the pinned one" >&2; exit 4; }; }
check models/frame.joblib e6acb0769863784795d7f71aba1c390743ddaf196b809b98a30e3aede9057aa3
check models/deadcheck/check.json 92491a34bc524436e30273134125a9a960546e3858b68448a883ca44b643bc46

[ -x "$PCRF/.venv/bin/python" ] && [ -f "$PCRF/repo/saved/760/args.json" ] || "$ROOT/scripts/setup_pathcrf.sh" >&2
UV_PROJECT_ENVIRONMENT=$PCRF/.venv uv sync --project "$ROOT/pathcrf" --locked --exact --no-install-project --quiet
"$ROOT/scripts/pathcrf_checkpoint.sh" verify >&2
[ -f "$ROOT/third_party/more-than-a-hack-2027/games.py" ] || "$ROOT/scripts/fetch_official.sh" >&2

cd "$ROOT/backend"
uv sync --locked --quiet
for run in 1 2; do
  uv run possession submit "$DATA" --out "$OUT/run$run.csv" --work-dir "$OUT/work$run" \
    > "$OUT/run$run.log" 2>&1 || { echo "run $run failed: see $OUT/run$run.log" >&2; exit 1; }
done
if cmp -s "$OUT/run1.csv" "$OUT/run2.csv"; then SAME=yes; else SAME=no; fi
GOT=$(sha "$OUT/run1.csv")
echo "csv_sha256=$GOT"
echo "identical_runs=$SAME"
echo "matches_submission=$([ "$GOT" = "$WANT" ] && echo yes || echo no)"
uv run python - "$OUT/run1.manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = m["summary"]
print("label_counts=" + " ".join(f"{k}={v}" for k, v in sorted(s["label_counts"].items())))
print(f"possessions={s['possessions']}")
for k, v in sorted(m["environment"].items()):
    print(f"{k}={v}")
PY
[ "$SAME" = yes ] && [ "$GOT" = "$WANT" ]
