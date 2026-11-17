# flicker

Frame-level football possession for **More Than a Hack 2027** (Metrica Sports × FC Barcelona), Phase 1.

For each of the 199,319 frames of FC Barcelona vs Athletic Club, the system says who has the ball: the home
team (H), the away team (A), nobody because play is stopped (D), or no answer outside the graded halves (X).

Team **flicker**: Dev Bhatt, Josh Noble, Alan Daring.

The method, its training data, the possession rules and the limits are described in the technical document
submitted with the entry (`flicker.pdf`).

Submitted file: `flicker.csv`, sha256 `1fbc1db7ada3f7f2e54c2b28c962aabaf8485edb86467364a77961460499b1b0`.

## Reproduce the submitted file

You need [`uv`](https://docs.astral.sh/uv/), `git`, `curl` and the competition data bundle. The match data is
not in this repository.

```bash
scripts/fetch_official.sh        # the organisers' validator and constants, at the pinned commit
scripts/setup_pathcrf.sh         # the path model's environment and its sha256-verified checkpoint
# unpack the data bundle as delivered into third_party/more-than-a-hack-2027/data/
scripts/reproduce.sh third_party/more-than-a-hack-2027/data
```

`reproduce.sh` runs the whole system twice, compares the two files byte for byte and prints the result's
sha256. It succeeds only if the two runs are identical and the file is the submitted one:

```
csv_sha256=1fbc1db7ada3f7f2e54c2b28c962aabaf8485edb86467364a77961460499b1b0
identical_runs=yes
matches_submission=yes
```

To write the CSV in one run:

```bash
(cd backend && uv run possession submit ../third_party/more-than-a-hack-2027/data --out ../flicker.csv)
```

Everything runs locally on CPU, on one thread, with fixed seeds and hash-pinned environments. It takes a few
minutes on a laptop and calls no outside service.

## What is here

| Path | What |
|---|---|
| `backend/` | The code that turns the tracking into the possession file, with its pinned Python environment |
| `models/` | The trained frame model and the models of the dead-ball check |
| `pathcrf/` | The pinned environment and runner of the path model (PathCRF, Kim et al., KDD 2026; code under MPL-2.0, fetched at a fixed commit). Its checkpoint is a Release asset of this repository |
| `scripts/` | Setup and reproduction |

## Licence

MIT, see [`LICENSE`](LICENSE). The match data is licensed by the organisers for the competition only and is not
redistributed here.
