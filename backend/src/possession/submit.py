"""One command from the competition data folder to the submission CSV.

    tracking -> stitch (11 tracks per team, gaps filled) -> frame model -> decode
    -> path model on the live stretches (its own environment, CPU, one thread)
    -> combine -> decode -> dead-ball check -> (second pass) -> frame,label CSV + manifest.

CPU, fixed seeds, single-threaded models: the same inputs give the same CSV byte for byte.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from possession import official
from possession.deadhead import load_bundle, veto_runs_bundle
from possession.decoding import decode_halves
from possession.labels import A, D, H, X, write_submission
from possession.learned import learned_scores, load_model
from possession.stitch import stitch
from possession.tracking import file_sha256, load_tracking

REPO_ROOT = official.REPO_ROOT
WORK = REPO_ROOT / "work"
PATHCRF_DIR = Path(os.environ.get("PATHCRF_HOME") or WORK / "pathcrf")
GAME = "BAR-ATH"


@dataclass(frozen=True)
class SubmitSettings:
    frame_model: Path = REPO_ROOT / "models/frame.joblib"
    dead_check: Path = REPO_ROOT / "models/deadcheck/check.json"
    # The path model runs in its own environment; the checkpoint is a trial under <repo>/saved/.
    pathcrf_python: Path = PATHCRF_DIR / ".venv/bin/python"
    pathcrf_repo: Path = PATHCRF_DIR / "repo"
    pathcrf_trial: str = "760"
    pathcrf_script: Path = REPO_ROOT / "pathcrf/infer.py"
    pathcrf_smooth_frames: int = 25
    pathcrf_threads: int = 1


@dataclass
class SubmitResult:
    csv: Path
    manifest: Path
    codes: npt.NDArray[np.int8]
    summary: dict[str, Any] = field(default_factory=dict)


def environment() -> dict[str, str]:
    """Versions that decide the bytes of the output."""
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "machine": platform.machine(), "system": platform.system(),
            "uv_lock_sha256": file_sha256(REPO_ROOT / "backend/uv.lock")}


def fuse(frame: npt.NDArray[np.float64], p_home_src: npt.NDArray[np.float64],
         p_away_src: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Combine the two models: the path model's team evidence where it has an output (else the
    frame model's own team split), geometric mean with the frame model's team split, and the
    frame model's dead-ball probability."""
    have = np.isfinite(p_home_src) & np.isfinite(p_away_src) & ((p_home_src + p_away_src) > 0)
    soft = np.where(have, p_home_src / np.where(have, p_home_src + p_away_src, 1.0), np.nan)
    frame_home = frame[:, H] / np.maximum(frame[:, H] + frame[:, A], 1e-9)
    team_home = np.where(have, soft, frame_home)
    p_dead = frame[:, D]
    mix_home = np.sqrt(np.clip(team_home, 1e-6, 1) * np.clip(frame_home, 1e-6, 1))
    mix_away = np.sqrt(np.clip(1 - team_home, 1e-6, 1) * np.clip(1 - frame_home, 1e-6, 1))
    mix_home = mix_home / (mix_home + mix_away)
    fused: npt.NDArray[np.float64] = np.column_stack(
        [(1 - p_dead) * mix_home, (1 - p_dead) * (1 - mix_home), p_dead])
    return fused


def pathcrf_evidence(players: npt.NDArray[np.float32], teams: npt.NDArray[np.int8],
                     alive_codes: npt.NDArray[np.int8], halves: list[tuple[int, int, int]],
                     ball: npt.NDArray[np.float32], work: Path, s: SubmitSettings
                     ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
    """The path model's team evidence on gap-filled tracks, in its own environment.

    `alive_codes` only marks the live stretches the path model runs in; it never sees a team
    label. Returns per-frame p_home_src, p_away_src (NaN where the path model has no output)."""
    import pandas as pd

    work.mkdir(parents=True, exist_ok=True)
    npz = work / "pathcrf-in.npz"
    np.savez(npz, allow_pickle=False, players=players, teams=teams, labels=alive_codes,
             halves=np.array(halves), ball=ball)
    out = work / "pathcrf"
    log = work / "pathcrf.log"
    cmd = [str(s.pathcrf_python), str(s.pathcrf_script), "--repo", str(s.pathcrf_repo),
           "--npz", str(npz), "--trial", s.pathcrf_trial, "--out", str(out),
           "--smooth", str(s.pathcrf_smooth_frames), "--threads", str(s.pathcrf_threads)]
    with open(log, "w") as handle:
        done = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"the path model failed (exit {done.returncode}); see {log}")
    frames = pd.read_csv(out / "frames.csv", index_col="frame_id")
    n = len(alive_codes)
    ph = np.full(n, np.nan)
    pa = np.full(n, np.nan)
    idx = frames.index.to_numpy()
    ph[idx] = frames["p_home_src"].to_numpy()
    pa[idx] = frames["p_away_src"].to_numpy()
    info: dict[str, Any] = json.loads((out / "log.json").read_text())
    checkpoint = s.pathcrf_repo / "saved" / s.pathcrf_trial / "model/state_dict_best_acc.pt"
    info["checkpoint_sha256"] = file_sha256(checkpoint)
    return ph, pa, info


def run_submit(data_dir: Path, out_csv: Path, settings: SubmitSettings | None = None,
               work_dir: Path | None = None) -> SubmitResult:
    s = settings or SubmitSettings()
    paths = official.games().asset_paths(data_dir=str(data_dir))
    atd_raw, atd_meta = Path(paths["atd_raw"]), Path(paths["atd_meta"])
    n_frames, halves = official.n_video_frames(), official.half_blocks()
    kickoffs, pitch, fps = official.kickoff_frames(), official.pitch_metres(), float(official.fps())
    work = work_dir or WORK / "submit" / out_csv.stem
    work.mkdir(parents=True, exist_ok=True)
    team_of_slot = official.starter_baseline().team_of_each_track(str(atd_meta))
    tracking = load_tracking(atd_raw, team_of_slot, n_frames, work / "cache")

    st = stitch(tracking, halves, pitch, fps)
    filled = st.tracking(tracking, filled=True, blocks=halves)
    frame = learned_scores(load_model(s.frame_model), filled, halves, pitch, fps)
    frame_codes = decode_halves(frame, halves, kickoffs, fps)

    ph, pa, info = pathcrf_evidence(filled.players, filled.team_of_slot, frame_codes, halves,
                                    tracking.ball, work, s)
    codes = decode_halves(fuse(frame, ph, pa), halves, kickoffs, fps)

    bundle = load_bundle(s.dead_check)
    frame2, check = veto_runs_bundle(bundle, frame, codes, filled, tracking.players, halves,
                                     pitch, fps, GAME)
    if check["vetoed"]:  # second pass, same order
        frame2_codes = decode_halves(frame2, halves, kickoffs, fps)
        ph2, pa2, _info2 = pathcrf_evidence(filled.players, filled.team_of_slot, frame2_codes,
                                            halves, tracking.ball, work / "pass2", s)
        codes = decode_halves(fuse(frame2, ph2, pa2), halves, kickoffs, fps)
    write_submission(out_csv, codes)

    summary: dict[str, Any] = {
        "label_counts": {k: int(np.sum(codes == v)) for k, v in
                         {"H": H, "A": A, "D": D, "X": X}.items()},
        "possessions": _spells(codes, halves),
        "stoppages_pass1": check["runs_pass1"],
        "stoppages_removed": len(check["vetoed"]),
        "stoppages_removed_seconds": check["vetoed_seconds"],
        "path_model": {k: info.get(k) for k in ("trial", "checkpoint_sha256", "n_episodes")},
    }
    manifest: dict[str, Any] = {
        "game": GAME,
        "csv_sha256": file_sha256(out_csv),
        "inputs_sha256": {"atd_raw": file_sha256(atd_raw), "atd_meta": file_sha256(atd_meta)},
        "frame_model_sha256": file_sha256(s.frame_model),
        "dead_check_sha256": file_sha256(s.dead_check),
        "environment": environment(),
        "summary": summary,
    }
    manifest_path = out_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    passed, report = official.validate_file(out_csv)
    summary["official_validator"] = "PASS" if passed else f"FAIL: {report[-400:]}"
    return SubmitResult(out_csv, manifest_path, codes, summary)


def _spells(codes: npt.NDArray[np.int8], halves: list[tuple[int, int, int]]) -> int:
    """Number of team possessions (runs of H or A) inside the graded halves."""
    total = 0
    for _h, a, b in halves:
        block = codes[a:b]
        team = np.isin(block, [H, A])
        starts = team & np.concatenate(([True], block[1:] != block[:-1]))
        total += int(starts.sum())
    return total
