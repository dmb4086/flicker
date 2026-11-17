"""Run the path model (PathCRF) on stitched, gap-filled player tracks and write its per-frame
team evidence. The model never sees the ball. Runs in the PathCRF environment:

    python infer.py --repo <pathcrf clone> --npz <in.npz> --trial 760 --out <dir>

The npz holds players (normalised), teams, labels (only used to mark live stretches), halves and
ball. Writes frames.csv (p_home_src, p_away_src per frame) and log.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

PITCH = np.array([105.0, 68.0])
FPS = 25
H, A = 0, 1


def _smooth_spans(players, present, window):
    """Centred moving average of each slot's x/y inside its active span (edge frames use shorter windows)."""
    k = np.ones(window) / window
    for s in range(players.shape[1]):
        idx = np.flatnonzero(present[:, s])
        if len(idx) == 0:
            continue
        a, b = idx[0], idx[-1] + 1
        for d in range(2):
            col = players[a:b, s, d]
            num = np.convolve(col, k, mode="same")
            den = np.convolve(np.ones_like(col), k, mode="same")
            players[a:b, s, d] = num / den
    return players


def build_tracking(c, smooth=0):
    players = c["players"].astype(np.float64) * PITCH  # metres in [0,105]x[0,68]; NaN when absent
    teams, labels, halves = c["teams"], c["labels"], c["halves"]
    ball = c["ball"].astype(np.float64) * PITCH
    T, S, _ = players.shape
    info = {}
    present = np.isfinite(players[:, :, 0])
    if smooth > 1:
        players = _smooth_spans(players, present, smooth)
        info["smooth_window_frames"] = smooth
    period = np.zeros(T, int)
    ts = np.zeros(T)
    for half, start, stop in halves:
        period[start:stop] = half
        ts[start:stop] = (np.arange(start, stop) - start) / FPS
    df = pd.DataFrame({"frame_id": np.arange(T), "period_id": period, "timestamp": np.round(ts, 2)})
    df["ball_state"] = np.where(np.isin(labels, [H, A]), "alive", "dead")
    owning = np.where(labels == H, "home", np.where(labels == A, "away", None))
    df["ball_owning_team_id"] = pd.Series(owning, dtype=object)
    df["ball_x"], df["ball_y"] = ball[:, 0], ball[:, 1]
    cols = {}
    for s in range(S):
        if not present[:, s].any():
            continue
        name = f"{'home' if teams[s] == 0 else 'away'}_{s}"
        cols[f"{name}_x"], cols[f"{name}_y"] = players[:, s, 0], players[:, s, 1]
    df = pd.concat([df, pd.DataFrame(cols)], axis=1)
    df = df[df["period_id"] > 0].reset_index(drop=True)  # drop the one-frame half gap
    return df, info


def soft_team_evidence(inf, utils, model, tracking, fps, sample_freq, window_seconds):
    """Per-frame edge-probability mass by sender team / receiver team (independent softmax, no CRF)."""
    tracking = tracking.copy().dropna(axis=1, how="all").set_index("frame_id").sort_index()
    tracking = inf._add_outside_nodes(tracking)
    out = {}
    n_skipped = 0
    for phase in sorted(tracking["phase_id"].unique()):
        if phase == 0:
            continue
        pt = tracking[tracking["phase_id"] == phase]
        lgk, rgk = utils.detect_keepers(pt, left_first=True)
        for ep in sorted(e for e in pt["episode_id"].unique() if e > 0):
            ept = pt[pt["episode_id"] == ep].sort_index()
            _, _, micro_full, order, _ = inf.inference_episode(
                model, ept, lgk, rgk, use_crf=False, decode="indep", evaluate=False,
                window_seconds=window_seconds, fps=fps, sample_freq=sample_freq, device=torch.device("cpu"))
            if micro_full is None:
                n_skipped += 1
                continue
            cols = list(micro_full.columns)
            src = np.array([c.split("-", 1)[0] for c in cols])
            dst = np.array([c.split("-", 1)[1] for c in cols])
            P = micro_full.to_numpy(dtype=np.float64)
            rec = {}
            for who, arr in (("src", src), ("dst", dst)):
                for team in ("home", "away", "out"):
                    rec[f"p_{team}_{who}"] = P[:, np.char.startswith(arr, team + "_")].sum(1)
            rec["frame_id"] = micro_full.index.to_numpy()
            for k, v in rec.items():
                out.setdefault(k, []).append(v)
    soft = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()}).set_index("frame_id")
    return soft, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the PathCRF clone (scripts/setup_pathcrf.sh)")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.npz, a.out = os.path.abspath(a.npz), os.path.abspath(a.out)
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    from datatools import utils
    from models.utils import build_model, load_trial_args, resolve_model_path
    import inference as inf

    args = load_trial_args(f"saved/{a.trial}")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    c = dict(np.load(a.npz))
    t0 = time.time()
    df, info = build_tracking(c, smooth=a.smooth)
    log = {"npz": os.path.basename(a.npz), "trial": a.trial, **info}
    df, _ = utils.label_frames_and_episodes(df, fps=FPS)
    df, phases = utils.label_phases(df)
    df = utils.calculate_running_features(df, fps=FPS, denoise=True)
    log.update(n_frames=int(len(df)), n_phases=int(len(phases)),
               n_episodes=int(df["episode_id"].nunique() - 1), prep_s=round(time.time() - t0, 1))
    model = build_model(args, torch.device("cpu"))
    sd = torch.load(resolve_model_path(f"saved/{a.trial}", "state_dict_best_acc.pt"),
                    map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    sample_freq = int(FPS / args.get("target_fps", 5.0))
    t0 = time.time()
    soft, n_skipped = soft_team_evidence(inf, utils, model, df, FPS, sample_freq,
                                         args.get("window_seconds", 10.0))
    log.update(soft_inference_s=round(time.time() - t0, 1), episodes_skipped_nan=n_skipped)
    out = df[["frame_id", "ball_state"]].set_index("frame_id").join(soft, how="left")
    out[["p_home_src", "p_away_src"]].to_csv(f"{a.out}/frames.csv")
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=2)
    print(json.dumps(log))


if __name__ == "__main__":
    main()
