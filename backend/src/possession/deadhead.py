"""The dead-ball check: a second look at every stoppage of the first pass.

After the normal pass, every dead-ball run of the decode gets a score, the mean of a small
model's P(dead) over the run. A run below its threshold gets the frame model's P(dead) := 0.001
on all its frames (the team split kept), and the chain runs again. The check can only remove
dead ball, never add it.

The models' inputs come from the same gap-filled tracking the frame model reads, plus the raw
tracking's visible player count: 97 columns. CPU, one thread, deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from possession.labels import A, D, H

# The tracking signals the check reads.
SIGNALS = (
    "ctx_still_share", "ctx_fast_share", "motion_v1", "motion_sx3", "motion_sy3", "present",
    "gap_s", "ball_speed~25", "dmin", "ball_edge_x", "ball_edge_y", "dead_since_kick_s",
    "dead_until_kick_s", "ctx_cent_speed_h", "ctx_beyond~10s")
WINDOWS_S = (2, 5, 10)
SIDE_S = 5
CAP_S = 60.0
VIS_WINDOW_S = 5


def _window_mean(x: npt.NDArray[np.float64], lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
                 ) -> npt.NDArray[np.float64]:
    """Mean of the finite values of x[lo:hi] per position (NaN if none)."""
    fin = np.isfinite(x)
    s = np.concatenate([[0.0], np.cumsum(np.where(fin, x, 0.0))])
    c = np.concatenate([[0.0], np.cumsum(fin.astype(np.float64))])
    tot, cnt = s[hi] - s[lo], c[hi] - c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out: npt.NDArray[np.float64] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def _since_until(seen: npt.NDArray[np.bool_], fps: float
                 ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    since = np.where(last >= 0, (idx - last) / fps, CAP_S)
    until = np.where(nxt < n, (nxt - idx) / fps, CAP_S)
    return np.minimum(since, CAP_S), np.minimum(until, CAP_S)


def tracking_signals(tracking: Any, blocks: list[tuple[int, int, int]], pitch: tuple[float, float],
                     fps: float) -> npt.NDArray[np.float32]:
    """The 15 signals per frame."""
    from possession.ctxfeat import ctx_features
    from possession.learned import all_features, quantile_ball

    feats, names = all_features(tracking, blocks, pitch, fps, True, True)
    feats = quantile_ball(feats, names, blocks)
    c, c_names = ctx_features(tracking, blocks, pitch, fps)
    allf, alln = np.hstack([feats, c]), names + c_names
    return allf[:, [alln.index(s) for s in SIGNALS]].astype(np.float32)


def features(tracking: Any, raw_players: npt.NDArray[np.float32], pdead: npt.NDArray[np.float64],
             blocks: list[tuple[int, int, int]], pitch: tuple[float, float], fps: float
             ) -> tuple[npt.NDArray[np.float32], list[str]]:
    """(n_frames, 97) head inputs and their names; NaN outside the blocks."""
    sig = tracking_signals(tracking, blocks, pitch, fps).astype(np.float64)
    n = len(pdead)
    clip = np.clip(pdead, 1e-4, 1 - 1e-4)
    lp = np.log(clip / (1 - clip))
    seen = np.isfinite(tracking.ball[:, 0])
    names: list[str] = []
    per_block = []
    for _h, a, b in blocks:
        m = b - a
        i = np.arange(m)

        def win(x: npt.NDArray[np.float64], back: int, fwd: int, i: Any = i, m: int = m
                ) -> npt.NDArray[np.float64]:
            return _window_mean(x, np.maximum(i - back, 0), np.minimum(i + fwd + 1, m))

        cb, nb = [], []
        for k, s in enumerate(SIGNALS):
            x = sig[a:b, k]
            cb.append(x)
            nb.append(s)
            for w in WINDOWS_S:
                cb.append(win(x, int(w * fps), int(w * fps)))
                nb.append(f"{s}~c{w}")
            cb.append(win(x, int(SIDE_S * fps), 0))
            nb.append(f"{s}~past{SIDE_S}")
            cb.append(win(x, 0, int(SIDE_S * fps)))
            nb.append(f"{s}~next{SIDE_S}")
        x = lp[a:b]
        cb += [x, win(x, int(fps), int(fps)), win(x, int(3 * fps), int(3 * fps))]
        nb += ["e4_logit_pdead", "e4_logit_pdead~c1", "e4_logit_pdead~c3"]
        since, until = _since_until(seen[a:b], fps)
        cb += [since, until]
        nb += ["ball_since_seen_s", "ball_until_seen_s"]
        per_block.append((a, b, np.column_stack(cb)))
        names = nb
    base = np.full((n, len(names)), np.nan)
    for a, b, block in per_block:
        base[a:b] = block
    x32 = base.astype(np.float32)
    vis = np.isfinite(raw_players[..., 0]).sum(axis=1).astype(np.float64)
    vc = np.full(len(vis), np.nan)
    w = int(VIS_WINDOW_S * fps)
    for _h, a, b in blocks:
        i = np.arange(b - a)
        vc[a:b] = _window_mean(vis[a:b], np.maximum(i - w, 0), np.minimum(i + w + 1, b - a))
    out = np.column_stack([x32, vis.astype(np.float32), vc.astype(np.float32)]).astype(np.float32)
    return out, names + ["raw_visible_players", "raw_visible_players~c5"]


def load_head(path: Path) -> dict[str, Any]:
    import joblib

    head: dict[str, Any] = joblib.load(path)
    if not {"clf", "names"} <= set(head):
        raise ValueError(f"{path}: not a model payload (needs clf and names)")
    return head


def _with_pdead(e4: npt.NDArray[np.float64], pd: npt.NDArray[np.float64],
                mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """The frame model's matrix with P(dead) := pd on mask, the team split kept."""
    out = e4.copy()
    team = e4[:, H:A + 1] / np.maximum(e4[:, H:A + 1].sum(axis=1, keepdims=True), 1e-9)
    out[mask, D] = pd[mask]
    out[mask, H:A + 1] = team[mask] * (1 - pd[mask])[:, None]
    return out


# The frame model's P(dead) on a run the check overturns.
VETO_P = 0.001


def d_runs(codes: npt.NDArray[np.int8], blocks: list[tuple[int, int, int]]
           ) -> list[tuple[int, int]]:
    """Maximal runs of D inside each graded block, as [start, stop) frame pairs."""
    out: list[tuple[int, int]] = []
    for _h, a, b in blocks:
        d = np.diff(np.concatenate([[0], (codes[a:b] == D).astype(np.int8), [0]]))
        out += [(a + int(s), a + int(e))
                for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)]
    return out


def load_bundle(path: Path) -> dict[str, Any]:
    """The bundle (models/deadcheck/check.json): its spec and every model it pins, each checked
    against the spec's sha256 before use."""
    import json

    from possession.tracking import file_sha256

    spec: dict[str, Any] = json.loads(path.read_text())
    if spec.get("schema") != 1:
        raise ValueError(f"{path}: not a bundle (schema 1)")
    heads: dict[str, dict[str, Any]] = {}
    for name, want in spec["heads_sha256"].items():
        got = file_sha256(path.parent / name)
        if got != want:
            raise ValueError(f"{path.parent / name}: sha256 {got} != bundle {want}")
        heads[name] = load_head(path.parent / name)
    used = {s["head"] for s in spec["stretches"]} | {spec["outside"]["head"]}
    if not used <= set(heads):
        missing = sorted(used - set(heads))
        raise ValueError(f"{path}: the bundle uses a head it does not pin: {missing}")
    return {"spec": spec, "heads": heads}


def veto_runs_bundle(bundle: dict[str, Any], e4: npt.NDArray[np.float64],
                     codes: npt.NDArray[np.int8], tracking: Any,
                     raw_players: npt.NDArray[np.float32], blocks: list[tuple[int, int, int]],
                     pitch: tuple[float, float], fps: float, game: str
                     ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Score every D run of the first pass with the bundle's models and thresholds. Runs below
    their threshold get P(dead) := VETO_P (team split kept). Returns the new matrix for the
    second pass and a few counts."""
    from threadpoolctl import threadpool_limits

    spec, heads = bundle["spec"], bundle["heads"]
    X, names = features(tracking, raw_players, e4[:, D], blocks, pitch, fps)
    for name, head in heads.items():
        if names != list(head["names"]):
            raise ValueError(f"model {name}: feature names differ from its features")
    n = len(e4)
    graded = np.zeros(n, dtype=bool)
    for _h, a, b in blocks:
        graded[a:b] = True
    use_map = game == spec["stretch_map"]["game"] and n == spec["stretch_map"]["n_frames"]
    stretch = np.full(n, -1, dtype=np.int64)
    if use_map:
        for k, s in enumerate(spec["stretches"]):
            stretch[s["start"]:s["stop"]] = k
    p = np.full(n, np.nan)
    with threadpool_limits(limits=1):
        for k, s in enumerate(spec["stretches"]):
            m = graded & (stretch == k)
            if m.any():
                p[m] = heads[s["head"]]["clf"].predict_proba(X[m])[:, 1]
        m = graded & (stretch < 0)
        if m.any():
            p[m] = heads[spec["outside"]["head"]]["clf"].predict_proba(X[m])[:, 1]
    runs = d_runs(codes, blocks)
    n_str = len(spec["stretches"])
    vetoed: list[tuple[int, int]] = []
    margins: list[float] = []
    by: dict[str, int] = {}
    for a, b in runs:
        ins = stretch[a:b] >= 0
        if ins.any():
            k = int(np.argmax(np.bincount(stretch[a:b][ins], minlength=n_str)))
            v, tau, who = (float(np.mean(p[a:b][ins])), float(spec["stretches"][k]["tau"]),
                           str(spec["stretches"][k]["name"]))
        else:
            v, tau, who = float(np.mean(p[a:b])), float(spec["outside"]["tau"]), "outside"
        margins.append(abs(v - tau))
        if v < tau:
            vetoed.append((a, b))
            by[who] = by.get(who, 0) + 1
    mask = np.zeros(n, dtype=bool)
    for a, b in vetoed:
        mask[a:b] = True
    out = _with_pdead(e4, np.full(n, VETO_P), mask) if vetoed else e4.copy()
    info = {"stretch_map": use_map, "runs_pass1": len(runs), "vetoed": [[a, b] for a, b in vetoed],
            "vetoed_by": by, "vetoed_seconds": round(sum(b - a for a, b in vetoed) / fps, 1),
            "min_margin": round(min(margins, default=float("nan")), 6)}
    return out, info
