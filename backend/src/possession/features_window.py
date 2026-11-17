"""Per-frame windowed features for the frame model.

Inputs are what the automatic tracking has: player x/y per slot, team per slot, ball x/y. No ball
height, no persistent player ids, no events. Future frames are used. Frames are processed per
half so nothing leaks across halftime. Missing values stay NaN.

The ball is linearly interpolated across missing-ball gaps, and distances are measured to that
interpolated position, so a frame inside a gap still carries a team signal. Gap descriptors tell
the model how much to trust it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from possession.tracking import AWAY_TEAM, HOME_TEAM, FloatArray, Tracking, run_lengths

FEATURE_VERSION = 2
HALF_WIDTHS = (12, 25, 50)  # centred windows of +-0.5 s, +-1 s, +-2 s at 25 fps
SHIFTS = (-50, -25, 25, 50)
TOUCH_LAG = 5  # 0.2 s


def _shift(x: FloatArray, k: int) -> FloatArray:
    """x[t + k], NaN outside the block."""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _window_mean(x: FloatArray, lo: int, hi: int) -> FloatArray:
    """NaN-aware mean of x[t + lo .. t + hi] (inclusive), clipped to the block."""
    n = len(x)
    finite = np.isfinite(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    ccount = np.concatenate(([0], np.cumsum(finite)))
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    count = ccount[b] - ccount[a]
    total = csum[b] - csum[a]
    out = np.full(n, np.nan)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out


def _window_max(x: FloatArray, lo: int, hi: int) -> FloatArray:
    """NaN-aware max of x[t + lo .. t + hi], clipped to the block."""
    pad_lo, pad_hi = -lo, hi
    padded = np.concatenate([np.full(pad_lo, np.nan), x, np.full(pad_hi, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, pad_lo + pad_hi + 1)
    with np.errstate(all="ignore"):
        out: FloatArray = np.where(np.isfinite(view).any(axis=1),
                                   np.nanmax(np.where(np.isfinite(view), view, -np.inf), axis=1),
                                   np.nan)
    return out


def _team_nearest(dist: FloatArray, mask: npt.NDArray[np.bool_]) -> tuple[FloatArray, FloatArray,
                                                                          FloatArray, FloatArray,
                                                                          FloatArray, FloatArray,
                                                                          npt.NDArray[np.intp]]:
    """Nearest and second-nearest distance, counts within 2 m and 5 m, index of nearest slot."""
    d = dist[:, mask]
    filled = np.where(np.isfinite(d), d, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    d1 = np.take_along_axis(filled, order[:, :1], axis=1)[:, 0]
    d2 = (np.take_along_axis(filled, order[:, 1:2], axis=1)[:, 0] if d.shape[1] > 1
          else np.full(len(d), np.inf))
    d1[~np.isfinite(d1)] = np.nan
    d2[~np.isfinite(d2)] = np.nan
    n05 = (filled < 0.5).sum(axis=1).astype(np.float64)
    n1 = (filled < 1.0).sum(axis=1).astype(np.float64)
    n2 = (filled < 2.0).sum(axis=1).astype(np.float64)
    n5 = (filled < 5.0).sum(axis=1).astype(np.float64)
    slot = np.flatnonzero(mask)[order[:, 0]]
    return d1, d2, n2, n5, n05, n1, slot


def _block_features(players_m: FloatArray, ball_m: FloatArray, team_of_slot: npt.NDArray[np.int8],
                    pitch: tuple[float, float], fps: float
                    ) -> tuple[FloatArray, list[str]]:
    n = len(ball_m)
    present = np.all(np.isfinite(ball_m), axis=1)
    idx = np.arange(n)
    names: list[str] = []
    cols: list[FloatArray] = []

    def add(name: str, values: FloatArray) -> None:
        names.append(name)
        cols.append(np.asarray(values, dtype=np.float64))

    if not present.any():
        raise ValueError("block has no visible ball frame")
    seen_idx = idx[present]
    filled = np.column_stack([np.interp(idx, seen_idx, ball_m[present, 0]),
                              np.interp(idx, seen_idx, ball_m[present, 1])])
    last = np.maximum.accumulate(np.where(present, idx, -1))
    nxt = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < n
    since = np.where(has_last, (idx - last) / fps, np.nan)
    until = np.where(has_next, (nxt - idx) / fps, np.nan)
    gap_s = run_lengths(~present) / fps

    delta = players_m - filled[:, None, :]
    dist = np.sqrt((delta**2).sum(axis=2))
    home = team_of_slot == HOME_TEAM
    away = team_of_slot == AWAY_TEAM
    dh1, dh2, nh2, nh5, nh05, nh1, sh = _team_nearest(dist, home)
    da1, da2, na2, na5, na05, na1, sa = _team_nearest(dist, away)
    diff1 = da1 - dh1  # > 0: home closer
    diff2 = da2 - dh2
    dmin = np.fmin(dh1, da1)

    # Nearest player's own speed (m/s over 0.2 s): is the closest man running onto the ball?
    k = TOUCH_LAG
    vel = (_shift(players_m.reshape(n, -1), k) - _shift(players_m.reshape(n, -1), -k))
    pspeed = np.sqrt((vel.reshape(n, -1, 2) ** 2).sum(axis=2)) * fps / (2 * k)
    rows = np.arange(n)
    sph = np.where(np.isfinite(dh1), pspeed[rows, sh], np.nan)
    spa = np.where(np.isfinite(da1), pspeed[rows, sa], np.nan)

    add("present", present.astype(np.float64))
    add("since_seen_s", since)
    add("until_seen_s", until)
    add("gap_s", gap_s)
    add("dh1", dh1)
    add("da1", da1)
    add("dh2", dh2)
    add("da2", da2)
    add("diff1", diff1)
    add("diff2", diff2)
    add("dmin", dmin)
    add("nh2", nh2)
    add("na2", na2)
    add("nh5", nh5)
    add("na5", na5)
    add("n2_diff", nh2 - na2)
    add("n5_diff", nh5 - na5)
    add("speed_h1", sph)
    add("speed_a1", spa)
    add("nh05", nh05)
    add("na05", na05)
    add("nh1", nh1)
    add("na1", na1)

    # Team call at the last and next visible frames, and how the ball moved across the gap.
    d_last = np.where(has_last, diff1[np.clip(last, 0, n - 1)], np.nan)
    d_next = np.where(has_next, diff1[np.clip(nxt, 0, n - 1)], np.nan)
    add("diff1_last_seen", d_last)
    add("diff1_next_seen", d_next)
    span = np.sqrt(((filled[np.clip(nxt, 0, n - 1)] - filled[np.clip(last, 0, n - 1)]) ** 2)
                   .sum(axis=1))
    span = np.where(has_last & has_next, span, np.nan)
    add("gap_span_m", np.where(gap_s > 0, span, 0.0))
    add("gap_speed_ms", np.where(gap_s > 0, span / np.maximum(
        (np.clip(nxt, 0, n - 1) - np.clip(last, 0, n - 1)) / fps, 1 / fps), np.nan))

    # Ball kinematics from observed positions only (never from interpolated ones).
    p_now = np.where(present[:, None], ball_m, np.nan)
    p_prev, p_next = _shift(p_now, -k), _shift(p_now, k)
    v_prev = (p_now - p_prev) * fps / k
    v_next = (p_next - p_now) * fps / k
    s_prev = np.linalg.norm(v_prev, axis=1)
    s_next = np.linalg.norm(v_next, axis=1)
    dv = np.linalg.norm(v_next - v_prev, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (v_prev * v_next).sum(axis=1) / (s_prev * s_next)
    add("ball_speed_prev", s_prev)
    add("ball_speed_next", s_next)
    add("ball_dv", dv)
    add("ball_dspeed", s_next - s_prev)
    add("ball_cos_turn", np.where(np.isfinite(cosang), np.clip(cosang, -1, 1), np.nan))
    length, width = pitch
    bx, by = filled[:, 0], filled[:, 1]
    # Control gates (Link & Hoernig, Vidal-Codina): strongest ball acceleration / turn nearby.
    accel = np.abs(s_next - s_prev) * fps / k
    add("ball_accel", accel)
    for h in (12, 25, 50):
        add(f"ball_dv_max~{h}", _window_max(dv, -h, h))
        add(f"ball_accel_max~{h}", _window_max(accel, -h, h))
    add("ball_edge_x", np.minimum(bx, length - bx))
    add("ball_edge_y", np.minimum(by, width - by))
    add("ball_centre_dist", np.hypot(bx - length / 2, by - width / 2))

    # Closing speed: did the nearest distance fall or rise over +-1 s?
    for label, series in (("dh1", dh1), ("da1", da1), ("diff1", diff1), ("dmin", dmin)):
        for s in SHIFTS:
            add(f"{label}@{s}", _shift(series, s))

    smooth_inputs = {"diff1": diff1, "diff2": diff2, "dh1": dh1, "da1": da1, "dmin": dmin,
                     "n2_diff": nh2 - na2, "n5_diff": nh5 - na5, "present": present * 1.0,
                     "ball_dv": dv, "ball_speed": s_next, "speed_h1": sph, "speed_a1": spa}
    for label, series in smooth_inputs.items():
        for h in HALF_WIDTHS:
            add(f"{label}~{h}", _window_mean(series, -h, h))
    for label in ("diff1", "dmin", "n2_diff"):
        for h in (25, 50):
            add(f"{label}<{h}", _window_mean(smooth_inputs[label], -h, 0))
            add(f"{label}>{h}", _window_mean(smooth_inputs[label], 0, h))
    return np.column_stack(cols), names


def window_features(tracking: Tracking, blocks: list[tuple[int, int, int]],
                    pitch: tuple[float, float], fps: float = 25.0
                    ) -> tuple[FloatArray, list[str]]:
    """(n_frames, n_features) matrix, NaN outside the half blocks, plus the feature names."""
    scale = np.array(pitch, dtype=np.float64)
    players_m = tracking.players.astype(np.float64) * scale
    ball_m = tracking.ball.astype(np.float64) * scale
    out: FloatArray | None = None
    names: list[str] = []
    for _half, start, stop in blocks:
        block, names = _block_features(players_m[start:stop], ball_m[start:stop],
                                       tracking.team_of_slot, pitch, fps)
        if out is None:
            out = np.full((tracking.n_frames, block.shape[1]), np.nan)
        out[start:stop] = block
    if out is None:
        raise ValueError("no blocks")
    return out, names


__all__ = ["FEATURE_VERSION", "AWAY_TEAM", "window_features"]
