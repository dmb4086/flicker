"""Seconds-of-context features on how the players move around a frame.

- share of visible players that are still, slow or running, from each player's displacement
  over 1 s, and those shares over longer windows, past and future apart;
- each team's centroid speed and the change of the centroid's x velocity across the frame;
- the ball seen beyond a line: seconds since and until, and whether it happens within 10 s.

The still-share feature follows an idea listed in another team's public write-up (no code of
theirs). Observed positions only, per half; NaN outside the halves.
"""

from __future__ import annotations

import numpy as np

from possession.features_window import _window_max, _window_mean
from possession.tracking import HOME_TEAM, FloatArray, Tracking

STILL_MS = 1.0
SLOW_MS = 2.0
FAST_MS = 4.0
MAX_SPEED_MS = 12.0  # faster displacements are identity swaps, not running
BEYOND_M = 1.0
CAP_S = 60.0
CTX_NAMES = [
    "still_share", "slow_share", "fast_share",
    "still~1s", "still~3s", "still~10s", "still_past3s", "still_next3s", "still_min~5s",
    "fast~3s", "fast_past3s", "fast_next3s", "fast_max~5s",
    "cent_speed_h", "cent_speed_a", "cent_flip_h", "cent_flip_a",
    "since_beyond_s", "until_beyond_s", "beyond~10s",
]


def _shift(x: FloatArray, k: int) -> FloatArray:
    """x[t + k], NaN past the ends."""
    out = np.full_like(x, np.nan)
    if k >= 0:
        out[:len(x) - k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def _share(m: np.ndarray, ok: np.ndarray, n_ok: np.ndarray) -> FloatArray:
    out: FloatArray = np.where(n_ok > 0, (m & ok).sum(axis=1) / np.maximum(n_ok, 1), np.nan)
    return out


def _block(players_m: FloatArray, home: np.ndarray, ball_m: FloatArray, pitch: tuple[float, float],
           fps: float) -> FloatArray:
    n = len(players_m)
    out = np.full((n, len(CTX_NAMES)), np.nan)

    def s(sec: float) -> int:
        return max(1, int(round(sec * fps)))

    h = max(1, int(round(0.5 * fps)))
    ahead = np.full_like(players_m, np.nan)
    behind = np.full_like(players_m, np.nan)
    ahead[:n - h] = players_m[h:]
    behind[h:] = players_m[:n - h]
    speed = np.sqrt(((ahead - behind) ** 2).sum(axis=2)) / (2 * h / fps)
    with np.errstate(invalid="ignore"):
        speed = np.where(speed > MAX_SPEED_MS, np.nan, speed)
        ok = np.isfinite(speed)
        n_ok = ok.sum(axis=1)
        still = _share(speed < STILL_MS, ok, n_ok)
        slow = _share(speed < SLOW_MS, ok, n_ok)
        fast = _share(speed > FAST_MS, ok, n_ok)
    out[:, 0], out[:, 1], out[:, 2] = still, slow, fast
    out[:, 3] = _window_mean(still, -s(1), s(1))
    out[:, 4] = _window_mean(still, -s(3), s(3))
    out[:, 5] = _window_mean(still, -s(10), s(10))
    out[:, 6] = _window_mean(still, -s(3), -1)
    out[:, 7] = _window_mean(still, 0, s(3) - 1)
    out[:, 8] = -_window_max(-still, -s(5), s(5))
    out[:, 9] = _window_mean(fast, -s(3), s(3))
    out[:, 10] = _window_mean(fast, -s(3), -1)
    out[:, 11] = _window_mean(fast, 0, s(3) - 1)
    out[:, 12] = _window_max(fast, -s(5), s(5))
    for c, team in ((13, home), (14, ~home)):
        with np.errstate(all="ignore"):
            cent = np.nanmean(players_m[:, team], axis=1) if team.any() else np.full((n, 2), np.nan)
        vel = (_shift(cent, h) - _shift(cent, -h)) / (2 * h / fps)
        vel = np.column_stack([_window_mean(vel[:, 0], -h, h), _window_mean(vel[:, 1], -h, h)])
        out[:, c] = np.sqrt((vel ** 2).sum(axis=1))
        out[:, c + 2] = np.abs(_shift(vel[:, 0], s(2)) - _shift(vel[:, 0], -s(2)))
    present = np.isfinite(ball_m).all(axis=1)
    with np.errstate(invalid="ignore"):
        touch = np.minimum(ball_m[:, 1], pitch[1] - ball_m[:, 1])
        goal = np.minimum(ball_m[:, 0], pitch[0] - ball_m[:, 0])
        beyond = present & (np.minimum(touch, goal) < -BEYOND_M)
    idx = np.arange(n)
    last_b = np.maximum.accumulate(np.where(beyond, idx, -10**9))
    next_b = np.minimum.accumulate(np.where(beyond, idx, 10**9)[::-1])[::-1]
    out[:, 17] = np.minimum((idx - last_b) / fps, CAP_S)
    out[:, 18] = np.minimum((next_b - idx) / fps, CAP_S)
    out[:, 19] = _window_max(beyond.astype(np.float64), -s(10), s(10))
    return out


def ctx_features(tracking: Tracking, blocks: list[tuple[int, int, int]],
                 pitch: tuple[float, float], fps: float) -> tuple[FloatArray, list[str]]:
    """(n_frames, len(CTX_NAMES)); NaN outside the half blocks."""
    scale = np.array(pitch, dtype=np.float64)
    players_m = tracking.players.astype(np.float64) * scale
    ball_m = tracking.ball.astype(np.float64) * scale
    home = tracking.team_of_slot == HOME_TEAM
    out = np.full((tracking.n_frames, len(CTX_NAMES)), np.nan)
    for _h, a, b in blocks:
        out[a:b] = _block(players_m[a:b], home, ball_m[a:b], pitch, fps)
    return out, [f"ctx_{n}" for n in CTX_NAMES]
