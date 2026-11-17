"""Dead-ball context features that do not equate "ball missing" with "dead".

- where the ball was last seen and where it is next seen (distance to the touchline and goal
  line, signed: negative beyond it);
- the observed ball's stillness: how long it has been still, and how long it stays still from
  the frame it is next seen;
- kicks: seconds since the last and until the next fast observed ball;
- longer context of ball presence and the ball beyond the lines.

Observed positions only, per half; NaN where undefined.
"""

from __future__ import annotations

import numpy as np

from possession.features_window import _window_max, _window_mean
from possession.tracking import FloatArray, Tracking

KICK_MS = 6.0  # observed ball speed over 0.2 s that counts as a kick or pass
STILL_MS = 0.5  # metres moved over 1 s below which the ball is still
CAP_S = 60.0
DEAD_NAMES = ["last_touch_m", "last_goal_m", "next_touch_m", "next_goal_m", "still_run_s",
              "next_still_s", "since_kick_s", "until_kick_s", "present~125", "present~250",
              "beyond~50"]


def _run_from(mask: np.ndarray) -> np.ndarray:
    """For each frame, how many consecutive True frames start there (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int64)
    run = 0
    for i in range(len(mask) - 1, -1, -1):
        run = run + 1 if mask[i] else 0
        out[i] = run
    return out


def _run_to(mask: np.ndarray) -> np.ndarray:
    """For each frame, how many consecutive True frames end there (0 where False)."""
    return _run_from(mask[::-1])[::-1]


def _block(ball_m: FloatArray, pitch: tuple[float, float], fps: float) -> FloatArray:
    n = len(ball_m)
    idx = np.arange(n)
    present = np.isfinite(ball_m).all(axis=1)
    out = np.full((n, len(DEAD_NAMES)), np.nan)
    if not present.any():
        return out
    touch = np.minimum(ball_m[:, 1], pitch[1] - ball_m[:, 1])
    goal = np.minimum(ball_m[:, 0], pitch[0] - ball_m[:, 0])
    last = np.maximum.accumulate(np.where(present, idx, -1))
    nxt = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < n
    li, ni = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
    out[:, 0] = np.where(has_last, touch[li], np.nan)
    out[:, 1] = np.where(has_last, goal[li], np.nan)
    out[:, 2] = np.where(has_next, touch[ni], np.nan)
    out[:, 3] = np.where(has_next, goal[ni], np.nan)
    k1 = int(round(fps))
    move = np.full(n, np.nan)
    move[k1:] = np.sqrt(((ball_m[k1:] - ball_m[:-k1]) ** 2).sum(axis=1))
    with np.errstate(invalid="ignore"):
        still = move < STILL_MS
    out[:, 4] = np.where(present, _run_to(still) / fps, np.nan)
    ahead = _run_from(still) / fps
    out[:, 5] = np.where(has_next, ahead[ni], np.nan)
    k = max(1, int(round(0.2 * fps)))
    speed = np.full(n, np.nan)
    speed[k:] = np.sqrt(((ball_m[k:] - ball_m[:-k]) ** 2).sum(axis=1)) * fps / k
    with np.errstate(invalid="ignore"):
        kick = speed > KICK_MS
    last_k = np.maximum.accumulate(np.where(kick, idx, -10**9))
    next_k = np.minimum.accumulate(np.where(kick, idx, 10**9)[::-1])[::-1]
    out[:, 6] = np.minimum((idx - last_k) / fps, CAP_S)
    out[:, 7] = np.minimum((next_k - idx) / fps, CAP_S)
    pres = present.astype(np.float64)
    out[:, 8] = _window_mean(pres, -125, 125)
    out[:, 9] = _window_mean(pres, -250, 250)
    with np.errstate(invalid="ignore"):
        beyond = np.where(present, (np.minimum(touch, goal) < -1.0) * 1.0, np.nan)
    out[:, 10] = _window_max(beyond, -50, 50)
    return out


def dead_context_features(tracking: Tracking, blocks: list[tuple[int, int, int]],
                          pitch: tuple[float, float], fps: float
                          ) -> tuple[FloatArray, list[str]]:
    """(n_frames, len(DEAD_NAMES)); NaN outside the half blocks."""
    scale = np.array(pitch, dtype=np.float64)
    ball_m = tracking.ball.astype(np.float64) * scale
    out = np.full((tracking.n_frames, len(DEAD_NAMES)), np.nan)
    for _h, a, b in blocks:
        out[a:b] = _block(ball_m[a:b], pitch, fps)
    return out, list(DEAD_NAMES)
