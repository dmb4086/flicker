"""Player-motion features that never look at the ball.

Players walk when play is stopped and run when it is live. These features describe how the
players move over the seconds around a frame. Computed per half; NaN outside the halves.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from possession.tracking import Tracking

FloatArray = npt.NDArray[np.float64]
MAX_SPEED_MS = 12.0  # faster steps are identity swaps or slot reuse, not running


def motion_features(tracking: Tracking, blocks: list[tuple[int, int, int]],
                    pitch: tuple[float, float], fps: float) -> tuple[FloatArray, list[str]]:
    """(n_frames, k) player-only features; NaN outside the half blocks. No ball, no counts of
    visible players (the real camera loses players when it loses the ball)."""
    scale = np.array(pitch, dtype=np.float64)
    names = ["v1", "v3", "v8", "p90_3", "dv3", "vmin10", "vmax10", "sx3", "sy3"]
    out = np.full((tracking.n_frames, len(names)), np.nan)
    for _h, a, b in blocks:
        p = tracking.players[a:b].astype(np.float64) * scale
        step = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=2)) * fps
        step = np.vstack([step[:1], step])
        with np.errstate(invalid="ignore"):
            step = np.where(step > MAX_SPEED_MS, np.nan, step)
        ok = np.isfinite(step)
        n_ok = ok.sum(axis=1)
        mean = np.where(n_ok > 0, np.nansum(step, axis=1) / np.maximum(n_ok, 1), np.nan)
        with np.errstate(all="ignore"):
            p90 = np.nanpercentile(np.where(ok, step, np.nan), 90, axis=1) if ok.any() else mean
            sx = np.nanstd(p[:, :, 0], axis=1)
            sy = np.nanstd(p[:, :, 1], axis=1)
        mean, p90, sx, sy = (_fill(x) for x in (mean, p90, sx, sy))

        def smooth(x: FloatArray, s: float) -> FloatArray:
            sm: FloatArray = uniform_filter1d(x, max(1, int(round(s * fps))), mode="nearest")
            return sm

        v1, v3 = smooth(mean, 1), smooth(mean, 3)
        k = int(round(3 * fps))
        ahead = np.concatenate([v3[k:], np.repeat(v3[-1:], k)])
        behind = np.concatenate([np.repeat(v3[:1], k), v3[:-k]])
        w = int(round(10 * fps))
        out[a:b] = np.column_stack([
            v1, v3, smooth(mean, 8), smooth(p90, 3), ahead - behind,
            minimum_filter1d(v1, w, mode="nearest"), maximum_filter1d(v1, w, mode="nearest"),
            smooth(sx, 3), smooth(sy, 3)])
    return out, names


def _fill(x: FloatArray) -> FloatArray:
    """Carry the last finite value forward (then backward) over NaN frames."""
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y: FloatArray = x[idx]
    first = int(np.argmax(ok))
    y[:first] = x[first]
    return y
