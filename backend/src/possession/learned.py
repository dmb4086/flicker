"""The frame model: gradient boosting over windowed tracking features.

For every frame it gives the probability of home, away and dead ball. Prediction is pinned to one
thread, so the same inputs give the same probabilities bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from threadpoolctl import threadpool_limits

from possession.deadfeat import dead_context_features
from possession.features_window import FEATURE_VERSION, window_features
from possession.labels import N_STATES
from possession.motion import motion_features
from possession.tracking import Tracking

FloatArray = npt.NDArray[np.float64]

# Ball-kinematic columns that are replaced by their rank within the match (`quantile_ball`).
QUANTILE_COLUMNS = (
    "ball_speed_prev", "ball_speed_next", "ball_dv", "ball_dspeed", "ball_accel",
    "ball_dv_max~12", "ball_dv_max~25", "ball_dv_max~50",
    "ball_accel_max~12", "ball_accel_max~25", "ball_accel_max~50",
    "ball_dv~12", "ball_dv~25", "ball_dv~50", "ball_speed~12", "ball_speed~25", "ball_speed~50",
    "gap_speed_ms")


@dataclass
class FrameModel:
    clf: Any
    feature_names: list[str]
    motion: bool
    dead: bool
    quantile: bool

    def predict_proba(self, features: npt.NDArray[Any]) -> FloatArray:
        with threadpool_limits(limits=1):
            proba: FloatArray = self.clf.predict_proba(features)
        if proba.shape[1] != N_STATES:
            raise ValueError("the model does not have all three classes")
        return proba


def all_features(tracking: Tracking, blocks: list[tuple[int, int, int]],
                 pitch: tuple[float, float], fps: float, motion: bool, dead: bool = False
                 ) -> tuple[FloatArray, list[str]]:
    """The window features, plus the ball-free motion features when `motion` and the dead-ball
    context features when `dead`."""
    feats, names = window_features(tracking, blocks, pitch, fps)
    if motion:
        extra, extra_names = motion_features(tracking, blocks, pitch, fps)
        feats, names = np.hstack([feats, extra]), names + [f"motion_{n}" for n in extra_names]
    if dead:
        extra, extra_names = dead_context_features(tracking, blocks, pitch, fps)
        feats, names = np.hstack([feats, extra]), names + [f"dead_{n}" for n in extra_names]
    return feats, names


def quantile_ball(feats: npt.NDArray[Any], names: list[str], blocks: list[tuple[int, int, int]]
                  ) -> npt.NDArray[Any]:
    """A copy of `feats` with each `QUANTILE_COLUMNS` column, over the frames inside the half
    `blocks`, replaced by its empirical CDF there: (average rank - 0.5) / n of the finite values,
    in (0, 1). NaN stays NaN; frames outside the blocks are left as they are."""
    from scipy.stats import rankdata

    out = feats.copy()
    inside = np.zeros(len(feats), dtype=bool)
    for _half, start, stop in blocks:
        inside[start:stop] = True
    rows = np.flatnonzero(inside)
    for name in QUANTILE_COLUMNS:
        c = names.index(name)
        v = feats[rows, c].astype(np.float64)
        ok = np.isfinite(v)
        if ok.any():
            v[ok] = (rankdata(v[ok], method="average") - 0.5) / ok.sum()
        out[rows, c] = v.astype(feats.dtype)
    return out


def load_model(path: Path) -> FrameModel:
    payload = joblib.load(path)
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError("the model was built with a different feature version")
    return FrameModel(payload["clf"], payload["feature_names"], bool(payload["motion_features"]),
                      bool(payload["dead_features"]), bool(payload["ball_quantile"]))


def learned_scores(model: FrameModel, tracking: Tracking, blocks: list[tuple[int, int, int]],
                   pitch: tuple[float, float], fps: float) -> FloatArray:
    """(n_frames, 3) H/A/D probabilities; uniform outside the half blocks."""
    feats, names = all_features(tracking, blocks, pitch, fps, model.motion, model.dead)
    if names != model.feature_names:
        raise ValueError("feature names differ from the model's features")
    if model.quantile:
        feats = quantile_ball(feats, names, blocks)
    scores = np.full((tracking.n_frames, N_STATES), 1 / N_STATES)
    inside = np.zeros(tracking.n_frames, dtype=bool)
    for _half, start, stop in blocks:
        inside[start:stop] = True
    scores[inside] = model.predict_proba(feats[inside].astype(np.float32))
    return scores
