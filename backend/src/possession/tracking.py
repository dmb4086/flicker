"""Automatic tracking data (ATD): parse once into arrays and cache.

Raw line format: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed" with NaN for a
missing observation. frame_id is one-based; arrays here are indexed by zero-based video frame.
Missing values stay NaN all the way through, never zero, which is a valid coordinate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
HOME_TEAM, AWAY_TEAM = 0, 1
# Bump whenever parse_atd_lines changes what it extracts, so stale caches are never reused.
PARSER_VERSION = 2


@dataclass(frozen=True)
class Tracking:
    players: npt.NDArray[np.float32]  # (n_frames, n_slots, 2) normalised x, y; NaN if absent
    ball: npt.NDArray[np.float32]  # (n_frames, 2) normalised x, y; NaN if absent
    ball_speed: npt.NDArray[np.float32]  # (n_frames,) supplied ball speed; NaN if absent
    team_of_slot: npt.NDArray[np.int8]  # (n_slots,) HOME_TEAM or AWAY_TEAM
    skipped_lines: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])


@dataclass(frozen=True)
class ParsedATD:
    players: npt.NDArray[np.float32]
    ball: npt.NDArray[np.float32]
    ball_speed: npt.NDArray[np.float32]
    skipped_lines: int


def parse_atd_lines(lines: list[str], n_frames: int, n_slots: int) -> ParsedATD:
    """Parse raw ATD lines. Lines without three ':' fields are skipped and counted, as the
    official reader skips them; duplicates, out-of-range frames and wrong slot counts raise."""
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    skipped = 0
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) != 3:
            skipped += 1
            continue
        index = int(parts[0]) - 1  # one-based frame_id -> zero-based video frame
        if not 0 <= index < n_frames:
            raise ValueError(f"ATD line {line_no}: frame_id outside 1..{n_frames}")
        if seen[index]:
            raise ValueError(f"ATD line {line_no}: duplicate frame_id")
        seen[index] = True
        values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
        if values.size != n_slots * 3:
            raise ValueError(f"ATD line {line_no}: {values.size // 3} slots, expected {n_slots}")
        players[index] = values.reshape(n_slots, 3)[:, :2]
        ball_values = np.array(parts[2].split(","), dtype=np.float32)
        ball[index] = ball_values[:2]
        if ball_values.size >= 3:
            ball_speed[index] = ball_values[2]
    return ParsedATD(players, ball, ball_speed, skipped)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(raw_sha256: str) -> str:
    return f"atd-v{PARSER_VERSION}-{raw_sha256[:16]}"


def load_tracking(raw_path: Path, team_of_slot: npt.NDArray[np.int8], n_frames: int,
                  cache_dir: Path) -> Tracking:
    """Parse the raw ATD, reusing a private .npz cache keyed by content hash and parser version."""
    cache_path = cache_dir / f"{cache_key(file_sha256(raw_path))}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return Tracking(cached["players"], cached["ball"], cached["ball_speed"], team_of_slot,
                        int(cached["skipped_lines"]))
    with open(raw_path, encoding="utf-8") as handle:
        parsed = parse_atd_lines(handle.readlines(), n_frames, len(team_of_slot))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, players=parsed.players, ball=parsed.ball, ball_speed=parsed.ball_speed,
             skipped_lines=np.int64(parsed.skipped_lines))
    tmp.replace(cache_path)
    return Tracking(parsed.players, parsed.ball, parsed.ball_speed, team_of_slot,
                    parsed.skipped_lines)


def run_lengths(mask: BoolArray) -> npt.NDArray[np.int64]:
    """For each True frame, the length of the run of True it belongs to; 0 elsewhere."""
    out = np.zeros(len(mask), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        out[start:stop] = stop - start
    return out
