"""Bridge to the pinned official competition repository.

The validator and the game constants are imported from third_party/more-than-a-hack-2027
unchanged (scripts/fetch_official.sh fetches it at the pinned commit).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

PINNED_COMMIT = "f8f2629ff1e0985f81d7e59d8413678514ee1530"
REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(
    os.environ.get("MTAH_REPO_DIR", REPO_ROOT / "third_party/more-than-a-hack-2027"))


class OfficialRepoMissing(RuntimeError):
    pass


def _ensure_path() -> None:
    if not (OFFICIAL_ROOT / "games.py").exists():
        raise OfficialRepoMissing(
            f"official repo not found at {OFFICIAL_ROOT}; run scripts/fetch_official.sh"
        )
    for path in (OFFICIAL_ROOT, OFFICIAL_ROOT / "evaluation", OFFICIAL_ROOT / "starter"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


@cache
def games() -> ModuleType:
    _ensure_path()
    return importlib.import_module("games")


@cache
def starter_baseline() -> ModuleType:
    _ensure_path()
    return importlib.import_module("baseline")


@cache
def starter_module(name: str) -> ModuleType:
    _ensure_path()
    return importlib.import_module(name)


def kickoff_frames() -> dict[int, int]:
    """Video frame of each half's kickoff, as published in the official repo."""
    published = starter_module("load_smart_tagging").KICKOFF_VIDEO_FRAMES
    return {1: int(published["1st Half"]), 2: int(published["2nd Half"])}


def n_video_frames(game: str = "BAR-ATH") -> int:
    return int(games().GAMES[game]["n_video_frames"])


def fps(game: str = "BAR-ATH") -> int:
    return int(games().GAMES[game]["fps"])


def pitch_metres(game: str = "BAR-ATH") -> tuple[float, float]:
    length, width = games().GAMES[game]["pitch"]
    return float(length), float(width)


def half_blocks(game: str = "BAR-ATH") -> list[tuple[int, int, int]]:
    """[(half, start, end_exclusive)] graded video-frame blocks, half-open."""
    return [(int(h), int(first), int(last) + 1)
            for h, first, last in games().half_video_blocks(game)]


def validate_file(path: Path) -> tuple[bool, str]:
    """Run the official validator script on the exact file."""
    script = OFFICIAL_ROOT / "evaluation/validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()
