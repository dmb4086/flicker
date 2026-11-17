"""Label codes, shared with the official scorer: H=0, A=1, D=2, X=3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
N_STATES = 3  # the decoder's states are H, A, D; X is an output policy, not a state

Codes = npt.NDArray[np.int8]


def to_letters(codes: Codes) -> npt.NDArray[np.str_]:
    return LETTERS[codes]


def write_submission(path: Path, codes: Codes) -> None:
    """frame,label CSV with stable row order, UTF-8 and '\\n' line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    letters = to_letters(codes)
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{label}\n" for i, label in enumerate(letters))
    tmp.replace(path)


def swap_teams(codes: Codes) -> Codes:
    out = codes.copy()
    out[codes == H] = A
    out[codes == A] = H
    return out
