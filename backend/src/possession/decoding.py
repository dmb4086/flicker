"""Three-state (H, A, D) Viterbi decoder on the canonical frame grid.

Emissions are per-frame log-probabilities. Neighbouring frames are not independent observations
(25-60 frames often share one underlying piece of evidence), so emission costs are scaled by
`evidence_per_second / fps`. That makes transition costs mean the same thing at 25 fps and 60 fps:
a switch is worth taking only when the evidence for it, integrated over time, exceeds its cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from possession.labels import N_STATES, A, Codes, D, H, X

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TransitionCosts:
    """Costs in nats. Indexed [from_state, to_state] by matrix()."""

    team_switch: float = 6.0  # H <-> A while the ball stays alive
    to_dead: float = 4.0  # H/A -> D
    from_dead: float = 4.0  # D -> H/A

    def matrix(self) -> FloatArray:
        m = np.zeros((N_STATES, N_STATES))
        m[H, A] = m[A, H] = self.team_switch
        m[H, D] = m[A, D] = self.to_dead
        m[D, H] = m[D, A] = self.from_dead
        return m


def emission_costs(
    probs: FloatArray, fps: float, evidence_per_second: float, floor: float = 1e-6
) -> FloatArray:
    """Negative log evidence, scaled so one second counts `evidence_per_second` times."""
    if probs.ndim != 2 or probs.shape[1] != N_STATES:
        raise ValueError(f"expected (n, {N_STATES}) probabilities, got {probs.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    clipped = np.maximum(probs, floor)
    normalised = clipped / clipped.sum(axis=1, keepdims=True)
    scale = evidence_per_second / fps
    costs: FloatArray = -np.log(normalised) * scale
    return costs


def viterbi(costs: FloatArray, transitions: FloatArray, initial: FloatArray | None = None
            ) -> npt.NDArray[np.int8]:
    """Minimum-cost state path. Ties resolve to the lowest state index (H, then A, then D)."""
    n = len(costs)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    start = np.zeros(N_STATES) if initial is None else initial
    back = np.zeros((n, N_STATES), dtype=np.int8)
    total = start + costs[0]
    for t in range(1, n):
        # candidate[prev, cur] = total[prev] + transition[prev, cur]
        candidate = total[:, None] + transitions
        best_prev = np.argmin(candidate, axis=0)  # argmin returns the first minimum
        back[t] = best_prev
        total = candidate[best_prev, np.arange(N_STATES)] + costs[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# The decoder's fixed costs. A change of team costs 2 units and entering or leaving a dead ball
# costs 8, while evidence counts 3 units per second.
EVIDENCE_PER_SECOND = 3.0
TEAM_SWITCH = 2.0
TO_DEAD = 8.0
FROM_DEAD = 8.0
PROBABILITY_FLOOR = 1e-6
FORCED = 1e6  # cost that rules a state out; finite so ties and sums stay well defined
COST_DECIMALS = 9  # quantise costs so last-ulp platform differences cannot flip a decision


def decode_halves(scores: FloatArray, blocks: list[tuple[int, int, int]],
                  kickoffs: dict[int, int], fps: float) -> Codes:
    """Viterbi per graded half; X outside. Frames before a half's kickoff are forced to D, and
    each half starts in D. Decoding restarts each half; nothing leaks across halftime."""
    costs = emission_costs(scores, fps, EVIDENCE_PER_SECOND, PROBABILITY_FLOOR)
    costs = np.round(costs, COST_DECIMALS)
    transitions = TransitionCosts(TEAM_SWITCH, TO_DEAD, FROM_DEAD).matrix()
    initial = np.array([FROM_DEAD, FROM_DEAD, 0.0])
    codes = np.full(len(scores), X, dtype=np.int8)
    for half, start, stop in blocks:
        window = costs[start:stop].copy()
        if half in kickoffs:
            pre = max(0, min(kickoffs[half], stop) - start)
            window[:pre, [H, A]] = FORCED
            window[:pre, D] = 0.0
        codes[start:stop] = viterbi(window, transitions, initial)
    return codes
