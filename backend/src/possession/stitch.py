"""Stitch anonymous tracking slots into at most 11 continuous tracks per team.

The automatic tracking has 46 slots for 22 players; a slot carries many short tracklets of
different people, duplicates appear, and identities swap. Per half and per team this module cuts
every slot into tracklets, suppresses duplicates, links tracklets into at most 11 tracks and
flags the goalkeeper. Gaps are filled separately (`fill_gaps`).

Deterministic, CPU only, no randomness.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from possession.tracking import AWAY_TEAM, HOME_TEAM, Tracking

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
LOG_2PI = float(np.log(2 * np.pi))
# A stitched track holds its first/last position to the ends of the half (None = no limit): with
# at most 11 tracks per team this cannot invent a twelfth player, and PathCRF keeps every frame
# (a 10 s limit left ~3 % of live frames without PathCRF output and scored lower).
HOLD_FRAMES: int | None = None


@dataclass(frozen=True)
class StitchParams:
    method: str = "lap"  # "lap": global gap closing; "greedy": event-driven frame assignment
    max_tracks: int = 11  # per team
    jump_m: float = 2.0  # one-frame displacement that ends a tracklet (50 m/s at 25 fps)
    dup_radius_m: float = 1.2  # duplicate offsets are ~0.3-1 m, plus 0.25 m jitter on each
    dup_min_s: float = 1.0  # real teammates rarely stay this close this long
    dup_cross_team: bool = False  # also merge observations of slots of different teams
    # Gap closing (lap): links an end to a later start at most max_gap_s later, gated at
    # gate_sigmas; a link between slots of different teams pays team_cost (team misfiling);
    # starting or ending a chain mid-half costs new_chain_cost (link if cost < 2x that).
    max_gap_s: float = 60.0
    gate_sigmas: float = 4.0
    team_cost: float = 3.0
    misfiled_max_s: float = 20.0  # a cross-team piece this short between two others is misfiled
    new_chain_cost: float = 8.0
    # A link that continues the same slot after a gap of at most same_slot_max_s adds
    # same_slot_cost (negative: a bonus). 0 = slots carry no identity.
    same_slot_cost: float = 0.0
    same_slot_max_s: float = 30.0
    # Track allocation: a chain may take over another chain's track only if that owner is not
    # seen again for at least this long.
    steal_after_s: float = 5.0
    sigma0_m: float = 0.7  # position uncertainty of a track right after it was lost
    sigma_speed_ms: float = 1.0  # growth of that uncertainty per second lost
    sigma_max_m: float = 30.0
    velocity_window_s: float = 0.4  # end-of-tracklet velocity estimate
    max_speed_ms: float = 8.0
    coast_s: float = 1.0  # motion prediction uses at most this much of the gap
    # Beyond the coasting part of a gap, a lost player is assumed to drift with his team's
    # centroid (players keep their place in the formation).
    formation_drift: bool = True
    keeper_prior: bool = True
    keeper_reserve: bool = True  # lap: only keeper-like chains may use a team's track 0
    keeper_cost: float = 6.0
    keeper_depth_m: float = 25.0  # outfield tracks pay the keeper cost only this close to goal


@dataclass(frozen=True)
class Stitched:
    players: npt.NDArray[np.float32]  # (n_frames, 2 * max_tracks, 2) normalised; NaN if unknown
    team_of_track: npt.NDArray[np.int8]  # (2 * max_tracks,)
    keeper_tracks: tuple[int, int]  # home keeper track, away keeper track
    confidence: npt.NDArray[np.float32]  # (n_frames, 2 * max_tracks) association confidence
    source_slot: npt.NDArray[np.int16]  # (n_frames, 2 * max_tracks) input slot used; -1 if none
    stats: dict[str, float]

    @property
    def frame_confidence(self) -> npt.NDArray[np.float32]:
        """Per frame: mean association confidence over the tracks (0 for unobserved tracks)."""
        out: npt.NDArray[np.float32] = self.confidence.mean(axis=1).astype(np.float32)
        return out

    def tracking(self, ball: Tracking, filled: bool = False,
                 blocks: list[tuple[int, int, int]] | None = None,
                 max_hold_frames: int | None = HOLD_FRAMES) -> Tracking:
        """A Tracking with the stitched players and the ball of `ball` (gaps filled if asked)."""
        players = self.players
        if filled:
            if blocks is None:
                raise ValueError("filling needs the half blocks")
            players = fill_gaps(players, blocks, max_hold_frames)
        return Tracking(players, ball.ball, ball.ball_speed, self.team_of_track,
                        ball.skipped_lines)


def _tracklets(pos: FloatArray, jump_m: float) -> IntArray:
    """(n, n_slots) tracklet id per observed frame (-1 where missing). A tracklet is a run of
    observed frames of one slot with no jump larger than `jump_m` between consecutive frames."""
    n, n_slots = pos.shape[:2]
    seen = np.isfinite(pos[:, :, 0])
    step = np.full((n, n_slots), np.inf)
    step[1:] = np.sqrt(((pos[1:] - pos[:-1]) ** 2).sum(axis=2))
    start = seen & ~(np.vstack([np.zeros((1, n_slots), bool), seen[:-1]]) & (step <= jump_m))
    order = np.cumsum(start.T.reshape(-1)).reshape(n_slots, n).T - 1  # slot-major numbering
    ids: IntArray = np.where(seen, order, -1).astype(np.int64)
    return ids


def _suppress_duplicates(pos: FloatArray, team_of_slot: npt.NDArray[np.int8], p: StitchParams,
                         fps: float) -> BoolArray:
    """Observations to drop: the younger of two observations (same team unless
    `dup_cross_team`) that stay within `dup_radius_m` of each other for at least `dup_min_s`."""
    n, n_slots = pos.shape[:2]
    seen = np.isfinite(pos[:, :, 0])
    age = np.zeros((n, n_slots), dtype=np.int64)  # frames since the slot's run started
    for s in range(n_slots):
        col = seen[:, s]
        starts = np.flatnonzero(col & ~np.concatenate(([False], col[:-1])))
        marks = np.zeros(n, dtype=np.int64)
        marks[starts] = starts
        run_start = np.maximum.accumulate(marks)
        age[:, s] = np.where(col, np.arange(n) - run_start, 0)
    drop = np.zeros((n, n_slots), dtype=bool)
    min_frames = int(round(p.dup_min_s * fps))
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            if team_of_slot[i] != team_of_slot[j] and not p.dup_cross_team:
                continue
            both = seen[:, i] & seen[:, j]
            if not both.any():
                continue
            d = np.sqrt(((pos[:, i] - pos[:, j]) ** 2).sum(axis=1))
            close = both & (np.nan_to_num(d, nan=np.inf) < p.dup_radius_m)
            if close.sum() < min_frames:
                continue
            edges = np.diff(np.concatenate(([0], close.astype(np.int8), [0])))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1),
                            strict=True):
                if b - a < min_frames:
                    continue
                younger = j if age[a, j] <= age[a, i] else i
                drop[a:b, younger] = True
    return drop


def _stitch_team_greedy(pos: FloatArray, own_goal_x: float, p: StitchParams, fps: float
                        ) -> tuple[IntArray, FloatArray, dict[str, float]]:
    """The first, simple method: associate one team's slots (one half block) with `max_tracks`
    tracks, deciding at each tracklet start among the tracks free at that moment.

    Returns (slot used per frame and track, -1 if none; confidence per frame and track; stats)."""
    n, n_slots = pos.shape[:2]
    k_tracks = p.max_tracks
    drop = _suppress_duplicates(pos, np.zeros(n_slots, dtype=np.int8), p, fps)
    pos = np.where(drop[:, :, None], np.nan, pos)
    tid = _tracklets(pos, p.jump_m)
    n_tl = int(tid.max()) + 1 if (tid >= 0).any() else 0
    tl_slot = np.zeros(n_tl, dtype=np.int64)
    tl_start = np.full(n_tl, n, dtype=np.int64)
    tl_stop = np.zeros(n_tl, dtype=np.int64)
    ts, ss = np.nonzero(tid >= 0)
    ids = tid[ts, ss]
    tl_slot[ids] = ss
    np.minimum.at(tl_start, ids, ts)
    np.maximum.at(tl_stop, ids, ts + 1)

    depth = np.abs(pos[:, :, 0] - own_goal_x)  # metres from own goal line; NaN if unseen
    with np.errstate(invalid="ignore"):
        filled_depth = np.where(np.isfinite(depth), depth, np.inf)
        order = np.sort(filled_depth, axis=1)
    deepest = np.argmin(filled_depth, axis=1)
    second = order[:, 1] if n_slots > 1 else np.full(n, np.inf)

    out_slot = np.full((n, k_tracks), -1, dtype=np.int64)
    conf = np.zeros((n, k_tracks), dtype=np.float64)
    track_of = np.full(n_tl, -1, dtype=np.int64)
    holder = np.full(k_tracks, -1, dtype=np.int64)  # tracklet currently on each track
    last_pos = np.full((k_tracks, 2), np.nan)
    last_vel = np.zeros((k_tracks, 2))
    last_t = np.full(k_tracks, -1, dtype=np.int64)
    assoc_conf = np.ones(n_tl)
    assigned_at = np.zeros(n_tl, dtype=np.int64)
    pending: set[int] = set()
    starts_at: dict[int, list[int]] = {}
    stops_at: dict[int, list[int]] = {}
    for i in range(n_tl):
        starts_at.setdefault(int(tl_start[i]), []).append(i)
        stops_at.setdefault(int(tl_stop[i]), []).append(i)
    vel_frames = max(1, int(round(p.velocity_window_s * fps)))
    n_forced_far = 0
    n_assign = 0
    for t in sorted(set(starts_at) | set(stops_at)):
        for i in stops_at.get(t, []):
            pending.discard(i)
            k = track_of[i]
            if k < 0 or holder[k] != i:
                continue
            s, a = tl_slot[i], tl_start[i]
            last_pos[k] = pos[t - 1, s]
            back = max(a, t - 1 - vel_frames)
            if t - 1 > back:
                v = (pos[t - 1, s] - pos[back, s]) * fps / (t - 1 - back)
                speed = float(np.hypot(*v))
                last_vel[k] = v * min(1.0, p.max_speed_ms / speed) if speed > 0 else 0.0
            else:
                last_vel[k] = 0.0
            last_t[k] = t - 1
            holder[k] = -1
        if t >= n:
            break
        pending.update(starts_at.get(t, []))
        free = np.flatnonzero(holder < 0)
        if not pending or not len(free):
            continue
        cand = sorted(pending)
        cost = np.zeros((len(cand), len(free)))
        for r, i in enumerate(cand):
            s = tl_slot[i]
            x = pos[t, s]
            is_deep = (deepest[t] == s) and np.isfinite(depth[t, s])
            clear_keeper = (is_deep and depth[t, s] < p.keeper_depth_m
                            and second[t] - depth[t, s] > 5)
            for c, k in enumerate(free):
                if last_t[k] < 0:
                    sigma = p.sigma_max_m
                    d2 = 0.0
                else:
                    dt = (t - last_t[k]) / fps
                    sigma = min(p.sigma0_m + p.sigma_speed_ms * dt, p.sigma_max_m)
                    pred = last_pos[k] + last_vel[k] * min(dt, p.coast_s)
                    d2 = float(((x - pred) ** 2).sum())
                cost[r, c] = d2 / (2 * sigma**2) + 2 * np.log(sigma) + LOG_2PI
                if p.keeper_prior:
                    if k == 0 and not is_deep:
                        cost[r, c] += p.keeper_cost
                    elif k != 0 and clear_keeper:
                        cost[r, c] += p.keeper_cost
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            i, k = cand[r], int(free[c])
            track_of[i] = k
            holder[k] = i
            assigned_at[i] = t
            pending.discard(i)
            # Softmin over the free tracks: how clearly this tracklet belongs to this track.
            z = -(cost[r] - cost[r].min())
            assoc_conf[i] = float(np.exp(z[c]) / np.exp(z).sum())
            n_assign += 1
            if cost[r, c] > 12:
                n_forced_far += 1
    for i in np.flatnonzero(track_of >= 0).tolist():
        # a tracklet that waited for a free track (pending) is used from its assignment on
        k, s, a, b = (int(v) for v in (track_of[i], tl_slot[i], assigned_at[i], tl_stop[i]))
        out_slot[a:b, k] = s
        conf[a:b, k] = assoc_conf[i]
    stats = {"tracklets": float(n_tl), "assignments": float(n_assign),
             "never_assigned": float((track_of < 0).sum()),
             "duplicate_frames_dropped": float(drop.sum()), "far_assignments": float(n_forced_far)}
    return out_slot, conf, stats


@dataclass(frozen=True)
class _Tracklets:
    ids: IntArray  # (n, n_slots) tracklet id per observation, -1 if none
    slot: IntArray
    team: IntArray
    start: IntArray  # first frame
    stop: IntArray  # last frame + 1
    start_pos: FloatArray
    end_pos: FloatArray
    start_vel: FloatArray
    end_vel: FloatArray
    centroid: FloatArray  # (n, 2 teams, 2) mean position of each slot team's observations

    def __len__(self) -> int:
        return len(self.slot)


def _tracklet_table(pos: FloatArray, team_of_slot: npt.NDArray[np.int8], p: StitchParams,
                    fps: float) -> _Tracklets:
    n = len(pos)
    tid = _tracklets(pos, p.jump_m)
    n_tl = int(tid.max()) + 1 if (tid >= 0).any() else 0
    slot = np.zeros(n_tl, dtype=np.int64)
    start = np.full(n_tl, n, dtype=np.int64)
    stop = np.zeros(n_tl, dtype=np.int64)
    ts, ss = np.nonzero(tid >= 0)
    ids = tid[ts, ss]
    slot[ids] = ss
    np.minimum.at(start, ids, ts)
    np.maximum.at(stop, ids, ts + 1)
    w = max(1, int(round(p.velocity_window_s * fps)))
    last = stop - 1
    a_end = np.maximum(start, last - w)
    b_start = np.minimum(last, start + w)

    def vel(a: IntArray, b: IntArray) -> FloatArray:
        frames = np.maximum(b - a, 1)
        v = (pos[b, slot] - pos[a, slot]) * fps / frames[:, None]
        v[b == a] = 0.0
        speed = np.hypot(v[:, 0], v[:, 1])
        scale = np.minimum(1.0, p.max_speed_ms / np.maximum(speed, 1e-9))
        out: FloatArray = v * scale[:, None]
        return out

    centroid = np.zeros((n, 2, 2))
    frames = np.arange(n)
    for team in (HOME_TEAM, AWAY_TEAM):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmean(pos[:, team_of_slot == team], axis=1)
        ok = np.isfinite(c[:, 0])
        for d in range(2):
            centroid[:, team, d] = (np.interp(frames, frames[ok], c[ok, d]) if ok.any()
                                    else 0.0)
    return _Tracklets(tid, slot, team_of_slot[slot].astype(np.int64), start, stop,
                      pos[start, slot], pos[last, slot], vel(start, b_start), vel(a_end, last),
                      centroid)


def _link_costs(tl: _Tracklets, ends: IntArray, starts: IntArray, p: StitchParams, fps: float
                ) -> FloatArray:
    """Negative log-likelihood that tracklet `starts[k]` continues tracklet `ends[k]`: both are
    extrapolated half way into the gap (at most `coast_s`) and must meet within a Gaussian
    whose width grows with the gap."""
    gap_s = (tl.start[starts] - (tl.stop[ends] - 1)) / fps
    half = np.minimum(gap_s / 2, p.coast_s)[:, None]
    fwd = tl.end_pos[ends] + tl.end_vel[ends] * half
    bwd = tl.start_pos[starts] - tl.start_vel[starts] * half
    if p.formation_drift:
        coast = np.round(half[:, 0] * fps).astype(np.int64)
        a = np.minimum(tl.stop[ends] - 1 + coast, tl.start[starts])
        b = np.maximum(tl.start[starts] - coast, a)
        team = tl.team[ends]
        fwd = fwd + tl.centroid[b, team] - tl.centroid[a, team]
    d2 = ((fwd - bwd) ** 2).sum(axis=1)
    sigma = np.minimum(p.sigma0_m + p.sigma_speed_ms * gap_s, p.sigma_max_m)
    cost = d2 / (2 * sigma**2) + 2 * np.log(sigma) + LOG_2PI
    cost = cost + p.team_cost * (tl.team[ends] != tl.team[starts])
    if p.same_slot_cost:
        cost = cost + p.same_slot_cost * ((tl.slot[ends] == tl.slot[starts])
                                          & (gap_s <= p.same_slot_max_s))
    gated = d2 <= (p.gate_sigmas * sigma) ** 2
    out: FloatArray = np.where(gated, cost, np.inf)
    return out


def _gap_close(tl: _Tracklets, p: StitchParams, fps: float) -> tuple[IntArray, FloatArray]:
    """Globally optimal linking of tracklet ends to later starts (Jaqaman et al. 2008): one
    sparse assignment where every end either links to a start or ends its chain, and every
    start is either linked or begins a chain. Returns (next tracklet or -1, link confidence)."""
    n = len(tl)
    nxt = np.full(n, -1, dtype=np.int64)
    conf = np.ones(n)
    if n == 0:
        return nxt, conf
    order = np.argsort(tl.start, kind="stable")
    sorted_start = tl.start[order]
    max_gap = int(round(p.max_gap_s * fps))
    lo = np.searchsorted(sorted_start, tl.stop, side="left")  # start >= last + 1
    hi = np.searchsorted(sorted_start, tl.stop - 1 + max_gap, side="right")
    ends = np.repeat(np.arange(n), hi - lo)
    starts = order[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True)])
                   if n else np.array([], dtype=np.int64)].astype(np.int64)
    cost = _link_costs(tl, ends, starts, p, fps) if len(ends) else np.zeros(0)
    ok = np.isfinite(cost)
    ends, starts, cost = ends[ok], starts[ok], cost[ok]
    eps = 1e-3
    b = p.new_chain_cost
    rows = np.concatenate([ends, np.arange(n), n + np.arange(n), n + starts])
    cols = np.concatenate([starts, n + np.arange(n), np.arange(n), n + ends])
    vals = np.concatenate([np.maximum(cost, eps), np.full(n, b), np.full(n, b),
                           np.full(len(ends), eps)])
    graph = coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    row_ind, col_ind = min_weight_full_bipartite_matching(graph)
    linked = (row_ind < n) & (col_ind < n)
    nxt[row_ind[linked]] = col_ind[linked]
    # Confidence of a link: softmin of its cost against the other candidates of the same end
    # and of the same start, with "no link" (2 x new_chain_cost) as one more candidate.
    if len(ends):
        lse_end = np.full(n, -2 * b)
        lse_start = np.full(n, -2 * b)
        np.logaddexp.at(lse_end, ends, -cost)
        np.logaddexp.at(lse_start, starts, -cost)
        chosen = nxt[ends] == starts
        e, s_, c = ends[chosen], starts[chosen], cost[chosen]
        conf[s_] = np.minimum(np.exp(-c - lse_end[e]), np.exp(-c - lse_start[s_]))
    return nxt, conf


def _stitch_half_lap(pos: FloatArray, team_of_slot: npt.NDArray[np.int8],
                     goals: dict[int, float], p: StitchParams, fps: float
                     ) -> tuple[IntArray, FloatArray, dict[str, float]]:
    """Both teams of one half block: tracklets -> global gap closing -> chains -> chain team by
    majority -> placement of chains on `max_tracks` tracks per team, keeper chains on track 0.

    Returns (input slot per frame and output track, -1 if none; confidence; stats)."""
    n = len(pos)
    k_tracks = p.max_tracks
    drop = _suppress_duplicates(pos, team_of_slot, p, fps)
    pos = np.where(drop[:, :, None], np.nan, pos)
    tl = _tracklet_table(pos, team_of_slot, p, fps)
    nxt, link_conf = _gap_close(tl, p, fps)
    n_tl = len(tl)
    has_prev = np.zeros(n_tl, dtype=bool)
    has_prev[nxt[nxt >= 0]] = True
    # Confidence of a chain head: 1 at the half start, otherwise unknown identity (0.5).
    link_conf[~has_prev] = np.where(tl.start[~has_prev] == 0, 1.0, 0.5)
    length = tl.stop - tl.start
    # Chains, cut where a link crosses teams: each piece keeps its slot team, except a short
    # piece sandwiched between two pieces of the other team (a misfiled tracklet), which is
    # folded into them. A cross-team swap inside a tracklet thus costs only that tracklet's tail.
    chains: list[list[int]] = []
    chain_teams: list[int] = []
    short = int(round(p.misfiled_max_s * fps))
    for head in np.flatnonzero(~has_prev):
        chain = [int(head)]
        while nxt[chain[-1]] >= 0:
            chain.append(int(nxt[chain[-1]]))
        pieces: list[list[int]] = [[chain[0]]]
        for i in chain[1:]:
            if tl.team[i] != tl.team[pieces[-1][-1]]:
                pieces.append([])
            pieces[-1].append(i)
        teams = [int(tl.team[piece[0]]) for piece in pieces]
        for q in range(1, len(pieces) - 1):
            if (teams[q - 1] == teams[q + 1] != teams[q]
                    and int(length[pieces[q]].sum()) <= short):
                teams[q] = teams[q - 1]
        merged: list[list[int]] = []
        merged_teams: list[int] = []
        for piece, team in zip(pieces, teams, strict=True):
            if merged and merged_teams[-1] == team:
                merged[-1] += piece
            else:
                merged.append(list(piece))
                merged_teams.append(team)
        chains += merged
        chain_teams += merged_teams

    # Keeper evidence: share of a chain's frames on which it is the deepest of its slot team.
    deepest = np.zeros(pos.shape[:2], dtype=bool)
    for team, goal in goals.items():
        cols = np.flatnonzero(team_of_slot == team)
        d = np.nan_to_num(np.abs(pos[:, cols, 0] - goal), nan=np.inf)
        ok = np.isfinite(d).any(axis=1)
        deepest[np.flatnonzero(ok), cols[np.argmin(d[ok], axis=1)]] = True
    deep_count = np.zeros(n_tl)
    obs = tl.ids >= 0
    np.add.at(deep_count, tl.ids[obs], deepest[obs])

    out_slot = np.full((n, 2 * k_tracks), -1, dtype=np.int64)
    conf = np.zeros((n, 2 * k_tracks))
    stats = {"tracklets": float(n_tl), "links": float((nxt >= 0).sum()),
             "chains": float(len(chains)), "chain_frames_dropped": 0.0,
             "duplicate_frames_dropped": float(drop.sum())}
    info = []
    for chain, team in zip(chains, chain_teams, strict=True):
        idx = np.array(chain)
        frames = length[idx]
        info.append((chain, team, float(frames.sum()),
                     float(deep_count[idx].sum() / frames.sum()), int(tl.start[idx[0]])))

    # Event-driven allocation, per team: at any frame at most `max_tracks` chains are shown,
    # those with the most observed frames (real players beat duplicates and stray fragments).
    # A chain keeps its track while it lives; it takes a free track when it (re)appears, the
    # keeper chain preferring track 0, otherwise the track whose last position fits best; if
    # no track is free, it takes the one whose owner is absent longest from now on.
    chain_of = np.full(n_tl, -1, dtype=np.int64)
    for c, (chain, *_rest) in enumerate(info):
        chain_of[chain] = c
    weight = np.array([c[2] for c in info])
    keeper_like = np.array([c[3] > 0.5 for c in info], dtype=bool)
    chain_team = np.array([c[1] for c in info], dtype=np.int64)
    # Next frame at which each chain is observed again, for the eviction rule.
    tl_by_chain = {c: np.array(chain) for c, (chain, *_r) in enumerate(info)}

    chain_starts = {c: np.sort(tl.start[idx]) for c, idx in tl_by_chain.items()}
    steal_frames = int(round(p.steal_after_s * fps))

    def next_seen(c: int, t: int) -> int:
        starts = chain_starts[c]
        k = int(np.searchsorted(starts, t, side="left"))
        return int(starts[k]) if k < len(starts) else n

    events = np.unique(np.concatenate([tl.start, tl.stop, [0]]))
    events = events[events < n]
    owner = np.full(2 * k_tracks, -1, dtype=np.int64)
    track_of_chain = np.full(len(info), -1, dtype=np.int64)
    last_seen = np.full(2 * k_tracks, -1, dtype=np.int64)  # frame of the track's last output
    active_tl = np.zeros(n_tl, dtype=bool)
    starts_by = {int(f): np.flatnonzero(tl.start == f) for f in np.unique(tl.start)}
    stops_by = {int(f): np.flatnonzero(tl.stop == f) for f in np.unique(tl.stop)}
    shown = np.zeros((len(events), 2 * k_tracks), dtype=np.int64) - 1
    for e, t in enumerate(events.tolist()):
        active_tl[stops_by.get(t, np.array([], dtype=np.int64))] = False
        active_tl[starts_by.get(t, np.array([], dtype=np.int64))] = True
        live = np.flatnonzero(active_tl)
        for team in (HOME_TEAM, AWAY_TEAM):
            base = team * k_tracks
            tracks = np.arange(base, base + k_tracks)
            mine = live[chain_team[chain_of[live]] == team]
            chains_now = np.unique(chain_of[mine])
            if len(chains_now) > k_tracks:
                order = np.lexsort((chains_now, -weight[chains_now]))
                chains_now = chains_now[order[:k_tracks]]
            keep = set(chains_now.tolist())
            for c in sorted(keep, key=lambda c: (-weight[c], c)):
                k = track_of_chain[c]
                if k >= 0 and owner[k] == c:
                    continue
                # A track is free if unowned, or if its owner is absent for at least
                # steal_after_s from now; otherwise the chain is not shown yet.
                # Track 0 is reserved for keeper-like chains (`keeper_reserve`).
                allowed = [int(j) for j in tracks
                           if keeper_like[c] or not p.keeper_reserve or j != base]
                unowned = [j for j in allowed if owner[j] < 0]
                stealable = [j for j in allowed if owner[j] >= 0 and owner[j] not in keep
                             and next_seen(int(owner[j]), t) - t >= steal_frames]
                pool = unowned or stealable
                if not pool:
                    continue
                first = tl_by_chain[c][0]
                start_pos = tl.start_pos[first]

                def fit(j: int, c: int = c, start_pos: FloatArray = start_pos, base: int = base,
                        t: int = t) -> tuple[float, ...]:
                    wants_keeper = keeper_like[c]
                    pref = 0.0 if (j == base) == wants_keeper else 1.0
                    if last_seen[j] < 0:
                        geo = 2 * np.log(p.sigma_max_m)
                    else:
                        f = last_seen[j]
                        here = pos[f, out_slot[f, j]]
                        gap_s = (t - f) / fps
                        sigma = min(p.sigma0_m + p.sigma_speed_ms * gap_s, p.sigma_max_m)
                        geo = float(((here - start_pos) ** 2).sum()) / (2 * sigma**2) + \
                            2 * np.log(sigma)
                    absent = -next_seen(int(owner[j]), t) if owner[j] >= 0 else -n
                    return (pref, absent, geo, j)

                k = min(pool, key=fit)
                if owner[k] >= 0:
                    track_of_chain[owner[k]] = -1
                owner[k] = c
                track_of_chain[c] = k
            for j in tracks:
                if owner[j] in keep and track_of_chain[owner[j]] == j:
                    shown[e, j] = owner[j]
        # write this event span
        stop_t = int(events[e + 1]) if e + 1 < len(events) else n
        track_now = {int(c): int(j) for j, c in enumerate(shown[e]) if c >= 0}
        for i in live.tolist():  # tracklets are contiguous: live over the whole event span
            shown_on = track_now.get(int(chain_of[i]), -1)
            if shown_on >= 0:
                out_slot[t:stop_t, shown_on] = tl.slot[i]
                conf[t:stop_t, shown_on] = link_conf[i]
                last_seen[shown_on] = stop_t - 1
    placed = int((out_slot >= 0).sum())
    stats["chain_frames_dropped"] = float(int(length.sum()) - placed)
    return out_slot, conf, stats



def own_goal_x(players_m: FloatArray, team_of_slot: npt.NDArray[np.int8], team: int,
               length: float) -> float:
    """x of the goal a team defends in a block: its mean x is on that side of the other team's."""
    with np.errstate(invalid="ignore"):
        mine = np.nanmean(players_m[:, team_of_slot == team, 0])
        theirs = np.nanmean(players_m[:, team_of_slot != team, 0])
    return 0.0 if mine < theirs else length


def stitch(tracking: Tracking, blocks: list[tuple[int, int, int]], pitch: tuple[float, float],
           fps: float = 25.0, params: StitchParams | None = None) -> Stitched:
    p = params or StitchParams()
    scale = np.array(pitch, dtype=np.float64)
    players_m = tracking.players.astype(np.float64) * scale
    n = tracking.n_frames
    k = p.max_tracks
    out = np.full((n, 2 * k, 2), np.nan, dtype=np.float64)
    conf = np.zeros((n, 2 * k), dtype=np.float64)
    source = np.full((n, 2 * k), -1, dtype=np.int64)
    stats: dict[str, float] = {}
    for _half, start, stop in blocks:
        block = players_m[start:stop]
        goals = {team: own_goal_x(block, tracking.team_of_slot, team, pitch[0])
                 for team in (HOME_TEAM, AWAY_TEAM)}
        if p.method == "lap":
            used_all, c_all, st = _stitch_half_lap(block, tracking.team_of_slot, goals, p, fps)
            ok = used_all >= 0
            rows_, tracks_ = np.nonzero(ok)
            out[start + rows_, tracks_] = block[rows_, used_all[ok]]
            source[start:stop] = used_all
            conf[start:stop] = c_all
            for key, v in st.items():
                stats[key] = stats.get(key, 0.0) + v
            continue
        for team in (HOME_TEAM, AWAY_TEAM):
            slots = np.flatnonzero(tracking.team_of_slot == team)
            goal = goals[team]
            used, c, st = _stitch_team_greedy(block[:, slots], goal, p, fps)
            cols = slice(team * k, (team + 1) * k)
            ok = used >= 0
            rows_, tracks_ = np.nonzero(ok)
            src = slots[used[ok]]
            out[start + rows_, team * k + tracks_] = block[rows_, src]
            source[start:stop, cols] = np.where(ok, slots[np.maximum(used, 0)], -1)
            conf[start:stop, cols] = c
            for key, v in st.items():
                stats[key] = stats.get(key, 0.0) + v
    team_of_track = np.array([HOME_TEAM] * k + [AWAY_TEAM] * k, dtype=np.int8)
    return Stitched((out / scale).astype(np.float32), team_of_track, (0, k),
                    conf.astype(np.float32), source.astype(np.int16), stats)


def fill_gaps(players: npt.NDArray[np.float32], blocks: list[tuple[int, int, int]],
              max_hold_frames: int | None = None) -> npt.NDArray[np.float32]:
    """Linear interpolation of every track inside each half; before its first and after its
    last observation the track holds that position for at most `max_hold_frames` (all the way
    to the half's ends if None). Tracks never observed in a half stay NaN there."""
    out = players.copy()
    for _half, start, stop in blocks:
        idx = np.arange(stop - start)
        for j in range(players.shape[1]):
            col = out[start:stop, j]
            ok = np.isfinite(col[:, 0])
            if ok.all() or not ok.any():
                continue
            seen = np.flatnonzero(ok)
            lo, hi = seen[0], seen[-1] + 1
            if max_hold_frames is not None:
                lo, hi = max(0, lo - max_hold_frames), min(len(col), hi + max_hold_frames)
            else:
                lo, hi = 0, len(col)
            todo = idx[lo:hi][~ok[lo:hi]]
            for d in range(2):
                col[todo, d] = np.interp(todo, seen, col[seen, d])
    return out
