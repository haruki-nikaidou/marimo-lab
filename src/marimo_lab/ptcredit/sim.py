"""Agent-based simulation of the point system (handoff §8, experiment 1).

One site-hour per tick, everything vectorised over memberships.

What is modelled
----------------
* heterogeneous nodes: uplink, downlink, uptime, disk, all correlated;
* a skewed-popularity catalogue with the four size classes;
* the full §3.2 estimator chain (pessimistic Beta availability, shrunk 90th
  percentile capacity, concurrency EWMA) fed by *simulated observations*, not
  by the ground truth;
* per-torrent health, leave-one-out marginal and the §3.4 reward;
* seeders that periodically drop their lowest reward-per-GiB torrent in favour
  of a better one, subject to disk and to the points they can afford;
* download demand that burns points, and the §3.7 ``kappa`` controller.

Deliberately not modelled: partial seeders (every member holds a complete
copy), invites and user growth, and the announce/queue plumbing of §5 — none of
them change the controller behaviour the acceptance test is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .credit import NodeEstimator, RewardParams, health, slowdown
from .world import (
    DEFAULT_BANDWIDTH,
    DEFAULT_MIX,
    MEGABIT_PER_GIB,
    BandwidthModel,
    Catalog,
    Population,
    sample_catalog,
    sample_population,
)

__all__ = ["SimParams", "SimResult", "build_world", "run"]


@dataclass(frozen=True)
class SimParams:
    """Everything about the run that is not a reward-formula parameter."""

    days: int = 45
    n_torrents: int = 2000
    n_users: int = 400
    pin_uploader: bool = False
    disk_scale: float = 1.0
    initial_fill: float = 0.8
    starts_per_user_day: float = 3.0
    #: §8 as written: at each decision tick *every* seeder makes at most one
    #: drop/join.  ``decision_share``/``moves_per_decision`` relax that into
    #: staggered batched re-planning, which mixes far faster; the notebook
    #: treats the relaxed form as a sensitivity variant, not the baseline.
    decide_every_h: int = 6
    decision_share: float = 1.0
    n_candidates: int = 24
    moves_per_decision: int = 1
    candidate_bias: float = 0.0
    swap_margin: float = 0.15
    keep_after_download: float = 0.7
    stay_bonus: float = 1.0
    kappa_control: bool = True
    kappa_target_ratio: float = 1.0
    kappa_lambda: float = 0.1
    kappa_clamp: float = 0.05
    kappa_window_days: int = 7
    start_balance: float = 200.0
    record_every_h: int = 6
    n_tracked: int = 48
    seed: int = 0


@dataclass
class SimResult:
    """Recorded history plus the final state of one run."""

    hours: np.ndarray
    #: (T, 3) p10 / p50 / p90 of C_i over the **whole catalogue**, with an
    #: unavailable torrent counted as ``inf``.  Conditioning on available
    #: torrents hides exactly the failure mode where capacity concentrates and
    #: the rest of the catalogue drops below the completeness threshold.
    c_quantiles_all: np.ndarray
    #: (T, 3) the same quantiles conditional on ``A_i >= pi_min``.
    c_quantiles: np.ndarray
    c_by_class: np.ndarray  # (T, n_classes): median available C_i per class
    #: Fraction of the catalogue with ``A_i < pi_min``.  Every membership in
    #: this engine is a *complete* seeder, so this is not "no copy exists" —
    #: it is "the probability that at least one copy is online has fallen
    #: below the threshold", which is what drives ``C_i`` to infinity.
    unavailable_fraction: np.ndarray
    health_mean: np.ndarray
    delta_health_mean: np.ndarray
    #: Points minted / burned per recording interval (not per hour).
    mint: np.ndarray
    burn: np.ndarray
    kappa: np.ndarray
    members_total: np.ndarray
    seeders_tracked: np.ndarray  # (T, n_tracked)
    balance_quantiles: np.ndarray  # (T, 3)
    #: Otherwise-eligible download requests refused for insufficient balance,
    #: per recording interval.
    refused: np.ndarray
    d_ref: float
    catalog: Catalog = field(repr=False)
    population: Population = field(repr=False)
    final_capacity: np.ndarray = field(repr=False)
    final_slowdown: np.ndarray = field(repr=False)
    final_members: np.ndarray = field(repr=False)
    final_contribution: np.ndarray = field(repr=False)
    final_balance: np.ndarray = field(repr=False)
    #: Total disk *budget* provisioned by each user (summed over their nodes).
    #: This is capacity owned, not bytes currently held.
    user_disk_budget: np.ndarray = field(repr=False)
    #: Hours represented by one recorded sample.
    record_every_h: int = 6
    pinned_coverage: float = 0.0

    def tail(self, days: float = 7.0) -> slice:
        """Index slice covering the last ``days`` of the run.

        Timestamps mark interval *ends*, so the sample stamped exactly at the
        cutoff represents the interval *before* the window and is excluded;
        ``side="right"`` is what makes a 7-day tail 28 six-hour records rather
        than 29.
        """
        cut = self.hours[-1] - days * 24.0
        return slice(int(np.searchsorted(self.hours, cut, side="right")), None)

    def tail_hours(self, days: float = 7.0) -> float:
        """Hours actually represented by :meth:`tail`, for rate conversions."""
        return float(self.hours[self.tail(days)].size * self.record_every_h)

    def oscillation(self, days: float = 14.0) -> float:
        """Relative amplitude of the cross-torrent mean seeder count.

        Hunting is a *common* period across torrents, so the mean over the
        tracked torrents is the right series; the metric is its detrended
        peak-to-peak swing as a fraction of its level.  ``C`` quantiles are a
        poor oscillation proxy because they also move with the unavailable
        mass, and they can be ``inf``.
        """
        series = self.seeders_tracked[self.tail(days)].mean(axis=1)
        if series.size < 4 or series.mean() <= 0.0:
            return float("nan")
        trend = np.polyval(
            np.polyfit(np.arange(series.size), series, 1), np.arange(series.size)
        )
        return float(np.ptp(series - trend) / series.mean())

    def acceptance(self, days: float = 7.0) -> dict[str, float]:
        """Summary against the §8 acceptance criterion.

        ``*_all`` keys cover the whole catalogue and may be ``inf``; the
        ``avail_*`` keys condition on ``A_i >= pi_min`` and are only meaningful
        next to ``unavailable_fraction``.
        """
        sl = self.tail(days)
        median_c = self.c_quantiles_all[sl, 1]
        finite_median = median_c[np.isfinite(median_c)]
        span_days = max(self.tail_hours(days) / 24.0, 1e-9)
        return {
            "median_C_all": float(np.mean(median_c)),
            "p90_C_all": float(np.mean(self.c_quantiles_all[sl, 2])),
            "avail_median_C": float(np.mean(self.c_quantiles[sl, 1])),
            "avail_p90_C": float(np.mean(self.c_quantiles[sl, 2])),
            # ptp over a series that may contain inf is nan; report the swing
            # of the finite part and let oscillation() carry the real signal.
            "swing_C": float(np.ptp(finite_median))
            if finite_median.size
            else float("nan"),
            "oscillation": self.oscillation(),
            "unavailable_fraction": float(np.mean(self.unavailable_fraction[sl])),
            "mint_burn": float(self.mint[sl].sum() / max(self.burn[sl].sum(), 1e-9)),
            "kappa": float(self.kappa[-1]),
            # refused is an interval total, so divide by the hours the tail
            # actually represents rather than assuming one sample per hour.
            "broke_per_day": float(self.refused[sl].sum() / span_days),
        }


def build_world(
    params: SimParams,
    *,
    bandwidth: BandwidthModel = DEFAULT_BANDWIDTH,
    mix=DEFAULT_MIX,
    n_super: int = 3,
    **population_kwargs,
) -> tuple[Catalog, Population]:
    """Draw the catalogue and the population a run will use."""
    rng = np.random.default_rng(params.seed)
    catalog = sample_catalog(params.n_torrents, rng, mix=mix, n_super=n_super)
    population = sample_population(
        params.n_users, rng, bandwidth=bandwidth, **population_kwargs
    )
    return catalog, population


class _Site:
    """Mutable simulation state.  One instance per run."""

    def __init__(
        self,
        params: SimParams,
        reward: RewardParams,
        catalog: Catalog,
        population: Population,
    ):
        self.p = params
        self.r = reward
        self.cat = catalog
        self.pop = population
        self.rng = np.random.default_rng(params.seed + 1)

        self.n_t = catalog.n_torrents
        self.n_n = population.n_nodes
        self.n_u = population.n_users

        self.size = catalog.size_gib
        self.rel = catalog.rel_size
        self.weight = reward.torrent_weight(catalog.importance)
        self.target = reward.torrent_target(catalog.importance)
        self.charge = catalog.charge
        self.demand = catalog.demand_share

        self.user = population.user_of_node
        self.up = population.up_mbps
        self.down = population.down_mbps
        self.avail = population.availability
        self.disk = population.disk_gib * params.disk_scale

        self.d_ref = float(np.median(self.down))
        self.kappa = reward.kappa
        self.balance = np.full(self.n_u, params.start_balance)

        self.est = NodeEstimator(
            self.n_n,
            m=reward.m,
            half_life_h=reward.half_life_h,
            prior_p=0.55,
            prior_b=float(np.quantile(self.up, 0.25)),
        )
        self.all_active = np.ones(self.n_n, dtype=bool)

        self.holds = np.zeros((self.n_n, self.n_t), dtype=bool)
        self.fetching = np.zeros((self.n_n, self.n_t), dtype=bool)
        self.used = np.zeros(self.n_n)

        self.mem_node = np.empty(0, dtype=np.int64)
        self.mem_torrent = np.empty(0, dtype=np.int64)
        self.mem_key = np.empty(0, dtype=np.int64)

        self.lch_node = np.empty(0, dtype=np.int64)
        self.lch_torrent = np.empty(0, dtype=np.int64)
        self.lch_left = np.empty(0)
        self.lch_invest = np.empty(0, dtype=bool)

        self.pin_node = np.full(self.n_t, -1, dtype=np.int64)
        self.pinned_coverage = 0.0
        self._seed_initial_allocation()
        if params.pin_uploader:
            self._pin_uploaders()
        self._rebuild_pairs()

    def _pin_uploaders(self) -> None:
        """Sensitivity experiment: give each torrent a seeder that never leaves.

        Not part of the specified mechanism.  It exists so the notebook can
        separate "the controller cannot hold a target" from "the torrent fell
        below the completeness threshold and §3.3 pays nothing to bring it
        back".

        Torrents that already have a holder keep their most available one; the
        rest are placed on the most available node with room, largest torrent
        first.  On a disk-poor site some torrents cannot be placed at all, so
        coverage is recorded rather than assumed to be 1.
        """
        if self.mem_node.size:
            score = self.avail[self.mem_node]
            best = np.full(self.n_t, -np.inf)
            np.maximum.at(best, self.mem_torrent, score)
            is_best = score >= best[self.mem_torrent]
            self.pin_node[self.mem_torrent[is_best]] = self.mem_node[is_best]

        free = self.disk - self.used
        by_uptime = np.argsort(-self.avail)
        add_nodes, add_torrents = [], []
        uncovered = np.flatnonzero(self.pin_node < 0)
        for i in uncovered[np.argsort(-self.size[uncovered])]:
            room = (free[by_uptime] >= self.size[i]) & ~self.holds[by_uptime, i]
            hit = np.flatnonzero(room)
            if hit.size == 0:
                continue
            v = int(by_uptime[hit[0]])
            free[v] -= self.size[i]
            self.pin_node[i] = v
            add_nodes.append(v)
            add_torrents.append(int(i))
        if add_nodes:
            self._add_members(np.asarray(add_nodes), np.asarray(add_torrents))
            self.used[:] = np.bincount(
                self.mem_node, self.size[self.mem_torrent], minlength=self.n_n
            )
        self.pinned_coverage = float(np.mean(self.pin_node >= 0))

    # -- membership bookkeeping -------------------------------------------

    def _seed_initial_allocation(self) -> None:
        """Fill part of every node's disk so the run does not start empty."""
        budget = self.disk * self.p.initial_fill
        nodes, torrents = [], []
        for v in range(self.n_n):
            left = budget[v]
            picks = self.rng.choice(
                self.n_t,
                size=min(512, self.n_t),
                replace=False,
                p=self._candidate_probabilities(),
            )
            for i in picks:
                if self.size[i] > left:
                    continue
                left -= self.size[i]
                nodes.append(v)
                torrents.append(i)
                if left < self.size.min():
                    break
        self._add_members(np.asarray(nodes), np.asarray(torrents))
        self.used[:] = np.bincount(
            self.mem_node, self.size[self.mem_torrent], minlength=self.n_n
        )
        if np.any(self.used > self.disk + 1e-9):
            raise AssertionError("initial allocation overflowed a node's disk")

    def _add_members(self, nodes, torrents) -> None:
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        torrents = np.asarray(torrents, dtype=np.int64).reshape(-1)
        if nodes.size == 0:
            return
        key = self.user[nodes].astype(np.int64) * self.n_t + torrents
        order = np.argsort(key, kind="stable")
        key, nodes, torrents = key[order], nodes[order], torrents[order]
        at = np.searchsorted(self.mem_key, key)
        self.mem_key = np.insert(self.mem_key, at, key)
        self.mem_node = np.insert(self.mem_node, at, nodes)
        self.mem_torrent = np.insert(self.mem_torrent, at, torrents)
        self.holds[nodes, torrents] = True

    def _drop_members(self, mask) -> None:
        if not mask.any():
            return
        self.holds[self.mem_node[mask], self.mem_torrent[mask]] = False
        self.used[:] -= np.bincount(
            self.mem_node[mask], self.size[self.mem_torrent[mask]], self.n_n
        )
        keep = ~mask
        self.mem_key = self.mem_key[keep]
        self.mem_node = self.mem_node[keep]
        self.mem_torrent = self.mem_torrent[keep]

    def _rebuild_pairs(self) -> None:
        """Group memberships into (user, torrent) pairs; §3.3 merge rule."""
        n = self.mem_key.size
        if n == 0:
            self.pair_inv = np.empty(0, dtype=np.int64)
            self.pair_torrent = np.empty(0, dtype=np.int64)
            self.pair_user = np.empty(0, dtype=np.int64)
            self.pair_count = np.empty(0, dtype=np.int64)
            return
        fresh = np.empty(n, dtype=bool)
        fresh[0] = True
        np.not_equal(self.mem_key[1:], self.mem_key[:-1], out=fresh[1:])
        self.pair_inv = np.cumsum(fresh) - 1
        starts = np.flatnonzero(fresh)
        self.pair_torrent = self.mem_torrent[starts]
        self.pair_user = self.user[self.mem_node[starts]]
        self.pair_count = np.diff(np.append(starts, n))

    # -- per-hour physics --------------------------------------------------

    def _serve(self):
        """One hour of actual transfers; returns per-node observed rates."""
        online = self.rng.random(self.n_n) < self.avail
        leech_count = np.bincount(self.lch_torrent, minlength=self.n_t)

        contributing = online[self.mem_node] & (leech_count[self.mem_torrent] > 0)
        busy_nodes = np.bincount(
            self.mem_node[contributing], minlength=self.n_n
        ).astype(float)
        offer = np.zeros(self.mem_node.size)
        if contributing.any():
            idx = self.mem_node[contributing]
            offer[contributing] = self.up[idx] / busy_nodes[idx]
        capacity_true = np.bincount(self.mem_torrent, offer, minlength=self.n_t)

        rate = np.zeros(self.lch_node.size)
        if rate.size:
            lt = self.lch_torrent
            share = capacity_true[lt] / np.maximum(leech_count[lt], 1)
            rate = np.minimum(self.down[self.lch_node], share)
            self.lch_left -= rate * 3600.0 / MEGABIT_PER_GIB

        served = np.bincount(self.lch_torrent, rate, minlength=self.n_t)
        frac = np.where(
            capacity_true > 0.0, served / np.maximum(capacity_true, 1e-12), 0.0
        )
        node_rate = np.bincount(
            self.mem_node, offer * frac[self.mem_torrent], minlength=self.n_n
        )
        self.est.observe(self.all_active, online, busy_nodes > 0, node_rate, busy_nodes)
        return online

    def _torrent_state(self, g, p_tilde):
        """Capacity, completeness and predicted slowdown for every torrent."""
        capacity = np.bincount(self.mem_torrent, g[self.mem_node], minlength=self.n_t)
        miss = np.log1p(-np.minimum(p_tilde, 1.0 - 1e-12))
        log_miss = np.bincount(
            self.mem_torrent, miss[self.mem_node], minlength=self.n_t
        )
        completeness = -np.expm1(log_miss)
        c = slowdown(capacity, completeness, self.d_ref, self.r.pi_min)
        return capacity, completeness, log_miss, c

    def _pair_reward(self, g, p_tilde, capacity, log_miss, c, target):
        """Points/hour each (user, torrent) pair would earn while online.

        ``target`` is the per-torrent ``C*``.  Passing an inflated target is how
        the §9 hysteresis variant values an *incumbent* membership more highly
        than the same membership would be valued by a joiner.
        """
        g_pair = np.bincount(
            self.pair_inv, g[self.mem_node], minlength=self.pair_torrent.size
        )
        miss = np.log1p(-np.minimum(p_tilde, 1.0 - 1e-12))
        miss_pair = np.bincount(
            self.pair_inv, miss[self.mem_node], minlength=self.pair_torrent.size
        )
        t = self.pair_torrent
        c_loo = slowdown(
            capacity[t] - g_pair,
            -np.expm1(log_miss[t] - miss_pair),
            self.d_ref,
            self.r.pi_min,
        )
        delta = health(c[t], target[t], self.r.gamma) - health(
            c_loo, target[t], self.r.gamma
        )
        rate = (
            self.kappa
            * self.weight[t]
            * self.rel[t] ** self.r.eta
            * (self.r.alpha + delta)
        )
        return rate, delta

    # -- decisions ---------------------------------------------------------

    @staticmethod
    def _running_total(keys, values):
        """Cumulative ``values`` within each ``keys`` group, this item included.

        Budget checks need this: several requests in one batch can belong to
        the same node or user, and each must be tested against what the earlier
        ones in the batch already consumed.
        """
        order = np.argsort(keys, kind="stable")
        grouped = keys[order]
        cumulative = np.cumsum(values[order])
        fresh = np.ones(order.size, dtype=bool)
        fresh[1:] = grouped[1:] != grouped[:-1]
        starts = np.flatnonzero(fresh)
        base = np.repeat(
            cumulative[starts] - values[order][starts],
            np.diff(np.append(starts, order.size)),
        )
        running = np.empty(order.size)
        running[order] = cumulative - base
        return running

    def _start_downloads(self, nodes, torrents, invest):
        """Charge the points and enqueue.

        Returns ``(started, burn, broke)`` where ``broke`` counts only the
        requests an otherwise-eligible user could not afford — the metric that
        sizes the startup gift and tells whether the economy is too tight.

        Eligibility is resolved in a fixed order so the counters mean what
        they say: structural rejections (already held, already downloading,
        duplicate in this batch) are removed *first*, then the per-node disk
        prefix, and only the survivors are tested against the per-user balance
        prefix.  Running the prefixes over rejected rows would let a request
        that was never going to happen make a later valid one look
        unaffordable, inflating ``broke``.
        """
        if nodes.size == 0:
            return 0, 0.0, 0
        size = self.size[torrents]
        structural = ~self.holds[nodes, torrents] & ~self.fetching[nodes, torrents]
        if structural.any():
            # Drop duplicate (node, torrent) rows within the batch too.
            keys = nodes.astype(np.int64) * self.n_t + torrents
            seen = np.flatnonzero(structural)
            _, first = np.unique(keys[seen], return_index=True)
            unique = np.zeros(nodes.size, dtype=bool)
            unique[seen[first]] = True
            structural &= unique

        fits = np.zeros(nodes.size, dtype=bool)
        rows = np.flatnonzero(structural)
        if rows.size:
            running = self._running_total(nodes[rows], size[rows])
            fits[rows] = self.used[nodes[rows]] + running <= self.disk[nodes[rows]]

        affordable = np.zeros(nodes.size, dtype=bool)
        rows = np.flatnonzero(fits)
        cost = self.charge[torrents] * size
        if rows.size:
            users = self.user[nodes[rows]]
            spend = self._running_total(users, cost[rows])
            affordable[rows] = self.balance[users] >= spend

        refused = int((fits & ~affordable).sum())
        ok = affordable
        nodes, torrents, cost = nodes[ok], torrents[ok], cost[ok]
        if nodes.size == 0:
            return 0, 0.0, refused
        self.balance -= np.bincount(self.user[nodes], cost, minlength=self.n_u)
        self.fetching[nodes, torrents] = True
        self.used += np.bincount(nodes, self.size[torrents], minlength=self.n_n)
        self.lch_node = np.concatenate([self.lch_node, nodes])
        self.lch_torrent = np.concatenate([self.lch_torrent, torrents])
        self.lch_left = np.concatenate([self.lch_left, self.size[torrents]])
        self.lch_invest = np.concatenate([self.lch_invest, np.full(nodes.size, invest)])
        return nodes.size, float(cost.sum()), refused

    def _finish_downloads(self) -> bool:
        done = self.lch_left <= 0.0
        if not done.any():
            return False
        nodes, torrents = self.lch_node[done], self.lch_torrent[done]
        self.fetching[nodes, torrents] = False
        keep = self.lch_invest[done] | (
            self.rng.random(nodes.size) < self.p.keep_after_download
        )
        self.used -= np.bincount(
            nodes[~keep], self.size[torrents[~keep]], minlength=self.n_n
        )
        self._add_members(nodes[keep], torrents[keep])
        alive = ~done
        self.lch_node = self.lch_node[alive]
        self.lch_torrent = self.lch_torrent[alive]
        self.lch_left = self.lch_left[alive]
        self.lch_invest = self.lch_invest[alive]
        return True

    def _candidate_probabilities(self) -> np.ndarray:
        """Where a browsing seeder looks for something to pick up.

        ``candidate_bias = 0`` means the whole catalogue is equally visible and
        the *reward* alone routes seeders; ``1`` means they only ever consider
        what is already popular, which starves exactly the torrents the
        marginal-contribution term is supposed to rescue.
        """
        bias = self.p.candidate_bias
        uniform = np.full(self.n_t, 1.0 / self.n_t)
        if bias <= 0.0:
            return uniform
        return (1.0 - bias) * uniform + bias * self.demand

    def _reallocate(self, g, p_tilde, capacity, completeness, pair_rate):
        """Re-plan a node's disk: fill free space, then swap up (§8).

        A deciding node values every torrent it holds and every candidate it
        sees in points per hour per GiB of disk — the quantity a finite disk
        actually trades off — and performs up to ``moves_per_decision``
        improvements, pairing its j-th worst holding against its j-th best
        candidate.  One move per decision would take a node holding hundreds of
        torrents most of a simulated year to reshuffle its library.
        """
        chosen = np.flatnonzero(self.rng.random(self.n_n) < self.p.decision_share)
        if chosen.size == 0 or self.mem_node.size == 0:
            return False
        k_nodes = chosen.size
        moves = int(self.p.moves_per_decision)

        per_member = (
            pair_rate[self.pair_inv]
            / self.pair_count[self.pair_inv]
            / self.size[self.mem_torrent]
        )
        # A pinned uploader never becomes the "worst holding".
        per_member = np.where(
            self.pin_node[self.mem_torrent] == self.mem_node, np.inf, per_member
        )
        picked = np.zeros(self.n_n, dtype=bool)
        picked[chosen] = True
        rows = np.flatnonzero(picked[self.mem_node])
        hold_row = np.full((moves, k_nodes), -1, dtype=np.int64)
        if rows.size:
            order = rows[np.lexsort((per_member[rows], self.mem_node[rows]))]
            owner = self.mem_node[order]
            fresh = np.ones(order.size, dtype=bool)
            fresh[1:] = owner[1:] != owner[:-1]
            starts = np.flatnonzero(fresh)
            rank = np.arange(order.size) - np.repeat(
                starts, np.diff(np.append(starts, order.size))
            )
            slot = np.searchsorted(chosen, owner)
            for j in range(moves):
                take = rank == j
                hold_row[j, slot[take]] = order[take]
        has_hold = hold_row >= 0
        safe_row = np.maximum(hold_row, 0)
        hold_value = np.where(has_hold, per_member[safe_row], 0.0)
        hold_size = np.where(has_hold, self.size[self.mem_torrent[safe_row]], 0.0)

        cand = self.rng.choice(
            self.n_t,
            size=(k_nodes, self.p.n_candidates),
            p=self._candidate_probabilities(),
        )
        g_v = g[chosen][:, None]
        cap_c, avail_c = capacity[cand], completeness[cand]
        joined = 1.0 - (1.0 - avail_c) * (1.0 - p_tilde[chosen][:, None])
        target_c = self.target[cand]
        gain = health(
            slowdown(cap_c + g_v, joined, self.d_ref, self.r.pi_min),
            target_c,
            self.r.gamma,
        ) - health(
            slowdown(cap_c, avail_c, self.d_ref, self.r.pi_min),
            target_c,
            self.r.gamma,
        )
        size_c = self.size[cand]
        value = (
            self.kappa
            * self.weight[cand]
            * self.rel[cand] ** self.r.eta
            * (self.r.alpha + gain)
            / size_c
        )
        usable = (
            (self.charge[cand] * size_c <= self.balance[self.user[chosen]][:, None])
            & ~self.holds[chosen[:, None], cand]
            & ~self.fetching[chosen[:, None], cand]
            & (size_c <= self.disk[chosen][:, None])
        )
        value = np.where(usable, value, -np.inf)
        rank_c = np.argsort(-value, axis=1)
        cand = np.take_along_axis(cand, rank_c, axis=1)
        value = np.take_along_axis(value, rank_c, axis=1)
        size_c = self.size[cand]

        free = (self.disk - self.used)[chosen]
        margin = 1.0 + self.p.swap_margin
        take_cand = np.zeros((moves, k_nodes), dtype=bool)
        take_drop = np.zeros((moves, k_nodes), dtype=bool)
        for j in range(min(moves, self.p.n_candidates)):
            offered = np.isfinite(value[:, j])
            c_size = size_c[:, j]
            add = offered & (c_size <= free)
            swap = (
                offered
                & ~add
                & has_hold[j]
                & (value[:, j] > hold_value[j] * margin)
                & (c_size <= free + hold_size[j])
            )
            free = free + np.where(swap, hold_size[j], 0.0)
            free = free - np.where(add | swap, c_size, 0.0)
            take_cand[j] = add | swap
            take_drop[j] = swap

        if take_drop.any():
            drop = np.zeros(self.mem_node.size, dtype=bool)
            drop[hold_row[take_drop]] = True
            self._drop_members(drop)

        move_idx, node_idx = np.nonzero(take_cand)
        nodes = chosen[node_idx]
        torrents = cand[node_idx, move_idx]
        if nodes.size:
            _, keep = np.unique(
                nodes.astype(np.int64) * self.n_t + torrents, return_index=True
            )
            nodes, torrents = nodes[keep], torrents[keep]
        started, _, _ = self._start_downloads(nodes, torrents, True)
        return bool(take_drop.any() or started)


def run(
    params: SimParams,
    reward: RewardParams,
    *,
    catalog: Catalog | None = None,
    population: Population | None = None,
    bandwidth: BandwidthModel = DEFAULT_BANDWIDTH,
) -> SimResult:
    """Run the simulation and return its recorded history."""
    if catalog is None or population is None:
        catalog, population = build_world(params, bandwidth=bandwidth)
    site = _Site(params, reward, catalog, population)

    total_h = int(params.days * 24)
    window = int(params.kappa_window_days * 24)
    mint_h = np.zeros(total_h)
    burn_h = np.zeros(total_h)
    refused_h = np.zeros(total_h)
    tracked = np.argsort(-catalog.demand_share)[: params.n_tracked]
    starts_per_hour = params.starts_per_user_day * params.n_users / 24.0
    n_classes = len(catalog.class_names)

    rec: dict[str, list] = {
        k: []
        for k in (
            "hours",
            "cqall",
            "cq",
            "cclass",
            "dead",
            "hmean",
            "dmean",
            "mint",
            "burn",
            "kappa",
            "members",
            "tracked",
            "bq",
            "refused",
        )
    }

    dirty = True
    for hour in range(total_h):
        site.est.decay(1.0)
        if dirty:
            site._rebuild_pairs()
            dirty = False
        online = site._serve()

        p_tilde, _, g = site.est.contribution()
        capacity, completeness, log_miss, c = site._torrent_state(g, p_tilde)
        h = health(c, site.target, reward.gamma)
        pair_rate, delta = site._pair_reward(
            g, p_tilde, capacity, log_miss, c, site.target
        )

        online_pair = np.bincount(
            site.pair_inv,
            online[site.mem_node].astype(float),
            minlength=site.pair_torrent.size,
        )
        earned = np.where(online_pair > 0, pair_rate, 0.0)
        site.balance += np.bincount(site.pair_user, earned, minlength=site.n_u)
        mint_h[hour] = earned.sum()

        n_start = site.rng.poisson(starts_per_hour)
        nodes = site.rng.integers(0, site.n_n, size=n_start)
        torrents = site.rng.choice(site.n_t, size=n_start, p=site.demand)
        _, burned, refused = site._start_downloads(nodes, torrents, False)
        burn_h[hour] = burned
        refused_h[hour] = refused

        if hour % params.decide_every_h == 0:
            incumbent = pair_rate
            if params.stay_bonus != 1.0:
                incumbent, _ = site._pair_reward(
                    g,
                    p_tilde,
                    capacity,
                    log_miss,
                    c,
                    site.target * params.stay_bonus,
                )
            dirty |= site._reallocate(g, p_tilde, capacity, completeness, incumbent)
        dirty |= site._finish_downloads()

        if (hour + 1) % 24 == 0:
            site.est.refresh_population_prior(site.all_active)
            if params.kappa_control:
                lo = max(0, hour + 1 - window)
                minted = mint_h[lo : hour + 1].sum()
                burnt = burn_h[lo : hour + 1].sum()
                if minted > 0.0 and burnt > 0.0:
                    step = (params.kappa_target_ratio * burnt / minted) ** (
                        params.kappa_lambda
                    )
                    lo_clamp = 1.0 - params.kappa_clamp
                    hi_clamp = 1.0 + params.kappa_clamp
                    step = float(np.clip(step, lo_clamp, hi_clamp))
                    site.kappa *= step

        if (hour + 1) % params.record_every_h == 0:
            span = slice(hour + 1 - params.record_every_h, hour + 1)
            available = np.isfinite(c)
            rec["hours"].append(float(hour + 1))
            # method="lower" is an order statistic, so inf never enters an
            # interpolation and the unavailable tail reads inf, not nan.
            rec["cqall"].append(np.quantile(c, [0.1, 0.5, 0.9], method="lower"))
            rec["cq"].append(
                np.quantile(c[available], [0.1, 0.5, 0.9])
                if available.any()
                else [np.nan] * 3
            )
            rec["cclass"].append(
                [
                    float(np.median(c[available & (catalog.class_idx == j)]))
                    if (available & (catalog.class_idx == j)).any()
                    else np.nan
                    for j in range(n_classes)
                ]
            )
            rec["dead"].append(float(1.0 - available.mean()))
            rec["hmean"].append(float(h.mean()))
            rec["dmean"].append(float(delta.mean()) if delta.size else 0.0)
            # Interval sums, not the single hour that happens to be sampled:
            # mint and burn are flows and acceptance() adds these up.
            rec["mint"].append(float(mint_h[span].sum()))
            rec["burn"].append(float(burn_h[span].sum()))
            rec["refused"].append(float(refused_h[span].sum()))
            rec["kappa"].append(float(site.kappa))
            rec["members"].append(float(site.mem_node.size))
            rec["tracked"].append(
                np.bincount(site.mem_torrent, minlength=site.n_t)[tracked].astype(float)
            )
            rec["bq"].append(np.quantile(site.balance, [0.1, 0.5, 0.9]))

    p_tilde, _, g = site.est.contribution()
    capacity, _, _, c = site._torrent_state(g, p_tilde)
    return SimResult(
        hours=np.asarray(rec["hours"]),
        c_quantiles_all=np.asarray(rec["cqall"], dtype=float),
        c_quantiles=np.asarray(rec["cq"], dtype=float),
        c_by_class=np.asarray(rec["cclass"], dtype=float),
        unavailable_fraction=np.asarray(rec["dead"]),
        health_mean=np.asarray(rec["hmean"]),
        delta_health_mean=np.asarray(rec["dmean"]),
        mint=np.asarray(rec["mint"]),
        burn=np.asarray(rec["burn"]),
        kappa=np.asarray(rec["kappa"]),
        members_total=np.asarray(rec["members"]),
        seeders_tracked=np.asarray(rec["tracked"], dtype=float),
        balance_quantiles=np.asarray(rec["bq"], dtype=float),
        refused=np.asarray(rec["refused"]),
        d_ref=site.d_ref,
        catalog=catalog,
        population=population,
        final_capacity=capacity,
        final_slowdown=c,
        final_members=np.bincount(site.mem_torrent, minlength=site.n_t),
        final_contribution=g,
        final_balance=site.balance.copy(),
        user_disk_budget=np.bincount(site.user, site.disk, minlength=site.n_u),
        record_every_h=params.record_every_h,
        pinned_coverage=site.pinned_coverage,
    )
