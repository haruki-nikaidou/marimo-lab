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

from .credit import NodeEstimator, RewardParams
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


def _weighted_quantile(values, weights, qs) -> list[float]:
    """Lower order statistics of ``values`` under ``weights``."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    picks = np.searchsorted(cumulative, np.asarray(qs) * cumulative[-1], side="left")
    return [float(values[order][min(int(k), order.size - 1)]) for k in picks]


def _mean_or_nan(values) -> float:
    """Mean of the finite entries, ``nan`` when there are none.

    Hourly telemetry is ``nan`` for an hour with no leecher, and a whole
    recording interval can be empty on a small or scarce site.
    """
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


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
    #: (T, 3) p10 / p50 / p90 of the *physical* ratio ``d_ref / x_i`` over the
    #: whole catalogue — the same quantity under every scoring rule, with
    #: neither clamp applied.  ``c_quantiles_all`` is what the mechanism pays
    #: on and is therefore censored differently by each rule; this is what the
    #: capacity actually is, and cross-rule comparisons belong here.
    c_physical_quantiles: np.ndarray
    #: (T, 3) the physical ratio conditional on ``A_i >= pi_min``.  The
    #: whole-catalogue p90 is dominated by torrents with no members at all —
    #: already reported by ``unavailable_fraction`` — so the tail of the
    #: *served* catalogue is read here.
    c_physical_avail: np.ndarray
    c_by_class: np.ndarray  # (T, n_classes): median available C_i per class
    #: Fraction of the catalogue with ``A_i < pi_min``.  Every membership in
    #: this engine is a *complete* seeder, so this is not "no copy exists" —
    #: it is "the probability that at least one copy is online has fallen
    #: below the threshold", which is what drives ``C_i`` to infinity.
    unavailable_fraction: np.ndarray
    health_mean: np.ndarray
    delta_health_mean: np.ndarray
    #: Points minted / burned per recording interval (not per hour).  ``burn``
    #: covers every charged download — demand *and* a seeder's acquisition of
    #: a new holding, which leaves the economy in exactly the same way.
    #: ``burn_invest`` is the acquisition part on its own, so the demand part
    #: is ``burn - burn_invest``.
    mint: np.ndarray
    burn: np.ndarray
    burn_invest: np.ndarray
    #: The part of ``mint`` paid by the flat floor ``alpha`` rather than by a
    #: marginal contribution — §8's "share of mint from alpha" instrument.
    #: ``mint - mint_floor`` is what the health term actually bought.
    mint_floor: np.ndarray
    #: Share of offered swarm capacity actually drawn, over torrents that
    #: had at least one leecher, and share of leecher-hours spent on a swarm
    #: with two or more concurrent leechers.  Together they say whether
    #: capacity beyond one reference downlink is reaching anybody.
    served_share: np.ndarray
    multi_leech_share: np.ndarray
    #: (T, 3) p10 / p50 / p90, over the *demand* downloads completed in the
    #: recording interval, of time taken relative to time at the reference
    #: rate: ``hours to complete / (size / min(down, d_ref))``.  This is the
    #: experienced counterpart of ``C_i`` — one number per download, no
    #: estimator and no scoring rule in it — so it reads the same under
    #: every rule.  Seeders' own re-planning acquisitions are excluded.
    download_slowdown_quantiles: np.ndarray
    #: (T, 3) the same ratio with each download weighted by its size, i.e.
    #: the slowdown per GiB downloaded.  Small files dominate the count and
    #: large ones the bytes; a rule can only be judged on both.
    download_slowdown_by_bytes: np.ndarray
    #: Demand downloads outstanding for more than 24 h at the record time.
    #: A download parked on a torrent whose only holder is rarely online
    #: waits for weeks — this engine has no giving up — and it belongs in a
    #: count of its own rather than in the completion-time quantiles.
    demand_backlog: np.ndarray
    #: Demand downloads accepted per recording interval, so the backlog can
    #: be read against what the site is asked for.
    demand_started: np.ndarray
    #: Demand requests per recording interval for a torrent no complete copy
    #: of exists.  They never become downloads, so the completion-time
    #: quantiles cannot see them; this is the survivorship denominator.
    demand_unsourced: np.ndarray
    #: Share of leecher-hours that are seeders' re-planning acquisitions
    #: rather than demand.  Those transfers draw on the same uplinks as
    #: demand does, so a rule that churns the library is also a rule that
    #: spends swarm capacity on moving copies around.
    invest_leech_share: np.ndarray
    #: Mean of the estimator's concurrency ``k_tilde`` over nodes — how many
    #: of a seeder's torrents have a leecher at once, which divides its
    #: credited contribution ``g_v`` and therefore every ``x_i`` it is in.
    concurrency_mean: np.ndarray
    #: Memberships released / acquired by re-planning, per recording interval.
    #: A holding whose marginal has collapsed is worth only ``alpha`` per GiB,
    #: so churn is how the mechanism recycles disk toward torrents that still
    #: pay; a rule under which nothing ever becomes worthless stops moving.
    replan_drops: np.ndarray
    replan_joins: np.ndarray
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
    #: ``A_i`` at the end of the run.  Reported separately from the slowdown
    #: because under a rule without the completeness cutoff ``C_i`` stays
    #: finite on a torrent no complete copy is reliably online for, and the
    #: comparison in part 4 needs the physical quantity in both worlds.
    final_completeness: np.ndarray = field(repr=False)
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
            "median_C_phys": float(np.mean(self.c_physical_quantiles[sl, 1])),
            "avail_median_C_phys": float(np.mean(self.c_physical_avail[sl, 1])),
            "avail_p90_C_phys": float(np.mean(self.c_physical_avail[sl, 2])),
            "served_share": float(np.nanmean(self.served_share[sl])),
            "multi_leech_share": float(np.nanmean(self.multi_leech_share[sl])),
            "download_p50": float(np.nanmean(self.download_slowdown_quantiles[sl, 1])),
            "download_p90": float(np.nanmean(self.download_slowdown_quantiles[sl, 2])),
            "bytes_p50": float(np.nanmean(self.download_slowdown_by_bytes[sl, 1])),
            "bytes_p90": float(np.nanmean(self.download_slowdown_by_bytes[sl, 2])),
            "demand_backlog": float(np.mean(self.demand_backlog[sl])),
            # Backlog in units of one day's accepted demand.
            "demand_backlog_days": float(
                np.mean(self.demand_backlog[sl])
                / max(self.demand_started[sl].sum() / span_days, 1e-9)
            ),
            # Share of demand that asked for a torrent no copy of exists.
            "demand_unsourced_share": float(
                self.demand_unsourced[sl].sum()
                / max(
                    self.demand_unsourced[sl].sum() + self.demand_started[sl].sum(),
                    1e-9,
                )
            ),
            "invest_leech_share": float(np.nanmean(self.invest_leech_share[sl])),
            "concurrency_mean": float(np.mean(self.concurrency_mean[sl])),
            # ptp over a series that may contain inf is nan; report the swing
            # of the finite part and let oscillation() carry the real signal.
            "swing_C": float(np.ptp(finite_median))
            if finite_median.size
            else float("nan"),
            "oscillation": self.oscillation(),
            "unavailable_fraction": float(np.mean(self.unavailable_fraction[sl])),
            "mint_burn": float(self.mint[sl].sum() / max(self.burn[sl].sum(), 1e-9)),
            "alpha_share": float(
                self.mint_floor[sl].sum() / max(self.mint[sl].sum(), 1e-9)
            ),
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
        self.lch_time = np.empty(0)
        #: Completion-time ratios and sizes of demand downloads finished
        #: since the recorder last drained them.
        self.finished: list[np.ndarray] = []
        self.finished_size: list[np.ndarray] = []
        #: Demand requests refused because no copy exists, since the
        #: recorder last drained the counter.
        self.demand_unsourced = 0

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
        """One hour of actual transfers; returns per-node observed rates.

        Each leecher is served at ``min(own downlink, equal share of the
        swarm's online uplink)``.  The estimator observes that rate — what a
        10-minute announce delta would show while the transfer is running —
        and so does ``served_share``; the mechanism is untouched by the rest
        of this method.  Bytes moved are capped by what the download had
        left, and the fraction of the hour a leecher was actually
        transferring is its weight in every leecher-side statistic, so a file
        that finishes in five minutes is not a full hour of experience.
        Uplink freed by an early finisher is not re-split within the hour.
        """
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
        moved = np.zeros(self.lch_node.size)
        weight = np.zeros(self.lch_node.size)
        if rate.size:
            lt = self.lch_torrent
            share = capacity_true[lt] / np.maximum(leech_count[lt], 1)
            rate = np.minimum(self.down[self.lch_node], share)
            possible = rate * 3600.0 / MEGABIT_PER_GIB
            moved = np.minimum(possible, self.lch_left)
            weight = np.divide(
                moved, possible, out=np.zeros_like(moved), where=possible > 0.0
            )
            self.lch_left -= moved

        served = np.bincount(self.lch_torrent, rate, minlength=self.n_t)
        frac = np.where(
            capacity_true > 0.0, served / np.maximum(capacity_true, 1e-12), 0.0
        )
        node_rate = np.bincount(
            self.mem_node, offer * frac[self.mem_torrent], minlength=self.n_n
        )
        # Demand-side telemetry.  ``served_share`` is the share of the
        # *instantaneous* offered rate that leechers could draw on torrents
        # with a leecher this hour — an average utilisation of allocated
        # rate, not bytes over the hour; it says how often a leecher's own
        # line rather than the swarm is the binding constraint, and nothing
        # about whether slack matters in the tail.  ``multi_leech_share`` is
        # the fraction of leecher time spent on a swarm with at least two
        # concurrent leechers.
        busy = leech_count > 0
        offered = float(capacity_true[busy].sum())
        self.served_share = float(served.sum() / offered) if offered > 0.0 else np.nan
        # Elapsed time per download: a transferring hour counts the fraction
        # actually used, a stalled hour (no seeder online) counts in full.
        elapsed = np.where(rate > 0.0, weight, 1.0)
        self.lch_time += elapsed
        total = float(elapsed.sum())
        self.multi_leech_share = (
            float((elapsed * (leech_count[self.lch_torrent] >= 2)).sum() / total)
            if total > 0.0
            else np.nan
        )
        self.invest_leech_share = (
            float((elapsed * self.lch_invest).sum() / total) if total > 0.0 else np.nan
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
        c = self.r.torrent_slowdown(capacity, completeness, self.d_ref)
        return capacity, completeness, log_miss, c

    def _pair_reward(self, g, p_tilde, capacity, completeness, log_miss, target):
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
        h_full = self.r.torrent_health(
            capacity[t], completeness[t], self.d_ref, target[t]
        )
        h_loo = self.r.torrent_health(
            capacity[t] - g_pair,
            -np.expm1(log_miss[t] - miss_pair),
            self.d_ref,
            target[t],
        )
        delta = h_full - h_loo
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
        they say: structural rejections (no complete copy anywhere, already
        held, already downloading, duplicate in this batch) are removed
        *first*, then the per-node disk prefix, and only the survivors are
        tested against the per-user balance prefix.  Running the prefixes over
        rejected rows would let a request that was never going to happen make
        a later valid one look unaffordable, inflating ``broke``.

        Every member holds a complete copy, so a torrent with no member has no
        copy anywhere and a download of it could never progress; it would
        still have burned the points and reserved the disk.  Such requests are
        refused here, for demand and re-planning alike, and never counted as
        ``broke``.
        """
        if nodes.size == 0:
            return 0, 0.0, 0
        size = self.size[torrents]
        sourced = np.bincount(self.mem_torrent, minlength=self.n_t) > 0
        if not invest:
            self.demand_unsourced += int((~sourced[torrents]).sum())
        structural = (
            sourced[torrents]
            & ~self.holds[nodes, torrents]
            & ~self.fetching[nodes, torrents]
        )
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
        self.lch_time = np.concatenate([self.lch_time, np.zeros(nodes.size)])
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
        demand = done & ~self.lch_invest
        if demand.any():
            ideal_h = (
                self.size[self.lch_torrent[demand]]
                * MEGABIT_PER_GIB
                / 3600.0
                / np.minimum(self.down[self.lch_node[demand]], self.d_ref)
            )
            self.finished.append(self.lch_time[demand] / ideal_h)
            self.finished_size.append(self.size[self.lch_torrent[demand]])
        alive = ~done
        self.lch_node = self.lch_node[alive]
        self.lch_torrent = self.lch_torrent[alive]
        self.lch_left = self.lch_left[alive]
        self.lch_invest = self.lch_invest[alive]
        self.lch_time = self.lch_time[alive]
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

        Returns ``(drops, joins, burn)``: memberships released, acquisitions
        started, and the points those acquisitions burned.  An investment
        download is charged exactly like a demand download, so its burn is
        part of the monetary loop of §3.7 and not a side effect to discard.
        """
        chosen = np.flatnonzero(self.rng.random(self.n_n) < self.p.decision_share)
        if chosen.size == 0 or self.mem_node.size == 0:
            return 0, 0, 0.0
        k_nodes = chosen.size
        moves = int(self.p.moves_per_decision)

        per_member = (
            pair_rate[self.pair_inv]
            / self.pair_count[self.pair_inv]
            / self.size[self.mem_torrent]
        )
        # A pinned uploader never becomes the "worst holding", and neither
        # does the only copy of a torrent somebody is downloading: an
        # accepted download keeps a source until it completes.  Without
        # this a swap can strand a leecher on a torrent no copy of exists.
        members = np.bincount(self.mem_torrent, minlength=self.n_t)
        leeching = np.bincount(self.lch_torrent, minlength=self.n_t) > 0
        held_fast = (self.pin_node[self.mem_torrent] == self.mem_node) | (
            (members[self.mem_torrent] == 1) & leeching[self.mem_torrent]
        )
        per_member = np.where(held_fast, np.inf, per_member)
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
        gain = self.r.torrent_health(
            cap_c + g_v, joined, self.d_ref, target_c
        ) - self.r.torrent_health(cap_c, avail_c, self.d_ref, target_c)
        size_c = self.size[cand]
        value = (
            self.kappa
            * self.weight[cand]
            * self.rel[cand] ** self.r.eta
            * (self.r.alpha + gain)
            / size_c
        )
        # A torrent nobody holds cannot be acquired, whatever it would pay.
        sourced = np.bincount(self.mem_torrent, minlength=self.n_t) > 0
        usable = (
            sourced[cand]
            & (self.charge[cand] * size_c <= self.balance[self.user[chosen]][:, None])
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

        # The per-holding guard above sees the membership count *before* the
        # batch; several deciding nodes can each release one copy of the same
        # torrent in one tick.  Cancel the surplus swaps so every torrent
        # being downloaded keeps at least one member.
        if take_drop.any():
            drop_torrent = self.mem_torrent[hold_row[take_drop]]
            order = np.argsort(drop_torrent, kind="stable")
            sorted_t = drop_torrent[order]
            fresh = np.ones(order.size, dtype=bool)
            fresh[1:] = sorted_t[1:] != sorted_t[:-1]
            starts = np.flatnonzero(fresh)
            rank = np.arange(order.size) - np.repeat(
                starts, np.diff(np.append(starts, order.size))
            )
            allowed = np.where(leeching[sorted_t], members[sorted_t] - 1, order.size)
            cancel = np.zeros(order.size, dtype=bool)
            cancel[order] = rank >= allowed
            if cancel.any():
                j_c, k_c = np.argwhere(take_drop)[cancel].T
                take_drop[j_c, k_c] = False
                take_cand[j_c, k_c] = False
        drops = int(take_drop.sum())
        if drops:
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
        started, burned, _ = self._start_downloads(nodes, torrents, True)
        return drops, started, burned


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
    mint_floor_h = np.zeros(total_h)
    burn_h = np.zeros(total_h)
    refused_h = np.zeros(total_h)
    invest_h = np.zeros(total_h)
    drops_h = np.zeros(total_h)
    joins_h = np.zeros(total_h)
    served_share_h = np.full(total_h, np.nan)
    multi_leech_h = np.full(total_h, np.nan)
    invest_leech_h = np.full(total_h, np.nan)
    demand_started_h = np.zeros(total_h)
    tracked = np.argsort(-catalog.demand_share)[: params.n_tracked]
    starts_per_hour = params.starts_per_user_day * params.n_users / 24.0
    n_classes = len(catalog.class_names)

    rec: dict[str, list] = {
        k: []
        for k in (
            "hours",
            "cqall",
            "cq",
            "cqphys",
            "cqphysavail",
            "cclass",
            "dead",
            "hmean",
            "dmean",
            "mint",
            "mintfloor",
            "invest",
            "drops",
            "joins",
            "servedshare",
            "multileech",
            "downloadq",
            "downloadqb",
            "backlog",
            "demandstarted",
            "demandunsourced",
            "investleech",
            "conc",
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
        served_share_h[hour] = site.served_share
        multi_leech_h[hour] = site.multi_leech_share
        invest_leech_h[hour] = site.invest_leech_share

        p_tilde, _, g = site.est.contribution()
        capacity, completeness, log_miss, c = site._torrent_state(g, p_tilde)
        h = site.r.torrent_health(capacity, completeness, site.d_ref, site.target)
        pair_rate, delta = site._pair_reward(
            g, p_tilde, capacity, completeness, log_miss, site.target
        )

        online_pair = np.bincount(
            site.pair_inv,
            online[site.mem_node].astype(float),
            minlength=site.pair_torrent.size,
        )
        earned = np.where(online_pair > 0, pair_rate, 0.0)
        site.balance += np.bincount(site.pair_user, earned, minlength=site.n_u)
        mint_h[hour] = earned.sum()
        # r = kappa w S**eta (alpha + dH), so the flat-floor part of a pair's
        # rate is that same rate scaled by alpha / (alpha + dH).  Recovering it
        # this way costs one divide instead of a second reward pass.
        floor_part = np.divide(
            reward.alpha,
            reward.alpha + delta,
            out=np.zeros_like(delta),
            where=earned > 0.0,
        )
        mint_floor_h[hour] = float((earned * floor_part).sum())

        n_start = site.rng.poisson(starts_per_hour)
        nodes = site.rng.integers(0, site.n_n, size=n_start)
        torrents = site.rng.choice(site.n_t, size=n_start, p=site.demand)
        started, burned, refused = site._start_downloads(nodes, torrents, False)
        demand_started_h[hour] = started
        burn_h[hour] = burned
        refused_h[hour] = refused

        if hour % params.decide_every_h == 0:
            incumbent = pair_rate
            if params.stay_bonus != 1.0:
                incumbent, _ = site._pair_reward(
                    g,
                    p_tilde,
                    capacity,
                    completeness,
                    log_miss,
                    site.target * params.stay_bonus,
                )
            drops, joins, invested = site._reallocate(
                g, p_tilde, capacity, completeness, incumbent
            )
            # A seeder acquiring a torrent pays for it exactly like a leecher
            # does, so those points leave the economy and belong in the burn
            # the kappa controller reads.
            burn_h[hour] += invested
            invest_h[hour] = invested
            drops_h[hour] = drops
            joins_h[hour] = joins
            dirty |= bool(drops or joins)
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
            # "Available" is the *physical* condition A_i >= pi_min, not
            # "C_i came out finite": a rule that drops the completeness cutoff
            # still leaves the catalogue with torrents no complete copy is
            # reliably online for, and they must keep being counted.
            available = (completeness >= reward.pi_min) & np.isfinite(c)
            rec["hours"].append(float(hour + 1))
            # method="lower" is an order statistic, so inf never enters an
            # interpolation and the unavailable tail reads inf, not nan.
            rec["cqall"].append(np.quantile(c, [0.1, 0.5, 0.9], method="lower"))
            physical = np.where(
                capacity > 0.0, site.d_ref / np.maximum(capacity, 1e-300), np.inf
            )
            rec["cqphys"].append(np.quantile(physical, [0.1, 0.5, 0.9], method="lower"))
            rec["cqphysavail"].append(
                np.quantile(physical[available], [0.1, 0.5, 0.9], method="lower")
                if available.any()
                else [np.nan] * 3
            )
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
            rec["mintfloor"].append(float(mint_floor_h[span].sum()))
            rec["burn"].append(float(burn_h[span].sum()))
            rec["invest"].append(float(invest_h[span].sum()))
            rec["drops"].append(float(drops_h[span].sum()))
            rec["joins"].append(float(joins_h[span].sum()))
            rec["servedshare"].append(_mean_or_nan(served_share_h[span]))
            rec["multileech"].append(_mean_or_nan(multi_leech_h[span]))
            finished = np.concatenate(site.finished) if site.finished else np.empty(0)
            finished_size = (
                np.concatenate(site.finished_size)
                if site.finished_size
                else np.empty(0)
            )
            site.finished.clear()
            site.finished_size.clear()
            rec["downloadq"].append(
                np.quantile(finished, [0.1, 0.5, 0.9], method="lower")
                if finished.size
                else [np.nan] * 3
            )
            rec["downloadqb"].append(
                _weighted_quantile(finished, finished_size, (0.1, 0.5, 0.9))
                if finished.size
                else [np.nan] * 3
            )
            rec["backlog"].append(
                float((~site.lch_invest & (site.lch_time > 24.0)).sum())
            )
            rec["demandstarted"].append(float(demand_started_h[span].sum()))
            rec["demandunsourced"].append(float(site.demand_unsourced))
            site.demand_unsourced = 0
            rec["investleech"].append(_mean_or_nan(invest_leech_h[span]))
            rec["conc"].append(float(site.est.concurrency.mean()))
            rec["refused"].append(float(refused_h[span].sum()))
            rec["kappa"].append(float(site.kappa))
            rec["members"].append(float(site.mem_node.size))
            rec["tracked"].append(
                np.bincount(site.mem_torrent, minlength=site.n_t)[tracked].astype(float)
            )
            rec["bq"].append(np.quantile(site.balance, [0.1, 0.5, 0.9]))

    p_tilde, _, g = site.est.contribution()
    capacity, completeness, _, c = site._torrent_state(g, p_tilde)
    return SimResult(
        hours=np.asarray(rec["hours"]),
        c_quantiles_all=np.asarray(rec["cqall"], dtype=float),
        c_quantiles=np.asarray(rec["cq"], dtype=float),
        c_physical_quantiles=np.asarray(rec["cqphys"], dtype=float),
        c_physical_avail=np.asarray(rec["cqphysavail"], dtype=float),
        c_by_class=np.asarray(rec["cclass"], dtype=float),
        unavailable_fraction=np.asarray(rec["dead"]),
        health_mean=np.asarray(rec["hmean"]),
        delta_health_mean=np.asarray(rec["dmean"]),
        mint=np.asarray(rec["mint"]),
        mint_floor=np.asarray(rec["mintfloor"]),
        burn=np.asarray(rec["burn"]),
        burn_invest=np.asarray(rec["invest"]),
        served_share=np.asarray(rec["servedshare"]),
        multi_leech_share=np.asarray(rec["multileech"]),
        download_slowdown_quantiles=np.asarray(rec["downloadq"], dtype=float),
        download_slowdown_by_bytes=np.asarray(rec["downloadqb"], dtype=float),
        demand_backlog=np.asarray(rec["backlog"], dtype=float),
        demand_started=np.asarray(rec["demandstarted"], dtype=float),
        demand_unsourced=np.asarray(rec["demandunsourced"], dtype=float),
        invest_leech_share=np.asarray(rec["investleech"], dtype=float),
        concurrency_mean=np.asarray(rec["conc"], dtype=float),
        replan_drops=np.asarray(rec["drops"]),
        replan_joins=np.asarray(rec["joins"]),
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
        final_completeness=completeness,
        final_members=np.bincount(site.mem_torrent, minlength=site.n_t),
        final_contribution=g,
        final_balance=site.balance.copy(),
        user_disk_budget=np.bincount(site.user, site.disk, minlength=site.n_u),
        record_every_h=params.record_every_h,
        pinned_coverage=site.pinned_coverage,
    )
