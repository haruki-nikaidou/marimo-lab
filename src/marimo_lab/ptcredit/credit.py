"""Health, reward and the per-node estimators of the point-system handoff.

Everything here is vectorised over nodes / torrents / memberships so the agent
simulation can run a whole site-hour in a handful of numpy passes.

Section numbers refer to ``point-system-handoff.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

__all__ = [
    "NodeEstimator",
    "RewardParams",
    "health",
    "health_marginal",
    "reward_rate",
    "slowdown",
    "slowdown_plain",
    "softplus",
    "target_capacity",
]


def softplus(t):
    """``ln(1 + e**t)``, overflow-free."""
    return np.logaddexp(0.0, np.asarray(t, dtype=float))


@dataclass(frozen=True)
class RewardParams:
    """Global parameters of §3.6.

    ``importance_mode`` selects the open decision of §9: ``"pay"`` scales the
    reward by ``w_i = 2**z_i`` (the current design), ``"target"`` instead moves
    the per-torrent target to ``C* * 2**-z_i`` and pays every torrent alike.

    ``eta`` defaults to ``1.0``, the size-neutral exponent derived in part 2:
    only at ``eta = 1`` does a GiB of disk earn the same reward whatever the
    torrent's size. Earlier drafts of §3.6 started it at ``0.7``.

    ``slowdown_rule`` selects which of the two clamps of §3.3 are applied when
    the mechanism scores a torrent, which is what part 4 puts in motion:

    ``"clamped"``
        both, i.e. §3.3 as written — ``max(1, d_ref/x)`` and
        ``A_i < pi_min -> inf``;
    ``"no_floor"``
        the completeness cutoff only, so capacity beyond ``d_ref`` keeps
        earning;
    ``"raw"``
        neither: the bare ratio ``d_ref / x`` of part 2 §3, under which
        availability is priced only through the ``p_tilde`` discount already
        inside ``g_v``.
    """

    kappa: float = 1.0
    alpha: float = 0.05
    gamma: float = 4.0
    c_star: float = 1.5
    eta: float = 1.0
    pi_min: float = 0.9
    m: float = 24.0
    half_life_h: float = 14.0 * 24.0
    importance_mode: str = "pay"
    slowdown_rule: str = "clamped"

    def torrent_target(self, importance) -> np.ndarray:
        """Per-torrent ``C*``."""
        importance = np.asarray(importance, dtype=float)
        if self.importance_mode == "target":
            return self.c_star * 2.0**-importance
        return np.full(importance.shape, self.c_star)

    def torrent_weight(self, importance) -> np.ndarray:
        """Per-torrent ``w_i`` used in the pay-out."""
        importance = np.asarray(importance, dtype=float)
        if self.importance_mode == "target":
            return np.ones(importance.shape)
        return 2.0**importance

    def torrent_slowdown(self, capacity, completeness, d_ref) -> np.ndarray:
        """``C_i`` under the configured ``slowdown_rule``.

        The mechanism scores torrents through this method and nothing else, so
        a rule change reaches the health, the leave-one-out marginal and the
        seeders' own candidate valuation at once.
        """
        ratio = slowdown_plain(capacity, d_ref)
        if self.slowdown_rule == "raw":
            return ratio
        if self.slowdown_rule != "no_floor":
            ratio = np.maximum(1.0, ratio)
        return np.where(
            np.asarray(completeness, dtype=float) >= self.pi_min, ratio, np.inf
        )


def health(slowdown_c, c_star=1.5, gamma=4.0):
    """``H(C) = 1 - sp(gamma (1 - C*/C)) / sp(gamma)`` (§3.3).

    ``C = inf`` (no complete copy) gives exactly 0.
    """
    slowdown_c = np.asarray(slowdown_c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(slowdown_c), c_star / slowdown_c, 0.0)
    return 1.0 - softplus(gamma * (1.0 - ratio)) / softplus(gamma)


def slowdown(capacity, completeness, d_ref, pi_min=0.9):
    """``C_i``: predicted slowdown, with both clamps of §3.2.

    Two clamps sit on top of the raw ratio ``d_ref / x``:

    * ``max(1, .)`` — a leecher can never beat their own downlink, so swarm
      capacity above ``d_ref`` buys that leecher nothing;
    * ``A_i < pi_min -> inf`` — capacity from nodes that are rarely online
      simultaneously does not constitute a servable copy.

    :func:`slowdown_plain` is the same quantity with neither clamp; part 2 of
    the notebook series compares the two.
    """
    capacity = np.asarray(capacity, dtype=float)
    completeness = np.asarray(completeness, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(capacity > 0.0, d_ref / np.maximum(capacity, 1e-300), np.inf)
    return np.where(completeness >= pi_min, np.maximum(1.0, ratio), np.inf)


def slowdown_plain(capacity, d_ref):
    """``C_i = d_ref / x_i``, unclamped — the comparison rule of part 2.

    Availability does not enter, and values below 1 are allowed, so health
    keeps rising with capacity a single leecher could never consume.
    """
    capacity = np.asarray(capacity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(capacity > 0.0, d_ref / np.maximum(capacity, 1e-300), np.inf)


def target_capacity(d_ref, c_star=1.5):
    """``x* = d_ref / C*``: the capacity the knee is aimed at."""
    return d_ref / c_star


def health_marginal(capacity, d_ref, c_star=1.5, gamma=4.0):
    """``dH/dx`` at capacity ``x`` (§3.3), ignoring the floor at ``C = 1``."""
    capacity = np.asarray(capacity, dtype=float)
    x_star = target_capacity(d_ref, c_star)
    logistic = 1.0 / (1.0 + np.exp(-gamma * (1.0 - capacity / x_star)))
    return gamma / (x_star * softplus(gamma)) * logistic


def reward_rate(params: RewardParams, weight, rel_size, delta_health):
    """``r = kappa w S**eta (alpha + dH)`` in points per hour (§3.4)."""
    return (
        params.kappa
        * np.asarray(weight, dtype=float)
        * np.asarray(rel_size, dtype=float) ** params.eta
        * (params.alpha + np.asarray(delta_health, dtype=float))
    )


class NodeEstimator:
    """Pessimistic per-node availability and capacity estimates (§3.2).

    The handoff stores upload-rate evidence in a decayed t-digest and reads its
    90th percentile.  Here the digest is replaced by an exponentially weighted
    ring buffer of the last ``buffer`` qualifying observations, which has the
    same decayed-quantile semantics at this scale and is exactly reproducible.

    Only *qualifying* intervals — those in which the node had at least one
    leecher on some torrent — feed the capacity estimate, so an idle seeder is
    never punished for showing a zero rate.
    """

    def __init__(
        self,
        n_nodes: int,
        *,
        m: float = 24.0,
        half_life_h: float = 14.0 * 24.0,
        buffer: int = 48,
        quantile: float = 0.9,
        pessimism: float = 0.25,
        prior_p: float = 0.55,
        prior_b: float = 20.0,
        k_half_life_h: float = 24.0,
    ):
        n_nodes = int(n_nodes)
        self.m = float(m)
        self.decay_factor = 0.5 ** (1.0 / float(half_life_h))
        self.quantile = float(quantile)
        self.pessimism = float(pessimism)
        self.k_alpha = 1.0 - 0.5 ** (1.0 / float(k_half_life_h))

        self.on = np.zeros(n_nodes)
        self.off = np.zeros(n_nodes)
        self.n_qual = np.zeros(n_nodes)
        self.concurrency = np.ones(n_nodes)

        self.rates = np.zeros((n_nodes, int(buffer)))
        self.age = np.full((n_nodes, int(buffer)), np.inf)
        self.slot = np.zeros(n_nodes, dtype=np.int64)

        self.pop_p = float(prior_p)
        self.pop_b = float(prior_b)
        self.prior_p = np.full(n_nodes, float(prior_p))
        self.prior_b = np.full(n_nodes, float(prior_b))

    # -- evidence ----------------------------------------------------------

    def decay(self, hours: float = 1.0) -> None:
        factor = self.decay_factor**hours
        self.on *= factor
        self.off *= factor
        self.n_qual *= factor
        self.age += hours

    def observe(self, active, online, qualifying, rates, concurrency, hours=1.0):
        """Fold one interval of evidence in.

        ``active`` masks nodes that exist yet; ``online`` those seen this
        interval; ``qualifying`` those that had a leecher to serve.
        """
        seen = np.asarray(active, dtype=bool)
        up = np.asarray(online, dtype=bool) & seen
        self.on += np.where(up, hours, 0.0)
        self.off += np.where(seen & ~up, hours, 0.0)

        qual = np.asarray(qualifying, dtype=bool) & up
        idx = np.flatnonzero(qual)
        if idx.size:
            slot = self.slot[idx]
            self.rates[idx, slot] = np.asarray(rates, dtype=float)[idx]
            self.age[idx, slot] = 0.0
            self.slot[idx] = (slot + 1) % self.rates.shape[1]
            self.n_qual[idx] += hours

        k_obs = np.maximum(1.0, np.asarray(concurrency, dtype=float))
        self.concurrency[up] += self.k_alpha * (k_obs[up] - self.concurrency[up])

    # -- estimates ---------------------------------------------------------

    def observed_capacity(self) -> np.ndarray:
        """``b_hat``: exponentially weighted ``quantile`` of observed rates."""
        weight = np.where(np.isfinite(self.age), self.decay_factor**self.age, 0.0)
        order = np.argsort(self.rates, axis=1, kind="stable")
        rates = np.take_along_axis(self.rates, order, axis=1)
        weight = np.take_along_axis(weight, order, axis=1)
        cumulative = np.cumsum(weight, axis=1)
        total = cumulative[:, -1]
        hit = cumulative >= (self.quantile * total)[:, None]
        pick = np.argmax(hit, axis=1)
        value = np.take_along_axis(rates, pick[:, None], axis=1)[:, 0]
        return np.where(total > 0.0, value, self.prior_b)

    def availability(self) -> np.ndarray:
        """``p_tilde``: lower quantile of the Beta posterior."""
        a = self.m * self.prior_p + self.on
        b = self.m * (1.0 - self.prior_p) + self.off
        return beta_dist.ppf(self.pessimism, a, b)

    def capacity(self) -> np.ndarray:
        """``b_tilde``: prior-shrunk observed capacity."""
        b_hat = self.observed_capacity()
        return (self.m * self.prior_b + self.n_qual * b_hat) / (self.m + self.n_qual)

    def contribution(self):
        """``(p_tilde, b_tilde, g)`` with ``g = p b / k``."""
        p = self.availability()
        b = self.capacity()
        return p, b, p * b / self.concurrency

    def refresh_population_prior(self, active) -> tuple[float, float]:
        """Daily job of §3.2: median ``p_tilde`` and 25th-pct ``b_tilde``.

        Only nodes with at least ``m`` hours of qualifying evidence count, so
        the prior is not dragged down by the nodes that are still sitting on it.
        """
        mature = np.asarray(active, dtype=bool) & (self.n_qual >= self.m)
        if mature.sum() >= 8:
            p, b, _ = self.contribution()
            self.pop_p = float(np.median(p[mature]))
            self.pop_b = float(np.quantile(b[mature], 0.25))
            self.prior_p[:] = self.pop_p
            self.prior_b[:] = self.pop_b
        return self.pop_p, self.pop_b
