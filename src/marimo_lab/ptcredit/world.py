"""Torrent catalogue and seeder population for the point-system simulations.

Units, everywhere in :mod:`marimo_lab.ptcredit`:

* **size** in GiB,
* **rate** in Mbps (decimal megabits per second),
* **time** in hours.

Size and rate meet only through :data:`MEGABIT_PER_GIB`.

The two exogenous distributions are fixed by the study brief:

* torrent size is a four-component mixture — small ``Lognormal(0, 0.5)``,
  medium ``Normal(12, 10)``, large ``Lognormal(4.5, 3)``, all truncated to a
  support, plus a handful of super-large torrents up to 2 TiB;
* node bandwidth is a **2-D truncated lognormal** with median 300 Mbps down /
  50 Mbps up and caps at 10 Gbps / 2 Gbps.

The bandwidth medians are the medians *of the truncated law*, not of the
underlying lognormal: :func:`calibrate_bandwidth` solves for the two log-means
that put the truncated marginal medians exactly on target.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import root
from scipy.stats import beta as beta_dist
from scipy.stats import lognorm, multivariate_normal, norm

__all__ = [
    "DEFAULT_CHARGE_PROBS",
    "DEFAULT_IMPORTANCE_PROBS",
    "DEFAULT_MIX",
    "DEFAULT_SIZE_CLASSES",
    "MEGABIT_PER_GIB",
    "BandwidthModel",
    "Catalog",
    "Population",
    "SizeClass",
    "calibrate_bandwidth",
    "hours_to_transfer",
    "sample_catalog",
    "sample_population",
]

#: 1 GiB expressed in decimal megabits, so ``gib * MEGABIT_PER_GIB / mbps``
#: is a duration in seconds.
MEGABIT_PER_GIB = 8.0 * 2.0**30 / 1.0e6


def hours_to_transfer(size_gib, rate_mbps):
    """Hours needed to move ``size_gib`` at ``rate_mbps`` (``inf`` at rate 0)."""
    size_gib = np.asarray(size_gib, dtype=float)
    rate_mbps = np.asarray(rate_mbps, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hours = size_gib * MEGABIT_PER_GIB / (rate_mbps * 3600.0)
    return np.where(rate_mbps > 0.0, hours, np.inf)


# --------------------------------------------------------------------------
# torrent sizes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeClass:
    """One component of the size mixture, truncated to ``[lo, hi]`` GiB.

    The second parameter is a **variance**, matching the brief's
    ``Lognormal(4.5, 3)`` / ``Normal(12, 10)`` notation; scipy wants a scale,
    so the square root is taken here rather than at every call site.  Getting
    this wrong silently rescales the whole catalogue, so the two are named
    apart:

    * ``"lognormal"`` — ``location`` is the mean of the log, ``variance`` the
      variance of the log;
    * ``"normal"`` — ``location`` and ``variance`` on the linear scale;
    * ``"loguniform"`` — both unused, the support is the distribution.

    Truncation is by inverse CDF, i.e. the *conditional* law on ``[lo, hi]``;
    no atom is piled up on the bounds.
    """

    name: str
    family: str
    location: float
    variance: float
    lo: float
    hi: float

    @property
    def sd(self) -> float:
        """Standard deviation, on whichever scale the family lives."""
        return float(np.sqrt(self.variance))

    def frozen(self):
        """The untruncated scipy distribution behind this class."""
        if self.family == "lognormal":
            return lognorm(s=self.sd, scale=float(np.exp(self.location)))
        if self.family == "normal":
            return norm(loc=self.location, scale=self.sd)
        if self.family == "loguniform":
            return None
        raise ValueError(f"unknown size family: {self.family!r}")

    @property
    def retained_mass(self) -> float:
        """Probability the untruncated law already falls inside ``[lo, hi]``.

        Far below 1 means the stated parameters are doing much less work than
        the truncation bounds; the world-model notebook reports it.
        """
        dist = self.frozen()
        if dist is None:
            return 1.0
        return float(dist.cdf(self.hi) - dist.cdf(self.lo))

    def ppf(self, q):
        """Quantile function of the truncated class."""
        q = np.asarray(q, dtype=float)
        dist = self.frozen()
        if dist is None:
            return self.lo * (self.hi / self.lo) ** q
        lo_c, hi_c = dist.cdf(self.lo), dist.cdf(self.hi)
        return np.asarray(dist.ppf(lo_c + q * (hi_c - lo_c)), dtype=float)

    def sample(self, n: int, rng: np.random.Generator):
        return self.ppf(rng.random(int(n)))


#: small / medium / large / super-large, with the brief's variances.  The
#: large class caps where the super-large class starts, so the two do not
#: overlap.
DEFAULT_SIZE_CLASSES: tuple[SizeClass, ...] = (
    SizeClass("small", "lognormal", 0.0, 0.5, 0.05, 20.0),
    SizeClass("medium", "normal", 12.0, 10.0, 0.5, 64.0),
    SizeClass("large", "lognormal", 4.5, 3.0, 20.0, 512.0),
    SizeClass("super", "loguniform", 0.0, 0.0, 512.0, 2048.0),
)

#: counts share of small / medium / large; super-large is an absolute count.
DEFAULT_MIX: tuple[float, float, float] = (0.30, 0.40, 0.30)

#: staff importance z_i in [-2, 2]; most torrents are ordinary.
DEFAULT_IMPORTANCE_PROBS: tuple[float, ...] = (0.02, 0.10, 0.76, 0.10, 0.02)

#: charge multiplier phi_i over (0, 1/2, 1).
DEFAULT_CHARGE_PROBS: tuple[float, float, float] = (0.10, 0.15, 0.75)


@dataclass(frozen=True)
class Catalog:
    """A site's torrent catalogue."""

    size_gib: np.ndarray
    class_idx: np.ndarray
    importance: np.ndarray
    charge: np.ndarray
    demand_share: np.ndarray
    class_names: tuple[str, ...]

    @property
    def n_torrents(self) -> int:
        return int(self.size_gib.size)

    @property
    def median_size(self) -> float:
        """Site median torrent size, the ``S_bar`` of the handoff."""
        return float(np.median(self.size_gib))

    @property
    def weight(self) -> np.ndarray:
        """``w_i = 2**z_i``."""
        return 2.0**self.importance

    @property
    def rel_size(self) -> np.ndarray:
        """``S_i = size_i / S_bar``."""
        return self.size_gib / self.median_size


def _mixture_counts(n_main: int, mix) -> np.ndarray:
    """Split ``n_main`` into shares with the largest-remainder rule."""
    weights = np.asarray(mix, dtype=float)
    weights = weights / weights.sum()
    exact = weights * n_main
    counts = np.floor(exact).astype(np.int64)
    short = n_main - int(counts.sum())
    if short:
        order = np.argsort(-(exact - counts))
        counts[order[:short]] += 1
    return counts


def sample_catalog(
    n_torrents: int,
    rng: np.random.Generator,
    *,
    mix=DEFAULT_MIX,
    classes: tuple[SizeClass, ...] = DEFAULT_SIZE_CLASSES,
    n_super: int = 3,
    zipf_s: float = 1.0,
    zipf_offset: float = 10.0,
    importance_probs=DEFAULT_IMPORTANCE_PROBS,
    charge_probs=DEFAULT_CHARGE_PROBS,
) -> Catalog:
    """Draw a catalogue of ``n_torrents`` torrents.

    ``mix`` gives the *count* shares of the first three classes; the last class
    contributes exactly ``n_super`` torrents.  Popularity is Zipf over a random
    ranking, independent of size.
    """
    n_torrents = int(n_torrents)
    n_super = int(min(n_super, n_torrents))
    counts = np.concatenate([_mixture_counts(n_torrents - n_super, mix), [n_super]])

    size = np.empty(n_torrents)
    class_idx = np.empty(n_torrents, dtype=np.int64)
    at = 0
    for j, (cls, count) in enumerate(zip(classes, counts, strict=True)):
        if count:
            size[at : at + count] = cls.sample(int(count), rng)
            class_idx[at : at + count] = j
            at += int(count)

    order = rng.permutation(n_torrents)
    size, class_idx = size[order], class_idx[order]

    rank = 1.0 + rng.permutation(n_torrents)
    demand = (rank + zipf_offset) ** (-zipf_s)
    demand /= demand.sum()

    importance = rng.choice(np.arange(-2, 3), size=n_torrents, p=importance_probs)
    charge = rng.choice([0.0, 0.5, 1.0], size=n_torrents, p=charge_probs)

    return Catalog(
        size_gib=size,
        class_idx=class_idx,
        importance=importance.astype(float),
        charge=charge,
        demand_share=demand,
        class_names=tuple(cls.name for cls in classes),
    )


# --------------------------------------------------------------------------
# bandwidth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BandwidthModel:
    """Joint downlink/uplink law: lognormal, truncated to a rate rectangle.

    ``median_down``/``median_up`` are the medians *after* truncation; ``rho`` is
    the correlation of the two logs.  ``sigma_up > sigma_down`` encodes that
    consumer links are asymmetric by varying amounts while seedboxes are not.
    """

    median_down: float = 300.0
    median_up: float = 50.0
    max_down: float = 10_000.0
    max_up: float = 2_000.0
    min_down: float = 2.0
    min_up: float = 0.5
    sigma_down: float = 0.9
    sigma_up: float = 1.2
    rho: float = 0.75

    @property
    def cap_quantiles(self) -> tuple[float, float]:
        """Where the two caps sit in the *untruncated* marginals."""
        mu_d, mu_u = calibrate_bandwidth(self)
        z_d = (np.log(self.max_down) - mu_d) / self.sigma_down
        z_u = (np.log(self.max_up) - mu_u) / self.sigma_up
        return float(norm.cdf(z_d)), float(norm.cdf(z_u))


@lru_cache(maxsize=64)
def calibrate_bandwidth(model: BandwidthModel) -> tuple[float, float]:
    """Log-means that place the truncated marginal medians on target.

    With a correlated pair truncated to a rectangle, the marginal of one
    coordinate is not a 1-D truncated normal, so the two log-means are solved
    jointly against rectangle probabilities of the standard bivariate normal.
    """
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, model.rho], [model.rho, 1.0]])

    def rect(x_lo, x_hi, y_lo, y_hi):
        corners = np.array(
            [[x_hi, y_hi], [x_lo, y_hi], [x_hi, y_lo], [x_lo, y_lo]], dtype=float
        )
        cdf = np.atleast_1d(mvn.cdf(corners))
        return float(cdf[0] - cdf[1] - cdf[2] + cdf[3])

    def residual(mu):
        mu_d, mu_u = float(mu[0]), float(mu[1])
        a_d = (np.log(model.min_down) - mu_d) / model.sigma_down
        b_d = (np.log(model.max_down) - mu_d) / model.sigma_down
        a_u = (np.log(model.min_up) - mu_u) / model.sigma_up
        b_u = (np.log(model.max_up) - mu_u) / model.sigma_up
        t_d = (np.log(model.median_down) - mu_d) / model.sigma_down
        t_u = (np.log(model.median_up) - mu_u) / model.sigma_up
        total = rect(a_d, b_d, a_u, b_u)
        below_d = rect(a_d, t_d, a_u, b_u)
        below_u = rect(a_d, b_d, a_u, t_u)
        return [below_d / total - 0.5, below_u / total - 0.5]

    guess = [np.log(model.median_down), np.log(model.median_up)]
    sol = root(residual, guess, method="hybr", tol=1e-12)
    # When truncation is negligible the residual is flat to within the
    # bivariate-CDF's own ~1e-8 accuracy, and hybr reports "no progress" on an
    # answer that is already exact.  Judge the residual, not the status flag —
    # but never accept a non-finite one, which would pass a `> tol` test.
    error = float(np.max(np.abs(residual(sol.x))))
    if not (np.all(np.isfinite(sol.x)) and np.isfinite(error) and error <= 1e-6):
        raise RuntimeError(
            f"bandwidth calibration failed ({sol.message}); residual {error:.2e}"
        )
    return float(sol.x[0]), float(sol.x[1])


#: Shared default so call sites need no mutable-looking default argument.
DEFAULT_BANDWIDTH = BandwidthModel()


def sample_bandwidth(
    n: int,
    rng: np.random.Generator,
    model: BandwidthModel = DEFAULT_BANDWIDTH,
):
    """Draw ``n`` (down, up) pairs plus the latent uplink z-score.

    The z-score is reused as the common factor that couples uptime and disk to
    uplink, so a seedbox tends to be fast *and* always-on *and* roomy.
    """
    mu_d, mu_u = calibrate_bandwidth(model)
    n = int(n)
    down = np.empty(n)
    up = np.empty(n)
    z_up = np.empty(n)
    tail = np.sqrt(max(0.0, 1.0 - model.rho**2))
    filled = 0
    while filled < n:
        draw = int((n - filled) * 1.3) + 32
        z_u = rng.standard_normal(draw)
        z_d = model.rho * z_u + tail * rng.standard_normal(draw)
        d = np.exp(mu_d + model.sigma_down * z_d)
        u = np.exp(mu_u + model.sigma_up * z_u)
        ok = (
            (d >= model.min_down)
            & (d <= model.max_down)
            & (u >= model.min_up)
            & (u <= model.max_up)
        )
        take = min(int(ok.sum()), n - filled)
        if take:
            idx = np.flatnonzero(ok)[:take]
            sl = slice(filled, filled + take)
            down[sl], up[sl], z_up[sl] = d[idx], u[idx], z_u[idx]
            filled += take
    return down, up, z_up


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """Nodes (client instances) and the users that own them."""

    user_of_node: np.ndarray
    down_mbps: np.ndarray
    up_mbps: np.ndarray
    availability: np.ndarray
    disk_gib: np.ndarray
    n_users: int

    @property
    def n_nodes(self) -> int:
        return int(self.up_mbps.size)

    @property
    def nodes_per_user(self) -> np.ndarray:
        return np.bincount(self.user_of_node, minlength=self.n_users)


def sample_population(
    n_users: int,
    rng: np.random.Generator,
    *,
    bandwidth: BandwidthModel = DEFAULT_BANDWIDTH,
    uptime_mean: float = 0.78,
    uptime_conc: float = 3.0,
    rho_uptime: float = 0.5,
    disk_median_gib: float = 1024.0,
    disk_sigma: float = 1.0,
    rho_disk: float = 0.55,
    multi_node_p: float = 0.08,
) -> Population:
    """Draw ``n_users`` users, each with one to three nodes.

    Uptime and disk are coupled to uplink through a Gaussian copula whose common
    factor *is* the uplink z-score, so all pairwise correlations are products of
    the two stated ones and the correlation matrix is positive semi-definite by
    construction.
    """
    n_users = int(n_users)
    extra = rng.binomial(2, multi_node_p, size=n_users)
    counts = 1 + extra
    user_of_node = np.repeat(np.arange(n_users), counts)
    n_nodes = int(counts.sum())

    down, up, z_up = sample_bandwidth(n_nodes, rng, bandwidth)

    def coupled(rho):
        return rho * z_up + np.sqrt(max(0.0, 1.0 - rho**2)) * rng.standard_normal(
            n_nodes
        )

    a = uptime_mean * uptime_conc
    b = (1.0 - uptime_mean) * uptime_conc
    avail = beta_dist.ppf(norm.cdf(coupled(rho_uptime)), a, b)
    avail = np.clip(avail, 0.02, 0.999)

    disk = np.exp(np.log(disk_median_gib) + disk_sigma * coupled(rho_disk))
    disk = np.clip(disk, 32.0, 500_000.0)

    return Population(
        user_of_node=user_of_node,
        down_mbps=down,
        up_mbps=up,
        availability=avail,
        disk_gib=disk,
        n_users=n_users,
    )
