"""Invariants of the point-system model.

Each test pins a property a plausible bug would break: a calibration that
silently misses its target, a health function whose concavity (the stability
argument of §4) fails, an estimator that is optimistic where the design says it
must be pessimistic, or a simulation that overdraws a budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from marimo_lab.ptcredit import (
    DEFAULT_SIZE_CLASSES,
    BandwidthModel,
    NodeEstimator,
    RewardParams,
    SimParams,
    build_world,
    calibrate_bandwidth,
    health,
    run,
    sample_bandwidth,
    sample_catalog,
    slowdown,
    slowdown_plain,
)
from marimo_lab.ptcredit.sim import _Site


@pytest.mark.parametrize(
    ("sigma_down", "sigma_up", "rho"),
    [(0.9, 1.2, 0.75), (1.3, 1.6, 0.6), (0.5, 0.7, 0.95)],
)
def test_bandwidth_medians_hit_the_truncated_target(sigma_down, sigma_up, rho):
    """The stated medians are medians of the *truncated* law, not the parent.

    Truncating a correlated pair to a rectangle shifts both marginals, so a
    naive implementation that sets mu = log(median) lands off target.
    """
    model = BandwidthModel(sigma_down=sigma_down, sigma_up=sigma_up, rho=rho)
    down, up, _ = sample_bandwidth(400_000, np.random.default_rng(1), model)

    assert np.median(down) == pytest.approx(model.median_down, rel=0.02)
    assert np.median(up) == pytest.approx(model.median_up, rel=0.02)
    assert down.max() <= model.max_down
    assert up.max() <= model.max_up
    assert down.min() >= model.min_down
    assert up.min() >= model.min_up
    log_corr = np.corrcoef(np.log(down), np.log(up))[0, 1]
    assert log_corr == pytest.approx(rho, abs=0.05)


def test_size_classes_respect_their_support_and_mix():
    catalog = sample_catalog(6000, np.random.default_rng(2), n_super=3)
    for j, cls in enumerate(DEFAULT_SIZE_CLASSES):
        sizes = catalog.size_gib[catalog.class_idx == j]
        assert sizes.min() >= cls.lo
        assert sizes.max() <= cls.hi
    counts = np.bincount(catalog.class_idx, minlength=4)
    assert counts[3] == 3
    shares = counts[:3] / counts[:3].sum()
    assert shares == pytest.approx([0.30, 0.40, 0.30], abs=0.001)


def test_size_classes_read_the_second_parameter_as_a_variance():
    """The brief states variances; scipy wants scales.

    Reading `Lognormal(4.5, 3)` as sigma = 3 instead of sigma = sqrt(3)
    rescales the large class by an order of magnitude at the tail and would
    silently change every downstream result, so the semantics are pinned.
    """
    small, medium, large, _ = DEFAULT_SIZE_CLASSES
    assert small.sd == pytest.approx(np.sqrt(0.5))
    assert medium.sd == pytest.approx(np.sqrt(10.0))
    assert large.sd == pytest.approx(np.sqrt(3.0))

    # Untruncated shape, i.e. the stated law itself.
    assert large.frozen().ppf(0.5) == pytest.approx(np.exp(4.5), rel=1e-9)
    assert large.frozen().ppf(0.975) == pytest.approx(2683.0, rel=0.01)
    assert medium.frozen().std() == pytest.approx(np.sqrt(10.0))

    # With sigma = sqrt(10) the normal class barely reaches zero, so the
    # truncation floor is a guard rather than a reshaping.
    assert medium.frozen().cdf(0.0) < 1e-4
    assert medium.retained_mass > 0.999
    assert medium.sample(50_000, np.random.default_rng(3)).min() > 0.0

    # A sampled class must reproduce the stated log-scale spread.
    drawn = np.log(small.sample(200_000, np.random.default_rng(11)))
    assert drawn.std() == pytest.approx(np.sqrt(0.5), rel=0.02)


def test_health_endpoints_and_target_value():
    assert float(health(np.inf, 1.5, 4.0)) == 0.0
    assert float(health(1.5, 1.5, 4.0)) == pytest.approx(
        1.0 - np.log(2.0) / np.log1p(np.exp(4.0)), rel=1e-12
    )
    assert float(health(1e-6, 1.5, 4.0)) == pytest.approx(1.0, abs=1e-6)


def test_health_is_concave_and_increasing_in_capacity():
    """The stability argument of §4 rests entirely on this."""
    d_ref, c_star, gamma = 300.0, 1.5, 4.0
    capacity = np.linspace(1.0, 4.0 * d_ref / c_star, 4000)
    values = health(slowdown(capacity, np.ones_like(capacity), d_ref), c_star, gamma)
    first = np.diff(values)
    assert np.all(first >= -1e-12)
    # Strictly decreasing marginal below the floor at C = 1.
    below_floor = capacity[:-1] < d_ref
    assert np.all(np.diff(first[below_floor]) <= 1e-12)


def test_completeness_cliff_leaves_no_marginal_below_pi_min():
    """§3.3 as written: below pi_min both H and its leave-one-out are zero.

    This is the coordination trap the notebooks report, and it must not be
    quietly smoothed away by the implementation.
    """
    p_tilde, pi_min = 0.78, 0.9
    capacity = 400.0
    solo = slowdown(capacity, p_tilde, 300.0, pi_min)
    none = slowdown(0.0, 0.0, 300.0, pi_min)
    assert not np.isfinite(solo)
    assert float(health(solo, 1.5, 4.0)) == 0.0
    assert float(health(none, 1.5, 4.0)) == 0.0

    pair = 1.0 - (1.0 - p_tilde) ** 2
    assert pair >= pi_min
    assert float(health(slowdown(capacity, pair, 300.0, pi_min), 1.5, 4.0)) > 0.0


def test_plain_slowdown_keeps_neither_clamp():
    """The part-2 comparison rule must stay a bare ratio.

    Both clamps re-added here would silently turn the comparison into a
    comparison of the rule with itself.
    """
    d_ref = 300.0
    # No availability argument at all, and no floor at C = 1.
    assert float(slowdown_plain(600.0, d_ref)) == pytest.approx(0.5)
    assert float(slowdown_plain(150.0, d_ref)) == pytest.approx(2.0)
    assert not np.isfinite(float(slowdown_plain(0.0, d_ref)))

    # Above the clamp the two rules must disagree, below it agree exactly.
    over, under = 600.0, 150.0
    assert float(slowdown(over, 1.0, d_ref)) == 1.0
    assert float(slowdown_plain(over, d_ref)) < 1.0
    assert float(slowdown(under, 1.0, d_ref)) == float(slowdown_plain(under, d_ref))

    # And health keeps rising past the clamp only for the plain rule.
    assert float(health(slowdown_plain(over, d_ref))) > float(
        health(slowdown(over, 1.0, d_ref))
    )


def test_completeness_modes_differ_only_in_how_pi_min_enters():
    """Identical seeders join one at a time; the marginal a joiner is paid
    is the leave-one-out difference.  The step pays nothing below the quorum
    and everything to the seeder that completes it; the cap is decreasing
    everywhere; possession ignores online availability altogether."""
    d_ref, p_tilde, g = 300.0, 0.55, 300.0 / 1.5 / 4.0
    n = np.arange(0, 9)
    avail = 1.0 - (1.0 - p_tilde) ** n
    target = np.full(n.shape, 1.5)

    def marginals(mode):
        rp = RewardParams(floor=False, completeness=mode)
        return np.diff(rp.torrent_health(n * g, avail, d_ref, target))

    step = marginals("online_step")
    assert step[0] == 0.0 and step[1] == 0.0 and step[2] > 0.5
    cap = marginals("online_cap")
    assert np.all(np.diff(cap) <= 1e-12)
    assert cap[0] > 0.0
    possession = marginals("possession")
    assert np.all(np.diff(possession) <= 1e-12)
    # The cap binds only where online availability is the bottleneck.
    assert np.all(cap <= possession + 1e-12)
    assert cap[-1] == pytest.approx(possession[-1])


def test_estimator_is_pessimistic_before_evidence_and_converges_after():
    est = NodeEstimator(2, m=24.0, prior_p=0.55, prior_b=20.0)
    cold = est.availability()[0]
    assert cold < 0.55  # a lower posterior quantile, never the prior mean

    rng = np.random.default_rng(4)
    active = np.ones(2, dtype=bool)
    for _ in range(2000):
        est.decay(1.0)
        online = rng.random(2) < 0.9
        est.observe(active, online, online, np.full(2, 500.0), np.ones(2))

    p_tilde = est.availability()
    b_tilde = est.capacity()
    assert 0.80 < p_tilde[0] < 0.90  # pessimistic, but close to the truth
    # The shrinkage drag never fully vanishes: decayed evidence saturates at
    # half_life / ln 2 hours, so m pseudo-hours of a low prior keep pulling
    # b_tilde a few percent below the truth.  That bias is the design.
    assert 0.90 * 500.0 < b_tilde[0] < 500.0


def test_estimator_ignores_intervals_without_leechers():
    """An idle seeder shows a zero rate through no fault of its own."""
    est = NodeEstimator(1, m=24.0, prior_b=20.0)
    active = np.ones(1, dtype=bool)
    online = np.ones(1, dtype=bool)
    idle = np.zeros(1, dtype=bool)
    for _ in range(500):
        est.decay(1.0)
        est.observe(active, online, idle, np.zeros(1), np.ones(1))
    assert est.capacity()[0] == pytest.approx(20.0, rel=1e-9)


def test_reward_merges_a_user_s_own_nodes():
    """§3.3: two clients of one user must not inflate that user's marginal."""
    params = SimParams(n_torrents=40, n_users=6, days=1, initial_fill=0.5, seed=5)
    catalog, population = build_world(params)
    site = _Site(params, RewardParams(), catalog, population)
    users = population.user_of_node[site.mem_node]
    combined = users.astype(np.int64) * site.n_t + site.mem_torrent
    assert site.pair_torrent.size == np.unique(combined).size
    assert int(site.pair_count.sum()) == site.mem_node.size


def test_simulation_never_overdraws_disk_or_balance():
    params = SimParams(days=6, n_torrents=300, n_users=60, seed=6)
    result = run(params, RewardParams(kappa=0.05))
    assert np.all(np.isfinite(result.members_total))
    assert result.members_total[-1] > 0
    # Balances are non-negative by §3.5 and disk is a hard budget.
    assert result.balance_quantiles.min() >= 0.0


def test_tail_window_covers_exactly_the_requested_days():
    """Timestamps mark interval *ends*, so the boundary sample is excluded.

    A left-inclusive search would take 29 six-hour records for a 7-day tail
    and then divide interval totals by 7 days, inflating every per-day rate
    by one interval's worth.
    """
    params = SimParams(days=10, n_torrents=200, n_users=40, record_every_h=6, seed=9)
    result = run(params, RewardParams(kappa=0.05))

    window = result.hours[result.tail(7.0)]
    assert window.size == 28
    assert result.tail_hours(7.0) == pytest.approx(168.0)
    assert window[-1] == pytest.approx(result.hours[-1])
    assert window[0] > result.hours[-1] - 7.0 * 24.0

    # broke_per_day must be a true rate over that window.
    acc = result.acceptance(days=7.0)
    expected = result.refused[result.tail(7.0)].sum() / 7.0
    assert acc["broke_per_day"] == pytest.approx(expected)


def test_simulation_respects_every_node_s_disk_budget():
    params = SimParams(days=8, n_torrents=300, n_users=60, seed=7)
    catalog, population = build_world(params)
    site = _Site(params, RewardParams(), catalog, population)
    rng = np.random.default_rng(8)
    for hour in range(120):
        site._serve()
        p_tilde, _, g = site.est.contribution()
        capacity, completeness, log_miss, c = site._torrent_state(g, p_tilde)
        rate, _ = site._pair_reward(
            g, p_tilde, capacity, completeness, log_miss, site.target
        )
        nodes = rng.integers(0, site.n_n, size=20)
        torrents = rng.integers(0, site.n_t, size=20)
        site._start_downloads(nodes, torrents, False)
        if hour % 6 == 0:
            if site._reallocate(g, p_tilde, capacity, completeness, rate):
                site._rebuild_pairs()
        if site._finish_downloads():
            site._rebuild_pairs()
        held = np.bincount(
            site.mem_node, site.size[site.mem_torrent], minlength=site.n_n
        )
        assert np.all(held <= site.disk + 1e-6)
        assert np.all(site.used >= held - 1e-6)
        assert np.all(site.used <= site.disk + 1e-6)
        assert np.all(site.balance >= -1e-9)


def _drive(site, rng, hours, *, demand_per_hour=20, decide_every=6):
    """Advance a site by hand, yielding after every hour's bookkeeping."""
    for hour in range(hours):
        site._serve()
        p_tilde, _, g = site.est.contribution()
        capacity, completeness, log_miss, c = site._torrent_state(g, p_tilde)
        rate, _ = site._pair_reward(
            g, p_tilde, capacity, completeness, log_miss, site.target
        )
        nodes = rng.integers(0, site.n_n, size=demand_per_hour)
        torrents = rng.integers(0, site.n_t, size=demand_per_hour)
        site._start_downloads(nodes, torrents, False)
        if hour % decide_every == 0:
            if site._reallocate(g, p_tilde, capacity, completeness, rate):
                site._rebuild_pairs()
        if site._finish_downloads():
            site._rebuild_pairs()
        yield hour


def test_download_of_a_torrent_nobody_holds_is_refused_unpaid():
    """No member means no copy anywhere: such a download could never finish,
    yet it would have burned the points and reserved the disk for good."""
    params = SimParams(n_torrents=200, n_users=10, days=1, initial_fill=0.05, seed=11)
    catalog, population = build_world(params)
    site = _Site(params, RewardParams(), catalog, population)
    held = np.bincount(site.mem_torrent, minlength=site.n_t) > 0
    empty = int(np.flatnonzero(~held)[0])
    node = int(np.flatnonzero(site.used + site.size[empty] <= site.disk)[0])
    balance, used = site.balance.copy(), site.used.copy()

    started, burned, broke = site._start_downloads(
        np.array([node]), np.array([empty]), False
    )

    assert (started, burned, broke) == (0, 0.0, 0)
    assert site.lch_node.size == 0
    assert np.array_equal(site.balance, balance)
    assert np.array_equal(site.used, used)
    # Refused for structure, not for money: counted where survivorship is read.
    assert site.demand_unsourced == 1


def test_every_download_keeps_a_source_until_it_completes():
    """A swap may not release the last copy of a torrent somebody is
    downloading — including when two holders each release one copy in the
    same re-planning batch."""
    params = SimParams(
        days=8,
        n_torrents=300,
        n_users=60,
        disk_scale=0.25,
        moves_per_decision=3,
        seed=7,
    )
    catalog, population = build_world(params)
    site = _Site(params, RewardParams(kappa=0.05), catalog, population)
    for _ in _drive(site, np.random.default_rng(8), 240, demand_per_hour=40):
        members = np.bincount(site.mem_torrent, minlength=site.n_t)
        assert np.all(members[site.lch_torrent] >= 1)
        assert np.all(site.lch_left >= 0.0)


def test_service_stops_at_the_end_of_the_file():
    """A download that finishes mid-hour moves only what it had left, is
    credited with only the fraction of the hour it used, and its completion
    time is measured against the reference rate."""
    params = SimParams(n_torrents=50, n_users=24, days=1, initial_fill=0.5, seed=12)
    catalog, population = build_world(params)
    site = _Site(params, RewardParams(), catalog, population)
    site.avail[:] = 1.0
    held = np.bincount(site.mem_torrent, minlength=site.n_t)
    eligible = np.flatnonzero((held >= 5) & (held < site.n_n))
    torrent = int(eligible[np.argmin(site.size[eligible])])
    node = int(np.argmin(np.where(site.holds[:, torrent], np.inf, site.down)))
    assert not site.holds[node, torrent]
    swarm_up = site.up[site.mem_node[site.mem_torrent == torrent]].sum()
    assert swarm_up >= site.down[node], "scenario needs a swarm faster than the line"

    site._start_downloads(np.array([node]), np.array([torrent]), False)
    site._serve()

    # Served at its own line, the small file finishes inside the hour.
    assert site.lch_left[0] == 0.0
    assert 0.0 < site.lch_time[0] < 1.0
    site._finish_downloads()
    ratio = float(np.concatenate(site.finished)[0])
    assert ratio == pytest.approx(min(site.down[node], site.d_ref) / site.down[node])


def test_importance_target_mode_moves_the_target_not_the_pay():
    importance = np.array([-1.0, 0.0, 2.0])
    pay = RewardParams(importance_mode="pay")
    tgt = RewardParams(importance_mode="target")
    assert pay.torrent_weight(importance) == pytest.approx([0.5, 1.0, 4.0])
    assert pay.torrent_target(importance) == pytest.approx([1.5, 1.5, 1.5])
    assert tgt.torrent_weight(importance) == pytest.approx([1.0, 1.0, 1.0])
    assert tgt.torrent_target(importance) == pytest.approx([3.0, 1.5, 0.375])


def test_calibration_is_cached_per_model():
    model = BandwidthModel()
    assert calibrate_bandwidth(model) is calibrate_bandwidth(BandwidthModel())
