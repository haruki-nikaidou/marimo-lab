import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from marimo_lab.ptcredit import RewardParams, SimParams, build_world, run

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9.0, "axes.grid": True, "grid.alpha": 0.3}
    )
    return RewardParams, SimParams, build_world, mo, np, plt, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point system, part 4 — the clamps on $C_i$, put in motion

    Part 2 §3 priced the two clamps on

    $$
    C^{\text{cur}}_i = \begin{cases}\max(1,\, d_{ref}/x_i) & A_i \ge \pi_{min}\\
    \infty & \text{else}\end{cases}
    \qquad\text{vs.}\qquad
    C^{\text{raw}}_i = \frac{d_{ref}}{x_i}
    $$

    on a **frozen** swarm. Part 3's engine removes that limitation: a
    catalogue, heterogeneous nodes, finite disks that must be *taken from
    somewhere*, and a price of disk that is whatever the competition for those
    disks makes it. This part runs the same world under three scoring rules:

    | rule | $\max(1, \cdot)$ | $A_i \ge \pi_{min}$ | what it is |
    |---|---|---|---|
    | `clamped` | yes | yes | §3.3 as written — part 3's baseline |
    | `no_floor` | no | yes | only the capacity floor removed |
    | `raw` | no | no | part 2's $C^{\text{raw}} = d_{ref}/x$ |

    The middle row makes the comparison a decomposition: `clamped` →
    `no_floor` prices the floor alone, `no_floor` → `raw` prices the
    completeness cutoff alone. Nothing else differs between the runs.

    > [!important] Two things about how this revision measures
    >
    > **The design has no monetary policy.** Points are a gate on download,
    > not a currency with a price level, so there is nothing for a falling
    > $\kappa$ to fix. Every run here holds $\kappa$ fixed; the §3.7
    > controller of part 3 is not evaluated.
    >
    > **Outcomes are read on what a downloader got, not on what the rule
    > scored.** Each rule's $C_i$ is its own definition — the clamped one
    > cannot report a value below 1 — so the comparison is made on two
    > rule-independent quantities: the physical ratio $C^{\text{phys}} =
    > d_{ref}/x_i$ (the *allocation*, the same formula in every world) and
    > the completion time of each demand download relative to the time it
    > would have taken at the reference rate (the *experience*: one number
    > per download, no estimator and no scoring rule in it). The first says
    > where the disk went; only the second says whether it mattered.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_days = mo.ui.slider(
        start=15, stop=90, step=5, value=40, label="days", show_value=True
    )
    ui_torrents = mo.ui.dropdown(
        options={"1k": 1000, "2k": 2000, "4k": 4000}, value="2k", label="torrents"
    )
    ui_users = mo.ui.dropdown(
        options={"200": 200, "400": 400, "800": 800}, value="400", label="users"
    )
    ui_kappa = mo.ui.number(
        start=1e-4, stop=10.0, step=1e-4, value=0.05, label=r"$\kappa$"
    )
    ui_alpha = mo.ui.slider(
        start=0.0,
        stop=0.2,
        step=0.01,
        value=0.05,
        label=r"floor $\alpha$",
        show_value=True,
    )
    ui_gamma = mo.ui.slider(
        start=2.0,
        stop=8.0,
        step=0.5,
        value=4.0,
        label=r"gain $\gamma$",
        show_value=True,
    )
    ui_seed = mo.ui.number(start=0, stop=999, step=1, value=0, label="first seed")
    ui_n_seeds = mo.ui.dropdown(
        options={"1": 1, "2": 2, "3": 3}, value="3", label="seeds"
    )
    mo.vstack(
        [
            mo.md(
                "## 1. Controls\n\nThree rules × three disk supplies × the"
                " chosen number of seeds, at roughly 4 s a run, memoised on"
                " the exact parameter pair. Tables average over seeds; time"
                " series and histograms show the first seed."
            ),
            mo.hstack([ui_days, ui_torrents, ui_users], widths="equal"),
            mo.hstack([ui_kappa, ui_alpha, ui_gamma], widths="equal"),
            mo.hstack([ui_seed, ui_n_seeds], widths="equal"),
        ]
    )
    return (
        ui_alpha,
        ui_days,
        ui_gamma,
        ui_kappa,
        ui_n_seeds,
        ui_seed,
        ui_torrents,
        ui_users,
    )


@app.cell(hide_code=True)
def _(
    RewardParams,
    SimParams,
    ui_alpha,
    ui_days,
    ui_gamma,
    ui_kappa,
    ui_n_seeds,
    ui_seed,
    ui_torrents,
    ui_users,
):
    rules = ("clamped", "no_floor", "raw")
    rule_label = {
        "clamped": r"clamped $C_i$ (§3.3)",
        "no_floor": r"no floor, $\pi_{min}$ kept",
        "raw": r"raw $d_{ref}/x$",
    }
    rule_colour = {"clamped": "k", "no_floor": "C0", "raw": "C1"}
    # (floor, completeness treatment) behind each rule name.
    rule_fields = {
        "clamped": (True, "online_step"),
        "no_floor": (False, "online_step"),
        "raw": (False, "possession"),
    }
    scales = (0.25, 1.0, 4.0)
    seeds = tuple(int(ui_seed.value) + _k for _k in range(int(ui_n_seeds.value)))

    rewards = {
        _rule: RewardParams(
            kappa=float(ui_kappa.value),
            alpha=float(ui_alpha.value),
            gamma=float(ui_gamma.value),
            eta=1.0,
            floor=rule_fields[_rule][0],
            completeness=rule_fields[_rule][1],
        )
        for _rule in rules
    }
    base = SimParams(
        days=int(ui_days.value),
        n_torrents=int(ui_torrents.value),
        n_users=int(ui_users.value),
        kappa_control=False,
        seed=seeds[0],
    )
    pi_min = float(rewards["clamped"].pi_min)
    c_star = float(rewards["clamped"].c_star)
    return (
        base,
        c_star,
        pi_min,
        rewards,
        rule_colour,
        rule_label,
        rules,
        scales,
        seeds,
    )


@app.cell(hide_code=True)
def _(build_world, run):
    from dataclasses import replace

    _runs_memo: dict = {}
    _worlds: dict = {}

    def simulate(reward, params, **overrides):
        """One run, memoised on the exact (reward, params) pair.

        The world is drawn once per (torrents, users, seed) and handed to
        every run, so the rules are compared on the *same* catalogue and the
        same nodes rather than on two draws that happen to share a seed.
        """
        p = replace(params, **overrides) if overrides else params
        world_key = (p.n_torrents, p.n_users, p.seed)
        if world_key not in _worlds:
            _worlds[world_key] = build_world(p)
        catalog, population = _worlds[world_key]
        key = (reward, p)
        if key not in _runs_memo:
            _runs_memo[key] = run(p, reward, catalog=catalog, population=population)
        return _runs_memo[key]

    return (simulate,)


@app.cell
def _(base, rewards, rules, scales, seeds, simulate):
    runs_all = {
        (_seed, _rule, _s): simulate(rewards[_rule], base, disk_scale=_s, seed=_seed)
        for _seed in seeds
        for _rule in rules
        for _s in scales
    }
    # First seed only: what the time series and histograms are drawn from.
    runs = {
        (_rule, _s): runs_all[seeds[0], _rule, _s] for _rule in rules for _s in scales
    }
    return runs, runs_all


@app.cell(hide_code=True)
def _(np, pi_min, runs, runs_all, seeds):
    def _one(r, days=7.0):
        sl = r.tail(days)
        span = max(r.tail_hours(days) / 24.0, 1e-9)
        out = dict(r.acceptance(days=days))
        out["mint_day"] = float(r.mint[sl].sum() / span)
        out["burn_day"] = float(r.burn[sl].sum() / span)
        out["invest_day"] = float(r.burn_invest[sl].sum() / span)
        out["joins_day"] = float(r.replan_joins[sl].sum() / span)
        out["drops_day"] = float(r.replan_drops[sl].sum() / span)
        out["members_per_torrent"] = float(r.members_total[-1] / r.catalog.n_torrents)
        return out

    def seed_stats(rule, scale, key, days=7.0):
        """One value per seed, in seed order."""
        return [_one(runs_all[_seed, rule, scale], days)[key] for _seed in seeds]

    def stats(rule, scale, days=7.0):
        """Acceptance summary plus flow and churn columns, averaged over seeds.

        Rates are per day over the tail window the acceptance criterion
        uses, so ``mint``, ``burn`` and the churn counters are directly
        comparable across rules and disk supplies.
        """
        per_seed = [_one(runs_all[_seed, rule, scale], days) for _seed in seeds]
        return {k: float(np.nanmean([p[k] for p in per_seed])) for k in per_seed[0]}

    def dead_mask(rule, scale):
        """Torrents whose completeness has fallen below ``pi_min``.

        Taken from ``A_i`` rather than from ``isfinite(C_i)``: under the two
        unclamped rules a torrent below the threshold still has a finite
        slowdown, and it would otherwise silently leave the tally.
        """
        return runs[rule, scale].final_completeness < pi_min

    def class_dead(rule, scale):
        r = runs[rule, scale]
        dead = dead_mask(rule, scale)
        return [
            float(np.mean(dead[r.catalog.class_idx == j]))
            for j in range(len(r.catalog.class_names))
        ]

    def physical_slowdown(rule, scale):
        """End-of-run ``d_ref / x_i``: the allocation, with no rule applied.

        ``inf`` marks a torrent nobody holds, which is a membership fact
        rather than a scoring one and reads the same under every rule.
        """
        r = runs[rule, scale]
        x = r.final_capacity
        with np.errstate(divide="ignore"):
            return np.where(x > 0.0, r.d_ref / np.maximum(x, 1e-300), np.inf)

    def dead_zone_fraction(rule, scale):
        """Share of the catalogue already at or past ``x = d_ref``.

        This is where the clamped rule's marginal is exactly zero, so it is
        the region the floor declares finished — measured physically, so the
        rules that do not clamp can be asked the same question.
        """
        return float(np.mean(physical_slowdown(rule, scale) <= 1.0))

    return (
        class_dead,
        dead_mask,
        dead_zone_fraction,
        physical_slowdown,
        seed_stats,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The same site, scored three ways

    Disk supply ×1: the world of part 3, over-provisioned in the middle of
    the catalogue and thin at its large end. The two *experience* columns are
    the p90 of completion time over demand downloads, once by count and once
    with each download weighted by its size — small files dominate the count
    and large ones the bytes, and a rule has to answer for both.
    """)
    return


@app.cell(hide_code=True)
def _(mo, rule_label, rules, stats):
    _cols = (
        ("median_C_phys", r"median $C^{\text{phys}}$", "{:.2f}"),
        ("avail_p90_C_phys", r"p90 $C^{\text{phys}}$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"$A_i<\pi_{min}$", "{:.1%}"),
        ("served_share", "rate utilisation", "{:.0%}"),
        ("download_p90", "p90 time, by count", "{:.2f}"),
        ("bytes_p90", "p90 time, by bytes", "{:.2f}"),
        ("demand_backlog_days", "stuck demand (days)", "{:.2f}"),
        ("demand_unsourced_share", "demand with no copy", "{:.1%}"),
        ("concurrency_mean", r"$\tilde k_v$", "{:.1f}"),
        ("invest_leech_share", "leech time = re-planning", "{:.0%}"),
        ("joins_day", "joins/day", "{:.0f}"),
        ("mint_burn", "mint/burn", "{:.1f}"),
        ("median_C_all", r"*scored* median $C$", "{:.2f}"),
    )
    _rows = [
        "| rule | " + " | ".join(_c[1] for _c in _cols) + " |",
        "|---" * (len(_cols) + 1) + "|",
    ]
    for _rule in rules:
        _s = stats(_rule, 1.0)
        _rows.append(
            f"| {rule_label[_rule]} | "
            + " | ".join(_c[2].format(_s[_c[0]]) for _c in _cols)
            + " |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(c_star, plt, rule_colour, rule_label, rules, runs):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.2), sharex=True)
    for _rule in rules:
        _r = runs[_rule, 1.0]
        _d = _r.hours / 24.0
        _col = rule_colour[_rule]
        _ax[0].plot(
            _d,
            _r.c_physical_quantiles[:, 1],
            color=_col,
            lw=1.3,
            label=rule_label[_rule],
        )
        _ax[0].plot(_d, _r.c_physical_avail[:, 2], color=_col, lw=0.7, ls=":")
        _ax[1].plot(_d, _r.download_slowdown_by_bytes[:, 2], color=_col, lw=1.3)
        _ax[1].plot(_d, _r.download_slowdown_by_bytes[:, 1], color=_col, lw=0.7, ls=":")
        _ax[2].plot(_d, _r.concurrency_mean, color=_col, lw=1.3)
    _ax[0].axhline(c_star, color="C3", ls="--", lw=0.9)
    _ax[0].axhline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].set_yscale("log")
    _ax[0].set_ylabel(r"$C^{\text{phys}} = d_{ref}/x_i$")
    _ax[0].set_title(r"allocation: median (solid), avail. p90 (dotted)")
    _ax[0].legend(fontsize=6.5)
    _ax[1].axhline(1.0, color="C7", ls=":", lw=0.9)
    _ax[1].set_ylabel("completion time / time at $d_{ref}$")
    _ax[1].set_title("experience by bytes: p90 (solid), p50 (dotted)")
    _ax[2].set_ylabel(r"mean $\tilde k_v$")
    _ax[2].set_title("swarms a seeder serves at once")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(dead_zone_fraction, mo, stats):
    _cur = stats("clamped", 1.0)
    _nof = stats("no_floor", 1.0)
    _raw = stats("raw", 1.0)
    mo.md(rf"""
    **The floor holds the allocation at one leecher's downlink.** Measured
    physically, the clamped rule's median torrent ends at
    $C^{{\text{{phys}}}} = {_cur["median_C_phys"]:.2f}$ — at $x = d_{{ref}}$,
    where its marginal reaches exactly zero (its *scored* median reads
    {_cur["median_C_all"]:.2f}, because the clamp cannot report anything below
    1). Without the floor the median goes to {_nof["median_C_phys"]:.2f}
    (`no_floor`) and {_raw["median_C_phys"]:.2f} (`raw`), and
    {dead_zone_fraction("raw", 1.0):.0%} of the catalogue sits past
    $x = d_{{ref}}$ against {dead_zone_fraction("clamped", 1.0):.0%}. Part 2's
    frozen-swarm intuition was that this surplus is idle, and the
    utilisation column looks like agreement ({_cur["served_share"]:.0%} of
    the instantaneous offered rate drawn under the clamped rule,
    {_nof["served_share"]:.0%} without the floor) — but that column is an
    average over allocated rate and says only how often a leecher's own
    line, not the swarm, is the binding constraint.

    **The experience columns are the ones that count, and they disagree.**
    The p90 completion time by bytes goes {_cur["bytes_p90"]:.2f} →
    {_nof["bytes_p90"]:.2f} → {_raw["bytes_p90"]:.2f} across the three rules;
    by count {_cur["download_p90"]:.2f} → {_nof["download_p90"]:.2f} →
    {_raw["download_p90"]:.2f}, with the same availability
    ({_cur["unavailable_fraction"]:.1%} / {_nof["unavailable_fraction"]:.1%} /
    {_raw["unavailable_fraction"]:.1%}) and the same share of demand refused
    for want of any copy ({_cur["demand_unsourced_share"]:.1%} /
    {_nof["demand_unsourced_share"]:.1%} / {_raw["demand_unsourced_share"]:.1%}),
    so the completed-download quantiles are comparing the same population.
    An average cannot see this because a swarm sized to exactly one
    reference downlink is fine *on average* and halves the moment a second
    leecher arrives. What the data show alongside is association, not a
    traced cause: under the clamped rule each seeder is serving
    {_cur["concurrency_mean"]:.1f} swarms at once by the estimator's count
    against {_nof["concurrency_mean"]:.1f} without the floor, and
    {_cur["invest_leech_share"]:.0%} against {_nof["invest_leech_share"]:.0%}
    of leech time is seeders' own re-planning. The slack the unclamped rules
    buy is where collisions land. That is the redundancy argument in its
    weakest possible form — one homogeneous capacity pool, every seeder
    equally reachable by every leecher — and it already shows.

    **The completeness cutoff does nothing here.** {_nof["unavailable_fraction"]:.1%}
    of the catalogue sits below $\pi_{{min}}$ with the cutoff priced and
    {_raw["unavailable_fraction"]:.1%} without it; §3 is where it bites.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Disk supply: what the floor holds constant, and what it does not

    §8 asks for the $C_i$ distribution to collapse toward $C^*$ **regardless
    of total disk supply**, and §4 says the knee delivers it: with a plateau
    above target and near-zero reward below, the crossing with the
    opportunity cost $\theta$ lands at the knee for any $\theta$ in a wide
    range. Here $\theta$ is endogenous, so scaling total disk by 1/4 and 4 is
    a direct test — run on both the allocation and the experience, because
    the two answer differently.
    """)
    return


@app.cell(hide_code=True)
def _(mo, rule_label, rules, scales, stats):
    _cols = (
        ("median_C_phys", r"median $C^{\text{phys}}$", "{:.2f}"),
        ("avail_p90_C_phys", r"p90 $C^{\text{phys}}$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"$A_i<\pi_{min}$", "{:.1%}"),
        ("bytes_p50", "p50 time, by bytes", "{:.2f}"),
        ("bytes_p90", "p90 time, by bytes", "{:.2f}"),
        ("demand_backlog_days", "stuck demand (days)", "{:.2f}"),
        ("demand_unsourced_share", "demand with no copy", "{:.1%}"),
        ("concurrency_mean", r"$\tilde k_v$", "{:.1f}"),
        ("joins_day", "joins/day", "{:.0f}"),
        ("oscillation", "oscillation", "{:.1%}"),
    )
    _rows = [
        "| rule | disk | " + " | ".join(_c[1] for _c in _cols) + " |",
        "|---" * (len(_cols) + 2) + "|",
    ]
    for _rule in rules:
        for _s in scales:
            _v = stats(_rule, _s)
            _rows.append(
                f"| {rule_label[_rule]} | ×{_s} | "
                + " | ".join(_c[2].format(_v[_c[0]]) for _c in _cols)
                + " |"
            )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(c_star, np, plt, rule_colour, rule_label, rules, scales, seed_stats, stats):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    _x = np.asarray(scales)
    for _rule in rules:
        _col = rule_colour[_rule]
        _med = [stats(_rule, _s)["median_C_phys"] for _s in scales]
        _p90 = [stats(_rule, _s)["avail_p90_C_phys"] for _s in scales]
        _exp = [stats(_rule, _s)["bytes_p90"] for _s in scales]
        _ax[0].plot(_x, _med, "o-", color=_col, lw=1.3, ms=3, label=rule_label[_rule])
        _ax[1].plot(_x, _p90, "o-", color=_col, lw=1.3, ms=3)
        _ax[2].plot(_x, _exp, "o-", color=_col, lw=1.3, ms=3)
        for _s in scales:
            _pts = seed_stats(_rule, _s, "bytes_p90")
            _ax[2].plot([_s] * len(_pts), _pts, ".", color=_col, ms=3, alpha=0.5)
    for _a, _t, _y in (
        (
            _ax[0],
            r"median $C^{\text{phys}}$: the floor pins it",
            r"median $d_{ref}/x_i$",
        ),
        (_ax[1], "p90 over available torrents", r"p90 $d_{ref}/x_i$"),
        (
            _ax[2],
            "experience: p90 completion time by bytes",
            "time / time at $d_{ref}$",
        ),
    ):
        _a.set_xscale("log")
        _a.set_xticks(_x, [f"×{_s}" for _s in scales])
        _a.set_xticks([], minor=True)
        _a.set_xlabel("total disk supply")
        _a.set_ylabel(_y)
        _a.set_title(_t)
    _ax[0].axhline(c_star, color="C3", ls="--", lw=0.9)
    _ax[0].axhline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].legend(fontsize=6.5)
    _ax[1].axhline(c_star, color="C3", ls="--", lw=0.9)
    _ax[2].axhline(1.0, color="C7", ls=":", lw=0.9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, rules, scales, seed_stats, stats):
    _med = {_r: [stats(_r, _s)["median_C_phys"] for _s in scales] for _r in rules}
    _exp = {_r: [stats(_r, _s)["bytes_p90"] for _s in scales] for _r in rules}
    _range = {_r: max(_med[_r]) - min(_med[_r]) for _r in rules}
    _cur0, _nof0, _raw0 = (stats(_r, 0.25) for _r in ("clamped", "no_floor", "raw"))
    _cur4, _nof4 = stats("clamped", 4.0), stats("no_floor", 4.0)
    _seed0 = {
        _r: " / ".join(f"{_v:.2f}" for _v in seed_stats(_r, 0.25, "bytes_p90"))
        for _r in ("clamped", "no_floor")
    }
    mo.md(rf"""
    > [!warning] The floor anchors the estimate, not the experience
    > Over a 16× swing in disk supply the clamped rule's median
    > $C^{{\text{{phys}}}}$ moves {_med["clamped"][0]:.2f} →
    > {_med["clamped"][1]:.2f} → {_med["clamped"][2]:.2f} (spread
    > {_range["clamped"]:.2f}); without the floor it moves
    > {_med["no_floor"][0]:.2f} → {_med["no_floor"][1]:.2f} →
    > {_med["no_floor"][2]:.2f} (spread {_range["no_floor"]:.2f}). Read
    > literally against §8 that is the floor passing the criterion and the
    > unclamped rules failing it. Read against what the criterion was *for* —
    > "torrents converge on $C^*$ rather than on whatever disk supply happens
    > to allow" — neither rule converges on $C^* = 1.5$: the floor pins the
    > median to $1$, its own hard zero, and the unclamped rules let surplus
    > disk become surplus capacity.
    >
    > The experience column moves the other way once disk is not scarce.
    > With the floor the p90 completion time by bytes gets *worse* as disk
    > gets cheaper — {_exp["clamped"][0]:.2f} → {_exp["clamped"][1]:.2f} →
    > {_exp["clamped"][2]:.2f}. What the run shows alongside, as
    > association rather than a traced cause, is more concurrent serving:
    > at ×4 the estimator counts $\tilde k_v = {_cur4["concurrency_mean"]:.1f}$
    > swarms per seeder with the floor against
    > {_nof4["concurrency_mean"]:.1f} without, on swarms the floor had
    > sized for exactly one leecher. Without the floor the p90 goes
    > {_exp["no_floor"][0]:.2f} → {_exp["no_floor"][1]:.2f} →
    > {_exp["no_floor"][2]:.2f}: better at ×1 and ×4, and **worse at
    > ×0.25**. The scarce-site row is too seed-sensitive to carry a
    > directional claim — per seed it reads {_seed0["no_floor"]} without the
    > floor against {_seed0["clamped"]} with it. The outlier is one world in
    > which a few large downloads sat on a torrent whose only holder was
    > almost never online, and without the floor the disk that would have
    > rescued it had been spent on slack elsewhere. The robust evidence is
    > the ×1/×4 pattern: on a site with more disk than its catalogue needs —
    > the case §8 worried about — holding $x_i = d_{{ref}}$ exactly refuses
    > redundancy that was free. Under scarcity the floor's refusal to buy
    > slack may be what keeps the tail honest, and this data cannot say
    > more than "may".
    >
    > The mechanism is simple once stated: $H$ is concave but it does not
    > *stop*. Without $\max(1,\cdot)$ there is always a little more health to
    > buy, so cheap disk keeps being spent until $H$'s own saturation
    > flattens the marginal somewhere below $C = 1$; with the floor, the
    > marginal hits exactly zero at $x = d_{{ref}}$ and surplus disk becomes
    > turnover instead. `no_floor` and `raw` coincide exactly at ×4 disk:
    > with enough disk nothing falls below $\pi_{{min}}$, so the cutoff is
    > never evaluated.

    On a scarce site the two clamps separate. At ×0.25 disk, dropping the
    floor alone moves the available-torrent p90 from
    {_cur0["avail_p90_C_phys"]:.2f} to {_nof0["avail_p90_C_phys"]:.2f} and
    unavailability from {_cur0["unavailable_fraction"]:.1%} to
    {_nof0["unavailable_fraction"]:.1%}; dropping the completeness cutoff on
    top of that moves the p90 to {_raw0["avail_p90_C_phys"]:.2f} and
    unavailability to {_raw0["unavailable_fraction"]:.1%}. **The scarce-site
    tail damage belongs to $\pi_{{min}}$, not to the floor** — an earlier
    revision of this notebook attributed it the other way round, on one seed,
    and the decomposition row says otherwise. Dropping the cutoff buys
    availability (no quorum step, so a lone seeder is paid, and the site
    spreads thinner over more torrents); it costs the p90 of the torrents
    that are available, and a small stuck-demand backlog
    ({_raw0["demand_backlog_days"]:.2f} days of demand under `raw`, on
    torrents whose only holder is rarely online and which the cutoff would
    have let go).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Where the capacity ends up

    The medians above are one number per site. The distributions say which
    torrents paid for the over-provisioning — again on $d_{ref}/x_i$, so the
    three histograms are the same measurement three times. First seed only.
    """)
    return


@app.cell(hide_code=True)
def _(
    c_star,
    class_dead,
    np,
    physical_slowdown,
    plt,
    rule_colour,
    rule_label,
    rules,
    runs,
):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    _bins = np.geomspace(0.2, 20.0, 60)
    for _rule in rules:
        _r = runs[_rule, 1.0]
        _phys = physical_slowdown(_rule, 1.0)
        _ax[0].hist(
            np.clip(_phys[np.isfinite(_phys)], 0.2, 20.0),
            bins=_bins,
            histtype="step",
            lw=1.3,
            color=rule_colour[_rule],
            label=rule_label[_rule],
        )
        _held = _r.final_members * _r.catalog.size_gib
        _srt = np.sort(_held)[::-1]
        _ax[2].plot(
            np.arange(1, _srt.size + 1) / _srt.size,
            np.cumsum(_srt) / max(_srt.sum(), 1e-9),
            color=rule_colour[_rule],
            lw=1.3,
            label=rule_label[_rule],
        )
    _ax[0].axvline(c_star, color="C3", ls="--", lw=0.9)
    _ax[0].axvline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].set_xscale("log")
    _ax[0].set_xlabel(r"$d_{ref}/x_i$ at end of run")
    _ax[0].set_ylabel("torrents")
    _ax[0].set_title(r"disk ×1: $C^*$ dashed, $x=d_{ref}$ dotted")
    _ax[0].legend(fontsize=6)

    _classes = runs["clamped", 0.25].catalog.class_names
    _width = 0.8 / len(rules)
    for _i, _rule in enumerate(rules):
        _ax[1].bar(
            np.arange(len(_classes)) + _i * _width,
            class_dead(_rule, 0.25),
            _width,
            color=rule_colour[_rule],
            label=rule_label[_rule],
        )
    _ax[1].set_xticks(np.arange(len(_classes)) + 0.4, _classes)
    _ax[1].set_ylabel(r"fraction with $A_i<\pi_{min}$")
    _ax[1].set_title("scarce site (×0.25): who dies")
    _ax[1].legend(fontsize=6)

    _ax[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    _ax[2].set_xlabel("fraction of catalogue (most held first)")
    _ax[2].set_ylabel("share of committed bytes")
    _ax[2].set_title("concentration of committed disk (×1)")
    _ax[2].legend(fontsize=6)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(class_dead, dead_mask, dead_zone_fraction, mo, np, physical_slowdown, runs):
    def _top_decile_bytes(rule, scale=1.0):
        _r = runs[rule, scale]
        _held = _r.final_members * _r.catalog.size_gib
        _n = max(_r.catalog.n_torrents // 10, 1)
        return float(np.sort(_held)[-_n:].sum() / max(_held.sum(), 1e-9))

    def _big_dead(rule, scale=0.25):
        """Unavailable share of the large + super-large classes, over their
        union rather than the mean of their rates: there are only a handful
        of super-large torrents."""
        _r = runs[rule, scale]
        _big = _r.catalog.class_idx >= 2
        return float(np.mean(dead_mask(rule, scale)[_big]))

    def _deep(rule, scale=1.0):
        """Share of the catalogue provisioned to twice a leecher's downlink."""
        return float(np.mean(physical_slowdown(rule, scale) <= 0.5))

    _names = runs["clamped", 0.25].catalog.class_names
    _dead_cur = class_dead("clamped", 0.25)
    _dead_nof = class_dead("no_floor", 0.25)
    _dead_raw = class_dead("raw", 0.25)
    _by_class = " / ".join(
        f"{_n}: {_c:.0%} → {_f:.0%} → {_w:.0%}"
        for _n, _c, _f, _w in zip(_names, _dead_cur, _dead_nof, _dead_raw, strict=True)
    )
    mo.md(rf"""
    **The floor shifts the whole distribution; it does not put a wall in
    it.** Every rule pushes most of the catalogue past $x = d_{{ref}}$ —
    {dead_zone_fraction("clamped", 1.0):.0%} of torrents under the clamped
    rule, {dead_zone_fraction("no_floor", 1.0):.0%} without the floor —
    because a seeder with free disk takes the best candidate it can see even
    where the health term has nothing left to pay, and $\alpha$ alone
    justifies holding it. What the floor changes is how far past that point
    the mass travels once the health term pays for the trip too:
    {_deep("clamped"):.0%} of the catalogue is provisioned to twice a
    reference leecher's downlink or better under the clamped rule against
    {_deep("no_floor"):.0%} without the floor. §2 showed that this is where
    the tail of the experience is bought.

    **On the scarce site the per-class bars name the culprit.** Large and
    super-large classes lose {_big_dead("clamped"):.0%} (clamped),
    {_big_dead("no_floor"):.0%} (no floor) and {_big_dead("raw"):.0%} (raw)
    of their torrents to $A_i < \pi_{{min}}$; per class,
    clamped → no floor → raw: {_by_class}. The two rules that keep the
    cutoff behave alike; the one that drops it is the one that changes who
    dies. That is part 2's continuous-completeness argument, confirmed, and
    it is a statement about $\pi_{{min}}$ alone.

    Concentration of committed bytes barely moves — the most-held tenth of
    the catalogue holds {_top_decile_bytes("clamped"):.0%} of committed bytes
    under the clamped rule and {_top_decile_bytes("no_floor"):.0%} without
    the floor. The over-provisioning is not a few torrents hoarding disk; it
    is the whole middle of the catalogue carrying some slack.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Turnover: what the floor makes seeders do

    Under the agent rule of §8 a seeder values each holding in points per
    hour per GiB and swaps its worst holding for a better candidate. Under
    the clamped rule an over-provisioned holding is worth exactly
    $\kappa w S^{\eta}\alpha$ per hour and nothing more, so it is cheap to
    abandon: the library keeps turning over. Under the unclamped rules no
    holding is ever worthless, the swap margin is rarely cleared, and the
    library settles.

    An earlier revision read this through the mint/burn ratio and called it
    inflation. With points as a gate rather than a currency that reading is
    empty; what remains is physical. Every acquisition is a download, and it
    runs on the same uplinks demand runs on — one way, though not the only
    way, a second leecher lands on a swarm sized for one.
    """)
    return


@app.cell(hide_code=True)
def _(plt, rule_colour, rule_label, rules, runs):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for _rule in rules:
        _r = runs[_rule, 1.0]
        _d = _r.hours / 24.0
        _col = rule_colour[_rule]
        _per_day = 24.0 / _r.record_every_h
        _ax[0].plot(
            _d,
            _r.replan_joins * _per_day,
            color=_col,
            lw=1.2,
            label=rule_label[_rule],
        )
        _ax[0].plot(_d, _r.replan_drops * _per_day, color=_col, lw=0.8, ls=":")
        _ax[1].plot(_d, _r.invest_leech_share, color=_col, lw=1.2)
        _ax[2].plot(_d, _r.multi_leech_share, color=_col, lw=1.2)
    _ax[0].set_ylabel("memberships / day")
    _ax[0].set_title("re-planning: joins (solid), drops (dotted)")
    _ax[0].legend(fontsize=6)
    _ax[1].set_ylabel("share of leech time")
    _ax[1].set_title("leech time that is re-planning")
    _ax[2].set_ylabel("share of leech time")
    _ax[2].set_title(r"leech time on a swarm with $\geq 2$ leechers")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, stats):
    _cur = stats("clamped", 1.0)
    _nof = stats("no_floor", 1.0)
    mo.md(rf"""
    At ×1 disk seeders start {_cur["joins_day"]:.0f} acquisitions a day under
    the clamped rule and {_nof["joins_day"]:.0f} without the floor;
    {_cur["invest_leech_share"]:.0%} against {_nof["invest_leech_share"]:.0%}
    of all leech time is that traffic, and {_cur["multi_leech_share"]:.0%}
    against {_nof["multi_leech_share"]:.0%} of leech time is spent sharing a
    swarm. The floor does not merely refuse to pay past $x = d_{{ref}}$; by
    driving the marginal to exactly zero it makes an over-provisioned holding
    worth abandoning, and abandoning it is a transfer.

    One caveat on the magnitude: this engine advances in one-hour ticks and
    splits a swarm's capacity among every leecher present at the start of the
    hour, so a five-minute download and a five-hour one collide as if they
    overlapped for the whole hour. The direction — slack absorbs collisions,
    an exact fit does not — does not depend on that; the size of the p90 gap
    is an upper bound.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pi_min, rules, runs, scales, seed_stats, stats):
    _med = {_r: [stats(_r, _s)["median_C_phys"] for _s in scales] for _r in rules}
    _exp = {_r: [stats(_r, _s)["bytes_p90"] for _s in scales] for _r in rules}
    _range = {_r: max(_med[_r]) - min(_med[_r]) for _r in rules}
    _cur1, _nof1, _raw1 = (stats(_r, 1.0) for _r in ("clamped", "no_floor", "raw"))
    _cur0, _nof0, _raw0 = (stats(_r, 0.25) for _r in ("clamped", "no_floor", "raw"))
    _better = int(
        np.sum(
            np.asarray(seed_stats("no_floor", 1.0, "bytes_p90"))
            <= np.asarray(seed_stats("clamped", 1.0, "bytes_p90"))
        )
    )
    _n_seeds = len(seed_stats("clamped", 1.0, "bytes_p90"))

    def _big_dead(rule, scale=0.25):
        _r = runs[rule, scale]
        _mask = _r.catalog.class_idx >= 2
        return float(np.mean((_r.final_completeness < pi_min)[_mask]))

    mo.md(rf"""
    ## 6. Verdict

    1. **Drop $\max(1,\,d_{{ref}}/x_i)$ — on a site that is not short of
       disk.** Its stated justification (§3.3: "capacity beyond the
       leecher's own downlink has no value to that leecher") is true of one
       leecher on a fungible pool and false of a site: the second leecher,
       the re-planning download and the badly-connected peer all draw on
       the slack. Measured on what demand downloads experienced, removing
       the floor moves the p90 completion time by bytes from
       {_cur1["bytes_p90"]:.2f} to {_nof1["bytes_p90"]:.2f} at ×1 disk
       (better or equal on {_better} of {_n_seeds} seeds) and from
       {_exp["clamped"][2]:.2f} to {_exp["no_floor"][2]:.2f} at ×4, with no
       change in availability ({_cur1["unavailable_fraction"]:.1%} vs
       {_nof1["unavailable_fraction"]:.1%}). What the floor delivered was a
       median $C^{{\text{{phys}}}}$ pinned at 1 across a 16× disk swing
       (spread {_range["clamped"]:.2f} vs {_range["no_floor"]:.2f}) — an
       anchor on the *estimate*, bought with a worse tail on the
       *experience* whenever disk is not scarce. Reading the unclamped
       median's drift as redundancy rather than as waste is the correct
       reading in this world, and only more so in one where seeders are not
       equally reachable.

       The condition matters. At ×0.25 disk the p90 by bytes is
       {_exp["clamped"][0]:.2f} with the floor against
       {_exp["no_floor"][0]:.2f} without, a gap made by one seed in which
       slack bought elsewhere was disk a trapped torrent needed (§3); three
       seeds cannot say whether that is systematic. If the site expects to
       run short of disk, keep the floor or watch that regime; if it
       expects to hold more disk than its catalogue needs, the floor is
       the thing throwing that disk away. "Trust retention" is a bet on the
       second case, and this data supports it there and is silent on the
       first.

    2. **The completeness cutoff is a separate trade-off, and its
       replacement is untested.** Isolating $\pi_{{min}}$ (`no_floor` →
       `raw`) on the scarce site: available-torrent p90
       {_nof0["avail_p90_C_phys"]:.2f} → {_raw0["avail_p90_C_phys"]:.2f},
       whole-catalogue unavailability {_nof0["unavailable_fraction"]:.1%} →
       {_raw0["unavailable_fraction"]:.1%}, large-class unavailability
       {_big_dead("no_floor"):.0%} → {_big_dead("raw"):.0%}, stuck demand
       {_nof0["demand_backlog_days"]:.2f} → {_raw0["demand_backlog_days"]:.2f}
       days. The hard cutoff concentrates capacity on the torrents that
       survive and lets the rest die; removing it keeps more of the
       catalogue alive and spreads the same disk thinner over it. Part 2's
       $\min(1, A_i/\pi_{{min}})\cdot H(C_i)$ is the natural candidate for
       getting both, but this engine has simulated only the step and its
       absence, and the scarce-site interactions above are not the kind
       that can be assumed to add up. It is a follow-up rule to run, not a
       conclusion of this notebook. Whatever is done with it is independent
       of the floor.

    3. **The monetary point is withdrawn.** With $\kappa$ fixed the
       mint/burn ratio is {_cur1["mint_burn"]:.1f} (clamped) against
       {_nof1["mint_burn"]:.1f} (no floor) at ×1 and no allocation or
       experience column reads it. What the churn measurement actually
       found is physical: the floor makes seeders move copies around
       ({_cur1["joins_day"]:.0f} vs {_nof1["joins_day"]:.0f} acquisitions a
       day), and the moving is traffic on the swarms demand is using.

    4. **What this model cannot say.** Every seeder here is equally
       reachable by every leecher. A swarm whose $x_i$ is fed by fast
       same-continent transfers while a far leecher crawls is exactly the
       case where the floor's premise fails hardest, and a region-blind
       reward cannot be shown to *fix* it — it can only be shown not to
       refuse the redundancy that happens to help. This notebook shows that
       refusal is already costly with reachability held equal; it does not
       measure the geographic gain, and the choice to trust retention over
       an exact fit rests on judgement there, not on this data.

    5. **What part 2 got right.** The two rules agree on the interior of the
       range on the allocation — at ×1 disk the available-torrent p90 of
       $C^{{\text{{phys}}}}$ differs by
       {abs(_nof1["avail_p90_C_phys"] - _cur1["avail_p90_C_phys"]):.2f} — and
       the frozen-swarm algebra was never the problem. What breaks when the
       clamp is added is the *allocation process* the rewards drive, and
       that needs agents, a finite disk and a demand stream that collides
       with it.

    > [!note] Engine changes in this revision
    > Three defects in part 3's engine were found while instrumenting this
    > comparison, and every number in this notebook is from the corrected
    > engine. (1) A download of a torrent nobody held was accepted, charged
    > and never completed — in the old runs 60 % of leecher-hours were such
    > downloads, reserving disk and burning points forever; a request with
    > no complete copy anywhere is now refused unpaid, for demand and
    > re-planning alike. (2) Re-planning could release the last copy of a
    > torrent somebody was downloading, including two copies in one batch;
    > an accepted download now keeps a source until it completes. (3) A
    > transfer was credited with a full hour at its rate even when the file
    > finished in minutes; bytes moved are now capped by what was left and
    > completion time is measured in the fraction of the hour used. The
    > estimator's observation — the rate while transferring, as a
    > ten-minute announce delta would see it — is unchanged. Instrumentation
    > added: per-download completion time (by count and by bytes), the stuck
    > demand backlog, the re-planning share of leech time, and the estimator
    > concurrency $\tilde k_v$.
    """)
    return


if __name__ == "__main__":
    app.run()
