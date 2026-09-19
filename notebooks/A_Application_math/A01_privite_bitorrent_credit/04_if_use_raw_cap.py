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

    on a **frozen** swarm, and the raw ratio came out looking good: identical
    payouts across almost the whole range, a continuous reward with no quorum
    step, more seeders held at a demanding outside option, and an extra mint
    that the $\alpha n$ term was large enough to hide. The honest caveats in
    that section were that the swarm was one torrent, the seeders were
    identical, memberships never moved, and $\theta$ — the price of a GiB of
    disk — was swept by hand rather than being produced by the site.

    Part 3's engine removes all four caveats: a catalogue, heterogeneous
    nodes, finite disks that must be *taken from somewhere*, and a $\theta$
    that is whatever the competition for those disks makes it. This part runs
    the same world under three scoring rules and asks whether part 2's verdict
    survives:

    | rule | $\max(1, \cdot)$ | $A_i \ge \pi_{min}$ | what it is |
    |---|---|---|---|
    | `clamped` | yes | yes | §3.3 as written — part 3's baseline |
    | `no_floor` | no | yes | only the capacity floor removed |
    | `raw` | no | no | part 2's $C^{\text{raw}} = d_{ref}/x$ |

    The middle row is not decoration: it is what makes the comparison a
    decomposition. `clamped` → `no_floor` prices the floor alone,
    `no_floor` → `raw` prices the completeness cutoff alone.

    Nothing else differs: the catalogue, the population, the seed, $\kappa$'s
    starting value, $\alpha$, $\gamma$, $\eta$ and the agent rule are shared.

    > [!important] Outcomes are measured on one common metric, not on each
    > rule's own score
    > Each rule's $C_i$ is *its own definition* — the clamped one cannot
    > report a value below 1 or a finite value below $\pi_{min}$ — so
    > comparing "median $C_i$" across rules would compare censorings as much
    > as allocations. Every outcome below is therefore read off two
    > rule-independent quantities:
    >
    > * **allocation**: the physical ratio $d_{ref}/x_i$, no clamps, the same
    >   formula in all three worlds. It is written $C^{\text{phys}}$ here;
    > * **availability**: the physical condition $A_i < \pi_{min}$, which a
    >   rule that declines to price completeness still has to answer for.
    >
    > The scored $C_i$ appears only where the subject is the *incentive* —
    > what a seeder is paid — and it is labelled as such.
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
        start=1e-4, stop=10.0, step=1e-4, value=0.05, label=r"initial $\kappa$"
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
    ui_seed = mo.ui.number(start=0, stop=999, step=1, value=0, label="seed")
    mo.vstack(
        [
            mo.md(
                "## 1. Controls\n\nNine runs — three rules × three disk"
                " supplies — at roughly 4 s each, memoised on the exact"
                " parameter pair."
            ),
            mo.hstack([ui_days, ui_torrents, ui_users], widths="equal"),
            mo.hstack([ui_kappa, ui_alpha, ui_gamma, ui_seed], widths="equal"),
        ]
    )
    return ui_alpha, ui_days, ui_gamma, ui_kappa, ui_seed, ui_torrents, ui_users


@app.cell(hide_code=True)
def _(
    RewardParams,
    SimParams,
    ui_alpha,
    ui_days,
    ui_gamma,
    ui_kappa,
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
    scales = (0.25, 1.0, 4.0)

    rewards = {
        _rule: RewardParams(
            kappa=float(ui_kappa.value),
            alpha=float(ui_alpha.value),
            gamma=float(ui_gamma.value),
            eta=1.0,
            slowdown_rule=_rule,
        )
        for _rule in rules
    }
    base = SimParams(
        days=int(ui_days.value),
        n_torrents=int(ui_torrents.value),
        n_users=int(ui_users.value),
        seed=int(ui_seed.value),
    )
    pi_min = float(rewards["clamped"].pi_min)
    c_star = float(rewards["clamped"].c_star)
    return base, c_star, pi_min, rewards, rule_colour, rule_label, rules, scales


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
def _(base, rewards, rules, scales, simulate):
    runs = {
        (_rule, _s): simulate(rewards[_rule], base, disk_scale=_s)
        for _rule in rules
        for _s in scales
    }
    return (runs,)


@app.cell(hide_code=True)
def _(np, pi_min, runs):
    def stats(rule, scale, days=7.0):
        """Acceptance summary plus the flow and churn columns part 4 needs.

        Rates are per day over the same tail window the acceptance criterion
        uses, so ``mint``, ``burn`` and the two churn counters are directly
        comparable across rules and disk supplies.
        """
        r = runs[rule, scale]
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

    return class_dead, dead_mask, dead_zone_fraction, physical_slowdown, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The same site, scored three ways

    Disk supply ×1: the world of part 3, which is over-provisioned in the
    middle of the catalogue and thin at its large end.
    """)
    return


@app.cell(hide_code=True)
def _(mo, rule_label, rules, stats):
    _cols = (
        ("median_C_phys", r"median $C^{\text{phys}}$", "{:.2f}"),
        ("avail_p90_C_phys", r"p90 $C^{\text{phys}}$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"$A_i<\pi_{min}$", "{:.1%}"),
        ("served_share", "capacity drawn", "{:.0%}"),
        ("multi_leech_share", r"leecher-h at $\ge 2$", "{:.0%}"),
        ("members_per_torrent", "members/torrent", "{:.1f}"),
        ("joins_day", "joins/day", "{:.0f}"),
        ("drops_day", "drops/day", "{:.0f}"),
        ("mint_day", "mint/day", "{:.0f}"),
        ("burn_day", "burn/day", "{:.0f}"),
        ("mint_burn", "mint/burn", "{:.2f}"),
        ("alpha_share", r"mint from $\alpha$", "{:.0%}"),
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
        _ax[1].plot(_d, _r.unavailable_fraction, color=_col, lw=1.3)
        _ax[2].plot(_d, _r.members_total / _r.catalog.n_torrents, color=_col, lw=1.3)
    _ax[0].axhline(c_star, color="C3", ls="--", lw=0.9)
    _ax[0].axhline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].set_yscale("log")
    _ax[0].set_ylabel(r"$C^{\text{phys}} = d_{ref}/x_i$")
    _ax[0].set_title(r"median (solid), available p90 (dotted); $C^*$, $C=1$")
    _ax[0].legend(fontsize=6.5)
    _ax[1].set_ylabel(r"fraction with $A_i<\pi_{min}$")
    _ax[1].set_title("availability is barely touched")
    _ax[2].set_ylabel("memberships / torrent")
    _ax[2].set_title("how much the site ends up holding")
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
    Two things are already visible, and neither was predictable from the
    frozen swarm.

    **The floor stops the site buying capacity past a single leecher's
    downlink — but it is a soft stop, not a wall.** Measured physically, the
    clamped rule's median torrent ends at
    $C^{{\text{{phys}}}} = {_cur["median_C_phys"]:.2f}$: just past
    $x = d_{{ref}}$, the point where a single reference leecher's own
    downlink becomes the binding constraint and the marginal reaches exactly
    zero.
    (Its *scored* median reads {_cur["median_C_all"]:.2f}, because the clamp
    cannot report anything below 1 — which is precisely why the comparison
    has to be made on the physical ratio.) Under the raw rule the median goes
    to {_raw["median_C_phys"]:.2f} and under `no_floor` to
    {_nof["median_C_phys"]:.2f}: the median torrent ends up carrying
    ${_cur["median_C_phys"] / max(_raw["median_C_phys"], 1e-9):.2f}\times$
    the capacity the clamped rule buys it, all of the excess beyond what one
    leecher can absorb. {dead_zone_fraction("clamped", 1.0):.0%}
    of the catalogue is already at or past $x = d_{{ref}}$ under the clamped
    rule, against {dead_zone_fraction("raw", 1.0):.0%} under the raw one.

    Whether that surplus is waste depends on how many leechers share a swarm
    at once — part 2's one honest argument for the raw form — and that is a
    measurement, not an assumption. It is a real effect in this world:
    {_cur["multi_leech_share"]:.0%} of leecher-hours under the clamped rule
    are spent on a swarm serving two or more leechers at once
    ({_raw["multi_leech_share"]:.0%} under the raw rule), so $x > d_{{ref}}$
    is genuinely consumable some of the time. What the utilisation column
    says is that it is not in fact consumed: of the capacity offered on
    torrents that had *any* leecher, {_cur["served_share"]:.0%} is drawn
    under the clamped rule and {_raw["served_share"]:.0%} under the raw one.
    The extra capacity the raw rule pays for is reaching a smaller share of
    demand, not a larger one — and it is bought with disk taken from
    somewhere else.

    **Removing the $\pi_{{min}}$ cliff does almost nothing for availability
    in a healthy world.** {_nof["unavailable_fraction"]:.1%} of the catalogue
    sits below the completeness threshold with the cutoff priced, and
    {_raw["unavailable_fraction"]:.1%} without it. Part 2's coordination trap
    is real but it is not what limits this site; §3 puts it where it does
    bite.

    The column that moves most is not a health column at all. Under the
    clamped rule the site's seeders start
    {_cur["joins_day"]:.0f} acquisitions a day and burn
    {_cur["burn_day"]:.0f} points; under the raw rule,
    {_raw["joins_day"]:.0f} and {_raw["burn_day"]:.0f}. §5 is about that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The acceptance criterion: insensitivity to disk supply

    §8 asks for the $C_i$ distribution to collapse toward $C^*$ **regardless
    of total disk supply**, and §4 says the knee is what delivers it: with a
    plateau above target and near-zero reward below, the crossing with the
    opportunity cost $\theta$ lands at the knee for any $\theta$ in a wide
    range, so the equilibrium does not track the price of disk.

    Part 2 could not test that, because it *set* $\theta$ by hand. Here
    $\theta$ is endogenous — it is whatever the competition for a finite disk
    makes it — so scaling total disk by 1/4 and 4 is a direct test of the
    claim. The test is run on $C^{\text{phys}} = d_{ref}/x_i$, so that a rule
    which cannot *report* an over-provisioned torrent is not credited with
    preventing one.
    """)
    return


@app.cell(hide_code=True)
def _(mo, rule_label, rules, scales, stats):
    _cols = (
        ("median_C_phys", r"median $C^{\text{phys}}$", "{:.2f}"),
        ("avail_p90_C_phys", r"p90 $C^{\text{phys}}$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"$A_i<\pi_{min}$", "{:.1%}"),
        ("members_per_torrent", "members/torrent", "{:.1f}"),
        ("joins_day", "joins/day", "{:.0f}"),
        ("mint_burn", "mint/burn", "{:.2f}"),
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
def _(c_star, np, plt, rule_colour, rule_label, rules, scales, stats):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    _x = np.asarray(scales)
    for _rule in rules:
        _col = rule_colour[_rule]
        _med = [stats(_rule, _s)["median_C_phys"] for _s in scales]
        _p90 = [stats(_rule, _s)["avail_p90_C_phys"] for _s in scales]
        _out = [stats(_rule, _s)["unavailable_fraction"] for _s in scales]
        _ax[0].plot(_x, _med, "o-", color=_col, lw=1.3, ms=3, label=rule_label[_rule])
        _ax[1].plot(_x, _p90, "o-", color=_col, lw=1.3, ms=3)
        _ax[2].plot(_x, _out, "o-", color=_col, lw=1.3, ms=3)
    for _a, _t, _y in (
        (
            _ax[0],
            r"median $C^{\text{phys}}$: the floor holds it",
            r"median $d_{ref}/x_i$",
        ),
        (_ax[1], "p90 over available torrents", r"p90 $d_{ref}/x_i$"),
        (_ax[2], r"catalogue below $\pi_{min}$", "unavailable fraction"),
    ):
        _a.set_xscale("log")
        _a.set_xticks(_x, [f"×{_s}" for _s in scales])
        _a.set_xlabel("total disk supply")
        _a.set_ylabel(_y)
        _a.set_title(_t)
    _ax[0].axhline(c_star, color="C3", ls="--", lw=0.9)
    _ax[0].axhline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].legend(fontsize=6.5)
    _ax[1].axhline(c_star, color="C3", ls="--", lw=0.9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, rules, scales, stats):
    _med = {_r: [stats(_r, _s)["median_C_phys"] for _s in scales] for _r in rules}
    _range = {_r: max(_med[_r]) - min(_med[_r]) for _r in rules}
    # How much of the last 4x of disk lands in swarms that are already
    # past d_ref: the drop in median C_phys from x1 to x4.
    _rich_drop = {_r: 1.0 - _med[_r][2] / max(_med[_r][1], 1e-9) for _r in rules}
    _scarce_cur = stats("clamped", 0.25)
    _scarce_gap = stats("raw", 0.25)["avail_p90_C_phys"] / max(
        stats("clamped", 0.25)["avail_p90_C_phys"], 1e-9
    )
    _scarce_nof = stats("no_floor", 0.25)
    _scarce_raw = stats("raw", 0.25)
    _rich_cur = stats("clamped", 4.0)
    _rich_raw = stats("raw", 4.0)
    mo.md(rf"""
    > [!warning] The floor is the robustness, and part 2 could not see it
    > Over a 16× swing in disk supply the clamped rule's median
    > $C^{{\text{{phys}}}}$ moves {_med["clamped"][0]:.2f} →
    > {_med["clamped"][1]:.2f} → {_med["clamped"][2]:.2f} — a spread of
    > {_range["clamped"]:.2f}. The raw rule moves {_med["raw"][0]:.2f} →
    > {_med["raw"][1]:.2f} → {_med["raw"][2]:.2f}, a spread of
    > {_range["raw"]:.2f}, and `no_floor` tracks it
    > ({_range["no_floor"]:.2f}). Read on the same physical quantity, in the
    > same world, **the equilibrium under the unclamped rules is a function
    > of how much disk the community happens to own** — the failure §4
    > designed the knee to avoid.
    >
    > The sharpest version is the last quadrupling of disk, where nothing is
    > scarce and the only question is where surplus capacity goes: the
    > clamped rule absorbs it almost entirely elsewhere — its median
    > $C^{{\text{{phys}}}}$ falls only {_rich_drop["clamped"]:.0%} from ×1 to
    > ×4 — while the raw rule keeps pushing it into swarms that are already
    > past $d_{{ref}}$ ({_rich_drop["raw"]:.0%}).
    >
    > The mechanism is simple once stated: $H$ is concave but it does not
    > *stop*. Without $\max(1,\cdot)$ there is always a little more health to
    > buy, so cheap disk keeps being spent until $H$'s own saturation finally
    > flattens the marginal somewhere below $C = 1$; with the floor, the
    > marginal hits exactly zero at $x = d_{{ref}}$ and surplus disk has to
    > go somewhere else. A hard zero is a stronger anchor than a small
    > number.
    >
    > Note also that `no_floor` and `raw` coincide exactly at ×4 disk
    > ({_rich_raw["median_C_phys"]:.2f} both,
    > {_rich_raw["unavailable_fraction"]:.1%} unavailable): with enough disk
    > nothing falls below $\pi_{{min}}$, so the cutoff is never evaluated.
    > Every difference between those two rows elsewhere is the completeness
    > term and nothing else.

    On a scarce site the ranking inverts relative to part 2's table. At ×0.25
    disk the clamped rule holds the available-torrent p90 at
    {_scarce_cur["avail_p90_C_phys"]:.2f} against the raw rule's
    {_scarce_raw["avail_p90_C_phys"]:.2f}
    (+{100 * (_scarce_gap - 1.0):.0f}%),
    while the raw rule buys back
    {_scarce_cur["unavailable_fraction"] - _scarce_raw["unavailable_fraction"]:.1%}
    of the catalogue in availability. Part 2 predicted the second effect (no
    quorum step, so a lone seeder is paid) and had no way to predict the
    first. At ×4 the two rules agree on availability
    ({_rich_cur["unavailable_fraction"]:.1%} vs
    {_rich_raw["unavailable_fraction"]:.1%}), so the remaining difference is
    purely how much idle capacity each rule keeps buying: under the raw
    rule even the p90 torrent is over-provisioned
    ({_rich_raw["avail_p90_C_phys"]:.2f} against
    {_rich_cur["avail_p90_C_phys"]:.2f}) on a site where only
    {_rich_raw["multi_leech_share"]:.0%} of leecher-hours see a second
    leecher at all — spare disk parked against demand that is not there,
    rather than a tail being repaired.

    The `no_floor` row separates the two clamps cleanly: at ×0.25 it sits at
    a p90 of {_scarce_nof["avail_p90_C_phys"]:.2f} with
    {_scarce_nof["unavailable_fraction"]:.1%} unavailable, so the scarce-site
    availability gain belongs to dropping $\pi_{{min}}$ and the tail damage
    belongs to dropping the floor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Where the capacity ends up

    The medians above are one number per site. The distributions say which
    torrents paid for the over-provisioning — again on $d_{ref}/x_i$, so the
    three histograms are the same measurement three times.
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
def _(
    class_dead,
    dead_mask,
    dead_zone_fraction,
    mo,
    np,
    physical_slowdown,
    runs,
    stats,
):
    def _top_decile_bytes(rule, scale=1.0):
        _r = runs[rule, scale]
        _held = _r.final_members * _r.catalog.size_gib
        _n = max(_r.catalog.n_torrents // 10, 1)
        return float(np.sort(_held)[-_n:].sum() / max(_held.sum(), 1e-9))

    def _big_dead(rule, scale=0.25):
        _r = runs[rule, scale]
        _big = _r.catalog.class_idx >= 2
        return float(np.mean(dead_mask(rule, scale)[_big]))

    def _deep(rule, scale=1.0):
        """Share of the catalogue provisioned to twice a leecher's downlink."""
        return float(np.mean(physical_slowdown(rule, scale) <= 0.5))

    _names = runs["clamped", 0.25].catalog.class_names
    _p90_cost = stats("raw", 0.25)["avail_p90_C_phys"] / max(
        stats("clamped", 0.25)["avail_p90_C_phys"], 1e-9
    )
    _dead_cur = class_dead("clamped", 0.25)
    _dead_raw = class_dead("raw", 0.25)
    _by_class = " / ".join(
        f"{_n}: {_c:.0%}→{_w:.0%}"
        for _n, _c, _w in zip(_names, _dead_cur, _dead_raw, strict=True)
    )
    mo.md(rf"""
    **The histogram says the floor shifts the whole distribution, it does
    not put a wall in it.** Both rules push most of the catalogue past
    $x = d_{{ref}}$ — {dead_zone_fraction("clamped", 1.0):.0%} of torrents
    under the clamped rule, {dead_zone_fraction("raw", 1.0):.0%} under the
    raw one — because a seeder with free disk takes the best candidate it can
    see even where the health term has nothing left to pay, and $\alpha$
    alone justifies holding it. What the floor changes is how far past that
    point the mass travels once the health term pays for the trip too:
    {_deep("clamped"):.0%} of the catalogue is provisioned to twice a
    reference leecher's downlink or better under the clamped rule against
    {_deep("raw"):.0%} under the raw one, and the body of the distribution
    shifts with it. Demand does not follow it there: utilisation of the
    offered capacity on torrents that have a leecher goes
    {stats("clamped", 1.0)["served_share"]:.0%} →
    {stats("raw", 1.0)["served_share"]:.0%}, so the disk that moved left
    came out of the right-hand tail and bought a smaller share of served
    bytes.

    **On a scarce site that transfer has a name.** At ×0.25 the large and
    super-large classes lose
    {_big_dead("clamped"):.0%} (clamped) against {_big_dead("raw"):.0%} (raw)
    of their torrents to $A_i < \pi_{{min}}$ — the raw rule is *better* here,
    and the per-class bars show it is better in exactly the classes where the
    quorum step bites: {_by_class}.
    That is part 2's continuous-completeness argument, confirmed. It is also
    the *only* column in this notebook where dropping a clamp helps, and §3
    showed it costs a
    {_p90_cost:.1f}×
    worse p90 to buy.

    Concentration of committed bytes barely moves — the most-held tenth of
    the catalogue holds {_top_decile_bytes("clamped"):.0%} of committed bytes
    under the clamped rule and {_top_decile_bytes("raw"):.0%} under the raw
    one. The over-provisioning is not a few torrents hoarding disk; it is the
    whole middle of the catalogue drifting a little past the point where extra
    capacity stops being usable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The economy: the clamp is what makes the site spend

    Part 2 expected the raw rule's cost to show up as **extra mint** — health
    paid for capacity past the floor — and argued the $\alpha n$ term was
    large enough to mask it. In motion the mint barely moves and the cost
    appears on the other side of the ledger.

    The reason is the agent rule of §8. A seeder values each holding in points
    per hour per GiB and swaps its worst holding for a better candidate. Under
    the clamped rule an over-provisioned holding is worth exactly
    $\kappa w S^{\eta}\alpha$ per hour and nothing more, so it is cheap to
    abandon: the library keeps turning over, and every acquisition is a
    charged download. Under the raw rule no holding is ever worthless, so the
    swap margin is rarely cleared, the library ossifies — and the site's
    largest points **sink** dries up.
    """)
    return


@app.cell(hide_code=True)
def _(plt, rule_colour, rule_label, rules, runs):
    _fig, _ax = plt.subplots(1, 4, figsize=(13.5, 3.1))
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
        _ax[1].plot(_d, _r.mint * _per_day, color=_col, lw=1.2)
        _ax[1].plot(_d, _r.burn * _per_day, color=_col, lw=0.8, ls="--")
        _ax[2].semilogy(
            _d, _r.mint / (_r.burn + 1e-9), color=_col, lw=1.1, label=rule_label[_rule]
        )
        _ax[3].semilogy(_d, _r.kappa, color=_col, lw=1.2)
    _ax[0].set_ylabel("memberships / day")
    _ax[0].set_title("re-planning: joins (solid), drops (dotted)")
    _ax[0].legend(fontsize=6)
    _ax[1].set_yscale("log")
    _ax[1].set_ylabel("points / day")
    _ax[1].set_title("mint (solid) and burn (dashed)")
    _ax[2].axhline(1.0, color="k", ls="--", lw=0.9)
    _ax[2].set_ylabel("mint / burn")
    _ax[2].set_title("monetary balance, target 1")
    _ax[2].legend(fontsize=6)
    _ax[3].set_ylabel(r"$\kappa$")
    _ax[3].set_title(r"$\kappa$ controller (disk ×1)")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig

    return


@app.cell(hide_code=True)
def _(mo, stats):
    _cur = stats("clamped", 1.0)
    _raw = stats("raw", 1.0)
    _nof = stats("no_floor", 1.0)
    _mint_gap = _raw["mint_day"] / max(_cur["mint_day"], 1e-9) - 1.0
    _burn_gap = _raw["burn_day"] / max(_cur["burn_day"], 1e-9) - 1.0
    _health_cur = _cur["mint_day"] * (1.0 - _cur["alpha_share"])
    _health_raw = _raw["mint_day"] * (1.0 - _raw["alpha_share"])
    mo.md(rf"""
    > [!important] The inflation under the raw rule is a burn deficit, not a
    > mint excess
    > Mint moves by {_mint_gap:+.0%} between the clamped and the raw rule
    > ({_cur["mint_day"]:.0f} → {_raw["mint_day"]:.0f} points/day) and the
    > health part of it by
    > {_health_raw / max(_health_cur, 1e-9) - 1.0:+.0%}
    > ({_health_cur:.0f} → {_health_raw:.0f} points/day) — part 2's predicted
    > extra payment for capacity past $d_{{ref}}$ is there, but it is small
    > and it is
    > partly self-cancelling, because a rule that pays everywhere also lets
    > $H$ saturate and flattens its own marginal.
    >
    > Burn moves by {_burn_gap:+.0%} ({_cur["burn_day"]:.0f} →
    > {_raw["burn_day"]:.0f} points/day), of which the acquisition component
    > goes {_cur["invest_day"]:.0f} → {_raw["invest_day"]:.0f}. Seeders start
    > {_cur["joins_day"]:.0f} acquisitions a day under the clamped rule and
    > {_raw["joins_day"]:.0f} under the raw one. The mint/burn ratio therefore
    > goes {_cur["mint_burn"]:.2f} → {_raw["mint_burn"]:.2f}
    > (`no_floor`: {_nof["mint_burn"]:.2f}), and the $\kappa$ controller —
    > which can only deflate everybody at once — has roughly twice as much to
    > absorb.
    >
    > This is a mechanism-design point that only a closed loop can show. The
    > floor does not merely refuse to pay past $x = d_{{ref}}$; by driving
    > the marginal to **exactly zero** it makes an over-provisioned holding
    > *worth abandoning*, and abandonment is what recycles disk and what keeps
    > the site's only sink flowing. A reward surface with no dead zone is a
    > reward surface with no reason to move.

    One caveat on reading the ratios: acquisitions are charged downloads in
    this engine, so a rule that suppresses churn suppresses burn almost
    mechanically. On a real site some of that turnover would be cross-seeding
    a library the user already owns, which §3.5 makes free — so treat the
    direction as robust and the magnitude as an upper bound.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pi_min, rules, runs, scales, stats):
    _med = {_r: [stats(_r, _s)["median_C_phys"] for _s in scales] for _r in rules}
    _range = {_r: max(_med[_r]) - min(_med[_r]) for _r in rules}
    _cur1, _raw1, _nof1 = (stats(_r, 1.0) for _r in ("clamped", "raw", "no_floor"))
    _cur0, _raw0, _nof0 = (stats(_r, 0.25) for _r in ("clamped", "raw", "no_floor"))

    def _big_dead(rule, scale=0.25):
        """Unavailable share of the large + super-large classes.

        Over the union of the two classes, not the mean of their rates:
        there are only a handful of super-large torrents and averaging
        rates would weight them like the hundreds of large ones.
        """
        _r = runs[rule, scale]
        _mask = _r.catalog.class_idx >= 2
        return float(np.mean((_r.final_completeness < pi_min)[_mask]))

    _big_cur = _big_dead("clamped")
    _big_nof = _big_dead("no_floor")
    _big_raw = _big_dead("raw")
    mo.md(rf"""
    ## 6. Verdict, and what changes in the handoff

    1. **Keep $\max(1,\,d_{{ref}}/x_i)$ — it is load-bearing, and part 2
       under-priced it.** Part 2 judged the floor "worth keeping and costs
       little" from a frozen swarm. In motion it is the clamp that delivers
       §8's acceptance criterion. Measured on the physical ratio
       $d_{{ref}}/x_i$, so that no rule is credited for a censoring, the
       median moves {_range["clamped"]:.2f} across a 16× disk swing with the
       floor and {_range["raw"]:.2f} without it
       ({_med["clamped"][0]:.2f}→{_med["clamped"][2]:.2f} against
       {_med["raw"][0]:.2f}→{_med["raw"][2]:.2f}). The knee alone does not
       anchor the target; the knee **plus a hard zero** does.

    2. **The floor is also the site's monetary policy.** It is what makes an
       over-provisioned holding worth only $\alpha$, hence worth swapping
       away, hence a charged acquisition. Removing it cuts burn by
       {1.0 - _raw1["burn_day"] / max(_cur1["burn_day"], 1e-9):.0%} and pushes
       mint/burn from {_cur1["mint_burn"]:.2f} to {_raw1["mint_burn"]:.2f} at
       ×1 disk. Mint itself moves by only
       {_raw1["mint_day"] / max(_cur1["mint_day"], 1e-9) - 1.0:+.0%}. The
       reward rule and the point supply are not separable questions.

    3. **$\pi_{{min}}$ remains the one clamp with a real cost, and the fix is
       still the one part 2 proposed.** Isolating it (`no_floor` → `raw`) on
       the scarce site: large and super-large unavailability
       {_big_nof:.0%} → {_big_raw:.0%} (baseline {_big_cur:.0%}),
       whole-catalogue
       {_nof0["unavailable_fraction"]:.1%} →
       {_raw0["unavailable_fraction"]:.1%}. That is the coordination trap,
       measured, and it argues for the continuous form
       $\min(1, A_i/\pi_{{min}})\cdot H(C_i)$ — which keeps the redundancy
       premium that $g_v$'s linear $\tilde p_v$ discount cannot express, while
       removing the step. It does **not** argue for dropping the floor, and
       the two questions should stop being bundled as "the clamps".

    4. **Where the raw rule genuinely wins, it wins for the completeness
       reason only.** Every raw-rule advantage in this notebook —
       {_cur0["unavailable_fraction"]:.1%} → {_raw0["unavailable_fraction"]:.1%}
       unavailable at ×0.25 — is reproduced by `no_floor` → `raw` and none of
       it by `clamped` → `no_floor`. Meanwhile the scarce-site p90 goes
       {_cur0["avail_p90_C_phys"]:.2f} → {_raw0["avail_p90_C_phys"]:.2f}. Taking
       the
       good half without the bad one is exactly what the continuous
       completeness term does.

    5. **What part 2 got right, and why the frozen analysis was still worth
       doing.** The two rules do agree on the interior of the range: at ×1
       disk the available-torrent p90 differs by
       {abs(_raw1["avail_p90_C_phys"] - _cur1["avail_p90_C_phys"]):.2f} and the
       mint by
       a few per cent. Part 2's error was not in the payout algebra, it was in
       assuming the payout is the whole mechanism. What breaks when the clamp
       goes is not any single torrent's reward but the *allocation process*
       the rewards drive, and that needs agents, a finite disk and a price of
       disk the site sets for itself.

    > [!note] Instrumentation this part adds
    > `c_physical_quantiles` / `c_physical_avail` (the unclamped
    > $d_{{ref}}/x_i$, which is what makes two scoring rules comparable at
    > all), `mint_floor` (the $\alpha$ share of mint, §8's permanent list),
    > `burn_invest` (acquisitions, which are burn and were previously missing
    > from the mint/burn ratio the $\kappa$ controller reads) and
    > `replan_joins` / `replan_drops` (library turnover). The churn counters
    > are what turned "the raw rule inflates" into "the raw rule stops the
    > site spending", and a production dashboard wants all of them.
    """)
    return


if __name__ == "__main__":
    app.run()
