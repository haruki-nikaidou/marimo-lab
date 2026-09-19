import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from marimo_lab.ptcredit import RewardParams, SimParams, run

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9.0, "axes.grid": True, "grid.alpha": 0.3}
    )
    return RewardParams, SimParams, mo, np, plt, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point system, part 3 — agent-based simulation

    This is experiment 1 of §8. Every hour of simulated site time:

    1. each node is online with its true probability;
    2. leechers are served — a node with $k$ busy torrents splits its uplink
       $k$ ways, each leecher takes the lesser of its downlink and its share;
    3. the §3.2 estimators see **only what a tracker would see** — online/offline
       hours and the observed upload rate on *qualifying* intervals (those with
       at least one leecher) — never the ground truth;
    4. $x_i$, $A_i$, $C_i$, $H_i$ and every leave-one-out $H_i^{(-u)}$ are
       recomputed and rewards accrue to the users online on each torrent;
    5. downloads start (burning $\phi_i \cdot \text{size}_i$ points, refused if
       the balance cannot cover it) and finish (becoming seeds);
    6. four times a day every seeder re-plans, exactly as §8 specifies: it
       values each holding and a sample of candidates in **points per hour per
       GiB of disk** — the quantity a finite disk actually trades off — and
       makes at most **one** move, taking free space if it has any and
       otherwise dropping its lowest-value holding for a better one;
    7. once a day the population priors refresh and the §3.7 $\kappa$
       controller moves on the trailing mint/burn ratio.

    **The acceptance criterion under test** (§8): *the $C_i$ distribution
    collapses toward $C^*$ regardless of total disk supply, with no sustained
    oscillation at $\gamma = 4$.*

    > [!important] What "unavailable" means here, and how $C_i$ is reported
    > Every membership in this engine is a **complete** seeder — partial
    > seeders are not modelled. So a torrent with $C_i = \infty$ is *not* one
    > where no copy exists; it is one whose completeness probability
    > $A_i = 1 - \prod_{v}(1 - \tilde p_v)$ has fallen below $\pi_{min}$, i.e.
    > the tracker's pessimistic estimate says a complete copy is not reliably
    > *online*. That is what §3.3 turns into $C_i = \infty$ and $H_i = 0$, and
    > it is the quantity reported as the **unavailable fraction** below.
    >
    > Quantiles that silently drop those torrents show a beautifully converged
    > distribution on a site that is quietly losing its catalogue, so every
    > quantile here is taken over the **whole catalogue** with unavailable
    > torrents counted as $\infty$. The available-only figures sit next to
    > them and are meaningful only together with the unavailable fraction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Controls

    A run is about 4–10 s; the disk sweep runs three of them and is cached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_days = mo.ui.slider(
        start=15, stop=90, step=5, value=40, label="days", show_value=True
    )
    ui_torrents = mo.ui.dropdown(
        options={"1k": 1000, "2k": 2000, "4k": 4000, "8k": 8000},
        value="2k",
        label="torrents",
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
    ui_eta = mo.ui.slider(
        start=0.9,
        stop=1.1,
        step=0.01,
        value=1.0,
        label=r"size exponent $\eta$",
        show_value=True,
    )
    ui_mode = mo.ui.dropdown(
        options={"pay (w_i = 2^z)": "pay", "target (C* = C*/2^z)": "target"},
        value="pay (w_i = 2^z)",
        label="importance acts on",
    )
    ui_stay = mo.ui.slider(
        start=1.0,
        stop=1.5,
        step=0.05,
        value=1.0,
        label=r"hysteresis $C^*_{stay}/C^*$",
        show_value=True,
    )
    ui_seed = mo.ui.number(start=0, stop=999, step=1, value=0, label="seed")
    mo.vstack(
        [
            mo.hstack([ui_days, ui_torrents, ui_users], widths="equal"),
            mo.hstack([ui_kappa, ui_alpha, ui_gamma], widths="equal"),
            mo.hstack([ui_eta, ui_mode, ui_stay, ui_seed], widths="equal"),
        ]
    )
    return (
        ui_alpha,
        ui_days,
        ui_eta,
        ui_gamma,
        ui_kappa,
        ui_mode,
        ui_seed,
        ui_stay,
        ui_torrents,
        ui_users,
    )


@app.cell(hide_code=True)
def _(
    RewardParams,
    SimParams,
    ui_alpha,
    ui_days,
    ui_eta,
    ui_gamma,
    ui_kappa,
    ui_mode,
    ui_seed,
    ui_stay,
    ui_torrents,
    ui_users,
):
    reward = RewardParams(
        kappa=float(ui_kappa.value),
        alpha=float(ui_alpha.value),
        gamma=float(ui_gamma.value),
        eta=float(ui_eta.value),
        importance_mode=ui_mode.value,
    )
    base = SimParams(
        days=int(ui_days.value),
        n_torrents=int(ui_torrents.value),
        n_users=int(ui_users.value),
        stay_bonus=float(ui_stay.value),
        seed=int(ui_seed.value),
    )
    return base, reward


@app.cell(hide_code=True)
def _(run):
    from dataclasses import replace

    _runs_memo: dict = {}

    def simulate(reward, base, **overrides):
        """One run, memoised on the exact (reward, params) pair.

        `mo.cache` is deliberately not used here: it collapsed variants that
        differ only in a reward-formula field onto one cache entry, which
        silently made every sensitivity row identical to the baseline.
        """
        params = replace(base, **overrides) if overrides else base
        key = (reward, params)
        if key not in _runs_memo:
            _runs_memo[key] = run(params, reward)
        return _runs_memo[key]

    def sweep(reward, base, scales):
        return {s: simulate(reward, base, disk_scale=s) for s in scales}

    return simulate, sweep


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The disk-supply sweep

    Total disk supply is scaled by 1/4, 1 and 4 around the population of
    part 1. The acceptance criterion asks that the $C_i$ distribution land in
    the same place in all three.
    """)
    return


@app.cell
def _(base, reward, sweep):
    scales = (0.25, 1.0, 4.0)
    runs = sweep(reward, base, scales)
    return runs, scales


@app.cell(hide_code=True)
def _(mo, runs, scales):
    _keys = (
        "median_C_all",
        "p90_C_all",
        "avail_median_C",
        "avail_p90_C",
        "unavailable_fraction",
        "swing_C",
        "mint_burn",
        "kappa",
    )
    _head = "| disk supply | " + " | ".join(_keys) + " | seeders/torrent |"
    _sep = "|---" * (len(_keys) + 2) + "|"
    _rows = []
    for _s in scales:
        _a = runs[_s].acceptance(days=7)
        _per = runs[_s].members_total[-1] / runs[_s].catalog.n_torrents
        _rows.append(
            f"| ×{_s} | "
            + " | ".join(f"{_a[_k]:.3g}" for _k in _keys)
            + f" | {_per:.1f} |"
        )
    mo.md("\n".join([_head, _sep, *_rows]))
    return


@app.cell(hide_code=True)
def _(plt, runs, scales):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.2), sharex=True)
    for _s, _c in zip(scales, ("C0", "C1", "C2"), strict=True):
        _r = runs[_s]
        _d = _r.hours / 24.0
        _ax[0].plot(_d, _r.c_quantiles_all[:, 1], color=_c, lw=1.2, label=f"×{_s}")
        _ax[0].plot(_d, _r.c_quantiles_all[:, 2], color=_c, lw=0.7, ls=":")
        _ax[1].plot(_d, _r.unavailable_fraction, color=_c, lw=1.2, label=f"×{_s}")
        _ax[2].plot(_d, _r.members_total / _r.catalog.n_torrents, color=_c, lw=1.2)
    _ax[0].axhline(1.5, color="k", ls="--", lw=0.9)
    _ax[0].set_yscale("log")
    _ax[0].set_ylabel(r"$C_i$ (whole catalogue)")
    _ax[0].set_title(r"median (solid) and p90 (dotted); $C^*$ dashed")
    _ax[0].legend(fontsize=7)
    _ax[1].set_ylabel("unavailable fraction")
    _ax[1].set_title(r"torrents with $A_i < \pi_{min}$")
    _ax[1].legend(fontsize=7)
    _ax[2].set_ylabel("memberships / torrent")
    _ax[2].set_title("how much the site is holding")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(np, plt, runs, scales):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for _s, _c in zip(scales, ("C0", "C1", "C2"), strict=True):
        _r = runs[_s]
        _fin = _r.final_slowdown
        _avail = np.isfinite(_fin)
        _ax[0].hist(
            np.clip(_fin[_avail], 1.0, 20.0),
            bins=np.geomspace(1.0, 20.0, 50),
            histtype="step",
            lw=1.3,
            color=_c,
            label=f"×{_s} ({1 - _avail.mean():.0%} unavailable)",
        )
    _ax[0].axvline(1.5, color="k", ls="--", lw=0.9)
    _ax[0].set_xscale("log")
    _ax[0].set_xlabel(r"$C_i$ at the end of the run (available torrents)")
    _ax[0].set_ylabel("torrents")
    _ax[0].set_title(r"final distribution, $C^*$ dashed")
    _ax[0].legend(fontsize=7)

    _r = runs[1.0]
    _names = _r.catalog.class_names
    for _j, _c in enumerate(("C0", "C1", "C2", "C3")):
        _m = _r.catalog.class_idx == _j
        _fin = _r.final_slowdown[_m]
        _ax[1].bar(
            _j,
            float(np.mean(~np.isfinite(_fin))),
            color=_c,
            label=f"{_names[_j]} (n={_m.sum()})",
        )
    _ax[1].set_xticks(range(4), _names)
    _ax[1].set_ylabel(r"fraction with $A_i < \pi_{min}$")
    _ax[1].set_title("which size class loses availability (disk ×1)")
    _ax[1].legend(fontsize=6.5)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, np, reward, runs, scales):
    _acc = {_s: runs[_s].acceptance(days=7) for _s in scales}
    _med = [_acc[_s]["median_C_all"] for _s in scales]
    _p90 = [_acc[_s]["avail_p90_C"] for _s in scales]
    _out = [_acc[_s]["unavailable_fraction"] for _s in scales]
    _osc = max(_acc[_s]["oscillation"] for _s in scales)
    _finite = [_m for _m in _med if np.isfinite(_m)]
    _spread = (max(_finite) - min(_finite)) if _finite else float("nan")
    _r1 = runs[1.0]
    _per = _r1.members_total[-1] / _r1.catalog.n_torrents
    _cat = _r1.catalog
    _held = _r1.final_members * _cat.size_gib
    _tenth = np.argsort(_r1.final_members)[-_cat.n_torrents // 10 :]
    _share_top = float(
        _r1.final_members[_tenth].sum() / max(_r1.final_members.sum(), 1)
    )
    _share_top_disk = float(_held[_tenth].sum() / max(_held.sum(), 1e-9))
    _big = _cat.class_idx >= 2
    _cat_share_big = float(_cat.size_gib[_big].sum() / _cat.size_gib.sum())
    _held_share_big = float(_held[_big].sum() / max(_held.sum(), 1e-9))
    _fin_big = _r1.final_slowdown[_big]
    _med_big = float(np.median(_fin_big[np.isfinite(_fin_big)]))
    _copies = float(_r1.user_disk_budget.sum() / _cat.size_gib.sum())
    _at_floor = float(np.mean(_r1.final_slowdown <= 1.0 + 1e-9))
    _tilt = reward.eta - 1.0
    if abs(_tilt) < 1e-9:
        _tilt_note = (
            "it is not a size bias: at the default $\\eta = 1$ the pay per "
            "GiB of disk is flat in size"
        )
    else:
        _tilt_note = (
            "the control adds a size tilt to it: pay per GiB of disk scales "
            f"as $\\text{{size}}^{{{_tilt:+.2f}}}$, favouring the "
            f"{'small' if _tilt < 0 else 'large'} end"
        )
    mo.md(rf"""
    > [!note] Two of the three clauses pass; the one that fails is the
    > interesting one
    > **Insensitivity to disk supply: yes.** Over a 16× range the
    > whole-catalogue median $C_i$ moves only
    > {_med[0]:.2f} → {_med[1]:.2f} → {_med[2]:.2f} (spread {_spread:.2f}).
    > That is the robustness the knee was designed for: the equilibrium does
    > not track the price of disk.
    >
    > **No sustained oscillation: yes.** Measured on the cross-torrent mean
    > seeder count — the series in which herding would actually show — the
    > detrended peak-to-peak amplitude is {_osc:.1%} of its level at
    > $\gamma = 4$. (The $C$ quantiles are a poor oscillation proxy: they also
    > move with the unavailable mass, and can be infinite.)
    >
    > **Collapse *onto* $C^*$: no.** The median settles at {_med[1]:.2f},
    > below $C^* = 1.5$ — the median torrent is over-provisioned, so
    > $\Delta H \approx 0$ and only the flat floor $\alpha$ pays it.
    >
    > The cause is **allocation, not aggregate abundance**, and at
    > $\eta = {reward.eta:.2f}$ {_tilt_note}. Measured in bytes this
    > population can afford only {_copies:.1f} whole copies of the
    > catalogue — fewer than the ~15
    > seeders a torrent needs to reach $x^* = d_{{ref}}/C^*$ — but measured
    > in *count* most of the catalogue sits far below the median size, so
    > the site still averages {_per:.0f} memberships per torrent and holds
    > {_at_floor:.0%} of the catalogue at the $C = 1$ floor. Where the disk
    > actually lands: large and
    > super-large releases are {_cat_share_big:.0%} of the catalogue's bytes
    > and hold {_held_share_big:.0%} of all committed disk, at a median
    > $C_i$ of {_med_big:.2f}. Memberships still look lopsided — the
    > most-seeded tenth of the catalogue absorbs {_share_top:.0%} of them —
    > but that same tenth is only {_share_top_disk:.0%} of the committed
    > bytes: a GiB of disk simply buys more small torrents, so membership
    > counts overstate how concentrated the committed disk is.
    >
    > **What disk supply moves is the tail, not the median.** The
    > available-torrent p90 goes {_p90[0]:.2f} → {_p90[1]:.2f} →
    > {_p90[2]:.2f} and the unavailable fraction
    > {_out[0]:.1%} → {_out[1]:.1%} → {_out[2]:.1%}. A scarce site does not
    > degrade uniformly to a worse $C$; it keeps the crowded part of the
    > catalogue at $C = 1$ and lets the rest fall below $\pi_{{min}}$. The
    > unavailable fraction has to be a first-class alarm alongside the $C_i$
    > histogram.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Is there hunting?

    §8 asks for no sustained oscillation at $\gamma = 4$. Seeder counts are
    tracked for the 48 most popular torrents. Herding would show as a common
    period across torrents, so what matters is not the wiggle of any one series
    but whether the *cross-torrent mean* oscillates.
    """)
    return


@app.cell
def _(np, plt, runs):
    _r = runs[1.0]
    _d = _r.hours / 24.0
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for _k in range(0, _r.seeders_tracked.shape[1], 4):
        _ax[0].plot(_d, _r.seeders_tracked[:, _k], lw=0.6, alpha=0.5)
    _ax[0].plot(_d, _r.seeders_tracked.mean(axis=1), "k", lw=1.6, label="mean")
    _ax[0].set_xlabel("day")
    _ax[0].set_ylabel("seeders")
    _ax[0].set_title("tracked torrents (48 most popular)")
    _ax[0].legend(fontsize=7)

    _tail = _r.tail(days=14)
    _series = _r.seeders_tracked[_tail].mean(axis=1)
    _series = _series - _series.mean()
    if _series.size > 8:
        _spec = np.abs(np.fft.rfft(_series)) ** 2
        _freq = np.fft.rfftfreq(_series.size, d=float(np.diff(_r.hours).mean()) / 24.0)
        _ax[1].semilogy(_freq[1:], _spec[1:] / _spec[1:].sum(), lw=1.0)
        _peak = float(_freq[1:][np.argmax(_spec[1:])])
        _ax[1].axvline(
            _peak,
            color="C3",
            ls="--",
            lw=0.9,
            label=f"peak at {1 / max(_peak, 1e-9):.1f} d",
        )
        _ax[1].legend(fontsize=7)
    _ax[1].set_xlabel("cycles per day")
    _ax[1].set_ylabel("power share")
    _ax[1].set_title("spectrum of the mean seeder count (last 14 d)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Monetary policy: does the $\kappa$ controller settle?

    $\kappa$ moves by at most ±5 %/day on the trailing 7-day mint/burn ratio.
    Two failure modes are visible here: a controller pinned at its clamp for
    the whole run (the initial $\kappa$ was orders of magnitude off), and a
    ratio that refuses to reach 1 no matter how small $\kappa$ gets.
    """)
    return


@app.cell(hide_code=True)
def _(plt, runs, scales):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for _s, _c in zip(scales, ("C0", "C1", "C2"), strict=True):
        _r = runs[_s]
        _d = _r.hours / 24.0
        _ax[0].semilogy(_d, _r.kappa, color=_c, lw=1.2, label=f"×{_s}")
        _ax[1].semilogy(_d, _r.mint / (_r.burn + 1e-9), color=_c, lw=0.9)
        _ax[2].plot(_d, _r.balance_quantiles[:, 1], color=_c, lw=1.2)
        _ax[2].fill_between(
            _d,
            _r.balance_quantiles[:, 0],
            _r.balance_quantiles[:, 2],
            color=_c,
            alpha=0.12,
        )
    _ax[0].set_ylabel(r"$\kappa$")
    _ax[0].set_title(r"$\kappa$ controller")
    _ax[0].legend(fontsize=7)
    _ax[1].axhline(1.0, color="k", ls="--", lw=0.9)
    _ax[1].set_ylabel("mint / burn (hourly)")
    _ax[1].set_title("monetary balance, target 1")
    _ax[2].set_yscale("log")
    _ax[2].set_ylabel("balance (points)")
    _ax[2].set_title("user balances: median, p10–p90")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, runs):
    _r = runs[1.0]
    _a = _r.acceptance(days=7)
    _pinned = float(_r.kappa[-1] / _r.kappa[0])
    mo.md(rf"""
    Over this run $\kappa$ moved by a factor {_pinned:.3g} and the mint/burn
    ratio ended at **{_a["mint_burn"]:.2f}** with
    **{_a["broke_per_day"]:.0f} downloads per day refused for insufficient
    balance**. Those two facts together are the interesting part: the site is
    simultaneously **minting faster than it burns** and leaving users unable to
    pay. Points are being created in the wrong place — the flat floor $\alpha$
    pays in proportion to memberships held, so the large seedboxes that already
    hold everything accumulate the supply, while ordinary users who hold little
    stay broke. A single scalar $\kappa$ cannot fix a distributional problem;
    it can only deflate everybody at once.

    This is the practical argument for keeping $\alpha$ very small, and for
    $I_{{min}}$ being a real subsidy rather than a rounding error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Do the estimators converge on the truth?

    The whole mechanism runs on $\tilde p_v$ and $\tilde b_v$, which are
    deliberately pessimistic. The question is not whether they are unbiased —
    they are not, by design — but whether they *order* nodes correctly and stop
    drifting.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, runs):
    _r = runs[1.0]
    _g = _r.final_contribution
    _pop = _r.population
    _truth = _pop.availability * _pop.up_mbps
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _ax[0].scatter(_truth, _g, s=4, alpha=0.3, lw=0)
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel(r"true $p_v b_v$ (Mbps)")
    _ax[0].set_ylabel(r"credited $g_v = \tilde p \tilde b / \tilde k$")
    _rank = np.corrcoef(np.argsort(np.argsort(_truth)), np.argsort(np.argsort(_g)))[
        0, 1
    ]
    _ax[0].set_title(f"node estimates (Spearman {_rank:.2f})")

    _share_true = np.sort(_truth)[::-1].cumsum() / _truth.sum()
    _share_cred = np.sort(_g)[::-1].cumsum() / _g.sum()
    _x = np.arange(1, _g.size + 1) / _g.size
    _ax[1].plot(_x, _share_true, lw=1.2, label="true capacity")
    _ax[1].plot(_x, _share_cred, lw=1.2, label="credited capacity")
    _ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    _ax[1].set_xlabel("fraction of nodes (largest first)")
    _ax[1].set_ylabel("share of capacity")
    _ax[1].set_title("concentration: who the site depends on")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sensitivity: the three open decisions of §9

    Each variant re-runs the same world with one change. Unlike §2–§5, this
    block does **not** follow the $\eta$ control above: its baseline is
    pinned to the size-neutral $\eta = 1$ so that the two $\eta$ rows always
    bracket it and every other row is read against the same reference.
    Everything else — $\kappa$, $\alpha$, $\gamma$, the world size and the
    seed — still comes from the controls.

    * **$\eta$** — part 2 showed analytically that only $\eta = 1$ is
      size-neutral, so $\eta = 1$ is the baseline; the two rows at the ends of
      the admissible band, $\eta = 0.9$ and $\eta = 1.1$, price a ±10 % tilt,
      measured as the dead fraction by size class.
    * **importance** — pay scaling vs target scaling.
    * **hysteresis** — $C^*_{stay} = 1.1\,C^*$ for incumbents.
    * **pinned uploader** — *not part of the specified mechanism*. It gives
      every torrent one seeder that never leaves, which isolates "the
      controller cannot hold a target" from "the torrent fell to zero seeders
      and §3.3 pays nothing to bring it back".
    * **batched replanning** — the agent rule itself. The baseline is §8 as
      written: every seeder makes at most one drop/join per decision tick. A
      node holding hundreds of torrents therefore needs months of simulated
      time to reshuffle its library, so this variant lets a quarter of the
      nodes make six moves each instead, to separate "the mechanism reaches
      this allocation" from "the mechanism has not finished mixing".
    """)
    return


@app.cell
def _(RewardParams, base, mo, reward, simulate):
    # Pinned to the size-neutral exponent: the eta rows below are meant to
    # bracket eta = 1, so the baseline cannot drift with the slider.
    neutral = RewardParams(**{**reward.__dict__, "eta": 1.0})
    variants = {
        "baseline": (neutral, {}),
        "eta = 0.9": (RewardParams(**{**neutral.__dict__, "eta": 0.9}), {}),
        "eta = 1.1": (RewardParams(**{**neutral.__dict__, "eta": 1.1}), {}),
        "importance -> target": (
            RewardParams(**{**neutral.__dict__, "importance_mode": "target"}),
            {},
        ),
        "hysteresis 1.1": (neutral, {"stay_bonus": 1.1}),
        "pinned uploader": (neutral, {"pin_uploader": True}),
        "alpha = 0": (RewardParams(**{**neutral.__dict__, "alpha": 0.0}), {}),
        "batched replanning": (
            neutral,
            {"decision_share": 0.25, "moves_per_decision": 6},
        ),
    }
    variant_runs = {
        _name: simulate(_rw, base, **_ov) for _name, (_rw, _ov) in variants.items()
    }
    mo.md(f"ran {len(variant_runs)} variants")
    return (variant_runs,)


@app.cell(hide_code=True)
def _(mo, variant_runs):
    _keys = (
        "median_C_all",
        "avail_median_C",
        "avail_p90_C",
        "unavailable_fraction",
        "swing_C",
        "mint_burn",
    )
    _rows = [
        "| variant | " + " | ".join(_keys) + " |",
        "|---" * (len(_keys) + 1) + "|",
    ]
    for _name, _r in variant_runs.items():
        _a = _r.acceptance(days=7)
        _rows.append(
            f"| {_name} | " + " | ".join(f"{_a[_k]:.3g}" for _k in _keys) + " |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(np, plt, variant_runs):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _names = list(variant_runs)
    _classes = variant_runs["baseline"].catalog.class_names
    _width = 0.8 / len(_names)
    for _i, _name in enumerate(_names):
        _r = variant_runs[_name]
        _dead = [
            float(np.mean(~np.isfinite(_r.final_slowdown[_r.catalog.class_idx == _j])))
            for _j in range(len(_classes))
        ]
        _ax[0].bar(np.arange(len(_classes)) + _i * _width, _dead, _width, label=_name)
        _ax[1].plot(_r.hours / 24.0, _r.c_quantiles_all[:, 2], lw=1.1, label=_name)
    _ax[0].set_xticks(np.arange(len(_classes)) + 0.4, _classes)
    _ax[0].set_ylabel("fraction dead at end of run")
    _ax[0].set_title("catalogue survival by size class")
    _ax[0].legend(fontsize=6)
    _ax[1].axhline(1.5, color="k", ls="--", lw=0.9)
    _ax[1].set_yscale("log")
    _ax[1].set_xlabel("day")
    _ax[1].set_ylabel(r"p90 $C_i$ (whole catalogue)")
    _ax[1].set_title("the tail the controller is supposed to fix")
    _ax[1].legend(fontsize=6)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Where do the points end up?

    The mint/burn ratio is an aggregate; it says nothing about *who* holds the
    supply. Since $\alpha$ pays per membership held, the natural hypothesis is
    that the point supply tracks disk. That is measurable rather than
    assertable, so it is measured.
    """)
    return


@app.cell
def _(np, plt, runs):
    _r = runs[1.0]
    _disk = _r.user_disk_budget
    _bal = _r.final_balance
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _ax[0].scatter(_disk, np.maximum(_bal, 1e-1), s=5, alpha=0.35, lw=0)
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel("user disk budget (GiB)")
    _ax[0].set_ylabel("final balance (points)")
    _rank = np.corrcoef(np.argsort(np.argsort(_disk)), np.argsort(np.argsort(_bal)))[
        0, 1
    ]
    _ax[0].set_title(f"balance vs provisioned disk (Spearman {_rank:.2f})")

    _srt = np.sort(_bal)[::-1]
    _share = np.cumsum(_srt) / max(_srt.sum(), 1e-9)
    _x = np.arange(1, _srt.size + 1) / _srt.size
    _top_decile = float(np.interp(0.1, _x, _share))
    _ax[1].plot(_x, _share, lw=1.3, label="points")
    _dsrt = np.sort(_disk)[::-1]
    _ax[1].plot(_x, np.cumsum(_dsrt) / _dsrt.sum(), lw=1.3, ls="--", label="disk")
    _ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    _ax[1].set_xlabel("fraction of users (richest first)")
    _ax[1].set_ylabel("cumulative share")
    _ax[1].set_title(f"top decile holds {_top_decile:.0%} of points")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, np, runs, variant_runs):
    def _unavail_large(name):
        _r = variant_runs[name]
        _m = _r.catalog.class_idx >= 2
        return float(np.mean(~np.isfinite(_r.final_slowdown[_m])))

    def _class_median(name, cls):
        """Median finite ``C_i`` within one size class."""
        _r = variant_runs[name]
        _f = _r.final_slowdown[_r.catalog.class_idx == cls]
        _f = _f[np.isfinite(_f)]
        return float(np.median(_f)) if _f.size else float("nan")

    def _class_medians(name):
        """Median ``C_i`` for small / medium / large, as a string."""
        return " / ".join(f"{_class_median(name, _j):.2f}" for _j in range(3))

    def _small_members(name):
        _r = variant_runs[name]
        return float(_r.final_members[_r.catalog.class_idx == 0].mean())

    # ``_base`` is the pinned eta = 1 reference of §6; ``_sweep`` is the
    # interactive run of §2-§5, which follows the controls. Never mix them
    # inside one claim.
    _base = variant_runs["baseline"].acceptance(days=7)
    _eta_lo = variant_runs["eta = 0.9"].acceptance(days=7)
    _eta_hi = variant_runs["eta = 1.1"].acceptance(days=7)
    _tgt = variant_runs["importance -> target"].acceptance(days=7)
    _hyst = variant_runs["hysteresis 1.1"].acceptance(days=7)
    _pin = variant_runs["pinned uploader"]
    _scarce = runs[0.25].acceptance(days=7)
    _r1 = runs[1.0]
    _sweep = runs[1.0].acceptance(days=7)
    _bal, _disk = _r1.final_balance, _r1.user_disk_budget
    _rank = float(
        np.corrcoef(np.argsort(np.argsort(_disk)), np.argsort(np.argsort(_bal)))[0, 1]
    )
    _top = float(
        np.sort(_bal)[-max(1, _bal.size // 10) :].sum() / max(_bal.sum(), 1e-9)
    )

    mo.md(rf"""
    ## 8. What to take back to the design

    1. **The knee is robust, but saturated at the median.** Disk supply barely
       moves the median $C_i$, which is the property §4 argued for. What it
       does not do is hold the median *at* $C^*$: the median torrent sits at
       {_sweep["median_C_all"]:.2f}, over-provisioned, so $\Delta H \approx 0$
       and the flat floor $\alpha$ is what actually pays most memberships.
       $\alpha$ is not a rounding term, it is the main reward. The share of
       mint coming from $\alpha$ belongs in §8's permanent instrumentation
       list next to the $\Delta H$ histogram.

    2. **Scarcity shows up as lost availability, not as slowdown.** At ×0.25
       disk the available-torrent median is still
       {_scarce["avail_median_C"]:.2f} while
       {_scarce["unavailable_fraction"]:.0%} of the catalogue has dropped
       below $\pi_{{min}}$. Note the metric does not separate the two ways
       that happens — a torrent with no members at all, and one whose
       complete seeders are individually too unreliable for
       $1 - \prod_v (1 - \tilde p_v)$ to clear the threshold. Both read as
       $C_i = \infty$. A dashboard watching only the $C_i$ histogram over
       available torrents would show a perfectly healthy site either way.

    3. **$\pi_{{min}}$ is a cliff with no gradient below it.** When
       $A_i < \pi_{{min}}$ both $H_i$ and $H_i^{{(-u)}}$ are zero, so the
       marginal is zero and nothing pays a seeder to restore availability
       until $\lceil \ln(1-\pi_{{min}})/\ln(1-\tilde p_v)\rceil$ of them
       arrive together. In this over-provisioned run the measured cost is
       modest — large-torrent unavailability
       {_unavail_large("baseline"):.1%} baseline vs
       {_unavail_large("pinned uploader"):.1%} with protected initial holders
       (covering {_pin.pinned_coverage:.0%} of the catalogue) — but it is
       what leaves the ×0.25 column without a *directed* way back: recovery
       depends on $\alpha$-driven random browsing rather than on any reward
       gradient pointing at the torrents that need seeders. Recommend making
       completeness continuous, e.g. paying
       $\min(1, A_i/\pi_{{min}}) \cdot H(C_i)$.

    4. **$\eta = 1$, and the band around it is tight** (§9, first open
       decision). Part 2 derives it in closed form: reward per GiB of disk is
       size-neutral iff $\eta = 1$, which is why it is the default here and
       why §3.6 now starts there rather than at the original $0.7$, with the
       slow job clamped to the band. The sweep prices a ±10 % tilt. Median
       $C_i$ by size class (small / medium / large) is flat only at the
       neutral point — {_class_medians("baseline")} at $\eta = 1$, against
       {_class_medians("eta = 0.9")} at $0.9$ and
       {_class_medians("eta = 1.1")} at $1.1$ — and the available-torrent p90
       is lowest there: {_base["avail_p90_C"]:.2f} against
       {_eta_lo["avail_p90_C"]:.2f} and {_eta_hi["avail_p90_C"]:.2f}. The two
       edges fail differently. At $0.9$ a GiB spent on a large torrent pays
       less than the same GiB on a small one, so seeders crowd the small end
       — {_small_members("eta = 0.9"):.0f} memberships on the average small
       torrent against {_small_members("baseline"):.0f} at $\eta = 1$ — and
       large-torrent unavailability roughly doubles,
       {_unavail_large("baseline"):.1%} → {_unavail_large("eta = 0.9"):.1%}.
       At $1.1$ the tilt reverses: the small end thins to
       {_small_members("eta = 1.1"):.0f} memberships, the site holds fewer
       memberships overall, and the middle of the catalogue drifts off the
       floor — medium-class median $C_i$
       {_class_median("eta = 1.1", 1):.2f} against
       {_class_median("baseline", 1):.2f} — while
       unavailability barely moves
       ({_unavail_large("baseline"):.1%} → {_unavail_large("eta = 1.1"):.1%}
       for large torrents), so that edge is paid in slowdown rather than in
       lost availability. Ten per cent of exponent is already
       this visible, so the exponent is not a tuning knob — if a
       small-torrent subsidy is wanted it should be an explicit $w_i$, not a
       silent side effect of the size exponent.

    5. **Importance should move the target, not the pay** (§9, second open
       decision). Target scaling reaches an available-torrent p90 of
       {_tgt["avail_p90_C"]:.2f} against {_base["avail_p90_C"]:.2f} for pay
       scaling: paying $4\times$ for a $z = 2$ torrent keeps recruiting
       seeders long after its health has saturated, whereas relocating its
       knee stops once the better health has actually been bought.

    6. **Hysteresis is not needed yet** (§9, third open decision). There is no
       oscillation to damp — detrended amplitude
       {_base["oscillation"]:.1%} of the mean seeder count — and
       $C^*_{{stay}} = 1.1 C^*$ moves the available-torrent p90 only from
       {_base["avail_p90_C"]:.2f} to {_hyst["avail_p90_C"]:.2f}. Keep it
       shelved, as §9 proposes.

    7. **A single $\kappa$ cannot fix a distributional problem.** The site
       mints faster than it burns (ratio {_sweep["mint_burn"]:.1f}) *and*
       refuses {_sweep["broke_per_day"]:.0f} downloads a day for empty
       balances. §7's panel shows why rather than assuming it: final balance
       ranks with user disk at Spearman {_rank:.2f}, and the richest decile of
       users ends up holding {_top:.0%} of all points. $\alpha$ pays per
       membership held, and memberships are bought with disk. Deflating
       $\kappa$ moves everyone down together; only a smaller $\alpha$, a
       larger $I_{{min}}$, or a per-user cap redistributes.

    8. **Mind the mixing time.** Under §8's literal rule — one drop/join per
       seeder per tick — a node holding hundreds of torrents needs months of
       simulated time to re-plan its library, so short runs measure the
       initial allocation as much as the equilibrium. The *batched
       replanning* row is the same mechanism allowed to mix faster; where it
       disagrees with the baseline, the baseline has not converged.

    9. **Scope caveat.** This is a deliberately small scenario
       ({_r1.catalog.n_torrents:,} torrents, {_r1.population.n_users:,} users)
       chosen so the sweep and {len(variant_runs)} variants run in about a
       minute. Part 1's default world is larger, and the catalogue-to-node
       ratio strongly controls both the unavailable fraction and whether $C$
       reaches the floor. The qualitative findings above should be re-checked
       at the larger size before any of them is treated as a production
       number.
    """)
    return


if __name__ == "__main__":
    app.run()
