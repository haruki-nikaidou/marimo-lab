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
       $k$ ways, each leecher takes the lesser of its downlink and its share,
       and a download that finishes mid-hour moves only what it had left;
    3. the §3.2 estimators see **only what a tracker would see** — online/offline
       hours and the observed upload rate on *qualifying* intervals (those with
       at least one leecher) — never the ground truth;
    4. $x_i$, $A_i$, $C_i$, $H_i$ and every leave-one-out $H_i^{(-u)}$ are
       recomputed and rewards accrue to the users online on each torrent;
    5. downloads start (burning $\phi_i \cdot \text{size}_i$ points, refused if
       the balance cannot cover it or if no complete copy exists anywhere) and
       finish (becoming seeds);
    6. four times a day every seeder re-plans, exactly as §8 specifies: it
       values each holding and a sample of candidates in **points per hour per
       GiB of disk** — the quantity a finite disk actually trades off — and
       makes at most **one** move, taking free space if it has any and
       otherwise dropping its lowest-value holding for a better one;
    7. once a day the population priors refresh. $\kappa$ is a constant:
       points are a gate on download, not a currency, so there is no
       monetary policy to run.

    **The rule under test is the design as it now stands**, not §3.3 as
    first written:

    * the capacity floor $\max(1, d_{ref}/x_i)$ is gone — the site is
      assumed never to be short of disk, and part 4 showed the floor turns
      surplus disk into turnover rather than into slack;
    * $\pi_{min}$ stays, applied to **copy completeness**. An uploader's copy
      is 100 % complete whatever their uptime, and every seeder in this
      engine holds a complete copy, so the threshold is retained but
      dormant here: it guards a swarm of partial seeders, which are not
      modelled, and validating it needs a partial-seeder engine (out of
      scope). Uptime is priced once, through $\tilde p_v$ inside $g_v$. §4
      shows what the alternative reading — $\pi_{min}$ on the *online*
      probability $A_i = 1 - \prod_v (1 - \tilde p_v)$ — does to the first
      two seeders, and why.

    **The acceptance criterion under test** (§8): *the $C_i$ distribution
    collapses toward $C^*$ regardless of total disk supply, with no sustained
    oscillation at $\gamma = 4$.*

    > [!important] What "unavailable" means here
    > Every membership is a **complete** seeder. The **unavailable fraction**
    > reported below is the share of the catalogue whose online-copy
    > probability $A_i$ has fallen below $\pi_{min}$ — the tracker's
    > pessimistic estimate says a complete copy is not reliably *online*. The
    > mechanism under test does not act on it; it is reported as a
    > diagnostic, and every $C_i$ quantile is taken over the whole catalogue
    > so that thin torrents cannot fall out of the picture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Controls

    A run is about 4 s; the disk sweep runs three of them and is memoised.
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
            mo.hstack([ui_alpha, ui_gamma], widths="equal"),
            mo.hstack([ui_eta, ui_mode, ui_stay, ui_seed], widths="equal"),
        ]
    )
    return (
        ui_alpha,
        ui_days,
        ui_eta,
        ui_gamma,
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
    ui_mode,
    ui_seed,
    ui_stay,
    ui_torrents,
    ui_users,
):
    reward = RewardParams(
        kappa=0.05,
        alpha=float(ui_alpha.value),
        gamma=float(ui_gamma.value),
        eta=float(ui_eta.value),
        importance_mode=ui_mode.value,
        floor=False,
        completeness="possession",
    )
    base = SimParams(
        days=int(ui_days.value),
        n_torrents=int(ui_torrents.value),
        n_users=int(ui_users.value),
        stay_bonus=float(ui_stay.value),
        kappa_control=False,
        seed=int(ui_seed.value),
    )
    pi_min = float(reward.pi_min)
    return base, pi_min, reward


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

    def with_fields(reward, **fields):
        """A copy of ``reward`` with some fields replaced."""
        return replace(reward, **fields)

    return simulate, sweep, with_fields


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The disk-supply sweep

    Total disk supply is scaled by 1/4, 1 and 4 around the population of
    part 1. The acceptance criterion asks that the $C_i$ distribution land in
    the same place in all three. Two experience columns sit next to the
    allocation: the p90 completion time of demand downloads relative to time
    at the reference rate, weighted by bytes, and the backlog of demand
    downloads stuck for more than a day, in days of demand.
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
        ("median_C_all", "median $C_i$", "{:.2f}"),
        ("p90_C_all", "p90 $C_i$", "{:.2f}"),
        ("avail_p90_C", "p90 $C_i$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"online-copy prob. $< \pi_{min}$", "{:.1%}"),
        ("bytes_p90", "p90 time, by bytes", "{:.2f}"),
        ("demand_backlog_days", "stuck demand (days)", "{:.2f}"),
        ("oscillation", "oscillation", "{:.1%}"),
    )
    _head = (
        "| disk supply | " + " | ".join(_k[1] for _k in _keys) + " | seeders/torrent |"
    )
    _sep = "|---" * (len(_keys) + 2) + "|"
    _rows = []
    for _s in scales:
        _a = runs[_s].acceptance(days=7)
        _per = runs[_s].members_total[-1] / runs[_s].catalog.n_torrents
        _rows.append(
            f"| ×{_s} | "
            + " | ".join(_k[2].format(_a[_k[0]]) for _k in _keys)
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
        _ax[2].plot(_d, _r.download_slowdown_by_bytes[:, 2], color=_c, lw=1.2)
    _ax[0].axhline(1.5, color="k", ls="--", lw=0.9)
    _ax[0].set_yscale("log")
    _ax[0].set_ylabel(r"$C_i$ (whole catalogue)")
    _ax[0].set_title(r"median (solid) and p90 (dotted); $C^*$ dashed")
    _ax[0].legend(fontsize=7)
    _ax[1].set_ylabel("unavailable fraction")
    _ax[1].set_title(r"torrents with $A_i < \pi_{min}$")
    _ax[1].legend(fontsize=7)
    _ax[2].axhline(1.0, color="k", ls=":", lw=0.9)
    _ax[2].set_ylabel("completion time / time at $d_{ref}$")
    _ax[2].set_title("demand p90 by bytes")
    for _a in _ax:
        _a.set_xlabel("day")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(np, pi_min, plt, runs, scales):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for _s, _c in zip(scales, ("C0", "C1", "C2"), strict=True):
        _r = runs[_s]
        _avail = _r.final_completeness >= pi_min
        _ax[0].hist(
            np.clip(_r.final_slowdown[_avail], 0.2, 20.0),
            bins=np.geomspace(0.2, 20.0, 50),
            histtype="step",
            lw=1.3,
            color=_c,
            label=f"×{_s} ({1 - _avail.mean():.0%} unavailable)",
        )
    _ax[0].axvline(1.5, color="k", ls="--", lw=0.9)
    _ax[0].axvline(1.0, color="C7", ls=":", lw=0.9)
    _ax[0].set_xscale("log")
    _ax[0].set_xlabel(r"$C_i$ at the end of the run (available torrents)")
    _ax[0].set_ylabel("torrents")
    _ax[0].set_title(r"final distribution, $C^*$ dashed, $x = d_{ref}$ dotted")
    _ax[0].legend(fontsize=7)

    _r = runs[1.0]
    _names = _r.catalog.class_names
    for _j, _c in enumerate(("C0", "C1", "C2", "C3")):
        _m = _r.catalog.class_idx == _j
        _ax[1].bar(
            _j,
            float(np.mean(_r.final_completeness[_m] < pi_min)),
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
    _exp = [_acc[_s]["bytes_p90"] for _s in scales]
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
    _past = float(np.mean(_r1.final_slowdown <= 1.0 + 1e-9))
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
    > [!note] One clause passes, one fails by design, one fails for a reason
    > worth keeping
    > **No sustained oscillation: yes.** Measured on the cross-torrent mean
    > seeder count — the series in which herding would actually show — the
    > detrended peak-to-peak amplitude is {_osc:.1%} of its level at
    > $\gamma = 4$. (The $C$ quantiles are a poor oscillation proxy: they also
    > move with the unavailable mass.)
    >
    > **Insensitivity to disk supply: no, and that is the design.** Over a
    > 16× range the whole-catalogue median $C_i$ moves
    > {_med[0]:.2f} → {_med[1]:.2f} → {_med[2]:.2f} (spread {_spread:.2f}).
    > With the capacity floor gone there is nothing to stop surplus disk
    > becoming surplus capacity, so the median tracks how much disk the
    > community owns. Part 4 priced that: the floor held the *estimate* at
    > $x_i = d_{{ref}}$ and paid for it with a worse tail on what downloads
    > actually experienced, because a swarm sized for exactly one leecher
    > has no slack for the second. Here the p90 completion time by bytes is
    > {_exp[0]:.2f} → {_exp[1]:.2f} → {_exp[2]:.2f}. Redundancy is what a
    > site with spare disk is supposed to buy.
    >
    > **Collapse *onto* $C^*$: no.** The median settles at {_med[1]:.2f},
    > below $C^* = 1.5$ — the median torrent is over-provisioned, so
    > $\Delta H \approx 0$ and only the flat floor $\alpha$ pays it. The
    > cause is **allocation, not aggregate abundance**, and at
    > $\eta = {reward.eta:.2f}$ {_tilt_note}. Measured in bytes this
    > population can afford only {_copies:.1f} whole copies of the
    > catalogue — fewer than the ~15 seeders a torrent needs to reach
    > $x^* = d_{{ref}}/C^*$ — but measured in *count* most of the catalogue
    > sits far below the median size, so the site still averages
    > {_per:.0f} memberships per torrent and holds {_past:.0%} of the
    > catalogue past $x = d_{{ref}}$. Where the disk actually lands: large
    > and super-large releases are {_cat_share_big:.0%} of the catalogue's
    > bytes and hold {_held_share_big:.0%} of all committed disk, at a
    > median $C_i$ of {_med_big:.2f}. Memberships still look lopsided — the
    > most-seeded tenth of the catalogue absorbs {_share_top:.0%} of them —
    > but that same tenth is only {_share_top_disk:.0%} of the committed
    > bytes: a GiB of disk simply buys more small torrents.
    >
    > **What disk supply moves is the tail, not the median.** The
    > available-torrent p90 goes {_p90[0]:.2f} → {_p90[1]:.2f} →
    > {_p90[2]:.2f} and the unavailable fraction
    > {_out[0]:.1%} → {_out[1]:.1%} → {_out[2]:.1%}. A scarce site does not
    > degrade uniformly to a worse $C$; it keeps the crowded part of the
    > catalogue well past $d_{{ref}}$ and lets the rest thin out. The
    > unavailable fraction has to be a first-class alarm alongside the $C_i$
    > histogram, even though the mechanism no longer acts on it.
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
    ## 4. The first two seeders: who pays them, and who does not

    Part 2 found that under §3.3 as written the first seeder onto a torrent
    is paid the bare floor $\alpha$ and the seeder who completes the quorum
    is paid a spike. **The floor $\max(1,\cdot)$ has nothing to do with it.**
    The cause is what $\pi_{min}$ is applied to. The handoff's

    $$A_i = 1 - \prod_{v \in F_i} (1 - \tilde p_v)$$

    is the probability a complete copy is *online*, not whether one exists.
    A fresh uploader holds a 100 % complete copy and still scores
    $A_i = \tilde p_v$ — with the pessimistic prior of §3.2 that is about
    $0.55$, well under $\pi_{min} = 0.9$ — so §3.3 sets $C_i = \infty$,
    $H_i = 0$, and the uploader earns $\alpha$ for their own release. It
    takes $\lceil \ln(1-\pi_{min}) / \ln(1-\tilde p_v) \rceil$ seeders
    arriving together before anyone is paid for health, and the one who
    tips $A_i$ over the line is paid all of it, because removing them drops
    $A_i$ back under the threshold and $H_i^{(-u)} = 0$.

    Three readings of the same $\pi_{min}$, all keeping it:

    | reading | $H_i$ |
    |---|---|
    | `online_step` | $H(C_i)$ if $A_i \ge \pi_{min}$, else $0$ — §3.3, a hard quorum |
    | `online_cap` | $\min(H(C_i),\ \min(1, A_i/\pi_{min}))$ — online copy as a cap |
    | `possession` | $H(C_i)$; $\pi_{min}$ on copy completeness, which is $1$ here |

    For identical seeders `online_cap` is a minimum of two concave functions
    of the seeder count, so in that probe its marginal is decreasing — no
    quorum, no spike, no region where losing one seeder lowers the next
    one's incentive. (Nothing is claimed for heterogeneous leave-one-out
    memberships; the engine tests only the identical-seeder ordering.) It
    coincides with `possession` except where a swarm's health would exceed
    the probability its copy is online at all, i.e. a few fast but rarely-on
    seeders. (Part 2's proposal, the *product*
    $\min(1, A_i/\pi_{min}) \cdot H(C_i)$, removes the spike but still pays
    the second seeder more than the first, so it is not shown.)

    Both panels are exact arithmetic: identical seeders with the same
    $\tilde p_v$ and the same $g_v$ join one at a time. On the left a
    typical seeder; on the right a fast, rarely-online one, where the cap
    binds. The table below is the simulated site.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, reward, with_fields):
    _d_ref = 300.0
    _x_star = _d_ref / reward.c_star
    _n = np.arange(0, 11)
    _target = np.full(_n.shape, reward.c_star)
    _modes = (
        ("online_step", "k", "-"),
        ("online_cap", "C0", "-"),
        ("possession", "C1", "--"),
    )
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    for _a, (_p, _gk) in zip(_ax, ((0.55, 4.0), (0.30, 1.5)), strict=True):
        _avail = 1.0 - (1.0 - _p) ** _n
        _g = _x_star / _gk
        for _mode, _col, _ls in _modes:
            _rw = with_fields(reward, floor=False, completeness=_mode)
            _h = _rw.torrent_health(_n * _g, _avail, _d_ref, _target)
            _a.plot(
                _n[1:],
                np.diff(_h),
                marker="o",
                color=_col,
                ls=_ls,
                ms=3,
                lw=1.2,
                label=_mode,
            )
        _a.axhline(0.0, color="C7", lw=0.6)
        _a.set_xlabel("seeders $n$ (the $n$-th joiner's marginal)")
        _a.set_title(rf"$\tilde p_v = {_p:.2f}$, $g_v = x^*/{_gk:g}$")
    _ax[0].set_ylabel(r"$\Delta H$ paid to the $n$-th seeder")
    _ax[0].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(base, reward, simulate, with_fields):
    completeness_modes = ("possession", "online_cap", "online_step")
    completeness_runs = {
        (_mode, _s): simulate(
            with_fields(reward, completeness=_mode), base, disk_scale=_s
        )
        for _mode in completeness_modes
        for _s in (0.25, 1.0)
    }
    return completeness_modes, completeness_runs


@app.cell(hide_code=True)
def _(completeness_modes, completeness_runs, mo, np, pi_min):
    def _lone_unpaid(r):
        """Among torrents held by exactly one seeder, the share whose online
        probability is under ``pi_min`` — the uploader-alone case."""
        lone = r.final_members == 1
        return (
            float(np.mean(r.final_completeness[lone] < pi_min)) if lone.any() else 0.0
        )

    def _big_dead(r):
        _m = r.catalog.class_idx >= 2
        return float(np.mean(r.final_completeness[_m] < pi_min))

    _cols = (
        ("median_C_all", "median $C_i$", "{:.2f}"),
        ("avail_p90_C", "p90 $C_i$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"online-copy prob. $< \pi_{min}$", "{:.1%}"),
        ("bytes_p90", "p90 time, by bytes", "{:.2f}"),
        ("demand_backlog_days", "stuck demand (days)", "{:.2f}"),
    )
    _rows = [
        "| completeness | disk | "
        + " | ".join(_c[1] for _c in _cols)
        + " | large+super dead | lone-seeder torrents with $A_i<\\pi_{min}$ |",
        "|---" * (len(_cols) + 4) + "|",
    ]
    for _mode in completeness_modes:
        for _s in (0.25, 1.0):
            _r = completeness_runs[_mode, _s]
            _a = _r.acceptance(days=7)
            _rows.append(
                f"| `{_mode}` | ×{_s} | "
                + " | ".join(_c[2].format(_a[_c[0]]) for _c in _cols)
                + f" | {_big_dead(_r):.0%} | {_lone_unpaid(_r):.1%} |"
            )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(completeness_runs, mo, np, pi_min):
    def _lone(r):
        lone = r.final_members == 1
        return (
            float(np.mean(r.final_completeness[lone] < pi_min)) if lone.any() else 0.0
        )

    def _big_dead(r):
        _m = r.catalog.class_idx >= 2
        return float(np.mean(r.final_completeness[_m] < pi_min))

    _pos0, _cap0, _step0 = (
        completeness_runs[_m, 0.25]
        for _m in ("possession", "online_cap", "online_step")
    )
    _pos1, _step1 = (
        completeness_runs["possession", 1.0],
        completeness_runs["online_step", 1.0],
    )
    _a_pos0, _a_cap0, _a_step0 = (
        _r.acceptance(days=7) for _r in (_pos0, _cap0, _step0)
    )
    _a_pos1, _a_step1 = _pos1.acceptance(days=7), _step1.acceptance(days=7)
    mo.md(rf"""
    At ×1 disk the three readings are indistinguishable — median $C_i$
    {_a_pos1["median_C_all"]:.2f} against {_a_step1["median_C_all"]:.2f},
    unavailability {_a_pos1["unavailable_fraction"]:.1%} against
    {_a_step1["unavailable_fraction"]:.1%} — because on a site with enough
    disk almost no torrent is ever down to one unreliable holder. The
    quorum only matters where torrents are thin. At ×0.25 the step leaves
    {_lone(_step0):.0%} of the torrents held by a single seeder under
    $\pi_{{min}}$, that seeder paid nothing for health, and the large and
    super-large classes lose {_big_dead(_step0):.0%} of their torrents to
    $A_i < \pi_{{min}}$ against {_big_dead(_pos0):.0%} under `possession`
    and {_big_dead(_cap0):.0%} under `online_cap`; whole-catalogue
    unavailability {_a_step0["unavailable_fraction"]:.1%} →
    {_a_pos0["unavailable_fraction"]:.1%} → {_a_cap0["unavailable_fraction"]:.1%}.
    The other side of the same trade: paying lone seeders spreads the
    scarce disk thinner, so the available-torrent p90 goes
    {_a_step0["avail_p90_C"]:.2f} (step) → {_a_pos0["avail_p90_C"]:.2f}
    (possession) → {_a_cap0["avail_p90_C"]:.2f} (cap) — more torrents alive,
    each held by fewer seeders. The demand side sees the p90 by bytes at
    {_a_step0["bytes_p90"]:.2f} / {_a_pos0["bytes_p90"]:.2f} /
    {_a_cap0["bytes_p90"]:.2f}.

    So: the quorum step is *entirely* $\pi_{{min}}$ applied to online
    probability, the fix is to apply it to copy completeness — where an
    uploader is complete by definition — and if the online probability is
    still to enter the score, `online_cap` does it without a step and
    without a region of increasing marginal. On a site that is not short of
    disk the choice between `possession` and `online_cap` is invisible; it
    decides only how a scarce site distributes its thin torrents.
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
    ## 6. Sensitivity: the open decisions of §9

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
      and nothing can bring it back".
    * **batched replanning** — the agent rule itself. The baseline is §8 as
      written: every seeder makes at most one drop/join per decision tick. A
      node holding hundreds of torrents therefore needs months of simulated
      time to reshuffle its library, so this variant lets a quarter of the
      nodes make six moves each instead, to separate "the mechanism reaches
      this allocation" from "the mechanism has not finished mixing".
    * **with the floor** — §3.3's $\max(1, d_{ref}/x_i)$ put back, for
      reference against part 4.
    """)
    return


@app.cell
def _(base, mo, reward, simulate, with_fields):
    # Pinned to the size-neutral exponent: the eta rows below are meant to
    # bracket eta = 1, so the baseline cannot drift with the slider.
    neutral = with_fields(reward, eta=1.0)
    variants = {
        "baseline": (neutral, {}),
        "eta = 0.9": (with_fields(neutral, eta=0.9), {}),
        "eta = 1.1": (with_fields(neutral, eta=1.1), {}),
        "importance -> target": (with_fields(neutral, importance_mode="target"), {}),
        "hysteresis 1.1": (neutral, {"stay_bonus": 1.1}),
        "pinned uploader": (neutral, {"pin_uploader": True}),
        "alpha = 0": (with_fields(neutral, alpha=0.0), {}),
        "batched replanning": (
            neutral,
            {"decision_share": 0.25, "moves_per_decision": 6},
        ),
        "with the floor": (with_fields(neutral, floor=True), {}),
    }
    variant_runs = {
        _name: simulate(_rw, base, **_ov) for _name, (_rw, _ov) in variants.items()
    }
    mo.md(f"ran {len(variant_runs)} variants")
    return (variant_runs,)


@app.cell(hide_code=True)
def _(mo, variant_runs):
    _keys = (
        ("median_C_all", "median $C_i$", "{:.2f}"),
        ("avail_median_C", "median $C_i$ (avail.)", "{:.2f}"),
        ("avail_p90_C", "p90 $C_i$ (avail.)", "{:.2f}"),
        ("unavailable_fraction", r"online-copy prob. $< \pi_{min}$", "{:.1%}"),
        ("bytes_p90", "p90 time, by bytes", "{:.2f}"),
        ("oscillation", "oscillation", "{:.1%}"),
    )
    _rows = [
        "| variant | " + " | ".join(_k[1] for _k in _keys) + " |",
        "|---" * (len(_keys) + 1) + "|",
    ]
    for _name, _r in variant_runs.items():
        _a = _r.acceptance(days=7)
        _rows.append(
            f"| {_name} | " + " | ".join(_k[2].format(_a[_k[0]]) for _k in _keys) + " |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(np, pi_min, plt, variant_runs):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _names = list(variant_runs)
    _classes = variant_runs["baseline"].catalog.class_names
    _width = 0.8 / len(_names)
    for _i, _name in enumerate(_names):
        _r = variant_runs[_name]
        _dead = [
            float(np.mean(_r.final_completeness[_r.catalog.class_idx == _j] < pi_min))
            for _j in range(len(_classes))
        ]
        _ax[0].bar(np.arange(len(_classes)) + _i * _width, _dead, _width, label=_name)
        _ax[1].plot(_r.hours / 24.0, _r.c_quantiles_all[:, 2], lw=1.1, label=_name)
    _ax[0].set_xticks(np.arange(len(_classes)) + 0.4, _classes)
    _ax[0].set_ylabel(r"fraction with $A_i < \pi_{min}$ at end of run")
    _ax[0].set_title("catalogue availability by size class")
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
def _(completeness_runs, mo, np, pi_min, runs, variant_runs):
    def _unavail_large(name):
        _r = variant_runs[name]
        _m = _r.catalog.class_idx >= 2
        return float(np.mean(_r.final_completeness[_m] < pi_min))

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

    def _big_dead(r):
        _m = r.catalog.class_idx >= 2
        return float(np.mean(r.final_completeness[_m] < pi_min))

    # ``_base`` is the pinned eta = 1 reference of §6; ``_sweep`` is the
    # interactive run of §2-§5, which follows the controls. Never mix them
    # inside one claim.
    _base = variant_runs["baseline"].acceptance(days=7)
    _eta_lo = variant_runs["eta = 0.9"].acceptance(days=7)
    _eta_hi = variant_runs["eta = 1.1"].acceptance(days=7)
    _tgt = variant_runs["importance -> target"].acceptance(days=7)
    _hyst = variant_runs["hysteresis 1.1"].acceptance(days=7)
    _floor = variant_runs["with the floor"].acceptance(days=7)
    _pin = variant_runs["pinned uploader"]
    _scarce = runs[0.25].acceptance(days=7)
    _r1 = runs[1.0]
    _sweep = runs[1.0].acceptance(days=7)
    _step0 = completeness_runs["online_step", 0.25]
    _pos0 = completeness_runs["possession", 0.25]

    mo.md(rf"""
    ## 7. What to take back to the design

    1. **The median tracks disk, and that is now intended.** Without the
       capacity floor the median $C_i$ at ×1 disk is
       {_sweep["median_C_all"]:.2f} and falls as disk grows; the *with the
       floor* row holds it at {_floor["median_C_all"]:.2f} and pays with a
       p90 completion time by bytes of {_floor["bytes_p90"]:.2f} against
       {_base["bytes_p90"]:.2f}. §8's "regardless of disk supply" clause
       should be rewritten as a statement about the *tail* — the
       available-torrent p90 and the unavailable fraction — not the median.

    2. **Scarcity shows up as lost availability, not as slowdown.** At ×0.25
       disk the available-torrent median is still
       {_scarce["avail_median_C"]:.2f} while
       {_scarce["unavailable_fraction"]:.0%} of the catalogue has an
       online-copy probability under $\pi_{{min}}$. A dashboard watching only
       the $C_i$ histogram over available torrents would show a healthy site.

    3. **The quorum step was $\pi_{{min}}$ on the wrong quantity.** §3.3 as
       written thresholds the *online* probability $A_i$, which a lone
       uploader with a pessimistic $\tilde p_v$ cannot clear alone, so the
       uploader is paid $\alpha$ for their own release and the seeder who
       completes the quorum is paid a spike. Apply $\pi_{{min}}$ to copy
       completeness — an uploader is complete by definition — and price
       uptime once, through $\tilde p_v$ in $g_v$. If the online probability
       must still enter the score, use the cap
       $\min(H(C_i), \min(1, A_i/\pi_{{min}}))$, whose marginal is decreasing
       in the identical-seeder probe. On the scarce site the step leaves the large and
       super-large classes at {_big_dead(_step0):.0%} unavailable against
       {_big_dead(_pos0):.0%} for possession; at ×1 the readings coincide.
       Recommend rewriting §3.3's $A_i$ accordingly and moving the
       online-probability cap to §9 as an option for scarce sites.

    4. **$\eta = 1$, and the band around it is tight** (§9, first open
       decision). Part 2 derives it in closed form: reward per GiB of disk is
       size-neutral iff $\eta = 1$. Median $C_i$ by size class (small /
       medium / large) is flat only at the neutral point —
       {_class_medians("baseline")} at $\eta = 1$, against
       {_class_medians("eta = 0.9")} at $0.9$ and
       {_class_medians("eta = 1.1")} at $1.1$ — and the available-torrent p90
       is {_base["avail_p90_C"]:.2f} there against
       {_eta_lo["avail_p90_C"]:.2f} and {_eta_hi["avail_p90_C"]:.2f}. At
       $0.9$ seeders crowd the small end —
       {_small_members("eta = 0.9"):.0f} memberships on the average small
       torrent against {_small_members("baseline"):.0f} at $\eta = 1$ — and
       large-torrent unavailability goes
       {_unavail_large("baseline"):.1%} → {_unavail_large("eta = 0.9"):.1%}.
       At $1.1$ the small end thins to {_small_members("eta = 1.1"):.0f}
       memberships and the medium-class median $C_i$ moves
       {_class_median("baseline", 1):.2f} → {_class_median("eta = 1.1", 1):.2f}.
       Ten per cent of exponent is already this visible, so the exponent is
       not a tuning knob — a small-torrent subsidy should be an explicit
       $w_i$.

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
       shelved.

    7. **Mind the mixing time.** Under §8's literal rule — one drop/join per
       seeder per tick — a node holding hundreds of torrents needs months of
       simulated time to re-plan its library, so short runs measure the
       initial allocation as much as the equilibrium. The *batched
       replanning* row is the same mechanism allowed to mix faster; where it
       disagrees with the baseline, the baseline has not converged.

    8. **Scope caveat.** This is a deliberately small scenario
       ({_r1.catalog.n_torrents:,} torrents, {_r1.population.n_users:,} users)
       chosen so the sweep and {len(variant_runs)} variants run in about a
       minute, on one seed. Part 1's default world is larger, and the
       catalogue-to-node ratio strongly controls both the unavailable
       fraction and where $C$ settles. The qualitative findings above should
       be re-checked at the larger size before any of them is treated as a
       production number. The pinned-uploader row (covering
       {_pin.pinned_coverage:.0%} of the catalogue) is the bound on what any
       completeness rule can recover: with possession completeness the gap
       to it is {_unavail_large("baseline"):.1%} →
       {_unavail_large("pinned uploader"):.1%} large-class unavailability.
    """)
    return


if __name__ == "__main__":
    app.run()
