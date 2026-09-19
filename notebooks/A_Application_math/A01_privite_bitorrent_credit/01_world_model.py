import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    from marimo_lab.ptcredit import world as W

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9.0, "axes.grid": True, "grid.alpha": 0.3}
    )
    return W, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point system, part 1 — the world the mechanism runs in

    Before anything can be said about rewards, the exogenous draws have to be
    pinned down and checked. Two of them are specified for this study:

    **Torrent size**, a four-component mixture, each component truncated to a
    support (truncation is by inverse CDF, i.e. the *conditional* law — no atom
    is piled on the bounds):

    | class | law | $\sigma$ | support (GiB) |
    |---|---|---|---|
    | small | $\mathrm{Lognormal}(0,\,\sigma^2{=}0.5)$ | $0.707$ | $[0.05,\,20]$ |
    | medium | $\mathcal{N}(12,\,\sigma^2{=}10)$ | $3.162$ | $[0.5,\,64]$ |
    | large | $\mathrm{Lognormal}(4.5,\,\sigma^2{=}3)$ | $1.732$ | $[20,\,512]$ |
    | super-large | log-uniform | — | $[512,\,2048]$ |

    counts split 30 % / 40 % / 30 % over the first three, with exactly three
    super-large torrents.

    **Node bandwidth**, a 2-D truncated lognormal: median 300 Mbps down and
    50 Mbps up, capped at 10 Gbps down and 2 Gbps up.

    Three details matter and are handled rather than assumed away.

    1. **The second size parameter is a variance, not a standard deviation.**
       scipy wants a scale, so `SizeClass` stores `variance` and takes the
       square root itself. The distinction is not cosmetic: read as a standard
       deviation, `Lognormal(4.5, 3)` would put its 97.5th percentile at
       32 TiB instead of 2.7 TiB, and `Normal(12, 10)` would put 11.5 % of the
       medium class below zero GiB instead of 0.007 %.
    2. The stated bandwidth medians are medians **of the truncated law**. With
       a correlated pair truncated to a rectangle the marginal of one
       coordinate is *not* a 1-D truncated normal, so the two log-means are
       solved jointly against rectangle probabilities of the standard
       bivariate normal (`calibrate_bandwidth`); the *calibrated law* has
       exactly the stated medians — CDF residual below 1e-6 — and a finite
       sample reproduces them up to ordinary sampling error.
    3. Truncation still does real work on the **large** class: with
       $\sigma = 1.732$ its untruncated 97.5th percentile is about 2.7 TiB,
       which would overlap and overshoot the super-large class, so it is
       capped at 512 GiB where super-large begins. The panel below reports
       what fraction of each stated law survives its support.

    Everything else here — uptime, disk, popularity, importance, charge
    multiplier — is *not* specified by the handoff and is a modelling choice,
    flagged as such where it appears.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Controls

    The bandwidth shape is not pinned down by the brief beyond its medians and
    caps, so its dispersion and coupling are exposed here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ui_sigma_down = mo.ui.slider(
        start=0.3,
        stop=1.8,
        step=0.05,
        value=0.9,
        label=r"$\sigma_{\downarrow}$ (log sd, downlink)",
        show_value=True,
    )
    ui_sigma_up = mo.ui.slider(
        start=0.3,
        stop=2.0,
        step=0.05,
        value=1.2,
        label=r"$\sigma_{\uparrow}$ (log sd, uplink)",
        show_value=True,
    )
    ui_rho = mo.ui.slider(
        start=0.0,
        stop=0.98,
        step=0.02,
        value=0.75,
        label=r"$\rho$ (log corr, down/up)",
        show_value=True,
    )
    ui_n_torrents = mo.ui.dropdown(
        options={"2k": 2000, "8k": 8000, "50k": 50_000},
        value="8k",
        label="catalogue size",
    )
    ui_n_users = mo.ui.dropdown(
        options={"400": 400, "2k": 2000, "20k": 20_000},
        value="2k",
        label="users",
    )
    ui_seed = mo.ui.number(start=0, stop=9999, step=1, value=0, label="seed")
    mo.vstack(
        [
            mo.hstack([ui_sigma_down, ui_sigma_up, ui_rho], widths="equal"),
            mo.hstack([ui_n_torrents, ui_n_users, ui_seed], widths="equal"),
        ]
    )
    return (
        ui_n_torrents,
        ui_n_users,
        ui_rho,
        ui_seed,
        ui_sigma_down,
        ui_sigma_up,
    )


@app.cell
def _(
    W,
    np,
    ui_n_torrents,
    ui_n_users,
    ui_rho,
    ui_seed,
    ui_sigma_down,
    ui_sigma_up,
):
    bw = W.BandwidthModel(
        sigma_down=float(ui_sigma_down.value),
        sigma_up=float(ui_sigma_up.value),
        rho=float(ui_rho.value),
    )
    rng = np.random.default_rng(int(ui_seed.value))
    catalog = W.sample_catalog(int(ui_n_torrents.value), rng)
    population = W.sample_population(int(ui_n_users.value), rng, bandwidth=bw)
    mu_down, mu_up = W.calibrate_bandwidth(bw)
    return bw, catalog, population, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Torrent sizes

    The mixture is drawn by class, then shuffled. `retained` is the probability
    mass the *stated* law already places inside the support: where it is far
    below 1, the truncation bounds, not the stated parameters, are setting the
    shape.
    """)
    return


@app.cell(hide_code=True)
def _(W, catalog, mo, np):
    _rows = []
    for _j, _cls in enumerate(W.DEFAULT_SIZE_CLASSES):
        _s = catalog.size_gib[catalog.class_idx == _j]
        _rows.append(
            f"| {_cls.name} | {_cls.family} | {_cls.sd:.3f} | {_s.size:,} | "
            f"{_s.min():.2f} | {np.median(_s):.2f} | {_s.max():,.0f} | "
            f"{_s.sum() / 1024:,.1f} | {_cls.retained_mass:.4f} |"
        )
    mo.md(
        "| class | law | σ | count | min | median | max | total TiB |"
        " retained |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def _(W, catalog, mo, np):
    _order = np.argsort(catalog.size_gib)
    _top = _order[-max(1, catalog.n_torrents // 100) :]
    _median_class = W.DEFAULT_SIZE_CLASSES[
        int(catalog.class_idx[_order[catalog.n_torrents // 2]])
    ].name
    mo.hstack(
        [
            mo.stat(
                label=r"median size $\bar S$",
                value=f"{catalog.median_size:.1f} GiB",
            ),
            mo.stat(label="mean size", value=f"{catalog.size_gib.mean():.1f} GiB"),
            mo.stat(
                label="catalogue",
                value=f"{catalog.size_gib.sum() / 1024:,.0f} TiB",
            ),
            mo.stat(
                label="top 1% of torrents hold",
                value=f"{catalog.size_gib[_top].sum() / catalog.size_gib.sum():.0%}",
            ),
            mo.stat(label="median-size class", value=_median_class),
        ],
        widths="equal",
    )
    return


@app.cell(hide_code=True)
def _(W, catalog, np, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _edges = np.logspace(-2, 3.5, 90)
    for _j, _cls in enumerate(W.DEFAULT_SIZE_CLASSES):
        _s = catalog.size_gib[catalog.class_idx == _j]
        _ax[0].hist(_s, bins=_edges, alpha=0.65, label=_cls.name)
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel("torrent size (GiB)")
    _ax[0].set_ylabel("torrents")
    _ax[0].set_title("size mixture, by class")
    _ax[0].legend(fontsize=7)

    _srt = np.sort(catalog.size_gib)
    _share = np.cumsum(_srt) / _srt.sum()
    _ax[1].plot(np.arange(1, _srt.size + 1) / _srt.size, _share)
    _ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="equal size")
    _ax[1].set_xlabel("fraction of torrents (smallest first)")
    _ax[1].set_ylabel("fraction of total bytes")
    _ax[1].set_title("where the bytes are")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The size distribution is the reason §3.1 needs $S_i^{\eta}$ at all: the
    catalogue spans four and a half orders of magnitude, and a reward that
    ignored size would pay the same for 1 GiB of disk as for 2 TiB.

    ## 3. Bandwidth

    Verification, not decoration: the sampled medians must sit on 300 / 50, the
    caps must bind exactly at 10 Gbps / 2 Gbps, and the log-correlation must
    come back as $\rho$.
    """)
    return


@app.cell(hide_code=True)
def _(W, bw, mo, np, rng):
    check_d, check_u, _ = W.sample_bandwidth(200_000, rng, bw)
    cap_d, cap_u = bw.cap_quantiles
    mo.hstack(
        [
            mo.stat(
                label="median down",
                value=f"{np.median(check_d):.1f} Mbps",
                caption=f"target {bw.median_down:.0f}",
            ),
            mo.stat(
                label="median up",
                value=f"{np.median(check_u):.2f} Mbps",
                caption=f"target {bw.median_up:.0f}",
            ),
            mo.stat(
                label="down cap sits at",
                value=f"p{100 * cap_d:.3f}",
                caption=f"{1e6 * (1 - cap_d):.0f} per million clipped",
            ),
            mo.stat(
                label="up cap sits at",
                value=f"p{100 * cap_u:.3f}",
                caption=f"{1e6 * (1 - cap_u):.0f} per million clipped",
            ),
            mo.stat(
                label="log corr",
                value=f"{np.corrcoef(np.log(check_d), np.log(check_u))[0, 1]:.3f}",
                caption=f"target {bw.rho:.2f}",
            ),
            mo.stat(
                label="mean / median uplink",
                value=f"{check_u.mean() / np.median(check_u):.2f}×",
                caption="lognormal skew",
            ),
        ],
        widths="equal",
    )
    return check_d, check_u


@app.cell(hide_code=True)
def _(bw, check_d, check_u, np, plt):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    _sub = slice(0, 20_000)
    _ax[0].scatter(check_d[_sub], check_u[_sub], s=1.2, alpha=0.12, lw=0)
    _ax[0].axvline(bw.median_down, color="C3", lw=0.9, ls="--")
    _ax[0].axhline(bw.median_up, color="C3", lw=0.9, ls="--")
    _ax[0].axvline(bw.max_down, color="k", lw=0.9)
    _ax[0].axhline(bw.max_up, color="k", lw=0.9)
    _ax[0].plot([1e0, 1e4], [1e0, 1e4], color="C2", lw=0.8, ls=":")
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel("downlink (Mbps)")
    _ax[0].set_ylabel("uplink (Mbps)")
    _ax[0].set_title("joint law (dashed = medians, solid = caps)")

    for _a, _v, _name, _cap in (
        (_ax[1], check_d, "downlink", bw.max_down),
        (_ax[2], check_u, "uplink", bw.max_up),
    ):
        _a.hist(_v, bins=np.logspace(np.log10(_v.min()), np.log10(_cap), 80))
        _a.axvline(np.median(_v), color="C3", lw=1.0, ls="--")
        _a.set_xscale("log")
        _a.set_xlabel(f"{_name} (Mbps)")
        _a.set_title(f"{_name} marginal")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cloud has two visible populations without either being put there by
    hand: a symmetric ridge on the $y = x$ diagonal (seedboxes and fibre) and a
    much wider asymmetric skirt below it (consumer links whose uplink is a
    small and variable fraction of their downlink). That skirt is what
    $\sigma_{\uparrow} > \sigma_{\downarrow}$ buys, and it is the reason the
    site's serving capacity is far more concentrated than its demand.

    ## 4. Uptime, disk and their coupling to uplink

    Not specified by the handoff. Uptime is Beta (mean 0.78) and disk is
    lognormal (median 1 TiB), both linked to uplink through a Gaussian copula
    whose common factor *is* the uplink z-score, so a seedbox tends to be fast,
    always-on and roomy at once. All pairwise correlations are then products of
    the two stated ones, which makes the correlation matrix PSD by
    construction rather than by repair.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, population):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.0))
    _ax[0].hist(population.availability, bins=60)
    _ax[0].set_xlabel(r"true uptime $p_v$")
    _ax[0].set_title("availability")
    _ax[1].hist(population.disk_gib, bins=np.logspace(1.5, 5, 60))
    _ax[1].set_xscale("log")
    _ax[1].set_xlabel("disk (GiB)")
    _ax[1].set_title("disk budget")
    _rank_up = np.argsort(np.argsort(population.up_mbps))
    _rank_p = np.argsort(np.argsort(population.availability))
    _spearman = np.corrcoef(_rank_up, _rank_p)[0, 1]
    _ax[2].scatter(population.up_mbps, population.availability, s=2, alpha=0.15, lw=0)
    _ax[2].set_xscale("log")
    _ax[2].set_xlabel("uplink (Mbps)")
    _ax[2].set_ylabel(r"uptime $p_v$")
    _ax[2].set_title(f"coupling (Spearman {_spearman:.2f})")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What this world implies for the mechanism

    The handoff's target is a slowdown $C^* = 1.5$ against a reference leecher
    of downlink $d_{ref}$, so every torrent wants aggregate capacity
    $x^* = d_{ref}/C^*$. With $d_{ref}$ at the site's median downlink, this
    world fixes the scale of everything the reward controller does — the two
    numbers below decide whether the knee is ever reached in practice.
    """)
    return


@app.cell(hide_code=True)
def _(catalog, mo, np, population):
    d_ref = float(np.median(population.down_mbps))
    x_star = d_ref / 1.5
    # g_v = p b / k: what one node credibly offers one torrent, at k = 3.
    g_typical = float(np.median(population.availability * population.up_mbps) / 3.0)
    site_capacity = float((population.availability * population.up_mbps).sum())
    site_disk = float(population.disk_gib.sum())
    catalog_bytes = float(catalog.size_gib.sum())
    mo.hstack(
        [
            mo.stat(label=r"$d_{ref}$", value=f"{d_ref:.0f} Mbps"),
            mo.stat(label=r"$x^* = d_{ref}/C^*$", value=f"{x_star:.0f} Mbps"),
            mo.stat(
                label=r"seeders to reach $x^*$",
                value=f"{x_star / g_typical:.1f}",
                caption=f"at g_v ≈ {g_typical:.0f} Mbps",
            ),
            mo.stat(
                label="site disk / catalogue",
                value=f"{site_disk / catalog_bytes:.1f}×",
                caption="copies of everything it could hold",
            ),
            mo.stat(
                label="site uplink",
                value=f"{site_capacity / 1000:,.0f} Gbps",
                caption="expected, all nodes",
            ),
        ],
        widths="equal",
    )
    return catalog_bytes, g_typical, site_disk, x_star


@app.cell(hide_code=True)
def _(catalog_bytes, g_typical, mo, site_disk, x_star):
    mo.md(rf"""
    > [!warning] The knee is easily overshot
    > A torrent needs only about **{x_star / g_typical:.0f} seeders** to sit
    > at $C^*$, and the site can hold **{site_disk / catalog_bytes:.1f} copies**
    > of its catalogue. Those two numbers decide everything downstream. Where
    > seeders concentrate — and part 2 shows the size exponent makes them
    > concentrate on small torrents — the count blows past
    > {x_star / g_typical:.0f} easily, the torrent lands on the floor $C = 1$
    > rather than on $C^* = 1.5$, the marginal $\Delta H$ collapses to zero,
    > and the only term left paying anybody is the flat floor $\alpha$.
    > Everything the seeders do *not* concentrate on gets whatever is left.
    >
    > That is not a defect of this world model — it is the operating point the
    > mechanism will actually meet, and part 3 measures what it does there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Catalogue attributes

    Popularity is Zipf over a random ranking (independent of size — a
    deliberate simplification: on a real tracker new large releases are also
    the popular ones). Importance $z_i$ and charge multiplier $\phi_i$ follow
    the ranges in §3.1 and §3.5.
    """)
    return


@app.cell(hide_code=True)
def _(catalog, np, plt):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 2.9))
    _srt = np.sort(catalog.demand_share)[::-1]
    _ax[0].loglog(np.arange(1, _srt.size + 1), _srt)
    _ax[0].set_xlabel("popularity rank")
    _ax[0].set_ylabel("share of demand")
    _ax[0].set_title("Zipf popularity")
    _vals, _counts = np.unique(catalog.importance, return_counts=True)
    _ax[1].bar(_vals, _counts / _counts.sum())
    _ax[1].set_xlabel(r"importance $z_i$")
    _ax[1].set_title(r"$w_i = 2^{z_i}$, 16× span")
    _vals2, _counts2 = np.unique(catalog.charge, return_counts=True)
    _ax[2].bar([str(v) for v in _vals2], _counts2 / _counts2.sum())
    _ax[2].set_xlabel(r"charge multiplier $\phi_i$")
    _ax[2].set_title("free / half / full")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    Part 2 (`02_health_and_reward.py`) takes this world as given and studies the
    reward formula on a frozen swarm; part 3 (`03_agent_simulation.py`) lets
    seeders move.
    """)
    return


if __name__ == "__main__":
    app.run()
