import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from marimo_lab.ptcredit import (
        RewardParams,
        health,
        health_marginal,
        sample_catalog,
        sample_population,
    )

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9.0, "axes.grid": True, "grid.alpha": 0.3}
    )
    return (
        RewardParams,
        health,
        health_marginal,
        mo,
        np,
        plt,
        sample_catalog,
        sample_population,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point system, part 2 — the reward on a frozen swarm

    Part 1 fixed the world. This part asks what the formulas of §3 *do*, with
    memberships held still, so that every effect seen here is a property of the
    mechanism and not of the agents' behaviour.

    $$
    H(C) = 1 - \frac{\operatorname{sp}(\gamma(1 - C^*/C))}{\operatorname{sp}(\gamma)},
    \qquad
    C_i = \begin{cases}\max(1, d_{ref}/x_i) & A_i \ge \pi_{min}\\
          \infty & \text{else}\end{cases},
    \qquad
    r_{u,i} = \kappa\, w_i\, S_i^{\eta}\,\bigl[\alpha + H_i - H_i^{(-u)}\bigr]
    $$

    Four questions, in order:

    1. is the reward really a stable controller, and where does its fixed point
       land;
    2. what happens at the availability threshold $\pi_{min}$;
    3. what value of $\eta$ actually makes the reward size-neutral (§9's first
       open decision);
    4. how big may $\alpha$ be, and how big must the startup gift $I_{min}$ be.
    """)
    return


@app.cell
def _(mo):
    ui_gamma = mo.ui.slider(
        start=1.0,
        stop=10.0,
        step=0.5,
        value=4.0,
        label=r"knee sharpness $\gamma$",
        show_value=True,
    )
    ui_cstar = mo.ui.slider(
        start=1.0,
        stop=4.0,
        step=0.1,
        value=1.5,
        label=r"target $C^*$",
        show_value=True,
    )
    ui_alpha = mo.ui.slider(
        start=0.0,
        stop=0.30,
        step=0.01,
        value=0.05,
        label=r"floor $\alpha$",
        show_value=True,
    )
    ui_eta = mo.ui.slider(
        start=0.0,
        stop=1.4,
        step=0.05,
        value=0.7,
        label=r"size exponent $\eta$",
        show_value=True,
    )
    ui_pimin = mo.ui.slider(
        start=0.5,
        stop=0.99,
        step=0.01,
        value=0.9,
        label=r"$\pi_{min}$",
        show_value=True,
    )
    ui_ptilde = mo.ui.slider(
        start=0.3,
        stop=0.99,
        step=0.01,
        value=0.78,
        label=r"per-seeder $\tilde p_v$",
        show_value=True,
    )
    mo.vstack(
        [
            mo.hstack([ui_gamma, ui_cstar, ui_alpha], widths="equal"),
            mo.hstack([ui_eta, ui_pimin, ui_ptilde], widths="equal"),
        ]
    )
    return ui_alpha, ui_cstar, ui_eta, ui_gamma, ui_pimin, ui_ptilde


@app.cell
def _(
    RewardParams,
    np,
    sample_catalog,
    sample_population,
    ui_alpha,
    ui_cstar,
    ui_eta,
    ui_gamma,
    ui_pimin,
):
    par = RewardParams(
        kappa=1.0,
        alpha=float(ui_alpha.value),
        gamma=float(ui_gamma.value),
        c_star=float(ui_cstar.value),
        eta=float(ui_eta.value),
        pi_min=float(ui_pimin.value),
    )
    _rng = np.random.default_rng(0)
    cat = sample_catalog(4000, _rng)
    pop = sample_population(1000, _rng)
    d_ref = float(np.median(pop.down_mbps))
    x_star = d_ref / par.c_star
    # One node's credited contribution g_v = p b / k, at a typical concurrency.
    g_unit = float(np.median(pop.availability * pop.up_mbps) / 3.0)
    return cat, d_ref, g_unit, par, pop, x_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The curve, and why its shape is the whole argument

    $H$ is plotted twice: against slowdown $C$ (what a leecher experiences) and
    against capacity $x = d_{ref}/C$ (what seeders actually add). Concavity in
    $x$ is the stability condition of §4 — it is what guarantees that each
    added seeder is worth strictly less than the last.
    """)
    return


@app.cell
def _(health, health_marginal, np, par, plt, x_star):
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    _c = np.linspace(1.0, 8.0, 600)
    for _g in (2.0, 4.0, 8.0):
        _ax[0].plot(_c, health(_c, par.c_star, _g), lw=1.0, label=rf"$\gamma$={_g:.0f}")
    _ax[0].plot(_c, health(_c, par.c_star, par.gamma), "k", lw=1.8, label="current")
    _ax[0].axvline(par.c_star, color="C3", ls="--", lw=0.9)
    _ax[0].set_xlabel(r"slowdown $C$")
    _ax[0].set_ylabel(r"$H$")
    _ax[0].set_title(r"health vs slowdown ($C^*$ dashed)")
    _ax[0].legend(fontsize=7)

    _x = np.linspace(0.02, 3.0, 600) * x_star
    _ax[1].plot(
        _x / x_star,
        health(np.maximum(1.0, x_star * par.c_star / _x), par.c_star, par.gamma),
        "k",
        lw=1.5,
    )
    _ax[1].axvline(1.0, color="C3", ls="--", lw=0.9)
    _ax[1].set_xlabel(r"capacity $x / x^*$")
    _ax[1].set_ylabel(r"$H$")
    _ax[1].set_title("health vs capacity: concave")

    _ax[2].plot(
        _x / x_star,
        health_marginal(_x, x_star * par.c_star, par.c_star, par.gamma) * x_star,
        "k",
        lw=1.5,
    )
    _ax[2].axvline(1.0, color="C3", ls="--", lw=0.9)
    _ax[2].set_xlabel(r"capacity $x / x^*$")
    _ax[2].set_ylabel(r"$x^*\,dH/dx$")
    _ax[2].set_title("marginal: plateau, knee, floor")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(health, mo, par):
    _h_target = float(health(par.c_star, par.c_star, par.gamma))
    _h_max = float(health(1.0, par.c_star, par.gamma))
    mo.hstack(
        [
            mo.stat(label=r"$H(C^*)$", value=f"{_h_target:.3f}"),
            mo.stat(label=r"$H(1)$ (attainable max)", value=f"{_h_max:.3f}"),
            mo.stat(
                label=r"$H(\infty)$",
                value=f"{float(health(float('inf'), par.c_star, par.gamma)):.3f}",
            ),
            mo.stat(
                label="max earnable per torrent",
                value=f"{par.alpha + _h_max:.2f}",
                caption=r"$\times\,\kappa w S^\eta$ points/h",
            ),
        ],
        widths="equal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The controller's fixed point, and the dead zone below $\pi_{min}$

    Add identical seeders one at a time. Each contributes $g_v$ to capacity
    *and* $\tilde p_v$ to completeness, so the marginal a joiner is paid is

    $$\Delta H(n) = H\bigl(C(n g,\, A_n)\bigr) - H\bigl(C((n-1) g,\, A_{n-1})\bigr),
      \qquad A_n = 1 - (1 - \tilde p_v)^n .$$

    A seeder joins while $\kappa w S^\eta (\alpha + \Delta H) $ exceeds their
    opportunity cost $\theta$; the equilibrium is where the curve crosses
    $\theta$ from above. The knee is what makes that crossing insensitive to
    $\theta$.
    """)
    return


@app.cell
def _(g_unit, health, np, par, plt, ui_ptilde, x_star):
    _p = float(ui_ptilde.value)
    _g = g_unit  # one node's credited contribution, from the sampled world
    _n = np.arange(0, 41)
    _cap = _n * _g
    _avail = 1.0 - (1.0 - _p) ** _n
    _c = np.where(
        _avail >= par.pi_min,
        np.maximum(1.0, np.divide(x_star * par.c_star, np.maximum(_cap, 1e-12))),
        np.inf,
    )
    _h = health(_c, par.c_star, par.gamma)
    _dh = np.diff(_h, prepend=0.0)

    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _ax[0].plot(_n, _h, "o-", ms=2.5, lw=1.0)
    _ax[0].axhline(
        float(health(par.c_star, par.c_star, par.gamma)),
        color="C3",
        ls="--",
        lw=0.9,
        label=r"$H(C^*)$",
    )
    _ax[0].set_xlabel("seeders $n$")
    _ax[0].set_ylabel(r"$H$")
    _ax[0].set_title("health vs seeder count")
    _ax[0].legend(fontsize=7)

    _ax[1].plot(_n, par.alpha + _dh, "o-", ms=2.5, lw=1.0, label=r"$\alpha + \Delta H$")
    _ax[1].axhline(par.alpha, color="C7", ls=":", lw=0.9, label=r"$\alpha$ floor")
    for _theta, _col in ((0.30, "C2"), (0.10, "C1"), (0.03, "C4")):
        _ax[1].axhline(_theta, color=_col, lw=0.8, ls="--")
        _cross = np.flatnonzero((par.alpha + _dh) < _theta)
        _cross = _cross[_cross > 1]
        # theta below the alpha floor is never crossed: the seeder keeps
        # adding torrents until disk, not reward, stops them.
        _eq = int(_cross[0]) if _cross.size else _n[-1]
        _tag = (
            rf"$\theta$={_theta}: n={_eq}"
            if _cross.size
            else rf"$\theta$={_theta}: below $\alpha$, fills disk"
        )
        _ax[1].annotate(
            _tag,
            (_eq, _theta),
            fontsize=6.5,
            color=_col,
            xytext=(3, 3),
            textcoords="offset points",
        )
    _ax[1].set_yscale("symlog", linthresh=0.01)
    _ax[1].set_xlabel("seeders $n$")
    _ax[1].set_ylabel("payout per seeder")
    _ax[1].set_title("marginal payout and where it crosses $\\theta$")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, np, par, ui_ptilde):
    n_alive = int(np.ceil(np.log1p(-par.pi_min) / np.log1p(-float(ui_ptilde.value))))
    if n_alive <= 1:
        _note = rf"""
    > [!note] No coordination gap at these settings
    > With $\tilde p_v = {float(ui_ptilde.value):.2f} \ge
    > \pi_{{min}} = {par.pi_min:.2f}$, a **single** seeder already satisfies
    > the completeness condition, so the very first seeder back onto an empty
    > torrent is paid the whole of $H_i$ (its leave-one-out is the empty
    > swarm). The reward surface has no flat gap and the mechanism recovers a
    > lost torrent on its own.
    >
    > This is the comfortable case, and it only holds while the *pessimistic*
    > estimate $\tilde p_v$ — not the true uptime — stays above
    > $\pi_{{min}}$. Lower the slider below {par.pi_min:.2f} to see what §3.3
    > does the rest of the time.
    """
    else:
        _note = rf"""
    > [!warning] The 0 → 1 step pays nothing: a lost torrent stays lost
    > With $\tilde p_v = {float(ui_ptilde.value):.2f}$ and
    > $\pi_{{min}} = {par.pi_min:.2f}$ it takes **{n_alive} simultaneous
    > seeders** before $A_i \ge \pi_{{min}}$ and the torrent counts as
    > available at all. Below that, $H_i = 0$ *and* $H_i^{{(-u)}} = 0$, so
    > $\Delta H = 0$: the first {n_alive - 1} seeder(s) are paid exactly the
    > flat floor $\alpha$ — the same as a seeder on a fully healthy torrent —
    > while the {n_alive}-th is paid the entire $H_i$.
    >
    > The marginal-contribution rule is doing precisely what it promises (the
    > seeder who makes the copy available captures the whole value), but the
    > consequence is a **coordination trap**: the reward surface is flat
    > across the gap, so nothing pulls a torrent from 0 seeders back to
    > {n_alive}. Recovery has to come from $\alpha$-driven random browsing.
    >
    > This is the largest gap between §3.3 as written and what a simulation
    > does with it, and part 3 measures what it costs. The cheap fix is to
    > make the completeness term continuous — pay
    > $\min(1, A_i/\pi_{{min}})\cdot H(C_i)$ — so the first seeder back is
    > already paid for the availability they restore.
    """
    mo.md(_note)
    return (n_alive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. What $\eta$ should be (§9, first open decision)

    $\eta$ is meant to make reward *per GiB-hour of disk* size-neutral. That
    condition can be written down rather than tuned. A seeder's binding
    constraint is disk, so at equilibrium every torrent they might hold must
    offer the same reward per GiB:

    $$\frac{\kappa\, w_i\, S_i^{\eta}\,(\alpha + \Delta H_i)}{\text{size}_i}
      = \theta
      \quad\Longrightarrow\quad
      \alpha + \Delta H_i = \frac{\theta\,\bar S}{\kappa\, w_i}\; S_i^{\,1-\eta}.$$

    The right-hand side is independent of size **iff $\eta = 1$**. For
    $\eta < 1$ a larger torrent has to be scarcer — carry a higher $\Delta H$,
    hence fewer seeders and a worse $C_i$ — to be worth the same disk. So the
    initial value $\eta = 0.7$ is not size-neutral; it is a deliberate
    ${\sim}S^{-0.3}$ subsidy to small torrents.

    Below, the required $\Delta H$ is inverted back into the equilibrium
    slowdown each size class would settle at, for the actual catalogue of
    part 1.
    """)
    return


@app.cell
def _(cat, g_unit, health, np, par, plt, x_star):
    def equilibrium_c(eta, sizes, gamma, c_star, alpha, median_size):
        r"""Slowdown at which a torrent of each size stops attracting seeders.

        The site-wide price of disk, $\theta$, is not a free parameter here:
        it is normalised so that the *median-size* torrent settles exactly on
        $C^*$.  What the curve then shows is purely the size tilt $\eta$
        introduces, with no arbitrary scale left in it.
        """
        grid_c = np.geomspace(1.0, 60.0, 1200)
        step = g_unit  # one seeder's credited contribution
        cap = x_star * c_star / grid_c
        marginal = health(
            np.maximum(1.0, x_star * c_star / (cap + step)), c_star, gamma
        ) - health(grid_c, c_star, gamma)
        at_target = float(np.interp(c_star, grid_c, marginal))
        need = (at_target + alpha) * (sizes / median_size) ** (1.0 - eta) - alpha
        # marginal increases with C, so it is a valid interpolation axis.
        return np.interp(need, marginal, grid_c, left=1.0, right=np.inf)

    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    _sizes = np.geomspace(0.1, 2048.0, 200)
    for _eta, _c in ((0.0, "C0"), (0.7, "C1"), (1.0, "C2"), (1.3, "C3")):
        _eq = equilibrium_c(
            _eta, _sizes, par.gamma, par.c_star, par.alpha, cat.median_size
        )
        _ax[0].plot(_sizes, _eq, color=_c, lw=1.2, label=rf"$\eta$={_eta}")
    _ax[0].axhline(par.c_star, color="k", ls="--", lw=0.9)
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("log")
    _ax[0].set_xlabel("torrent size (GiB)")
    _ax[0].set_ylabel("equilibrium slowdown $C$")
    _ax[0].set_title(r"only $\eta=1$ is flat in size")
    _ax[0].legend(fontsize=7)

    _rel = cat.size_gib / cat.median_size
    for _eta, _c in ((0.0, "C0"), (0.7, "C1"), (1.0, "C2"), (1.3, "C3")):
        _per_gib = _rel**_eta / cat.size_gib
        _order = np.argsort(cat.size_gib)
        _ax[1].plot(
            cat.size_gib[_order],
            (_per_gib / _per_gib.max())[_order],
            color=_c,
            lw=1.2,
            label=rf"$\eta$={_eta}",
        )
    _ax[1].set_xscale("log")
    _ax[1].set_yscale("log")
    _ax[1].set_xlabel("torrent size (GiB)")
    _ax[1].set_ylabel(r"$S^\eta/\text{size}$, normalised")
    _ax[1].set_title("pay per GiB of disk, at equal $\\Delta H$")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return (equilibrium_c,)


@app.cell(hide_code=True)
def _(cat, mo, np, par):
    _rel = cat.size_gib / cat.median_size
    _tilt = _rel**par.eta / cat.size_gib
    _small = float(np.median(_tilt[cat.class_idx == 0]))
    _large = float(np.median(_tilt[cat.class_idx == 2]))
    _super = float(np.median(_tilt[cat.class_idx == 3]))
    mo.md(rf"""
    At $\eta = {par.eta:.2f}$ and equal $\Delta H$, a GiB of disk spent on the
    median **small** torrent pays **{_small / _large:.1f}×** what the same GiB
    pays on the median **large** torrent, and **{_small / _super:.1f}×** what it
    pays on a super-large one. A rational seeder therefore fills its disk with
    the smallest torrents it can find, and the 2 TiB releases are left to
    whoever has spare disk and no better use for it — which, with the
    $\pi_{{min}}$ cliff above, is how a catalogue loses its large end first.

    There is a second, unmodelled tilt in the same direction: acquiring a
    torrent to seed it costs $\phi_i \cdot \text{{size}}_i$ points up front. That
    does not shift the long-run exponent on its own — dividing both the reward
    and the acquisition charge by size leaves the comparison intact once the
    torrent is held forever — but it is a real *finite-horizon* barrier: the
    shorter the expected holding time, the more of that one-off charge has to
    be amortised, and the worse large torrents look. Turnover therefore biases
    the effective break-even exponent upward by an amount this notebook does
    not model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Sizing $\alpha$, $\kappa$ and $I_{min}$

    $\alpha$ is paid on every membership regardless of health, so it is the
    dominant term in total mint whenever most torrents are healthy — exactly
    the regime part 1 predicted this world sits in. The plot shows the share of
    total mint attributable to the flat floor as a function of $\alpha$, given
    a catalogue where a fraction $\pi_{\text{healthy}}$ of memberships carry
    $\Delta H \approx 0$.
    """)
    return


@app.cell
def _(np, par, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
    _alpha = np.linspace(0.0, 0.3, 200)
    for _frac, _c in ((0.5, "C0"), (0.8, "C1"), (0.95, "C2")):
        _mint_alpha = _alpha
        _mint_delta = (1.0 - _frac) * 0.5
        _ax[0].plot(
            _alpha,
            _mint_alpha / (_mint_alpha + _mint_delta),
            color=_c,
            lw=1.2,
            label=rf"{_frac:.0%} of memberships at $\Delta H\approx 0$",
        )
    _ax[0].axvline(par.alpha, color="k", ls="--", lw=0.9)
    _ax[0].set_xlabel(r"$\alpha$")
    _ax[0].set_ylabel("share of mint from the flat floor")
    _ax[0].set_title("who the points are actually paid to")
    _ax[0].legend(fontsize=6.5)

    # Break-even: with alpha only, how long to earn one median download back.
    _kappa = np.geomspace(1e-3, 1.0, 200)
    for _held, _c in ((50, "C0"), (200, "C1"), (800, "C2")):
        _rate = _kappa * par.alpha * _held
        _ax[1].plot(
            _kappa,
            14.0 * 0.825 / np.maximum(_rate, 1e-12),
            color=_c,
            lw=1.2,
            label=f"{_held} memberships",
        )
    _ax[1].set_xscale("log")
    _ax[1].set_yscale("log")
    _ax[1].axhline(24.0, color="k", ls=":", lw=0.9)
    _ax[1].set_xlabel(r"$\kappa$")
    _ax[1].set_ylabel("hours to earn one median download")
    _ax[1].set_title(r"$\kappa$ sets the clock (dotted = 1 day)")
    _ax[1].legend(fontsize=6.5)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cat, mo, n_alive, np):
    phi_mean = float(np.mean(cat.charge))
    i_min = 4.0 * cat.median_size * phi_mean
    mo.hstack(
        [
            mo.stat(
                label=r"$\bar S$",
                value=f"{cat.median_size:.1f} GiB",
            ),
            mo.stat(label=r"mean $\phi_i$", value=f"{phi_mean:.2f}"),
            mo.stat(
                label="startup buffer, 4 median draws",
                value=f"{i_min:.0f} points",
                caption="expected charge, not a guaranteed $I_{min}$",
            ),
            mo.stat(
                label="one super-large download",
                value=f"{2048 * phi_mean:,.0f} points",
                caption=f"= {2048 / (4 * cat.median_size):.0f}× the startup gift",
            ),
            mo.stat(
                label=r"seeders for $A \ge \pi_{min}$",
                value=f"{n_alive}",
            ),
        ],
        widths="equal",
    )
    return i_min, phi_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Importance: pay more, or demand more? (§9, second open decision)

    $w_i = 2^{z_i}$ multiplies the pay; the alternative is a per-torrent target
    $C^*_i = C^* 2^{-z_i}$ that leaves pay alone. The difference is visible in
    one plot: scaling the pay shifts the whole marginal curve up, so a
    high-importance torrent keeps attracting seeders long past its target,
    whereas moving the target relocates the knee and stops recruiting once the
    better health is actually achieved.
    """)
    return


@app.cell
def _(g_unit, health, np, par, plt, x_star):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
    _g = g_unit
    _n = np.arange(1, 41)
    for _z, _c in ((0, "C0"), (1, "C1"), (2, "C2")):
        _cap = _n * _g
        _c_of_n = np.maximum(1.0, x_star * par.c_star / _cap)
        _dh_pay = 2.0**_z * np.diff(health(_c_of_n, par.c_star, par.gamma), prepend=0.0)
        _dh_tgt = np.diff(
            health(_c_of_n, par.c_star * 2.0**-_z, par.gamma), prepend=0.0
        )
        _ax[0].plot(
            _n, par.alpha * 2.0**_z + _dh_pay, color=_c, lw=1.2, label=rf"$z={_z}$"
        )
        _ax[1].plot(_n, par.alpha + _dh_tgt, color=_c, lw=1.2, label=rf"$z={_z}$")
    for _a, _t in (
        (_ax[0], r"pay scaling: $w_i = 2^{z_i}$"),
        (_ax[1], r"target scaling: $C^*_i = C^* 2^{-z_i}$"),
    ):
        _a.axhline(0.05, color="k", ls="--", lw=0.8)
        _a.set_yscale("log")
        _a.set_xlabel("seeders $n$")
        _a.set_ylabel("payout per seeder")
        _a.set_title(_t)
        _a.legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With pay scaling the $z = 2$ curve sits a full factor of four above the
    ordinary one *everywhere*, including deep in the over-provisioned region
    where the extra seeders buy nothing — that is the "provisioned past $C^*$"
    concern in §9, and it is real. With target scaling the curves converge once
    each torrent reaches its own knee, so the extra capacity is bought only
    until the better health exists. Target scaling is the better default; pay
    scaling remains useful as a separate, smaller lever for genuinely
    under-supplied content.

    ---

    Part 3 puts seeders in motion and checks which of these static predictions
    survive contact with the allocation dynamics.
    """)
    return


if __name__ == "__main__":
    app.run()
