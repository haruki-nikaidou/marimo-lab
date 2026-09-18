import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import expon, logistic, lognorm, norm, uniform

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9.0, "axes.grid": True, "grid.alpha": 0.3}
    )
    return expon, logistic, lognorm, minimize, mo, norm, np, plt, uniform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unfair tiered admissions — exact mechanism vs. fluid limit

    `n` people, a fraction $\alpha$ of them group **A**, the rest group **B**;
    skills iid from one distribution, so the groups are statistically identical.
    `q` tiers are filled best-to-worst; tier $k$ has $h_k$ seats, $f_k$ of them
    **reserved** for A and $g_k = h_k - f_k$ **open** to anyone. Within a tier,
    reserved seats take the best remaining A's, then open seats take the best
    remaining people of either group.

    Everything the mechanism does is carried by one running scalar, the
    **overhang** $e_k$ — A-mass admitted so far beyond A's natural share —
    which obeys Lindley's recursion (a single-server queue: reservations arrive,
    the natural share $\alpha h_k$ serves, and open seats can never un-admit):

    $$e_k = \max\bigl(e_{k-1} + f_k/n - \alpha\,\tau_k,\; 0\bigr),
      \qquad \tau_k = h_k/n .$$

    From it, the two admission cutoffs, the per-tier quality change and the
    weighted value loss follow in closed form:

    $$u_k = F^{-1}\!\bigl(1 - T_k - e_k/\alpha\bigr), \quad
      v_k = F^{-1}\!\bigl(1 - T_k + e_k/\beta\bigr), \quad
      T_k = \textstyle\sum_{j\le k}\tau_j,$$

    $$D_k = \underbrace{M(T_k)}_{\text{fair mass}} -
            \bigl[\alpha M(T_k + e_k/\alpha) + \beta M(T_k - e_k/\beta)\bigr],
      \qquad \text{Loss} = n\sum_k (L_k - L_{k+1})\,D_k,$$

    with $M(p)$ = mean skill mass of the top-$p$ fraction of a pool
    ($M(p) = \varphi(\Phi^{-1}(1-p))$ for the normal) and $L_k$ the value of a
    tier-$k$ seat. Because every configuration fills the same seats, the mean of
    the skill distribution cancels: all losses are in $\sigma$ units.

    **What this notebook adds over `sim.py`.** The discrete mechanism is
    re-simulated with common random numbers, but instrumented much further:
    per-tier *admission cutoffs* and *within-tier A/B segregation*, an
    interactive reservation profile, the dead zone and marginal swap price as
    functions of $\rho_k = f_k/h_k$, the wave's reach and the
    conservation/expulsion split, the two-shock interaction surface, the exact
    formula vs. its quadratic approximation, the **optimal placement of a fixed
    quota budget** (constrained convex program), the same theory
    re-derived numerically for **non-normal skill distributions**, and the
    $n^{-1/2}$ concentration of the discrete mechanism around the fluid limit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Parameters

    Cost warning: every panel re-simulates when these change. The heavy term is
    `reps × (sort of n)`; defaults (`n` = 100k, 48 reps) keep a full pass at a
    couple of seconds.
    """)
    return


@app.cell
def _(mo):
    ui_n = mo.ui.dropdown(
        options={"25k": 25_000, "50k": 50_000, "100k": 100_000, "200k": 200_000},
        value="100k",
        label="population $n$",
    )
    ui_alpha = mo.ui.slider(
        start=0.05,
        stop=0.50,
        step=0.01,
        value=0.30,
        label=r"A-share $\alpha$",
        show_value=True,
    )
    ui_q = mo.ui.slider(
        start=4, stop=12, step=1, value=8, label="tiers $q$", show_value=True
    )
    ui_share = mo.ui.slider(
        start=0.02,
        stop=0.10,
        step=0.005,
        value=0.06,
        label=r"seats per tier $\tau$",
        show_value=True,
    )
    ui_values = mo.ui.dropdown(
        options={
            "linear: q, q-1, ..., 1": "linear",
            "geometric: 2^-k": "geometric",
            "flat: all tiers equal": "flat",
            "elite only: tier 1 counts": "elite",
        },
        value="linear: q, q-1, ..., 1",
        label="tier values $L_k$",
    )
    ui_reps = mo.ui.slider(
        start=8, stop=128, step=8, value=48, label="replications", show_value=True
    )
    ui_seed = mo.ui.number(start=0, stop=9999, step=1, value=7, label="seed")
    mo.vstack(
        [
            mo.hstack([ui_n, ui_alpha, ui_q], widths="equal"),
            mo.hstack([ui_share, ui_values], widths="equal"),
            mo.hstack([ui_reps, ui_seed], widths="equal"),
        ]
    )
    return ui_alpha, ui_n, ui_q, ui_reps, ui_seed, ui_share, ui_values


@app.cell
def _(np):
    def tier_values(kind, q):
        """Value L_k of a seat in tier k. Only the decrements L_k - L_{k+1} matter."""
        ks = np.arange(q)
        if kind == "geometric":
            return 2.0 ** (-ks)
        if kind == "flat":
            return np.ones(q)
        if kind == "elite":
            return (ks == 0).astype(float)
        return np.arange(q, 0, -1, dtype=float)

    return (tier_values,)


@app.cell
def _(
    norm,
    np,
    tier_values,
    ui_alpha,
    ui_n,
    ui_q,
    ui_reps,
    ui_seed,
    ui_share,
    ui_values,
):
    n = int(ui_n.value)
    alpha = float(ui_alpha.value)
    beta = 1.0 - alpha
    q = int(ui_q.value)
    # Keep the system from admitting (almost) everybody: q * share <= 0.9.
    share = min(float(ui_share.value), 0.9 / q)
    seats = int(round(share * n))
    h = np.full(q, seats)
    tau = h / n
    T = np.cumsum(tau)
    tf = norm.ppf(1.0 - T)  # fair cumulative cutoffs t_k
    pdf_tf = norm.pdf(tf)
    L = tier_values(ui_values.value, q)
    lam = L - np.r_[L[1:], 0.0]  # value decrements; lam[-1] = expulsion price
    reps = int(ui_reps.value)
    reps_small = max(8, reps // 4)
    seed = int(ui_seed.value)
    mdl = {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "q": q,
        "share": share,
        "seats": seats,
        "h": h,
        "tau": tau,
        "T": T,
        "tf": tf,
        "pdf_tf": pdf_tf,
        "L": L,
        "lam": lam,
        "seed": seed,
    }
    return L, T, alpha, h, mdl, n, q, reps, reps_small, seats, seed, share, tf


@app.cell
def _(L, T, alpha, mo, n, q, seats, share, tf):
    mo.hstack(
        [
            mo.stat(label="seats per tier", value=f"{seats:,}"),
            mo.stat(label="admitted", value=f"{T[-1]:.1%} of n"),
            mo.stat(label="A-pool", value=f"{int(round(alpha * n)):,}"),
            mo.stat(label="top-tier fair cutoff", value=f"{tf[0]:+.3f}σ"),
            mo.stat(label="value decrement sum", value=f"{L[0]:.3g}"),
            mo.stat(label=r"τ actually used", value=f"{share:.3f} × {q}"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The exact mechanism

    No fluid approximation: sorted pools, two pointers, a top-$g_k$ selection per
    tier. Instrumented to return, per tier, the skill sum *and* count *and*
    weakest admitted skill **separately for each group** — that is what makes the
    cutoff and segregation panels below possible.

    One policy detail beyond `sim.py`: if the A-pool runs dry, unfillable
    reserved seats fall back to open seats (the fluid theory assumes this never
    happens, which the diagnostics flag when it does).
    """)
    return


@app.cell
def _(np):
    def admit(pool_a, pool_b, f, g):
        """Greedy tiered admission on descending-sorted pools.

        Returns per-tier (sum_a, sum_b, cnt_a, cnt_b, cut_a, cut_b); the cut_*
        entries are the weakest admitted member of that group in that tier,
        NaN if the group got no seat there.
        """
        q = f.size
        sum_a = np.zeros(q)
        sum_b = np.zeros(q)
        cnt_a = np.zeros(q)
        cnt_b = np.zeros(q)
        cut_a = np.full(q, np.nan)
        cut_b = np.full(q, np.nan)
        ia = ib = 0
        for k in range(q):
            fk = min(int(f[k]), pool_a.size - ia)  # A-pool may run dry
            if fk > 0:
                block = pool_a[ia : ia + fk]
                sum_a[k] += block.sum()
                cnt_a[k] += fk
                cut_a[k] = block[-1]
                ia += fk
            gk = int(g[k]) + int(f[k]) - fk  # unfilled reserved seats go open
            cand_a = pool_a[ia : ia + gk]
            cand_b = pool_b[ib : ib + gk]
            if gk >= cand_a.size + cand_b.size:
                m_a, m_b = cand_a.size, cand_b.size
            else:
                cand = np.concatenate((cand_a, cand_b))
                keep = np.argpartition(cand, cand.size - gk)[cand.size - gk :]
                m_a = int(np.count_nonzero(keep < cand_a.size))
                m_b = gk - m_a
            if m_a:  # candidates are sorted, so the winners are a prefix
                sum_a[k] += cand_a[:m_a].sum()
                cnt_a[k] += m_a
                cut_a[k] = cand_a[m_a - 1]
                ia += m_a
            if m_b:
                sum_b[k] += cand_b[:m_b].sum()
                cnt_b[k] += m_b
                cut_b[k] = cand_b[m_b - 1]
                ib += m_b
        return sum_a, sum_b, cnt_a, cnt_b, cut_a, cut_b

    return (admit,)


@app.cell
def _(expon, logistic, lognorm, norm, np, uniform):
    _ln = lognorm(0.8)
    _ln_m, _ln_s = float(_ln.mean()), float(_ln.std())
    # All standardized to mean 0, variance 1 so losses are comparable.
    DISTS = {
        "normal": norm(),
        "logistic": logistic(scale=np.sqrt(3.0) / np.pi),
        "uniform": uniform(loc=-np.sqrt(3.0), scale=2.0 * np.sqrt(3.0)),
        "exponential": expon(loc=-1.0),
        "lognormal": lognorm(0.8, loc=-_ln_m / _ln_s, scale=1.0 / _ln_s),
    }
    DIST_NOTES = {
        "normal": "reference",
        "logistic": "heavier tail",
        "uniform": "bounded, no tail",
        "exponential": "right-skewed",
        "lognormal": "very heavy right tail",
    }
    return DISTS, DIST_NOTES


@app.cell
def _(DISTS, admit, mo, np):
    @mo.cache
    def run_batch(n, alpha, q, seats, reps, seed, rho_rows, dist_name):
        """Score every configuration on the same skill draws (common random numbers)."""
        n_a = int(round(alpha * n))
        n_b = n - n_a
        h = np.full(q, int(seats))
        rows = np.asarray(rho_rows, float).reshape(-1, q)
        f = np.minimum(np.round(rows * h).astype(np.int64), h)
        g = h - f
        n_cfg = rows.shape[0]
        keys = ("sum", "sum_a", "sum_b", "cnt_a", "cnt_b", "cut_a", "cut_b")
        out = {key: np.zeros((n_cfg, reps, q)) for key in keys}
        dist = DISTS[dist_name]
        rng = np.random.default_rng(seed)
        for r in range(reps):
            pool_a = np.sort(dist.ppf(rng.random(n_a)))[::-1].copy()
            pool_b = np.sort(dist.ppf(rng.random(n_b)))[::-1].copy()
            for c in range(n_cfg):
                s_a, s_b, c_a, c_b, k_a, k_b = admit(pool_a, pool_b, f[c], g[c])
                out["sum"][c, r] = s_a + s_b
                out["sum_a"][c, r] = s_a
                out["sum_b"][c, r] = s_b
                out["cnt_a"][c, r] = c_a
                out["cnt_b"][c, r] = c_b
                out["cut_a"][c, r] = k_a
                out["cut_b"][c, r] = k_b
        return out

    def mc(mdl, rho_rows, reps, dist_name="normal"):
        """Simulate configurations `rho_rows` for model `mdl`; row 0 is the baseline."""
        rows = np.asarray(rho_rows, float).reshape(-1, mdl["q"])
        key = tuple(tuple(float(x) for x in row) for row in rows)
        return run_batch(
            mdl["n"],
            mdl["alpha"],
            mdl["q"],
            mdl["seats"],
            int(reps),
            mdl["seed"],
            key,
            dist_name,
        )

    return (mc,)


@app.cell
def _(np):
    def mean_finite(x):
        """Mean over replications, ignoring tiers where a group got no seat."""
        ok = np.isfinite(x)
        cnt = ok.sum(axis=0)
        tot = np.where(ok, x, 0.0).sum(axis=0)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    def paired(batch, L):
        """Paired differences of each configuration against row 0 of the batch."""
        d = batch["sum"] - batch["sum"][0]
        reps = d.shape[1]
        loss = -(d @ L)
        return {
            "dq": d.mean(axis=1),
            "dq_sem": d.std(axis=1, ddof=1) / np.sqrt(reps),
            "loss": loss.mean(axis=1),
            "loss_sem": loss.std(axis=1, ddof=1) / np.sqrt(reps),
        }

    def group_means(batch, cfg):
        """Mean admitted skill per tier, per group, plus the group cutoffs."""
        with np.errstate(invalid="ignore", divide="ignore"):
            m_a = np.where(
                batch["cnt_a"][cfg] > 0,
                batch["sum_a"][cfg] / np.maximum(batch["cnt_a"][cfg], 1.0),
                np.nan,
            )
            m_b = np.where(
                batch["cnt_b"][cfg] > 0,
                batch["sum_b"][cfg] / np.maximum(batch["cnt_b"][cfg], 1.0),
                np.nan,
            )
        return (
            mean_finite(m_a),
            mean_finite(m_b),
            mean_finite(batch["cut_a"][cfg]),
            mean_finite(batch["cut_b"][cfg]),
        )

    def fmt_profile(row):
        """Compact inline rendering of a per-tier profile for markdown tables."""
        return "[" + " ".join(f"{val:.2f}" for val in np.asarray(row, float)) + "]"

    def dry_note(ths):
        """Applicability gate: the closed forms assume the A-pool never runs dry."""
        bad = sum(1 for t in ths if t["dry"])
        if not bad:
            return ""
        return (
            f"⚠️ **out of scope:** the A-pool runs dry in {bad} of {len(ths)} "
            "profiles analysed here, so $p_A$ saturates at 1 and the fluid limit "
            "no longer describes them (the simulator spills unfillable reserved "
            "seats to open seats). Raise $\\alpha$ or lower the reservations."
        )

    return dry_note, fmt_profile, group_means, paired


@app.cell
def _(norm, np):
    EPS = 1e-15

    def overhang(rho, mdl):
        """Lindley recursion for the A-overhang, per capita."""
        e = np.empty(rho.size)
        cur = 0.0
        for k in range(rho.size):
            cur = max(cur + (rho[k] - mdl["alpha"]) * mdl["tau"][k], 0.0)
            e[k] = cur
        return e

    def top_mass(p):
        """Skill mass of the top-p fraction of a standard normal pool."""
        return norm.pdf(norm.ppf(np.clip(1.0 - p, EPS, 1.0 - EPS)))

    def band_mean(p_edges):
        """Mean skill of the pool band between consecutive depths."""
        dm = np.diff(top_mass(p_edges))
        dp = np.diff(p_edges)
        return np.where(dp > 1e-12, dm / np.where(dp > 1e-12, dp, 1.0), np.nan)

    def fluid(rho, mdl):
        """Fluid-limit solution for reservation profile rho (fraction per tier)."""
        alpha, beta, n = mdl["alpha"], mdl["beta"], mdl["n"]
        e = overhang(np.asarray(rho, float), mdl)
        p_a = np.clip(mdl["T"] + e / alpha, 0.0, 1.0)  # A-pool depth used
        p_b = np.clip(mdl["T"] - e / beta, 0.0, 1.0)  # B-pool depth used
        u = norm.ppf(np.clip(1.0 - p_a, EPS, 1.0 - EPS))
        v = norm.ppf(np.clip(1.0 - p_b, EPS, 1.0 - EPS))
        W = alpha * norm.pdf(u) + beta * norm.pdf(v)
        D = mdl["pdf_tf"] - W  # cumulative loss through tier k
        quad = e**2 / (2.0 * alpha * beta * mdl["pdf_tf"])  # small-e expansion
        return {
            "e": e,
            "p_a": p_a,
            "p_b": p_b,
            "u": u,
            "v": v,
            "D": D,
            "dQ": n * (np.r_[0.0, D[:-1]] - D),  # per-tier quality change
            "loss": float(mdl["lam"] @ D) * n,
            "loss_quad": float(mdl["lam"] @ quad) * n,
            "mean_a": band_mean(np.r_[0.0, p_a]),
            "mean_b": band_mean(np.r_[0.0, p_b]),
            "reach": int((e > 1e-12).sum()),
            "expelled": float(D[-1]) * n,
            "dry": bool(p_a[-1] >= 1.0 - 1e-9),
        }

    def scaled_model(mdl, n_new):
        """Same shape of problem at a different population size."""
        out = dict(mdl)
        out["n"] = int(n_new)
        out["seats"] = int(round(mdl["share"] * n_new))
        out["h"] = np.full(mdl["q"], out["seats"])
        out["tau"] = out["h"] / out["n"]
        out["T"] = np.cumsum(out["tau"])
        out["tf"] = norm.ppf(1.0 - out["T"])
        out["pdf_tf"] = norm.pdf(out["tf"])
        return out

    return fluid, overhang, scaled_model, top_mass


@app.cell
def _(mo):
    mo.md(r"""
    ### Fair baseline

    With no reservations the mechanism is a plain top-$T_k$ cut, so tier $k$
    holds the skill band between $t_{k-1}$ and $t_k$ and A holds exactly
    $\alpha$ of every tier. This is the control arm every later comparison is
    differenced against, so it doubles as a correctness check on the simulator.
    """)
    return


@app.cell
def _(T, group_means, h, mc, mdl, mo, n, np, q, reps, tf, top_mass):
    base_batch = mc(mdl, np.zeros((1, q)), reps)
    base_skill = base_batch["sum"][0].mean(axis=0)
    base_a, base_b, base_cut_a, base_cut_b = group_means(base_batch, 0)
    base_share = base_batch["cnt_a"][0].mean(axis=0) / h
    base_theory = n * np.diff(top_mass(np.r_[0.0, T]))
    mo.md(
        "| tier | fair cutoff $t_k$ | sim cutoff (A / B) | tier skill, theory | "
        "tier skill, sim | A-share |\n| --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {k + 1} | {tf[k]:+.4f} | {base_cut_a[k]:+.4f} / "
            f"{base_cut_b[k]:+.4f} | {base_theory[k]:,.0f}σ | "
            f"{base_skill[k]:,.0f}σ | {base_share[k]:.3f} |"
            for k in range(q)
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Playground: an arbitrary reservation profile

    Set $\rho_k = f_k/h_k$ per tier and watch all four theory objects and their
    Monte-Carlo counterparts at once: the overhang trajectory, the two cutoffs,
    the per-tier quality change, and the within-tier segregation that the quota
    creates even where tier quality barely moves.
    """)
    return


@app.cell
def _(mo, q):
    ui_rho = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=1.0,
                step=0.05,
                value=0.6 if k == 0 else 0.0,
                label=f"tier {k + 1}",
                show_value=True,
            )
            for k in range(q)
        ]
    )
    mo.vstack([mo.md(r"**Reserved fraction $\rho_k$ of each tier**"), ui_rho])
    return (ui_rho,)


@app.cell
def _(L, fluid, group_means, mc, mdl, np, paired, q, reps, ui_rho):
    rho_play = np.array(ui_rho.value, float)
    th_play = fluid(rho_play, mdl)
    batch_play = mc(mdl, np.vstack([np.zeros(q), rho_play]), reps)
    res_play = paired(batch_play, L)
    play_a, play_b, play_cut_a, play_cut_b = group_means(batch_play, 1)
    return play_a, play_b, play_cut_a, play_cut_b, res_play, rho_play, th_play


@app.cell
def _(mo, res_play, th_play):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="theory loss", value=f"{th_play['loss']:,.0f}σ"),
                    mo.stat(
                        label="simulated loss (±2 sem)",
                        value=f"{res_play['loss'][1]:,.0f} ± "
                        f"{2 * res_play['loss_sem'][1]:,.0f}σ",
                    ),
                    mo.stat(
                        label="quadratic approx",
                        value=f"{th_play['loss_quad']:,.0f}σ",
                    ),
                    mo.stat(label="wave reach", value=f"{th_play['reach']} tiers"),
                    mo.stat(
                        label="destroyed (expelled)",
                        value=f"{th_play['expelled']:,.0f}σ",
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                "⚠️ the A-pool runs dry in this configuration — the fluid limit "
                "no longer applies, the simulator falls back to open seats."
                if th_play["dry"]
                else ""
            ),
        ]
    )
    return


@app.cell
def _(
    np,
    play_a,
    play_b,
    play_cut_a,
    play_cut_b,
    plt,
    q,
    res_play,
    rho_play,
    th_play,
):
    fig_play, ax_play = plt.subplots(2, 2, figsize=(11.5, 7.0))
    _ks_play = np.arange(1, q + 1)

    ax_play[0, 0].step(_ks_play, rho_play, where="mid", color="tab:gray", lw=1.2)
    ax_play[0, 0].fill_between(
        _ks_play,
        0,
        rho_play,
        step="mid",
        color="tab:gray",
        alpha=0.25,
        label=r"$\rho_k$",
    )
    ax_play[0, 0].set_ylabel(r"reserved fraction $\rho_k$")
    _ax_twin = ax_play[0, 0].twinx()
    _ax_twin.plot(_ks_play, th_play["e"], "o-", color="tab:blue", ms=4)
    _ax_twin.set_ylabel(r"overhang $e_k$ (per capita)", color="tab:blue")
    _ax_twin.tick_params(axis="y", colors="tab:blue")
    _ax_twin.grid(False)
    ax_play[0, 0].set_xlabel("tier $k$")
    ax_play[0, 0].set_title("profile and the Lindley backlog it generates")
    ax_play[0, 0].legend(loc="upper right", fontsize=8)

    ax_play[0, 1].plot(
        _ks_play, th_play["u"], "-", color="tab:red", label="theory $u_k$"
    )
    ax_play[0, 1].plot(
        _ks_play, th_play["v"], "-", color="tab:blue", label="theory $v_k$"
    )
    ax_play[0, 1].plot(_ks_play, play_cut_a, "o", ms=4, color="tab:red", label="sim A")
    ax_play[0, 1].plot(_ks_play, play_cut_b, "s", ms=4, color="tab:blue", label="sim B")
    ax_play[0, 1].set_ylim(
        np.nanmin(np.r_[th_play["u"], play_cut_a]) - 0.2,
        np.nanmax(np.r_[th_play["v"][th_play["v"] < 5], play_cut_b]) + 0.2,
    )
    ax_play[0, 1].set_xlabel("tier $k$")
    ax_play[0, 1].set_ylabel(r"admission cutoff [$\sigma$]")
    ax_play[0, 1].set_title("cutoffs: A dug deeper, B squeezed up")
    ax_play[0, 1].legend(fontsize=8, ncols=2)

    ax_play[1, 0].bar(
        _ks_play,
        res_play["dq"][1],
        yerr=2 * res_play["dq_sem"][1],
        capsize=3,
        color="tab:red",
        alpha=0.6,
        label="Monte Carlo (±2 sem)",
    )
    ax_play[1, 0].plot(_ks_play, th_play["dQ"], "k_", ms=14, mew=2, label="theory")
    ax_play[1, 0].axhline(0, color="k", lw=0.8)
    ax_play[1, 0].set_xlabel("tier $k$")
    ax_play[1, 0].set_ylabel(r"tier quality change [$\sigma$, total]")
    ax_play[1, 0].set_title("per-tier quality change")
    ax_play[1, 0].legend(fontsize=8)

    ax_play[1, 1].plot(
        _ks_play,
        th_play["mean_a"] - th_play["mean_b"],
        "k_",
        ms=14,
        mew=2,
        label="theory (absent where a group's fluid intake is 0)",
    )
    ax_play[1, 1].plot(
        _ks_play, play_a - play_b, "o", ms=4, color="tab:purple", label="Monte Carlo"
    )
    ax_play[1, 1].axhline(0, color="k", lw=0.8)
    ax_play[1, 1].set_xlabel("tier $k$")
    ax_play[1, 1].set_ylabel(r"mean skill A $-$ mean skill B [$\sigma$]")
    ax_play[1, 1].set_title("within-tier segregation")
    ax_play[1, 1].legend(fontsize=7.5)

    fig_play.tight_layout()
    fig_play
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Q1 — the top tier: dead zone, swap price, convexity

    Sweep one tier's reservation and read off three regimes:

    1. **Dead zone** for $\rho \le \alpha$: the reserved seats go to A's who
       would have been admitted anyway. Exactly zero cost, not approximately.
    2. **Swap price** past it: each extra reserved seat kicks out the weakest
       admitted B (skill $v_k$) and pulls in the best remaining A (skill $u_k$),
       so the marginal cost is exactly $v_k - u_k$ — distribution-free.
    3. **Convex growth**: $u$ falls and $v$ rises as the gap widens, so total
       loss grows like $m^2/(2\alpha\beta\,\varphi(t_k) n)$ and then faster.
       The $1/\varphi(t_k)$ says elite tiers are exponentially fragile; the
       $1/(\alpha\beta)$ says small minorities are expensive per seat.
    """)
    return


@app.cell
def _(mo, q):
    ui_q1_tier = mo.ui.dropdown(
        options={f"tier {k + 1}": k for k in range(q)},
        value="tier 1",
        label="sweep which tier",
    )
    ui_q1_tier
    return (ui_q1_tier,)


@app.cell
def _(L, alpha, dry_note, fluid, h, mc, mdl, np, paired, q, reps, ui_q1_tier):
    q1_k = int(ui_q1_tier.value)
    q1_grid = np.linspace(0.0, 1.0, 161)
    q1_rows = np.zeros((q1_grid.size, q))
    q1_rows[:, q1_k] = q1_grid
    q1_th = [fluid(row, mdl) for row in q1_rows]
    q1_dq = np.array([t["dQ"][q1_k] for t in q1_th]) / h[q1_k]
    q1_quad = np.array([t["loss_quad"] for t in q1_th])
    q1_loss = np.array([t["loss"] for t in q1_th])
    # the instantaneous swap price diverges once the tier stops admitting B
    q1_open = np.array([t["p_b"][q1_k] for t in q1_th]) > 1e-12
    q1_gap = np.where(
        q1_open, np.array([t["v"][q1_k] - t["u"][q1_k] for t in q1_th]), np.nan
    )
    q1_shut = float(q1_grid[q1_open][-1]) if (~q1_open).any() else None

    q1_mc_rho = np.round(np.linspace(0.0, 1.0, 21), 3)
    q1_mc_rows = np.zeros((q1_mc_rho.size, q))
    q1_mc_rows[:, q1_k] = q1_mc_rho
    q1_batch = mc(mdl, np.vstack([np.zeros(q), q1_mc_rows]), reps)
    q1_res = paired(q1_batch, L)
    q1_dq_mc = q1_res["dq"][1:, q1_k] / h[q1_k]
    q1_sem_mc = q1_res["dq_sem"][1:, q1_k] / h[q1_k]
    # cost of one more reserved seat, averaged over each sweep interval; theory
    # and Monte Carlo share this estimand exactly, and it stays finite on the
    # last bin where the instantaneous swap price diverges.  The interval cost
    # is differenced per replication first, so the common random numbers keep
    # their covariance and the sem describes the plotted quantity.
    q1_bin_seats = np.diff(q1_mc_rho) * h[q1_k]
    q1_marg_rep = -np.diff(q1_batch["sum"][1:, :, q1_k], axis=0) / q1_bin_seats[:, None]
    q1_marg_mc = q1_marg_rep.mean(axis=1)
    q1_marg_sem = q1_marg_rep.std(axis=1, ddof=1) / np.sqrt(q1_marg_rep.shape[1])
    q1_th_mc = [fluid(row, mdl)["dQ"][q1_k] for row in q1_mc_rows]
    q1_marg_th = -np.diff(q1_th_mc) / q1_bin_seats
    q1_marg_x = 0.5 * (q1_mc_rho[1:] + q1_mc_rho[:-1])
    q1_dead = alpha
    q1_dry = dry_note(q1_th)
    return (
        q1_dead,
        q1_dq,
        q1_dq_mc,
        q1_dry,
        q1_gap,
        q1_grid,
        q1_k,
        q1_loss,
        q1_marg_mc,
        q1_marg_sem,
        q1_marg_th,
        q1_marg_x,
        q1_mc_rho,
        q1_quad,
        q1_sem_mc,
        q1_shut,
    )


@app.cell
def _(
    np,
    plt,
    q1_dead,
    q1_dq,
    q1_dq_mc,
    q1_dry,
    q1_gap,
    q1_grid,
    q1_k,
    q1_loss,
    q1_marg_mc,
    q1_marg_sem,
    q1_marg_th,
    q1_marg_x,
    q1_mc_rho,
    q1_quad,
    q1_sem_mc,
    q1_shut,
):
    fig_q1, ax_q1 = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax_q1[0].plot(q1_grid, q1_dq, "k-", lw=1.5, label="fluid limit")
    ax_q1[0].errorbar(
        q1_mc_rho,
        q1_dq_mc,
        yerr=2 * q1_sem_mc,
        fmt="o",
        ms=4,
        color="tab:red",
        capsize=2,
        label="Monte Carlo (±2 sem)",
    )
    ax_q1[0].axvline(q1_dead, color="tab:blue", ls="--", lw=1)
    ax_q1[0].text(
        q1_dead + 0.02,
        min(q1_dq.min(), q1_dq_mc.min()) * 0.9,
        r"natural share $\alpha$",
        color="tab:blue",
        rotation=90,
        va="bottom",
        fontsize=8,
    )
    ax_q1[0].set_xlabel(r"reserved fraction $\rho$ of the swept tier")
    ax_q1[0].set_ylabel(r"tier quality change per seat [$\sigma$]")
    ax_q1[0].set_title(f"tier {q1_k + 1}: dead zone, then convex loss")
    ax_q1[0].legend(loc="lower left", fontsize=8)

    ax_q1[1].step(
        q1_mc_rho[1:],
        q1_marg_th,
        where="pre",
        color="k",
        lw=1.4,
        label="theory, averaged over each bin",
    )
    ax_q1[1].errorbar(
        q1_marg_x,
        q1_marg_mc,
        yerr=2 * q1_marg_sem,
        fmt="s",
        ms=4,
        color="tab:red",
        capsize=2,
        label="Monte Carlo, same bins (±2 sem)",
    )
    ax_q1[1].plot(
        q1_grid,
        q1_gap,
        ":",
        color="tab:blue",
        lw=1.3,
        label=r"instantaneous swap price $v_k-u_k$",
    )
    ax_q1[1].axvline(q1_dead, color="tab:blue", ls="--", lw=1)
    if q1_shut is not None:
        ax_q1[1].axvspan(q1_shut, 1.0, color="tab:gray", alpha=0.18)
        ax_q1[1].text(
            q1_shut - 0.03,
            0.42,
            "tier admits no B past here:\n$v_k$ diverges, bin averages stay finite",
            transform=ax_q1[1].get_xaxis_transform(),
            ha="right",
            va="center",
            fontsize=7,
            color="dimgray",
        )
    ax_q1[1].set_ylim(0.0, max(q1_marg_th.max(), q1_marg_mc.max()) * 1.35)
    ax_q1[1].set_xlabel(r"reserved fraction $\rho$")
    ax_q1[1].set_ylabel(r"cost per extra reserved seat [$\sigma$]")
    ax_q1[1].set_title("the swap price (distribution-free)")
    ax_q1[1].legend(loc="upper left", fontsize=7.5)

    ax_q1[2].plot(q1_grid, q1_loss, "k-", lw=1.5, label="exact fluid loss")
    ax_q1[2].plot(
        q1_grid, q1_quad, "--", color="tab:green", lw=1.3, label=r"quadratic in $e$"
    )
    ax_q1[2].set_xlabel(r"reserved fraction $\rho$")
    ax_q1[2].set_ylabel(r"weighted value loss [$\sigma$]")
    ax_q1[2].set_title("small-overhang expansion vs. exact")
    ax_q1[2].legend(loc="upper left", fontsize=8)
    with np.errstate(divide="ignore", invalid="ignore"):
        _ratio = np.where(q1_loss > 1e-9, q1_quad / q1_loss, np.nan)
    ax_q1[2].text(
        0.03,
        0.62,
        f"approx / exact at $\\rho=1$: {_ratio[-1]:.3f}",
        transform=ax_q1[2].transAxes,
        fontsize=8,
    )
    if q1_dry:
        ax_q1[0].text(
            0.03,
            0.06,
            "A-pool runs dry in part of this sweep",
            transform=ax_q1[0].transAxes,
            fontsize=7.5,
            color="tab:red",
        )

    fig_q1.tight_layout()
    fig_q1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Q2 — the damage travels as a wave, and mostly reshuffles

    Shock one tier: the overhang drains downward, each following fair tier
    absorbing $\alpha\tau_k$ of it, so the disturbance reaches
    $\lceil e/(\alpha\tau)\rceil$ tiers and dies. Tiers past the wave are
    *exactly* untouched. Inside the wave the shocked tier loses and the next
    tiers **gain** — they inherit displaced B's stronger than their fair intake.

    Raw skill is only destroyed if the wave survives past the last tier
    ($D_q > 0$): then above-the-bar B's are expelled from the system. Hence

    $$\sum_k \Delta Q_k = -n\,D_q,$$

    which the panel below checks numerically against the simulator.
    """)
    return


@app.cell
def _(mo, q):
    ui_q2_tier = mo.ui.dropdown(
        options={f"tier {k + 1}": k for k in range(q)},
        value="tier 1",
        label="shock which tier",
    )
    ui_q2_rho = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=1.0,
        label=r"reserved fraction $\rho$",
        show_value=True,
    )
    mo.hstack([ui_q2_tier, ui_q2_rho], widths="equal")
    return ui_q2_rho, ui_q2_tier


@app.cell
def _(L, fluid, mc, mdl, np, paired, q, reps, ui_q2_rho, ui_q2_tier):
    q2_k = int(ui_q2_tier.value)
    q2_rho = np.zeros(q)
    q2_rho[q2_k] = float(ui_q2_rho.value)
    q2_th = fluid(q2_rho, mdl)
    q2_batch = mc(mdl, np.vstack([np.zeros(q), q2_rho]), reps)
    q2_res = paired(q2_batch, L)
    q2_sum_mc = q2_res["dq"][1].sum()
    q2_sum_sem = (q2_batch["sum"][1] - q2_batch["sum"][0]).sum(axis=1).std(
        ddof=1
    ) / np.sqrt(q2_batch["sum"].shape[1])
    return q2_k, q2_res, q2_sum_mc, q2_sum_sem, q2_th


@app.cell
def _(dry_note, mo, q2_k, q2_sum_mc, q2_sum_sem, q2_th):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="shocked tier", value=f"{q2_k + 1}"),
                    mo.stat(label="wave reach", value=f"{q2_th['reach']} tiers"),
                    mo.stat(
                        label="Σ tier changes, sim",
                        value=f"{q2_sum_mc:,.0f} ± {2 * q2_sum_sem:,.0f}σ",
                    ),
                    mo.stat(
                        label="theory: $-n D_q$ (expelled)",
                        value=f"{-q2_th['expelled']:,.0f}σ",
                    ),
                ],
                widths="equal",
            ),
            mo.md(dry_note([q2_th])),
        ]
    )
    return


@app.cell
def _(np, plt, q, q2_k, q2_res, q2_th):
    fig_q2, ax_q2 = plt.subplots(1, 2, figsize=(11.5, 4.0))
    _ks_q2 = np.arange(1, q + 1)

    ax_q2[0].bar(
        _ks_q2,
        q2_res["dq"][1],
        yerr=2 * q2_res["dq_sem"][1],
        capsize=3,
        color="tab:red",
        alpha=0.6,
        label="Monte Carlo (±2 sem)",
    )
    ax_q2[0].plot(_ks_q2, q2_th["dQ"], "k_", ms=16, mew=2, label="theory")
    ax_q2[0].axhline(0, color="k", lw=0.8)
    ax_q2[0].set_xlabel("tier $k$")
    ax_q2[0].set_ylabel(r"tier quality change [$\sigma$, total]")
    ax_q2[0].set_title(f"shock at tier {q2_k + 1}: loss up top, gains below")
    _ax_twin_q2 = ax_q2[0].twinx()
    _ax_twin_q2.plot(_ks_q2, q2_th["e"], "o--", color="tab:blue", ms=4, lw=1)
    _ax_twin_q2.set_ylabel(r"overhang $e_k$", color="tab:blue")
    _ax_twin_q2.tick_params(axis="y", colors="tab:blue")
    _ax_twin_q2.grid(False)
    ax_q2[0].legend(loc="center right", fontsize=8)

    _q2_cum_mc = np.cumsum(q2_res["dq"][1])
    ax_q2[1].plot(_ks_q2, np.cumsum(q2_th["dQ"]), "k-", lw=1.4)
    ax_q2[1].plot(_ks_q2, _q2_cum_mc, "o", ms=4, color="tab:purple")
    ax_q2[1].axhline(0, color="k", lw=0.8)
    ax_q2[1].set_xlabel("tier $k$")
    ax_q2[1].set_ylabel(r"cumulative change $-nD_k$ [$\sigma$]")
    ax_q2[1].set_title("cumulative loss: returns to 0 unless people are expelled")
    ax_q2[1].legend(["theory", "Monte Carlo"], fontsize=8, loc="lower right")

    fig_q2.tight_layout()
    fig_q2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Q3 — two shocks: additive when apart, superadditive when they overlap

    Two reservations interact through exactly one channel: whether the upper
    wave is still alive when it reaches the lower tier. Disjoint waves ⇒ losses
    add exactly. Overlapping waves ⇒ the overhangs stack, and since the loss is
    convex in $e$ there is a positive cross term; worse, the upstream overhang
    eats the downstream tier's absorption capacity, so a reservation that would
    have been free alone starts to bite.

    The surface below is theory over all placements of the second shock; the two
    marked cases are verified against the simulator.
    """)
    return


@app.cell
def _(mo):
    ui_q3_rho = mo.ui.slider(
        start=0.35,
        stop=1.0,
        step=0.05,
        value=0.6,
        label=r"both shocks use $\rho$",
        show_value=True,
    )
    ui_q3_rho
    return (ui_q3_rho,)


@app.cell
def _(L, dry_note, fluid, mc, mdl, np, paired, q, reps, ui_q3_rho):
    q3_rho = float(ui_q3_rho.value)
    q3_rho_grid = np.round(np.linspace(0.35, 1.0, 14), 3)
    q3_k2s = np.arange(1, q)

    def q3_profile(idx):
        row = np.zeros(q)
        for k, val in idx:
            row[k] = val
        return row

    # cells whose profiles exhaust the A-pool are masked: the fluid limit that
    # the ratio is computed from does not describe them
    q3_ratio = np.zeros((q3_rho_grid.size, q3_k2s.size))
    q3_masked = 0
    for _i, _rr in enumerate(q3_rho_grid):
        _t1 = fluid(q3_profile([(0, _rr)]), mdl)
        for _j, _k2 in enumerate(q3_k2s):
            _t2 = fluid(q3_profile([(_k2, _rr)]), mdl)
            _tj = fluid(q3_profile([(0, _rr), (_k2, _rr)]), mdl)
            _parts = _t1["loss"] + _t2["loss"]
            _bad = _t1["dry"] or _t2["dry"] or _tj["dry"] or _parts <= 1e-9
            q3_masked += int(_bad)
            q3_ratio[_i, _j] = np.nan if _bad else _tj["loss"] / _parts

    q3_near = 1
    q3_far = min(q - 1, max(2, int(np.ceil(q / 2))))
    q3_rows = np.vstack(
        [
            np.zeros(q),
            q3_profile([(0, q3_rho)]),
            q3_profile([(q3_near, q3_rho)]),
            q3_profile([(0, q3_rho), (q3_near, q3_rho)]),
            q3_profile([(q3_far, q3_rho)]),
            q3_profile([(0, q3_rho), (q3_far, q3_rho)]),
        ]
    )
    q3_batch = mc(mdl, q3_rows, reps)
    q3_res = paired(q3_batch, L)
    q3_ths = [fluid(row, mdl) for row in q3_rows]
    q3_th_loss = np.array([t["loss"] for t in q3_ths])
    q3_dry = dry_note(q3_ths)
    q3_pairs = [
        ("adjacent\n" + f"tiers 1 & {q3_near + 1}", 1, 2, 3),
        ("separated\n" + f"tiers 1 & {q3_far + 1}", 1, 4, 5),
    ]
    return (
        q3_dry,
        q3_far,
        q3_k2s,
        q3_masked,
        q3_near,
        q3_pairs,
        q3_ratio,
        q3_res,
        q3_rho,
        q3_rho_grid,
        q3_th_loss,
    )


@app.cell
def _(
    np,
    plt,
    q3_dry,
    q3_far,
    q3_k2s,
    q3_masked,
    q3_near,
    q3_pairs,
    q3_ratio,
    q3_res,
    q3_rho,
    q3_rho_grid,
    q3_th_loss,
):
    fig_q3, ax_q3 = plt.subplots(1, 2, figsize=(11.5, 4.0))

    _mesh = ax_q3[0].pcolormesh(
        q3_k2s + 1, q3_rho_grid, q3_ratio, shading="nearest", cmap="magma_r"
    )
    fig_q3.colorbar(_mesh, ax=ax_q3[0], label="joint loss / sum of parts")
    for _k2_marked in (q3_near, q3_far):
        ax_q3[0].plot(
            [_k2_marked + 1], [q3_rho], "o", mfc="none", mec="tab:cyan", ms=9, mew=1.6
        )
    ax_q3[0].set_xlabel("tier of the second shock")
    ax_q3[0].set_ylabel(r"$\rho$ of both shocks")
    ax_q3[0].set_title("superadditivity ratio (theory)\n1.0 = exactly additive")
    if q3_masked:
        ax_q3[0].text(
            0.5,
            -0.28,
            f"{q3_masked} cells blank: A-pool exhausted, fluid limit not valid",
            transform=ax_q3[0].transAxes,
            ha="center",
            fontsize=7.5,
            color="tab:red",
        )
    ax_q3[0].grid(False)

    _x_q3 = np.arange(len(q3_pairs))
    _width_q3 = 0.35
    _parts_q3 = [q3_res["loss"][a] + q3_res["loss"][b] for _, a, b, _ in q3_pairs]
    _joints_q3 = [q3_res["loss"][c] for _, _, _, c in q3_pairs]
    ax_q3[1].bar(
        _x_q3 - _width_q3 / 2,
        _parts_q3,
        _width_q3,
        color="tab:gray",
        label="sum of separate shocks",
    )
    ax_q3[1].bar(
        _x_q3 + _width_q3 / 2,
        _joints_q3,
        _width_q3,
        color="tab:red",
        alpha=0.75,
        label="both together",
    )
    ax_q3[1].plot(
        np.r_[_x_q3 - _width_q3 / 2, _x_q3 + _width_q3 / 2],
        [q3_th_loss[a] + q3_th_loss[b] for _, a, b, _ in q3_pairs]
        + [q3_th_loss[c] for _, _, _, c in q3_pairs],
        "k_",
        ms=16,
        mew=2,
        label="theory",
    )
    for _xi, (_pp, _jj) in enumerate(zip(_parts_q3, _joints_q3, strict=True)):
        ax_q3[1].text(
            _xi + _width_q3 / 2,
            _jj,
            f"×{_jj / _pp:.2f}" if _pp > 1e-9 else "n/a",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_q3[1].set_xticks(_x_q3, [label for label, *_ in q3_pairs])
    ax_q3[1].set_ylabel(r"weighted value loss [$\sigma$]")
    ax_q3[1].set_title("Monte Carlo check of the two marked cases")
    ax_q3[1].legend(fontsize=8)
    ax_q3[1].text(
        0.02,
        0.02,
        f"tier-1 wave reaches tier {q3_near + 1}; second case sits at tier "
        f"{q3_far + 1}",
        transform=ax_q3[1].transAxes,
        fontsize=7.5,
        va="bottom",
        color="dimgray",
    )
    if q3_dry:
        ax_q3[1].text(
            0.02,
            0.12,
            "A-pool runs dry in some profile here",
            transform=ax_q3[1].transAxes,
            fontsize=7.5,
            color="tab:red",
        )

    fig_q3.tight_layout()
    fig_q3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Q4 — the total bill in closed form

    Abel summation over tiers gives the exact loss, and the small-overhang
    expansion gives the readable version

    $$\text{Loss} \approx \frac{n}{2\alpha\beta}
      \sum_k (L_k - L_{k+1})\,\frac{e_k^2}{\varphi(t_k)} ,$$

    the weighted "energy" of the overhang trajectory. Only value *decrements*
    appear (shuffling between equally valued tiers is free), the last decrement
    prices people expelled from the system, and the mean skill never appears.

    Validation over randomly drawn reservation profiles, every configuration
    scored against the same skill draws. Common random numbers shrink the
    variance of the paired difference against the fair baseline but do not
    remove it, so the residual panel plots each relative gap with its ±2 sem
    band: residuals are finite-$n$ and seat-rounding effects *plus* the
    remaining Monte-Carlo error, and the bars say which is which.
    """)
    return


@app.cell
def _(L, fluid, mc, mdl, np, paired, q, reps, seed):
    q4_rng = np.random.default_rng(seed + 101)
    q4_rows = [np.zeros(q)]
    for _ in range(18):
        _row = np.zeros(q)
        _support = q4_rng.choice(q, size=int(q4_rng.integers(1, 4)), replace=False)
        _row[_support] = np.round(q4_rng.uniform(0.2, 1.0, _support.size), 2)
        q4_rows.append(_row)
    q4_rows = np.vstack(q4_rows)
    q4_batch = mc(mdl, q4_rows, reps)
    q4_res = paired(q4_batch, L)
    q4_th = [fluid(row, mdl) for row in q4_rows]
    q4_exact = np.array([t["loss"] for t in q4_th])
    q4_quad = np.array([t["loss_quad"] for t in q4_th])
    q4_sim = q4_res["loss"]
    q4_big = q4_exact > 1.0
    q4_relerr = (q4_sim[q4_big] - q4_exact[q4_big]) / q4_exact[q4_big]
    return q4_exact, q4_quad, q4_relerr, q4_res, q4_sim, q4_th


@app.cell
def _(dry_note, mo, np, q4_relerr, q4_th):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="configurations", value=f"{q4_relerr.size + 1}"),
                    mo.stat(
                        label="median |theory − sim| / theory",
                        value=f"{np.median(np.abs(q4_relerr)):.2%}",
                    ),
                    mo.stat(
                        label="worst |theory − sim| / theory",
                        value=f"{np.abs(q4_relerr).max():.2%}",
                    ),
                ],
                widths="equal",
            ),
            mo.md(dry_note(q4_th)),
        ]
    )
    return


@app.cell
def _(np, plt, q4_exact, q4_quad, q4_res, q4_sim):
    fig_q4, ax_q4 = plt.subplots(1, 3, figsize=(13.5, 3.9))

    _hi_q4 = max(q4_exact.max(), q4_sim.max()) * 1.05 + 1.0
    ax_q4[0].plot([0, _hi_q4], [0, _hi_q4], "k-", lw=1)
    ax_q4[0].errorbar(
        q4_exact,
        q4_sim,
        yerr=2 * q4_res["loss_sem"],
        fmt="o",
        ms=4,
        color="tab:purple",
        capsize=2,
    )
    ax_q4[0].set_xlabel(r"closed-form loss [$\sigma$]")
    ax_q4[0].set_ylabel(r"simulated loss [$\sigma$]")
    ax_q4[0].set_title("exact formula vs. exact mechanism")

    ax_q4[1].plot([0, _hi_q4], [0, _hi_q4], "k-", lw=1)
    ax_q4[1].plot(q4_exact, q4_quad, "o", ms=4, color="tab:green")
    ax_q4[1].set_xlabel(r"exact fluid loss [$\sigma$]")
    ax_q4[1].set_ylabel(r"quadratic approximation [$\sigma$]")
    ax_q4[1].set_title("where the small-$e$ expansion breaks")

    _order = np.argsort(q4_exact)
    _resid = np.where(
        q4_exact[_order] > 1.0,
        (q4_sim[_order] - q4_exact[_order]) / np.maximum(q4_exact[_order], 1.0),
        np.nan,
    )
    ax_q4[2].axhline(0, color="k", lw=1)
    ax_q4[2].errorbar(
        q4_exact[_order],
        _resid,
        yerr=2 * q4_res["loss_sem"][_order] / np.maximum(q4_exact[_order], 1.0),
        fmt="o",
        ms=4,
        color="tab:red",
        capsize=2,
    )
    ax_q4[2].set_xscale("log")
    ax_q4[2].set_xlabel(r"closed-form loss [$\sigma$, log]")
    ax_q4[2].set_ylabel("relative residual")
    ax_q4[2].set_title("residuals against Monte-Carlo noise")

    fig_q4.tight_layout()
    fig_q4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Q5 — where should a fixed quota budget go?

    Reserving up to $\alpha h_k$ in a tier is free, so a budget measured in raw
    reserved seats is not a budget at all: spread thinly enough, any amount up
    to $\alpha \sum_k h_k$ lands inside the dead zone and costs nothing. The
    quantity that actually buys displacement is the **excess**

    $$E = \sum_k \max(f_k - \alpha h_k,\ 0),$$

    which is also what makes the report's two placements comparable (all at
    tier 1 vs. spread over tiers 1, 3, 5, 7 — same excess, very different bill).
    So write $\rho_k = \alpha + x_k$ with $x_k \in [0, 1-\alpha]$ and minimize
    the exact fluid loss subject to $\sum_k x_k h_k = E$: a linear constraint,
    SLSQP, multistarted from the three hand-made policies it is compared
    against.

    What the optimizer trades off is not "avoid the bottom": with declining
    $L_k$, pushing overhang out of the system costs only the last decrement
    $L_q - 0$, which can be cheaper than displacing people in a selective tier
    where $\varphi(t_k)$ is small. So the prediction is a balance of three
    terms — the density $\varphi(t_k)$ at the tier, the value decrement
    $\lambda_k$ each unit of overhang crosses, and the squared overhang the
    placement leaves behind. With the default linear $L_k$ the optimum is the
    bottom-loaded policy itself: the search confirms the greedy placement
    rather than beating it, and the ×1.00 row in the table is that statement.
    Switch $L_k$ in §1 (geometric or elite-only values make the last decrement
    negligible, flat values make interior moves free) and the ranking moves.

    One invariant to read off the table: while every tier keeps
    $\rho_k \ge \alpha$, nothing drains, so the overhang leaving the bottom is
    $E/n$ for *every* placement and the expelled column is identical. The four
    policies differ purely in the interior path the overhang takes — which is
    exactly the $\sum_k \lambda_k e_k^2 / \varphi(t_k)$ energy the formula
    prices.
    """)
    return


@app.cell
def _(mo):
    ui_q5_budget = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=0.4,
        label="excess seats / displaceable seats",
        show_value=True,
    )
    ui_q5_budget
    return (ui_q5_budget,)


@app.cell
def _(
    L,
    alpha,
    dry_note,
    fluid,
    h,
    mc,
    mdl,
    minimize,
    np,
    paired,
    q,
    reps,
    ui_q5_budget,
):
    q5_frac = float(ui_q5_budget.value)
    q5_room = (1.0 - alpha) * h.sum()  # seats that can be moved at all
    q5_excess = q5_frac * q5_room

    def q5_fill(_order):
        """Greedily saturate tiers in the given _order until the excess is spent."""
        x = np.zeros(q)
        left = q5_excess
        for k in _order:
            take = min((1.0 - alpha) * h[k], left)
            x[k] = take / h[k]
            left -= take
            if left <= 0:
                break
        return x

    q5_x = {
        "uniform": np.full(q, q5_excess / h.sum()),
        "top-loaded": q5_fill(range(q)),
        "bottom-loaded": q5_fill(range(q - 1, -1, -1)),
    }

    def q5_obj(x):
        return fluid(alpha + np.clip(x, 0.0, 1.0 - alpha), mdl)["loss"]

    q5_cons = ({"type": "eq", "fun": lambda x: float((x * h).sum() - q5_excess)},)

    def q5_feasible(x):
        """Inside the box and spending exactly the excess budget."""
        return bool(
            np.all(x >= -1e-9)
            and np.all(x <= 1.0 - alpha + 1e-9)
            and abs(float((x * h).sum()) - q5_excess) <= 1e-6 * max(q5_excess, 1.0)
        )

    # multistart; the hand-made policies are feasible by construction, so they
    # stay in the candidate pool and a non-converged SLSQP run cannot make the
    # "optimized" row worse than the best policy it is compared against
    q5_cands = list(q5_x.values())
    for _start in list(q5_x.values()):
        _got = minimize(
            q5_obj,
            _start,
            method="SLSQP",
            bounds=[(0.0, 1.0 - alpha)] * q,
            constraints=q5_cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        _cand = np.clip(_got.x, 0.0, 1.0 - alpha)
        if q5_feasible(_cand):
            q5_cands.append(_cand)
    # Near-ties resolve to the earliest candidate, i.e. to a hand-made policy
    # rather than to a numerically jittered version of it, and excess below half
    # a seat is snapped away (the mechanism rounds seats anyway) so the reported
    # profile and its wave reach are the ones actually simulated.
    _q5_vals = np.array([q5_obj(cand) for cand in q5_cands])
    _q5_tol = _q5_vals.min() * (1.0 + 1e-6) + 1e-9
    _q5_pick = q5_cands[int(np.argmax(_q5_vals <= _q5_tol))]
    q5_x["optimized"] = np.where(_q5_pick * h >= 0.5, _q5_pick, 0.0)

    q5_names = list(q5_x)
    q5_policies = {name: alpha + q5_x[name] for name in q5_names}
    q5_rows = np.vstack([np.zeros(q)] + [q5_policies[name] for name in q5_names])
    q5_batch = mc(mdl, q5_rows, reps)
    q5_res = paired(q5_batch, L)
    q5_ths = [fluid(q5_policies[name], mdl) for name in q5_names]
    q5_th = np.array([t["loss"] for t in q5_ths])
    q5_reach = [t["reach"] for t in q5_ths]
    q5_expelled = np.array([t["expelled"] for t in q5_ths])
    q5_dry = dry_note(q5_ths)
    return (
        q5_dry,
        q5_excess,
        q5_expelled,
        q5_names,
        q5_policies,
        q5_reach,
        q5_res,
        q5_room,
        q5_th,
        q5_x,
    )


@app.cell
def _(
    alpha,
    fmt_profile,
    h,
    mo,
    q5_dry,
    q5_excess,
    q5_expelled,
    q5_names,
    q5_policies,
    q5_res,
    q5_room,
    q5_th,
    q5_x,
):
    q5_ref = max(q5_th.min(), 1e-9)
    mo.vstack(
        [
            mo.md(
                f"Excess budget **{q5_excess:,.0f} seats** moved past the natural "
                f"share, out of **{q5_room:,.0f}** displaceable seats "
                f"($(1-\\alpha)\\sum_k h_k$). Every policy below also holds the "
                f"free allowance $\\alpha\\sum_k h_k = "
                f"{alpha * h.sum():,.0f}$ seats, so all four reserve the same "
                f"{alpha * h.sum() + q5_excess:,.0f} seats in total and differ "
                f"only in where the excess sits."
            ),
            mo.md(
                "| policy | $\\rho_k$ profile | excess seats | theory loss | "
                "simulated loss | expelled | vs. best |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                + "\n".join(
                    f"| {name} | {fmt_profile(q5_policies[name])} | "
                    f"{(q5_x[name] * h).sum():,.0f} | {q5_th[idx]:,.0f}σ | "
                    f"{q5_res['loss'][idx + 1]:,.0f} ± "
                    f"{2 * q5_res['loss_sem'][idx + 1]:,.0f}σ | "
                    f"{q5_expelled[idx]:,.0f}σ | "
                    f"×{q5_th[idx] / q5_ref:,.2f} |"
                    for idx, name in enumerate(q5_names)
                )
            ),
            mo.md(q5_dry),
        ]
    )
    return


@app.cell
def _(
    alpha,
    np,
    plt,
    q,
    q5_expelled,
    q5_names,
    q5_policies,
    q5_reach,
    q5_res,
    q5_th,
):
    fig_q5, ax_q5 = plt.subplots(1, 2, figsize=(11.5, 4.0))
    _ks_q5 = np.arange(1, q + 1)
    _colors_q5 = ["tab:blue", "tab:red", "tab:orange", "tab:green"]

    for _name, _color in zip(q5_names, _colors_q5, strict=True):
        ax_q5[0].step(
            _ks_q5, q5_policies[_name], where="mid", lw=1.6, color=_color, label=_name
        )
    ax_q5[0].axhline(alpha, color="k", ls="--", lw=1)
    ax_q5[0].text(
        q * 0.55, alpha + 0.02, r"free up to $\alpha$", fontsize=8, color="dimgray"
    )
    ax_q5[0].set_xlabel("tier $k$")
    ax_q5[0].set_ylabel(r"reserved fraction $\rho_k$")
    ax_q5[0].set_title("same excess budget, four placements")
    ax_q5[0].legend(fontsize=8)

    _x_q5 = np.arange(len(q5_names))
    ax_q5[1].set_ylim(0.0, max(q5_res["loss"][1:].max(), q5_th.max()) * 1.22)
    ax_q5[1].bar(
        _x_q5,
        q5_res["loss"][1:],
        yerr=2 * q5_res["loss_sem"][1:],
        capsize=3,
        color=_colors_q5,
        alpha=0.7,
        label="Monte Carlo (±2 sem)",
    )
    ax_q5[1].plot(_x_q5, q5_th, "k_", ms=18, mew=2, label="theory")
    for _xi, (_reach, _exp) in enumerate(zip(q5_reach, q5_expelled, strict=True)):
        ax_q5[1].text(
            _xi,
            max(q5_res["loss"][1 + _xi], 0) * 1.02,
            f"reach {_reach}\nexpelled {_exp:,.0f}σ",
            ha="center",
            va="bottom",
            fontsize=7,
            color="dimgray",
        )
    ax_q5[1].set_xticks(_x_q5, q5_names)
    ax_q5[1].set_ylabel(r"weighted value loss [$\sigma$]")
    ax_q5[1].set_title(r"convexity, $\varphi(t_k)$ and $\lambda_k$ decide")
    ax_q5[1].legend(fontsize=8)

    fig_q5.tight_layout()
    fig_q5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Beyond the Gaussian

    Which parts of the theory need normality? The structural results — dead
    zone, Lindley recursion, wave locality, swap price $v-u$, convexity — are
    distribution-free; only the *closed forms* use
    $\int_u^\infty z\varphi(z)\,dz = \varphi(u)$.

    So the fluid limit is recomputed with the general top-mass functional
    $M(p) = \mathbb{E}[Z\,\mathbf{1}\{Z \ge F^{-1}(1-p)\}]$ (numerical
    integration) for five standardized skill distributions, and compared both to
    the simulator on that distribution and to the Gaussian formula applied
    blindly.
    """)
    return


@app.cell
def _(DISTS, mo, np, overhang):
    @mo.cache
    def mass_general(dist_name, p_tuple):
        dist = DISTS[dist_name]
        out = np.empty(len(p_tuple))
        for i, p in enumerate(p_tuple):
            if p <= 0.0:
                out[i] = 0.0
            elif p >= 1.0:
                out[i] = float(dist.mean())
            else:
                out[i] = float(dist.expect(lambda z: z, lb=dist.ppf(1.0 - p)))
        return out

    def fluid_general(dist_name, rho, mdl):
        """Same fluid limit, no normality: M(p) evaluated numerically."""
        e = overhang(np.asarray(rho, float), mdl)
        p_a = np.clip(mdl["T"] + e / mdl["alpha"], 0.0, 1.0)
        p_b = np.clip(mdl["T"] - e / mdl["beta"], 0.0, 1.0)
        fair = mass_general(dist_name, tuple(mdl["T"]))
        W = mdl["alpha"] * mass_general(dist_name, tuple(p_a)) + mdl[
            "beta"
        ] * mass_general(dist_name, tuple(p_b))
        D = fair - W
        return {
            "e": e,
            "D": D,
            "loss": float(mdl["lam"] @ D) * mdl["n"],
            "dQ": mdl["n"] * (np.r_[0.0, D[:-1]] - D),
        }

    return (fluid_general,)


@app.cell
def _(
    DISTS,
    L,
    dry_note,
    fluid,
    fluid_general,
    h,
    mc,
    mdl,
    np,
    paired,
    q,
    reps_small,
):
    q6_rho = 0.8
    q6_step = 0.05
    q6_rows = np.zeros((3, q))
    q6_rows[1, 0] = q6_rho
    q6_rows[2, 0] = q6_rho + q6_step
    q6_seats = q6_step * h[0]
    q6_gauss = fluid(q6_rows[1], mdl)["loss"]
    q6_dry = dry_note([fluid(row, mdl) for row in q6_rows[1:]])
    q6_table = []
    for _name in DISTS:
        _batch = mc(mdl, q6_rows, reps_small, _name)
        _res = paired(_batch, L)
        # interval cost per seat, differenced per replication so the common
        # random numbers keep their covariance
        _per_rep = -np.diff(_batch["sum"][1:, :, 0], axis=0)[0] / q6_seats
        _th = [fluid_general(_name, row, mdl)["dQ"][0] for row in q6_rows[1:]]
        q6_table.append(
            {
                "name": _name,
                "sim": _res["loss"][1],
                "sem": _res["loss_sem"][1],
                "general": fluid_general(_name, q6_rows[1], mdl)["loss"],
                "gauss": q6_gauss,
                "marginal": float(_per_rep.mean()),
                "marginal_sem": float(_per_rep.std(ddof=1) / np.sqrt(_per_rep.size)),
                "marginal_th": -(_th[1] - _th[0]) / q6_seats,
            }
        )
    return q6_dry, q6_rho, q6_seats, q6_step, q6_table


@app.cell
def _(DIST_NOTES, mo, q6_dry, q6_rho, q6_seats, q6_step, q6_table):
    mo.vstack(
        [
            mo.md(
                f"Tier 1 reserved at $\\rho = {q6_rho:.2f}$, skill distributions "
                f"standardized to unit variance. The marginal columns are the "
                f"cost of the next {q6_seats:,.0f} reserved seats "
                f"($\\Delta\\rho = {q6_step:.2f}$) divided by that seat count — "
                f"theory and simulation compute the same interval average, no "
                f"derivative approximation."
            ),
            mo.md(
                "| distribution | | sim loss | general fluid theory | Gaussian "
                "formula | marginal/seat, sim | marginal/seat, theory |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                + "\n".join(
                    f"| {row['name']} | {DIST_NOTES[row['name']]} | "
                    f"{row['sim']:,.0f} ± {2 * row['sem']:,.0f}σ | "
                    f"{row['general']:,.0f}σ | {row['gauss']:,.0f}σ | "
                    f"{row['marginal']:.3f} ± {2 * row['marginal_sem']:.3f}σ | "
                    f"{row['marginal_th']:.3f}σ |"
                    for row in q6_table
                )
            ),
            mo.md(q6_dry),
        ]
    )
    return


@app.cell
def _(np, plt, q6_table):
    fig_q6, ax_q6 = plt.subplots(1, 2, figsize=(11.5, 4.0))
    _names_q6 = [row["name"] for row in q6_table]
    _x_q6 = np.arange(len(_names_q6))
    _w_q6 = 0.27

    ax_q6[0].bar(
        _x_q6 - _w_q6,
        [row["sim"] for row in q6_table],
        _w_q6,
        yerr=[2 * row["sem"] for row in q6_table],
        capsize=2,
        color="tab:red",
        alpha=0.8,
        label="simulation",
    )
    ax_q6[0].bar(
        _x_q6,
        [row["general"] for row in q6_table],
        _w_q6,
        color="tab:blue",
        alpha=0.8,
        label="general fluid theory",
    )
    ax_q6[0].bar(
        _x_q6 + _w_q6,
        [row["gauss"] for row in q6_table],
        _w_q6,
        color="tab:gray",
        alpha=0.8,
        label="Gaussian formula",
    )
    ax_q6[0].set_xticks(_x_q6, _names_q6, rotation=20, ha="right")
    ax_q6[0].set_ylabel(r"weighted value loss [$\sigma$]")
    ax_q6[0].set_title("closed form needs normality; the mechanism does not")
    ax_q6[0].legend(fontsize=8)

    _th_q6 = np.array([row["marginal_th"] for row in q6_table])
    _marg_q6 = np.array([row["marginal"] for row in q6_table])
    _sem_q6 = np.array([row["marginal_sem"] for row in q6_table])
    _lim_q6 = max(_th_q6.max(), _marg_q6.max()) * 1.15
    ax_q6[1].plot([0, _lim_q6], [0, _lim_q6], "k-", lw=1)
    ax_q6[1].errorbar(
        _th_q6, _marg_q6, yerr=2 * _sem_q6, fmt="o", ms=6, color="tab:purple", capsize=3
    )
    for _name, _sx, _my in zip(_names_q6, _th_q6, _marg_q6, strict=True):
        ax_q6[1].annotate(
            _name, (_sx, _my), textcoords="offset points", xytext=(6, -3), fontsize=7.5
        )
    ax_q6[1].set_xlabel(r"general fluid theory, cost per seat [$\sigma$]")
    ax_q6[1].set_ylabel(r"simulated cost per seat [$\sigma$]")
    ax_q6[1].set_title("interval-averaged swap price, theory vs. mechanism")

    fig_q6.tight_layout()
    fig_q6
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. How fast does the discrete mechanism become the fluid limit?

    The theory is a law of large numbers for _order statistics, so two things
    should happen as $n$ grows: the per-replication loss should concentrate, and
    its mean should sit on the closed form.

    The first is directly measurable — the **relative spread** of the per-capita
    loss across replications, which the quantile-process heuristic puts at
    $n^{-1/2}$. The second is a bias below the noise floor of any affordable
    run: with a replication mean, the sem shrinks only as $(nR)^{-1/2}$, so the
    honest statement is that no systematic deviation is detectable at any rung
    of this ladder. Both are plotted; the reference lines make the rate visible.
    """)
    return


@app.cell
def _(L, dry_note, fluid, mc, mdl, np, paired, q, reps_small, scaled_model):
    q7_rho = np.zeros(q)
    q7_rho[0] = 0.7
    q7_rows = np.vstack([np.zeros(q), q7_rho])
    q7_ns = np.array([6_250, 12_500, 25_000, 50_000, 100_000, 200_000])
    q7_reps = 2 * reps_small
    q7_signed = np.zeros(q7_ns.size)
    q7_err = np.zeros(q7_ns.size)
    q7_sd = np.zeros(q7_ns.size)
    q7_ths = []
    for _i, _n_i in enumerate(q7_ns):
        _sub = scaled_model(mdl, int(_n_i))
        _res = paired(mc(_sub, q7_rows, q7_reps), L)
        _th = fluid(q7_rho, _sub)
        q7_ths.append(_th)
        _theory_pc = _th["loss"] / _n_i
        _sim_pc = _res["loss"][1] / _n_i
        _sem_pc = _res["loss_sem"][1] / _n_i
        q7_signed[_i] = (_sim_pc - _theory_pc) / _theory_pc
        q7_err[_i] = 2 * _sem_pc / _theory_pc
        # sem = sd / sqrt(R): recover the single-replication spread
        q7_sd[_i] = _sem_pc * np.sqrt(q7_reps) / _theory_pc
    q7_dry = dry_note(q7_ths)
    q7_inside = int((np.abs(q7_signed) <= q7_err).sum())
    return q7_dry, q7_err, q7_inside, q7_ns, q7_reps, q7_sd, q7_signed


@app.cell
def _(np, plt, q7_dry, q7_err, q7_inside, q7_ns, q7_reps, q7_sd, q7_signed):
    fig_q7, ax_q7 = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax_q7[0].plot(
        q7_ns, q7_sd, "o-", ms=5, color="tab:blue", label="one-replication spread"
    )
    ax_q7[0].plot(
        q7_ns,
        q7_sd[0] * np.sqrt(q7_ns[0] / q7_ns),
        "k--",
        lw=1.2,
        label=r"$n^{-1/2}$ reference",
    )
    _slope = np.polyfit(np.log(q7_ns), np.log(q7_sd), 1)[0]
    ax_q7[0].set_xscale("log")
    ax_q7[0].set_yscale("log")
    ax_q7[0].set_xlabel("population $n$")
    ax_q7[0].set_ylabel("relative spread of the per-capita loss")
    ax_q7[0].set_title(f"concentration (fitted slope {_slope:+.2f})")
    ax_q7[0].legend(fontsize=8)

    ax_q7[1].axhline(0, color="k", lw=1)
    ax_q7[1].fill_between(
        q7_ns,
        -q7_err,
        q7_err,
        color="tab:purple",
        alpha=0.12,
        label="±2 sem band",
    )
    ax_q7[1].errorbar(
        q7_ns,
        q7_signed,
        yerr=q7_err,
        fmt="s",
        ms=5,
        color="tab:purple",
        capsize=3,
        label=f"(mean of {q7_reps} reps − theory) / theory",
    )
    ax_q7[1].set_xscale("log")
    ax_q7[1].set_xlabel("population $n$")
    ax_q7[1].set_ylabel("signed relative residual")
    ax_q7[1].set_title(
        f"no detectable bias: {q7_inside} of {q7_ns.size} rungs inside ±2 sem"
    )
    ax_q7[1].legend(fontsize=8)
    if q7_dry:
        ax_q7[1].text(
            0.03,
            0.06,
            "A-pool runs dry at some rung",
            transform=ax_q7[1].transAxes,
            fontsize=7.5,
            color="tab:red",
        )

    fig_q7.tight_layout()
    fig_q7
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **One state variable.** The whole mechanism compresses into the overhang
      $e_k$ and its Lindley recursion; unfairness behaves like queue backlog.
    - **Free below the natural share.** $\rho_k \le \alpha$ costs exactly zero —
      relabelled seats, same people.
    - **Marginal cost is a swap price.** Past that, one seat costs $v_k - u_k$;
      the panels confirm it for normal, logistic, uniform, exponential and
      lognormal skills alike.
    - **Elite tiers are fragile.** The $1/\varphi(t_k)$ factor makes the same
      quota vastly more expensive at the top; the $1/(\alpha\beta)$ factor makes
      it expensive for small minorities.
    - **Interior reservations mostly reshuffle.** Tier losses are matched by
      gains a tier or two down; skill is destroyed only when the wave reaches
      past the last tier, and that residue is exactly $n D_q$.
    - **Placement dominates volume.** Same budget, different placement changes
      the bill by an _order of magnitude; the optimizer's answer is "natural
      share everywhere, excess parked low", i.e. convexity plus the dead zone.
    - **Segregation is invisible to the value metric.** Inside the wave every
      reserved A sits below every displaced B, so tiers become internally
      stratified even where their total quality hardly moved.
    - **The fluid limit is not an approximation in practice.** Per-replication
      spread falls off as $n^{-1/2}$ (measured slope ≈ $-0.46$) and the
      replication mean sits inside $\pm 2$ sem of the closed form at every
      population size tested, so the formulas can be used as the answer, with
      the simulator kept for the regimes the theory disclaims.

    **Fine print.** Closed forms assume normal skills and an A-pool that never
    runs dry (flagged when it does); the value metric $\sum_k L_k \times$ skill
    is deliberately narrow, so read the loss as the price tag of a policy's
    goals, not a verdict on them.
    """)
    return


if __name__ == "__main__":
    app.run()
