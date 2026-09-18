"""
Unfair tiered admissions: fluid-limit theory vs exact Monte Carlo.

Model: n people, fraction alpha of kind A, beta = 1-alpha of kind B,
skills iid N(0,1).  q tiers processed best-to-worst; tier k has h_k seats,
f_k reserved for A, g_k = h_k - f_k open.  Reserved seats take the best
remaining A's; open seats take the best remaining people of any kind.

Theory objects (per-capita, sigma units):
  overhang  e_k = max(e_{k-1} + f_k/n - alpha*tau_k, 0)      (Lindley)
  cutoffs   u_k = PPF(1 - T_k - e_k/alpha),  v_k = PPF(1 - T_k + e_k/beta)
  W_k = alpha*phi(u_k) + beta*phi(v_k)   (skill mass admitted through tier k)
  D_k(e_k) = phi(t_k) - W_k              (cumulative loss at depth k)
  per-tier quality change  dQ_k = D_{k-1} - D_k
  weighted loss = sum_k (L_k - L_{k+1}) D_k
"""

import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 200_000
alpha = 0.30
beta = 1 - alpha
a, b = int(alpha * n), n - int(alpha * n)
q = 8
h = np.full(q, int(0.06 * n))  # 12000 seats per tier
tau = h / n
T = np.cumsum(tau)
L = np.arange(q, 0, -1, dtype=float)  # tier values 8,7,...,1
lam = L - np.r_[L[1:], 0.0]  # value decrements (all 1 here)
tf = norm.ppf(1 - T)  # fair cumulative cutoffs t_k
pdf_tf = norm.pdf(tf)


# ---------------------------------------------------------------- theory ----
def overhang(f):
    e, cur = np.zeros(q), 0.0
    for k in range(q):
        cur = max(cur + f[k] / n - alpha * tau[k], 0.0)
        e[k] = cur
    return e


def Wvec(e):
    u = norm.ppf(1 - T - e / alpha)
    va = 1 - T + e / beta
    pv = np.where(va >= 1 - 1e-12, 0.0, norm.pdf(norm.ppf(np.minimum(va, 1 - 1e-12))))
    return alpha * norm.pdf(u) + beta * pv


def theory(f):
    e = overhang(f)
    D = pdf_tf - Wvec(e)
    dQ = np.r_[0.0, D[:-1]] - D
    return e, D, dQ, float(lam @ D)


def D1_of(e):  # tier-1 cumulative loss, scalar/array e
    e = np.asarray(e, float)
    u = norm.ppf(1 - tau[0] - e / alpha)
    va = 1 - tau[0] + e / beta
    pv = np.where(va >= 1 - 1e-12, 0.0, norm.pdf(norm.ppf(np.minimum(va, 1 - 1e-12))))
    return pdf_tf[0] - (alpha * norm.pdf(u) + beta * pv)


# ------------------------------------------------------- exact mechanism ----
def simulate(Ad, Bd, f, g):
    """Ad, Bd sorted descending. Returns per-tier skill sums."""
    iA = iB = 0
    out = np.empty(q)
    for k in range(q):
        fk, gk = int(f[k]), int(g[k])
        s = Ad[iA : iA + fk].sum() if fk else 0.0
        iA += fk
        if gk:
            ca, cb = Ad[iA : iA + gk], Bd[iB : iB + gk]
            cand = np.concatenate((ca, cb))
            keep = np.argpartition(cand, cand.size - gk)[cand.size - gk :]
            s += cand[keep].sum()
            mA = int(np.count_nonzero(keep < ca.size))
            iA += mA
            iB += gk - mA
        out[k] = s
    return out


# ------------------------------------------------------------- configs ------
def rho_at(pairs):
    r = np.zeros(q)
    for k, val in pairs:
        r[k] = val
    return r


sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
configs = {f"s{r:g}": rho_at([(0, r)]) for r in sweep}
configs["solo2"] = rho_at([(1, 0.6)])
configs["solo5"] = rho_at([(4, 0.6)])
configs["joint12"] = rho_at([(0, 0.6), (1, 0.6)])
configs["joint15"] = rho_at([(0, 0.6), (4, 0.6)])
configs["spread"] = rho_at([(0, 0.45), (2, 0.45), (4, 0.45), (6, 0.45)])

fg = {
    name: (np.round(r * h).astype(int), h - np.round(r * h).astype(int))
    for name, r in configs.items()
}

reps = 120
acc = {name: np.zeros((reps, q)) for name in configs}
for rep in range(reps):
    A = np.sort(rng.standard_normal(a))[::-1]
    B = np.sort(rng.standard_normal(b))[::-1]
    for name, (f, g) in fg.items():  # common random numbers
        acc[name][rep] = simulate(A, B, f, g)

fair = acc["s0"]
res = {}
for name in configs:
    d = acc[name] - fair  # paired differences, raw sigma
    lossr = -(d @ L)
    res[name] = dict(
        dq=d.mean(0),
        dq_sem=d.std(0, ddof=1) / np.sqrt(reps),
        loss=lossr.mean(),
        loss_sem=lossr.std(ddof=1) / np.sqrt(reps),
    )

th = {name: theory(f) for name, (f, g) in fg.items()}

# --------------------------------------------------------------- report -----
print(f"{'config':10s} {'theory loss':>12s} {'sim loss':>12s} {'sem':>8s}")
for name in configs:
    print(
        f"{name:10s} {n * th[name][3]:12.1f} {res[name]['loss']:12.1f} "
        f"{res[name]['loss_sem']:8.1f}"
    )

print("\nper-tier dQ, config s1 (fully reserved tier 1):")
print("theory:", np.array2string(n * th["s1"][2], precision=0, suppress_small=True))
print("sim   :", np.array2string(res["s1"]["dq"], precision=0, suppress_small=True))
print("sem   :", np.array2string(res["s1"]["dq_sem"], precision=0))
print("overhang e_k * n:", np.array2string(n * th["s1"][0], precision=0))

print("\nper-tier dQ, config s0.45 (small shock):")
print("theory:", np.array2string(n * th["s0.45"][2], precision=1, suppress_small=True))
print("sim   :", np.array2string(res["s0.45"]["dq"], precision=1, suppress_small=True))

print(
    "\nmarginal gap at rho=.6:  v1-u1 =",
    float(
        norm.ppf(1 - tau[0] + th["s0.6"][0][0] / beta)
        - norm.ppf(1 - tau[0] - th["s0.6"][0][0] / alpha)
    ),
)

# --------------------------------------------------------------- figures ----
plt.rcParams.update(
    {"figure.dpi": 150, "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3}
)

# --- figure 1: Q1 sweep + Q2 propagation
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

rho_grid = np.linspace(0, 1, 301)
e_grid = np.maximum((rho_grid - alpha) * tau[0], 0.0)
ax[0].plot(rho_grid, -D1_of(e_grid) / tau[0], "k-", lw=1.5, label="fluid-limit theory")
xs = sweep
ys = [res[f"s{r:g}"]["dq"][0] / h[0] for r in xs]
es = [res[f"s{r:g}"]["dq_sem"][0] / h[0] for r in xs]
ax[0].errorbar(
    xs,
    ys,
    yerr=np.array(es) * 2,
    fmt="o",
    ms=4,
    color="tab:red",
    capsize=2,
    label="Monte Carlo (±2 sem)",
)
ax[0].axvline(alpha, color="tab:blue", ls="--", lw=1)
ax[0].text(
    alpha + 0.015,
    -0.42,
    r"natural share $f_1=\alpha h_1$",
    color="tab:blue",
    rotation=90,
    va="bottom",
)
ax[0].set_xlabel(r"reserved fraction of tier 1,  $f_1/h_1$")
ax[0].set_ylabel(r"tier-1 quality change per seat  [$\sigma$]")
ax[0].set_title("Q1: dead zone, then convex loss")
ax[0].legend(loc="lower left")

ks = np.arange(1, q + 1)
ax[1].bar(
    ks,
    res["s1"]["dq"],
    yerr=2 * res["s1"]["dq_sem"],
    capsize=3,
    color="tab:red",
    alpha=0.65,
    label="Monte Carlo (±2 sem)",
)
ax[1].plot(ks, n * th["s1"][2], "k_", ms=16, mew=2, label="theory")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("tier $k$")
ax[1].set_ylabel(r"tier quality change  [$\sigma$, total]")
ax[1].set_title("Q2: shock at tier 1 ($f_1=h_1$) — wave dies out")
axe = ax[1].twinx()
axe.plot(
    ks, n * th["s1"][0], "o--", color="tab:blue", ms=4, lw=1, label="overhang $n\\,e_k$"
)
axe.set_ylabel("A-overhang $n\\,e_k$ (people)", color="tab:blue")
axe.tick_params(axis="y", colors="tab:blue")
axe.grid(False)
h1_, l1_ = ax[1].get_legend_handles_labels()
h2_, l2_ = axe.get_legend_handles_labels()
ax[1].legend(h1_ + h2_, l1_ + l2_, loc="lower right")
fig.tight_layout()
fig.savefig(
    "/mnt/user-data/outputs/fig1_first_tier_and_propagation.png", bbox_inches="tight"
)

# --- figure 2: Q3 superposition + budget shape + Q4 validation
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

groups = [
    (
        "tiers 1 & 2\n(adjacent)",
        res["s0.6"]["loss"] + res["solo2"]["loss"],
        res["joint12"]["loss"],
        n * (th["s0.6"][3] + th["solo2"][3]),
        n * th["joint12"][3],
    ),
    (
        "tiers 1 & 5\n(separated)",
        res["s0.6"]["loss"] + res["solo5"]["loss"],
        res["joint15"]["loss"],
        n * (th["s0.6"][3] + th["solo5"][3]),
        n * th["joint15"][3],
    ),
]
x0 = np.arange(len(groups))
w = 0.35
ax[0].bar(
    x0 - w / 2,
    [g[1] for g in groups],
    w,
    color="tab:gray",
    label="sum of separate shocks",
)
ax[0].bar(
    x0 + w / 2,
    [g[2] for g in groups],
    w,
    color="tab:red",
    alpha=0.75,
    label="both shocks together",
)
ax[0].plot(
    np.r_[x0 - w / 2, x0 + w / 2],
    [g[3] for g in groups] + [g[4] for g in groups],
    "k_",
    ms=18,
    mew=2,
    label="theory",
)
ax[0].set_xticks(x0, [g[0] for g in groups])
ax[0].set_ylabel(r"weighted value loss  [$\sigma$]")
ax[0].set_title("Q3: additive when apart,\nsuperadditive when overlapping")
ax[0].legend(fontsize=8)

names = ["s0.9", "spread"]
labels = ["all at tier 1\n($f_1=0.9h_1$)", "spread over tiers\n1,3,5,7 ($0.45h_k$)"]
ax[1].bar(
    labels, [res[m]["loss"] for m in names], color=["tab:red", "tab:green"], alpha=0.75
)
ax[1].plot([0, 1], [n * th[m][3] for m in names], "k_", ms=22, mew=2, label="theory")
ax[1].set_ylabel(r"weighted value loss  [$\sigma$]")
ax[1].set_title(
    "same moved quota, different placement\n(convexity: concentration hurts)"
)
ax[1].legend()

tx = np.array([n * th[m][3] for m in configs])
sx = np.array([res[m]["loss"] for m in configs])
ax[2].plot([0, tx.max() * 1.05], [0, tx.max() * 1.05], "k-", lw=1)
ax[2].plot(tx, sx, "o", ms=5, color="tab:purple")
ax[2].set_xlabel(r"theory loss  [$\sigma$]")
ax[2].set_ylabel(r"simulated loss  [$\sigma$]")
ax[2].set_title("Q4: closed form vs simulation\n(all 17 configurations)")
fig.tight_layout()
fig.savefig(
    "/mnt/user-data/outputs/fig2_superposition_and_total_loss.png", bbox_inches="tight"
)

print("\nfigures written.")
