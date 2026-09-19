---
title: Point System — Handoff
tags:
  - tracker
  - point-system
  - mechanism-design
status: final design, pre-simulation
---

# Point System — Handoff

> [!abstract] Summary
> Users spend points to download (1 point = 1 GiB chargeable) and earn points by keeping torrents healthy. A seeder's reward rate is the predicted drop in torrent health if they stopped seeding. Health is a concave, knee-shaped function of predicted download slowdown, so the reward acts as a feedback controller that pushes every torrent toward a target health $C^*$ from both sides. Points are non-transferable except as a startup gift attached to an invite. All predictions are built from small per-node statistics with pessimistic Bayesian priors; the system needs no announce history to operate.

---

## 1. Principles

1. **Points buy download, nothing else.** The only sink is chargeable download; the only mint is seeding. There is no market, no tipping, no bounties. Seeding is a hobby, not a job; a thin economy avoids the overjustification effect where extrinsic reward crowds out the intrinsic motive.
2. **Invites are the only transfer, and they move points rather than create them.** The inviter pays the new user's startup buffer, so an invite has a real cost and a sybil farm is self-funded by the attacker.
3. **Reward capacity and availability, not upload volume.** Upload volume rewards luck and popularity. Capacity and uptime are what actually keep a torrent downloadable. Measured upload is still used, but as *evidence* of capacity, not as the thing being paid for.
4. **Pay the marginal contribution.** A seeder is paid for the health the torrent would lose without them. This automatically pays scarcity, caps how much any one seeder can earn from one torrent, and pays a content uploader as the sole seeder of their upload without a separate mechanism.
5. **The reward curve is a controller, not a price list.** Its shape is chosen for stability of the seeder-count dynamics and robustness of the target, not for "fairness" per se.
6. **Predictions are small state, not history.** Every announce is folded into per-node sufficient statistics on ingest. Raw announces are an audit log, never a query target for rewards.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $u$ | user (account) |
| $v$ | node: one client instance of a user, keyed by passkey + client fingerprint + IP prefix |
| $V_u$ | the set of nodes belonging to $u$ |
| $i$ | torrent |
| $E_i$ | member nodes of torrent $i$ (currently seeding, see §5.3 for membership rules) |
| $F_i \subseteq E_i$ | member nodes holding a complete copy |
| $z_i$ | staff-assigned importance, $z_i \in [-2, 2]$ |
| $w_i$ | importance weight |
| $S_i$ | relative size, $\text{size}_i / \bar S$, $\bar S$ = site median torrent size |
| $\tilde p_v$ | pessimistic availability estimate of node $v$ |
| $\tilde b_v$ | pessimistic uplink capacity estimate of node $v$ (bytes/s) |
| $\tilde k_v$ | expected number of $v$'s torrents with active leechers at once |
| $g_v$ | expected per-torrent capacity contribution of node $v$ |
| $g_{u,i}$ | contribution of user $u$ to torrent $i$ (all their nodes merged) |
| $x_i$ | aggregate expected capacity of torrent $i$ |
| $A_i$ | probability a complete copy is available |
| $d_{ref}$ | reference leecher downlink (site median observed leecher download rate) |
| $C_i$ | predicted download slowdown factor relative to a reference leecher served at full rate |
| $H_i$ | health, $H(C_i)$ |
| $H_i^{(-u)}$ | health with user $u$ removed |
| $r_{u,i}$ | reward rate, points per hour |
| $\kappa, \alpha, \gamma, C^*, \eta, \pi_{min}, m$ | global parameters, §3.6 |

---

## 3. Formulas

### 3.1 Torrent factors

$$
w_i = 2^{z_i}, \qquad z_i \in [-2, 2]
$$

A 16× range between the least and most important torrent. $z_i$ is set by staff and defaults to $0$.

$$
S_i = \frac{\text{size}_i}{\bar S}
$$

$S_i^{\eta}$ scales reward with size so that disk spent on large and small torrents is compensated comparably. $\eta$ is a global exponent, slowly adjusted from the site-wide size distribution (see §9).

### 3.2 Node estimates

All node estimates are **pessimistic**: they use a lower quantile of a posterior, so an unproven node is treated as mediocre until it has demonstrated otherwise. Evidence decays exponentially so old behaviour fades.

**Availability.** Let $\mathrm{on}_v$ and $\mathrm{off}_v$ be decayed hours observed online and offline. With parent prior mean $\bar p_v$ and prior weight $m$ pseudo-hours:

$$
\tilde p_v = Q_{0.25}\Big[\mathrm{Beta}\big(m\,\bar p_v + \mathrm{on}_v,\;\; m\,(1-\bar p_v) + \mathrm{off}_v\big)\Big]
$$

**Capacity.** Observed upload rate is a *lower bound* on capacity (a seeder with no leechers shows zero through no fault of its own), so only intervals in which the node had at least one leecher on any of its torrents are *qualifying*. Let $\hat b_v$ be the 90th percentile of observed upload rates over qualifying intervals (from a decayed t-digest), and $n_v$ the decayed count of qualifying hours. With parent prior $\bar b_v$ taken at its 25th percentile:

$$
\tilde b_v = \frac{m\,\bar b_v + n_v\,\hat b_v}{m + n_v}
$$

**Concurrency.** A node seeding thousands of torrents cannot give each one its full uplink:

$$
\tilde k_v = \max\big(1,\; \mathrm{EWMA}\,[\#\text{ of } v\text{'s torrents with} \ge 1 \text{ leecher}]\big)
$$

**Contribution.**

$$
g_v = \frac{\tilde p_v\,\tilde b_v}{\tilde k_v}
$$

**Parent priors** form a hierarchy: population → user → node.

- Population prior: decayed median of $\tilde p$ and 25th percentile of $\tilde b$ over all nodes with $n_v \ge m$. Refreshed daily.
- User prior: the evidence-weighted pooled posterior of the user's *other* nodes, shrunk toward the population prior with the same weight $m$. Used as $\bar p_v, \bar b_v$ for a node whose user already has evidence; otherwise the population prior is used directly.

### 3.3 Torrent health

Nodes of the same user on the same torrent are merged, so a user cannot inflate their own marginal by running two clients:

$$
g_{u,i} = \sum_{v \in E_i \cap V_u} g_v, \qquad
x_i = \sum_{u} g_{u,i}
$$

Completeness: the probability at least one complete copy is online.

$$
A_i = 1 - \prod_{v \in F_i} (1 - \tilde p_v)
$$

Slowdown, relative to a reference leecher of downlink $d_{ref}$ served at full rate:

$$
C_i =
\begin{cases}
\max\!\left(1,\; \dfrac{d_{ref}}{x_i}\right) & A_i \ge \pi_{min} \\[8pt]
\infty & \text{otherwise}
\end{cases}
$$

The floor at $1$ encodes that capacity beyond the leecher's own downlink has no value to that leecher.

Health, with $\operatorname{sp}(t) = \ln(1 + e^{t})$ (softplus):

$$
H(C) = 1 - \frac{\operatorname{sp}\!\big(\gamma\,(1 - C^*/C)\big)}{\operatorname{sp}(\gamma)}
$$

Properties:

- $H(\infty) = 0$ (dead torrent), $H \to 1$ as $C \to 0$; with the floor, the attainable maximum is $H(1)$, slightly below $1$.
- At the target, $H(C^*) = 1 - \ln 2 / \operatorname{sp}(\gamma)$, about $0.83$ for $\gamma = 4$.
- Written in capacity $x = d_{ref}/C$ with $x^* = d_{ref}/C^*$, the marginal is
  $$
  \frac{dH}{dx} = \frac{\gamma}{x^*\,\operatorname{sp}(\gamma)}\;\sigma\!\big(\gamma\,(1 - x/x^*)\big)
  $$
  where $\sigma$ is the logistic function. It is positive and strictly decreasing in $x$, so $H$ is concave in capacity everywhere: nearly flat at its maximum while $x < x^*$, falling to nearly zero once $x > x^*$, with the transition spanning roughly a factor of two in $C$ for $\gamma \in [3, 6]$.

**Leave-one-out.** $H_i^{(-u)}$ is $H$ evaluated with $x_i - g_{u,i}$ and with $A_i$ recomputed over $F_i \setminus V_u$. If removing $u$ drops $A_i$ below $\pi_{min}$, then $H_i^{(-u)} = 0$. This is the sole-complete-seeder case and it pays the full $H_i$.

### 3.4 Reward

$$
r_{u,i} = \kappa\; w_i\; S_i^{\eta}\;\Big[\alpha + H_i - H_i^{(-u)}\Big]
\qquad \text{[points / hour]}
$$

Points accrue per announce: $\Delta\text{points} = r_{u,i}\,\Delta t$, where $\Delta t$ is the time since the node's previous announce for that torrent, capped at announce interval + grace. A user earns on a torrent only while at least one of their nodes is online on it.

Because $x_i$ is a sum, $H_i - H_i^{(-u)}$ is two function evaluations on cached values; no per-torrent re-prediction is needed per announce.

> [!info] Availability is paid for twice, on purpose
> An unreliable node has a small $\tilde p_v$, which shrinks its $g_v$ and therefore its marginal, *and* it is only paid during the hours it is actually online. Both effects reward the same thing, availability, which is what the site is buying. If this proves too harsh in simulation, the alternative knob is to pay on membership time rather than online time, but that invites "announce once, stay a member" gaming and is not the default.

### 3.5 Points, charging, invites

- 1 point = 1 GiB of chargeable download. Each torrent has a charge multiplier $\phi_i \in \{0, \tfrac12, 1\}$; cost of downloading $D$ GiB is $\phi_i D$. Freeleech ($\phi_i = 0$) is used for new-user onboarding and staff picks.
- Balances are non-negative; a download is refused if the balance cannot cover the torrent's remaining chargeable size.
- An invite transfers $I \ge I_{min}$ points from inviter to invitee. $I_{min}$ is guaranteed sufficient to bootstrap (download enough to start seeding) and is included in the invite's cost. There is no other transfer path.
- Cross-seeding a library the user already owns is free and is the intended zero-cost onboarding route.

### 3.6 Global parameters

| Parameter | Meaning | Initial value | Adjusted by |
|---|---|---|---|
| $\kappa$ | reward rate (points/hour at unit weight, unit size, $\Delta H = 1$) | tune in shadow mode | controller, §3.7 |
| $\alpha$ | floor reward for any seeder; "disk rent" | $\le 0.05$ (well below plateau $\approx 0.8$) | admin |
| $\gamma$ | knee sharpness, i.e. controller gain | $4$ | admin, rarely |
| $C^*$ | target slowdown | $1.5$ | admin |
| $\eta$ | size exponent | $0.7$ | slow job, §9 |
| $\pi_{min}$ | minimum completeness probability | $0.9$ | admin |
| $m$ | prior weight in pseudo-hours | $24$ | admin |
| $\lambda$ | evidence half-life | $14$ days | admin |
| $d_{ref}$ | reference leecher downlink | site median, refreshed daily | job |
| $T_{mem}$ | membership expiry without announce | $72$ h | admin |
| $I_{min}$ | minimum startup gift | sized so a new user can download and seed several median torrents | admin |

### 3.7 $\kappa$ controller (monetary policy)

Total supply is minted by seeding and burned by download, and a site that drifts seeding-heavy will inflate. Over a trailing window $W$ (7 days), with $M_W$ points minted and $B_W$ points burned, target ratio $\rho$ (initially $1$):

$$
\kappa \leftarrow \kappa \cdot \left(\rho\,\frac{B_W}{M_W}\right)^{\lambda_\kappa},
\qquad \lambda_\kappa \approx 0.1 \text{ per day}, \quad \text{clamped to } \pm 5\%/\text{day}
$$

> [!warning] Two slow controllers
> $\kappa$ and $\eta$ are both adjusted from aggregate behaviour that they themselves influence. Keep both slow (days, not hours) and never let them move on the same day, or they can oscillate against each other.

---

## 4. Why these formulas

**Marginal contribution.** Rewarding $H_i - H_i^{(-u)}$ pays exactly the value the site would lose. Consequences that fall out for free: rare torrents pay the most, the reward from any single torrent is bounded by $\kappa w_i S_i^\eta(\alpha + 1)$ regardless of bandwidth (so a large seedbox must go broad rather than dominate), and an uploader is paid as sole seeder of their upload with no separate bonus.

**Concavity in capacity is the stability condition.** Model the seeder count of torrent $i$ as a state that grows when the marginal reward exceeds a seeder's opportunity cost $\theta$ (the reward that disk would earn elsewhere) and shrinks otherwise:

$$
\dot n_i \;\propto\; \kappa\, w_i S_i^\eta\, \Delta H_i(n_i) - \theta
$$

The fixed point is stable iff $\Delta H$ is decreasing there. Since seeders add capacity roughly additively, $H$ concave in $x$ guarantees $\Delta H$ decreases with every added seeder, so the loop is stable from both sides everywhere: too few seeders means high reward and people join; too many means low reward and people leave. Any region where $\Delta H$ *increases* with $n$ is a positive-feedback cliff in which a torrent that loses one seeder loses the rest.

**The knee makes the target robust.** $\theta$ is not under admin control; it is the site-wide price of a GiB of disk and drifts with how much spare disk the community has. With a smoothly decaying $\Delta H$ the equilibrium would track $\theta$. With a plateau above target and near-zero below, the crossing lands at the knee for any $\theta$ in that range, so torrents converge on $C^*$ rather than on "whatever disk supply happens to allow". $\gamma$ is the gain: too high and delayed user reactions cause hunting (everyone joins, it overshoots, everyone leaves). If hunting appears in production, add hysteresis (a slightly higher $C^*$ for staying than for joining) rather than lowering $\gamma$.

**The target lives in health space, not seeder count.** "p90 slowdown at most $C^*$" is what users experience. The seeder count that achieves it adjusts automatically for bandwidth and uptime, which a raw count cannot.

**Pessimistic shrinkage estimates.** Every node estimate is a posterior quantile below the mean. This is one mechanism serving three needs: cold start (a new node is paid as a plausible seeder from hour one and converges to its true numbers in a day or two), sybil resistance (fresh accounts cannot harvest full rare-torrent rewards until they have demonstrated capacity), and noise tolerance (a single lucky fast interval does not spike a reward). The hierarchy means a reliable user's new seedbox inherits a good prior while a stranger's does not.

**A conservative point estimate instead of a sampled quantile.** The controller only needs $C$ to be monotone in capacity and availability and roughly calibrated; the knee absorbs miscalibration by shifting the target slightly. Pessimistic inputs give the conservatism a p90 would, at the cost of two multiplications.

**Additive capacity.** Because $x_i$ is a sum, leave-one-out is $O(1)$ per announce and torrent state is only recomputed on membership or estimate changes. This is what makes 1,700 announces/second cheap.

**Upload as evidence, not as reward.** Paying for capacity removes the natural proof-of-work that upload volume provided. Measured upload restores it: a node that throttles sees its $\hat b_v$ fall and its reward with it.

**$\alpha$.** With $\alpha = 0$, seeders on a healthy torrent earn nothing and leave abruptly the moment $x_i$ passes $x^*$. A small floor keeps them around as a buffer against churn and pays modest disk rent. It is also the dominant term in total mint on popular torrents, which is why it must stay small and why the $\kappa$ controller exists.

---

## 5. Algorithm

### 5.1 Pipeline

Tracker announce → batched events on RabbitMQ → Rust reward service (all derived state in memory) → periodic snapshot and ledger flush to DuckDB → Parquet cold storage.

The reward service is the single writer of derived state. It is rebuildable from the hot announce window plus the last snapshot.

### 5.2 Per announce (node $v$, torrent $i$)

1. Resolve the node key; create the node with its parent prior if unseen.
2. Compute $\Delta t$ since $v$'s previous announce on $i$, capped at interval + grace. Mark $v$ online.
3. Add $\Delta t$ to $\mathrm{on}_v$.
4. If $v$ had at least one leecher on any torrent during the interval, compute the observed rate from the uploaded-bytes delta, push it into $v$'s t-digest, add $\Delta t$ to $n_v$, and update $\tilde k_v$.
5. Update membership: on `started` or first sight, add $v$ to $E_i$ (and $F_i$ if complete); on `completed`, move into $F_i$; on `stopped`, remove. Any change marks $i$ dirty.
6. Accrue reward: with cached $x_i$, $A_i$, $H_i$ and $g_{u,i}$, compute $H_i^{(-u)}$ and add $r_{u,i}\,\Delta t$ to the user's in-memory accumulator, keyed by $(u, i)$.
7. Recompute $\tilde p_v$, $\tilde b_v$, $g_v$. If $g_v$ moved by more than a relative threshold $\varepsilon$ (e.g. 5%) since it was last folded into torrent state, mark all of $v$'s torrents dirty.

Steps 1 through 7 are $O(1)$ apart from step 7's fan-out, which is rate-limited by $\varepsilon$.

### 5.3 Offline sweeper (every minute)

- A node with no announce for interval + grace is marked offline; elapsed time is added to $\mathrm{off}_v$, and offline time keeps accumulating until it returns. Its memberships stay in $E_i$ and continue to contribute $g_v$ to $x_i$ (expected availability is already in $\tilde p_v$), but it earns nothing while offline.
- A membership with no announce for $T_{mem}$ is removed from $E_i$ and $F_i$; the torrent is marked dirty.

### 5.4 Dirty recompute (every minute)

For each dirty torrent: recompute $g_{u,i}$ for its members, $x_i$, $A_i$, $C_i$, $H_i$; clear the flag. Cost is $O(|E_i|)$ per dirty torrent, and most torrents are not dirty in a given minute.

### 5.5 Periodic jobs

- **Hourly:** apply evidence decay to all node counters; flush accumulators (§5.6).
- **Daily:** refresh population priors and $d_{ref}$; run the $\kappa$ controller; on alternate days, run the $\eta$ job; recompute user-level priors.
- **Weekly:** re-fit prior distributions from cold storage; compact Parquet.

### 5.6 Ledger flush

Accumulators are flushed hourly into two tables: a per-user-per-hour total (the authoritative balance ledger, append-only), and a per-user-per-torrent-per-day breakdown used only for the UI and retained for 14 days. Balances are the sum of the hourly ledger minus charges and invite transfers; they are never stored as the only copy.

---

## 6. Storage plan

Sizing assumes $10^6$ (torrent, node) memberships, $10^5$ nodes, $2 \times 10^5$ torrents, 10-minute announce interval.

| Tier | Contents | Size | Retention |
|---|---|---|---|
| Hot (DuckDB) | raw announces, 100 B/row | ~14 GB/day | 48 h, then dropped after Parquet export |
| Derived (in memory, snapshotted to DuckDB every 5 min) | node table | ~150 MB (t-digest ~1 KB, Beta counters, EWMAs, prior refs) | live |
| | membership table: node, torrent, complete flag, since, last-announce | ~32 MB | live |
| | torrent table: $x_i, A_i, C_i, H_i$, dirty flag, member count | ~15 MB | live |
| | reward accumulators $(u, i)$ | ~30 MB | flushed hourly |
| Ledger (DuckDB) | per-user-per-hour totals | ~1.2 M rows/day | forever |
| | per-user-per-torrent-per-day | ~1 M rows/day | 14 days |
| Cold (Parquet, zstd) | raw announces, columnar | ~2–3 GB/day | 90 days, then downsampled to hourly per-node aggregates |

The reward path never reads the hot or cold tiers. They exist for rebuilds after a crash (hot), for audits and re-fitting priors (cold), and for debugging disputes.

---

## 7. Cold start and edge cases

All cases below are handled by the prior hierarchy in §3.2; none needs special-case logic in the reward path.

- **New user, first node.** Starts at the population prior with weight $m$ pseudo-hours; $\tilde p_v$ and $\tilde b_v$ sit somewhat below the site median until the node has roughly $m$ hours of evidence. Their startup gift $I_{min}$ plus freeleech torrents provide the first downloads; any cross-seeded library earns immediately.
- **Existing user adds a seedbox.** The new node's parent prior is the user's pooled posterior, so it inherits that user's track record and converges quickly. The merge rule in §3.3 prevents two nodes of one user on one torrent from double-counting.
- **Client restart or IP change.** The node key uses client fingerprint and IP prefix rather than peer_id, so restarts keep the same node; a full IP change creates a new node that inherits the user prior.
- **Site launch.** No population data exists. Hand-set the population prior from external estimates with a small $m$ so it is overwritten quickly. Expect $\Delta H$ to be high on almost every torrent because most have one seeder; this is a mint spike for the $\kappa$ controller to absorb, not a predictor problem. Run in shadow mode first (§8).
- **Node with seeders but no leechers for weeks.** $n_v$ decays, so $\tilde b_v$ drifts back toward the prior; the node is paid as "plausible" rather than on stale evidence.
- **Torrent with only partial seeders.** They contribute capacity to $x_i$ but not to $A_i$; if no complete copy exists, $C_i = \infty$, $H_i = 0$, and everyone receives only $\alpha$ until a complete copy appears, at which point that node is paid the full $H_i$.

---

## 8. Launch plan and validation

1. **Agent-based simulation before launch.** $N$ torrents with skewed popularity, $M$ seeders with disk budgets and heterogeneous bandwidth and uptime; each tick, every seeder drops its lowest-reward torrent if a higher-reward one is available. Track the distribution of $C_i$ over time while varying total disk supply. Acceptance: the $C_i$ distribution collapses toward $C^*$ regardless of disk supply, with no sustained oscillation at $\gamma = 4$. Use the same run to size $\alpha$, $\kappa$'s initial value and $I_{min}$.
2. **Shadow mode for two to three weeks after launch.** Compute and display rewards in the UI but pay a flat rate. Watch node estimates settle and the mint/burn ratio stabilise before real points depend on the predictor.
3. **Switch to live rewards** with the $\kappa$ controller enabled and $\eta$ frozen for the first month.
4. **Instrumentation to keep permanently:** histogram of $C_i$, histogram of $\Delta H$ paid, mint/burn ratio, fraction of nodes still at prior weight, and per-torrent seeder-count time series for a sample of torrents (to detect hunting).

---

## 9. Open decisions

> [!todo] Rule for $\eta$
> $\eta$ is meant to make reward per GiB-hour of disk roughly size-neutral across the site's torrent size distribution. The concrete update rule (which statistic of the distribution, what step size) is not yet specified. Freeze $\eta$ at its initial value until the rule is written and simulated.

> [!todo] Should importance move the target instead of the pay?
> Currently $z_i$ scales the reward via $w_i$, which means high-importance torrents get provisioned past $C^*$ when disk is plentiful. The alternative is a per-torrent target $C_i^* = C^* \cdot 2^{-z_i}$, so importance demands better health rather than paying more for the same health. Both can coexist. Decide after simulation.

> [!todo] Hysteresis
> Only add it if production shows hunting on thin torrents. Implementation: use $C^*_{\text{stay}} = 1.1\,C^*$ for nodes already in $E_i$ and $C^*$ for the join decision shown in the UI.

> [!question] Active capacity probes
> Tracker-run probe leechers that download random pieces would give direct capacity and possession evidence independent of organic demand. Not required by the design; revisit if passive evidence proves too sparse for nodes on unpopular content.
