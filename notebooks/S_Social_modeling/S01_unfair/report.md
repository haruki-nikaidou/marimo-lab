# What quotas cost: a back-of-the-envelope theory that turned out exact

*Tiered admissions with reserved seats, studied with a bit of calculus and checked against a brute-force simulation. Informal write-up.*

## The setup, in plain words

There are `n` people (say 200,000). A fraction `α` of them are group A (say 30%), the rest are group B. Everyone's skill is drawn from the same normal distribution — the groups are statistically identical. There are `q` tiers of positions, best to worst; tier `k` has `h_k` seats, of which `f_k` are reserved for A and the rest `g_k` are open to anyone.

Admission is greedy: go tier by tier from the top. In each tier, first fill the reserved seats with the best available A-people, then fill the open seats with the best available people of any kind.

The question: if we keep tier sizes fixed but convert open seats into reserved seats, what happens to the quality of each tier, and to the total value `Σ L_k × (skill sum of tier k)` where `L_k` is how much we care about tier `k`?

One simplification for free: since every configuration fills exactly the same number of seats, the mean of the skill distribution cancels out of every comparison. Only the spread σ matters, so all losses below are in "σ units."

## The one number that runs the whole show

Everything reduces to a single running quantity per tier, which I call the **overhang** `e_k`: how many extra A-people have been admitted so far, beyond A's natural 30% share of the seats.

It obeys a beautifully simple update rule. Each tier's reservation *pushes in* `f_k` A-people, while the tier's overall capacity would *naturally absorb* `α·h_k` of them anyway. So:

```
e_k = max(e_{k-1} + f_k − α·h_k, 0) / n
```

If you've seen queueing theory, this is **Lindley's recursion** — the waiting-time formula for a single-server queue. That's not a coincidence, it's the right mental model: unfairness behaves like *backlog*. Reservations are arriving work, the natural share is service capacity, and the `max(…, 0)` is there because open seats can never *un-admit* anyone. Once you see this, all four answers fall out.

From the overhang you get the two admission cutoffs at each tier: A-people are taken down to a skill level `u_k` (lower than fair, because extras were forced in) and B-people only down to `v_k` (higher than fair, because they got squeezed). The gap `v_k − u_k` is the whole story.

## Q1: What happens to the top tier

Three regimes, and the simulation nails each one (fig1, left panel).

**First, a dead zone.** As long as you reserve *less than A's natural share* (`f_1 ≤ α·h_1`), the loss is exactly zero. Those reserved seats get filled by A-people who would have made the cut anyway — you've relabeled seats, not changed people. In the simulation, reserving up to 30% of tier 1 costs literally ~0.1σ out of a tier worth ~21,000σ. Noise.

**Then, a swap price.** Past the natural share, each extra reserved seat performs a swap: kick out the weakest admitted B (skill `v_1`) and pull in the best remaining A (skill `u_1`). So the marginal cost per seat is exactly `(v_1 − u_1)·σ` — and this part is true for *any* skill distribution, not just the normal. In my benchmark, at 60% reservation the swap price is already 0.65σ per seat.

**And it accelerates.** The total loss grows quadratically at first — for `m` seats past the natural share, loss ≈ `m²σ / (2αβ·φ(t_1)·n)` — and worse after. Two things in that denominator deserve names:

The `1/(αβ)` factor says reserving for a *small* minority is per-seat expensive: to find `e` extra A-mass you must dig `e/α` quantiles deep into A's distribution.

The `1/φ(t_1)` factor (φ = normal density at the tier's fair cutoff) says **elite tiers are exponentially fragile**. Deep in the Gaussian tail, people are spread out — equal displacement in *rank* means huge displacement in *skill*. Fully reserving the top-6% tier costs 0.586σ *per seat*: theory said 7027σ total, simulation said 7023 ± 6.

## Q2: The damage travels as a wave — and mostly it just reshuffles

Shock one tier and the overhang drains downward: each subsequent fair tier absorbs `α·h_k` of it, so the disturbance travels a finite number of tiers and dies. Tiers below the wave are **exactly** untouched — same people, same seats.

Inside the wave, something that surprised me at first: the shocked tier loses quality, but the next few tiers *gain* it. They inherit the displaced B-people, who are stronger than that tier's fair intake, while losing their best A-people to promotion. Fully reserved tier 1 gives per-tier changes of `(−7027, +5739, +1224, +65, 0, 0, 0, 0)` in theory; simulation: `(−7023, +5737, +1222, +64, 0, …)`.

Sum those up and you get (almost) zero. That's a theorem, not luck: **an interior reservation only reshuffles skill between adjacent tiers**. Raw total skill is destroyed only if the wave survives all the way past the last tier — that's when above-the-bar B-people get pushed out of admission entirely in exchange for weaker A's. For a small shock the conservation is comically exact: at 45% reservation, tier 1 loses 305.7σ and tier 2 gains… 305.7σ.

One side effect worth flagging: inside the wave, every reserved A sits strictly below every displaced B in skill, so shocked tiers become internally *segregated* — the quota shows up as a within-tier gap between the groups even where the tier's total quality barely moved.

## Q3: Two shocks — when do they add, when do they compound

Two reservations at tiers `k1 < k2` interact through exactly one channel: whether the first wave is still alive when it reaches `k2`.

If it isn't — the waves touch disjoint tiers — the losses are **exactly additive**. Not approximately: benchmark shocks at tiers 1 and 5 cost 1618σ together vs. 1623σ summed separately (difference is sim noise).

If the waves overlap, the overhangs stack, and because the loss is *convex* in overhang, stacking hurts more than the parts: there's a positive cross-term proportional to `e^(x) · e^(y)` on the overlapping tiers. Worse, upstream overhang eats into the downstream tier's absorption capacity, so a reservation that would've been harmless alone becomes costly. Benchmark: shocks at tiers 1 and 2 cost 4603σ together vs. 1919σ summed — **2.4× the parts** (fig2, left).

## Q4: The total bill, in closed form

With tier values `L_k`, a summation-by-parts trick (Abel summation) gives an exact formula, and a very readable approximation:

```
Loss ≈ (nσ / 2αβ) · Σ_k (L_k − L_{k+1}) · e_k² / φ(t_k)
```

Read it like a physics "energy": the loss is the weighted energy of the overhang trajectory. Everything qualitative is visible in it:

Only the *decrements* of L matter — shuffling people between equally-valued tiers is free. The last decrement `L_q − 0` is the price of the people expelled from the system entirely. The mean μ never appears. And since `e_k²` is convex, a fixed quota budget does the least damage **spread thin, placed far apart, and parked at unselective tiers** (where φ(t_k) is large). Same budget, two placements: all at tier 1 → 5553σ; spread over tiers 1, 3, 5, 7 → 659σ. **8.4× difference** (fig2, middle).

The exact formula matched all 17 simulated configurations to ~0.2% (fig2, right — the dots just sit on the line).

## How it was checked

`sim.py` runs the *actual* discrete mechanism — no fluid approximation, just sorted arrays, two pointers, and a top-`g_k` selection per tier — with n = 200,000, eight 6% tiers, α = 0.3, L = 8…1, 120 replications. The trick that makes the error bars invisible is **common random numbers**: every configuration is scored on the *same* skill draws and differenced against the fair baseline, so config-vs-fair comparisons have tiny variance. Cheap and very much worth stealing for any A/B-style simulation.

## Fine print

The closed forms use normality (via the neat identity ∫ᵤ z·φ dz = φ(u)); the structural results — dead zone, Lindley recursion, wave locality, swap-price rule, convexity — are distribution-free. The analysis assumes the A-pool never runs dry (fine here). And "value = Σ L_k × skill" is deliberately narrow: real reservation systems pursue goals this metric doesn't see, so read the loss as the *price tag* of those goals, not a verdict on them.

## Files

- `fig1_first_tier_and_propagation.png` — dead zone + convex growth; the wave and its draining overhang
- `fig2_superposition_and_total_loss.png` — additive vs. compounding shocks; concentration vs. spreading; theory-vs-sim scatter
- `sim.py` — the exact simulator + theory formulas (numpy/scipy, ports nicely to Rust as a fold over tiers)
