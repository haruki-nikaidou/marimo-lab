import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import json
    import math
    import struct

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    return Path, PathPatch, itertools, json, math, mo, np, plt, struct


@app.cell
def _(mo):
    mo.md("""
    # Bezier curve compression — 118 2D cubic Beziers

    Reparametrize, quantize, bit-pack, decode, measure the damage. The error
    budget is a slider: move it and the bit allocation, payload size,
    reconstruction error and every plot recompute.

    **Strategy.** Each curve `[P0, P1, P2, P3]` (8 floats) is stored as

    | component | meaning | span |
    | --- | --- | --- |
    | `P0` | absolute anchor | large |
    | `d = P3 - P0` | chord vector | medium |
    | `h1 = P1 - P0` | start handle offset | small |
    | `h2 = P2 - P3` | end handle offset | small |

    Each component gets its own bit budget from its measured span. That is the
    whole trick: handle offsets are far smaller than absolute positions, so they
    buy the same absolute accuracy with fewer bits.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data

    Flat 8-tuples below. `parse_curves` also accepts a JSON string, nested
    `[[x, y], ...]` points, or `P0`/`P1`/`P2`/`P3` dicts, so a real export can be
    dropped in unchanged.
    """)
    return


@app.cell
def _():
    # 118 cubic Beziers, flat: x0, y0, x1, y1, x2, y2, x3, y3.
    RAW_CURVES = [
        [170.799, 92.14, 82.956, 161.891, 86.821, 355.221, 56.552, 420.945],
        [56.552, 420.945, 26.284, 486.669, -1.181, 569.716, 16.693, 610.041],
        [16.693, 610.041, 34.567, 650.366, 56.552, 639.031, 56.552, 639.031],
        [101.702, 358.502, 86.444, 428.689, 25.412, 518.275, 17.129, 567.755],
        [17.129, 567.755, 8.846, 617.235, 6.23, 634.018, 32.605, 644.917],
        [32.605, 644.917, 58.98, 655.816, 67.48, 613.965, 67.48, 613.965],
        [107.37, 302.265, 97.779, 412.123, 64.212, 511.736, 82.521, 553.586],
        [82.521, 553.586, 100.831, 595.437, 190.635, 601.54, 212.65, 593.039],
        [139.629, 416.046, 123.935, 457.243, 107.587, 536.773, 194.994, 596.347],
        [180.172, 101.836, 148.13, 192.512, 166.44, 307.165, 154.451, 352.285],
        [154.451, 352.285, 142.463, 397.405, 122.341, 450.808, 77.146, 454.732],
        [77.146, 454.732, 31.951, 458.655, 6.448, 440.782, 25.194, 407.432],
        [180.172, 238.068, 186.493, 321.769, 172.979, 388.468, 112.165, 427.267],
        [112.165, 427.267, 51.351, 466.066, 9.492, 428.498, 25.242, 406.975],
        [139.987, 259.87, 147.18, 349.892, 188.813, 451.285, 247.883, 449.305],
        [144.898, 401.116, 163.31, 490.702, 213.444, 545.231, 247.883, 506.196],
        [166.58, 416.046, 185.108, 453.429, 232.19, 459.381, 247.883, 444.09],
        [188.595, 438.607, 191.429, 466.507, 224.778, 519.474, 253.769, 503.999],
        [181.587, 246.356, 190.121, 302.375, 206.469, 339.477, 227.176, 354.602],
        [204.382, 267.717, 209.521, 306.516, 227.177, 354.602, 227.177, 354.602],
        [213.226, 311.16, 245.704, 364.933, 289.734, 376.485, 317.852, 373.216],
        [247.883, 263.794, 269.026, 328.096, 298.67, 366.822, 317.852, 373.725],
        [289.952, 202.108, 309.134, 293.22, 308.044, 335.652, 257.256, 373.725],
        [289.952, 202.108, 308.48, 258.563, 323.302, 297.362, 310.005, 347.495],
        [306.954, 187.94, 322.648, 252.024, 331.585, 316.325, 310.006, 347.495],
        [302.529, 202.108, 351.202, 306.734, 428.582, 332.673, 483.075, 324.826],
        [483.075, 324.826, 537.568, 316.979, 566.558, 281.45, 556.313, 256.165],
        [337.252, 211.481, 367.986, 310.658, 410.055, 316.107, 473.92, 312.62],
        [473.92, 312.62, 537.786, 309.132, 557.555, 272.945, 558.351, 264.119],
        [235.677, 77.428, 363.408, 196.223, 406.349, 290.604, 396.104, 351.636],
        [396.104, 351.636, 385.859, 412.668, 364.28, 415.72, 347.714, 417.899],
        [418.555, 347.495, 411.144, 390.435, 401.771, 408.309, 346.624, 417.9],
        [242.543, 61.026, 333.274, 82.551, 448.254, 229.137, 483.129, 359.375],
        [483.129, 359.375, 518.004, 489.613, 538.984, 513.974, 414.195, 567.81],
        [447.802, 370.979, 465.965, 457.35, 453.704, 541.219, 412.289, 567.81],
        [439.262, 390.597, 439.534, 475.061, 405.476, 497.403, 364.334, 474.789],
        [426.457, 354.631, 421.28, 429.559, 395.941, 489.229, 364.335, 474.515],
        [309.229, 94.632, 453.976, 118.813, 479.86, 233.248, 531.288, 322.14],
        [531.288, 322.14, 582.716, 411.032, 627.672, 522.742, 546.274, 498.561],
        [474.71, 246.463, 503.046, 305.751, 582.17, 400.351, 582.17, 447.215],
        [582.17, 447.215, 582.17, 494.079, 561.463, 499.746, 545.987, 498.438],
        [170.799, 92.14, 194.982, 102.961, 205.444, 118.655, 205.444, 118.655],
        [242.543, 61.026, 225.498, 71.748, 204.573, 102.961, 205.445, 118.655],
        [212.568, 92.14, 152.861, 56.128, 143.236, 1.507, 213.824, -45.29],
        [213.824, -45.29, 284.412, -92.087, 325.983, -22.196, 305.058, 3.333],
        [212.568, 92.14, 175.6, 50.764, 165.695, -1.689, 246.188, -24.707],
        [212.568, 92.14, 203.64, 51.88, 239.771, 10.447, 271.996, 4.867],
        [271.996, 4.867, 304.221, -0.713, 327.797, 10.168, 327.657, 29.698],
        [327.657, 29.698, 327.517, 49.228, 319.287, 56.343, 300.315, 45.741],
        [217.272, 87.98, 215.218, 53.693, 250.233, 21.19, 272.832, 12.54],
        [272.832, 12.54, 295.431, 3.891, 316.217, 2.232, 330.167, 15.547],
        [330.167, 15.547, 344.117, 28.862, 337.735, 57.397, 308.143, 49.337],
        [217.186, 451.561, 254.654, 480.333, 322.661, 485.129, 322.661, 485.129],
        [405.807, 388.24, 386.963, 436.521, 332.906, 481.205, 322.662, 485.128],
        [163.719, 211.57, 174.321, 178.508, 192.317, 156.327, 207.662, 149.631],
        [207.662, 149.631, 223.007, 142.935, 265.555, 128.427, 290.247, 137.494],
        [208.499, 158.559, 203.477, 180.042, 191.48, 229.426, 191.48, 229.426],
        [191.48, 229.426, 214.079, 241.981, 214.079, 241.981, 214.079, 241.981],
        [214.079, 241.981, 210.173, 165.952, 210.173, 165.952, 210.173, 165.952],
        [224.96, 154.374, 248.675, 165.116, 277.831, 197.899, 277.831, 197.899],
        [214.079, 241.981, 265.415, 225.241, 265.415, 225.241, 265.415, 225.241],
        [265.415, 225.241, 224.122, 158.559, 224.122, 158.559, 224.122, 158.559],
        [202.779, 169.859, 232.632, 154.374, 232.632, 154.374, 232.632, 154.374],
        [163.719, 211.57, 189.387, 222.451, 189.387, 222.451, 189.387, 222.451],
        [290.666, 137.495, 296.525, 153.119, 287.039, 181.578, 287.039, 181.578],
        [240.445, 162.117, 264.579, 171.115, 287.039, 181.578, 287.039, 181.578],
        [166.648, 156.188, 197.896, 129.822, 225.099, 113.779, 279.505, 120.476],
        [277.003, 211.57, 291.607, 278.016, 297.71, 327.06, 270.246, 362.371],
        [96.959, 570.099, 81.265, 609.552, 81.169, 683.662, 163.719, 666.224],
        [132.052, 587.318, 117.23, 623.937, 114.711, 658.71, 158.911, 667.153],
        [226.216, 542.634, 175.211, 533.915, 158.911, 564.533, 158.911, 564.533],
        [96.18, 645.295, 61.212, 665.788, 0.0, 714.396, 0.0, 714.396],
        [174.126, 549.888, 195.699, 520.183, 196.353, 517.349, 238.858, 524.106],
        [187.273, 532.483, 199.623, 524.106, 218.151, 534.133, 261.309, 568.354],
        [261.309, 568.354, 304.467, 602.576, 378.578, 671.891, 378.578, 671.891],
        [233.08, 523.211, 233.08, 523.211, 268.25, 525.249, 277.596, 518.553],
        [277.596, 518.553, 286.943, 511.857, 290.291, 501.813, 290.291, 501.813],
        [290.291, 501.812, 324.794, 518.809, 359.298, 535.805, 393.801, 552.802],
        [393.801, 552.802, 391.918, 538.642, 390.034, 524.481, 385.152, 510.321],
        [297.638, 481.445, 305.776, 490.792, 301.601, 507.384, 301.601, 507.384],
        [381.106, 429.829, 377.897, 452.986, 380.18, 475.754, 380.18, 475.754],
        [380.536, 481.445, 381.524, 491.21, 385.151, 510.322, 385.151, 510.322],
        [384.048, 504.296, 392.267, 499.999, 392.546, 511.996, 404.264, 521.063],
        [404.264, 521.063, 415.982, 530.131, 434.396, 545.894, 434.396, 545.894],
        [378.578, 671.891, 437.221, 677.096, 483.78, 700.463, 483.78, 700.463],
        [442.317, 554.943, 472.878, 579.734, 503.439, 604.526, 534.001, 629.317],
        [534.001, 629.317, 541.209, 646.057, 548.416, 662.797, 555.624, 679.537],
        [555.624, 679.537, 504.59, 643.15, 453.557, 606.764, 402.521, 570.377],
        [402.521, 570.377, 429.608, 613.738, 456.694, 657.099, 483.781, 700.46],
        [0.0, 721.127, 17.887, 721.127, 44.262, 760.58, 59.956, 784.775],
        [59.956, 784.775, 75.65, 808.97, 123.822, 839.268, 194.88, 806.354],
        [194.88, 806.354, 265.938, 773.44, 284.798, 734.641, 378.578, 782.595],
        [297.315, 653.338, 341.937, 705.869, 386.559, 758.4, 431.179, 810.931],
        [143.003, 635.246, 158.432, 683.563, 173.86, 731.88, 189.289, 780.197],
        [376.246, 784.556, 410.672, 791.095, 431.414, 801.692, 431.179, 820.807],
        [228.779, 788.628, 233.609, 806.382, 238.439, 824.136, 243.269, 841.89],
        [383.275, 785.992, 405.558, 804.625, 427.841, 823.257, 450.125, 841.89],
        [411.544, 675.571, 439.589, 729.41, 467.635, 783.249, 495.681, 837.088],
        [490.014, 274.285, 510.067, 334.445, 538.84, 437.546, 508.978, 493.129],
        [521.851, 444.303, 541.891, 537.595, 545.814, 566.149, 490.013, 597.319],
        [512.693, 485.26, 519.222, 527.35, 517.042, 571.792, 495.681, 594.037],
        [480.225, 532.613, 518.36, 570.893, 556.495, 609.173, 594.629, 647.453],
        [400.478, 517.677, 419.254, 526.814, 427.798, 520.035, 443.492, 524.645],
        [0.0, 789.624, 24.991, 828.11, 46.107, 841.89, 46.107, 841.89],
        [500.782, 841.89, 535.521, 810.74, 556.331, 792.349, 594.629, 799.841],
        [216.623, 407.59, 228.118, 396.876, 246.867, 385.493, 246.867, 385.493],
        [333.023, 352.682, 347.643, 350.004, 372.641, 348.33, 372.641, 348.33],
        [222.563, 402.467, 197.873, 399.777, 194.637, 379.354, 199.436, 365.627],
        [199.436, 365.627, 204.235, 351.9, 230.573, 349.556, 236.264, 360.717],
        [236.264, 360.717, 241.956, 371.877, 244.632, 382.093, 243.35, 387.699],
        [334.92, 352.682, 321.863, 332.482, 318.626, 316.412, 330.791, 307.818],
        [330.791, 307.818, 342.956, 299.225, 371.972, 301.68, 372.642, 318.978],
        [372.642, 318.978, 373.312, 336.276, 367.947, 348.676, 367.947, 348.676],
        [154.185, 467.916, 155.057, 506.105, 159.765, 539.585, 159.765, 539.585],
        [410.17, 483.959, 413.658, 508.023, 414.529, 519.532, 414.529, 519.532],
        [230.911, 467.916, 229.516, 490.411, 226.462, 497.211, 226.462, 497.211],
        [272.739, 483.959, 272.739, 500.132, 266.472, 514.846, 266.472, 514.846],
        [439.54, 394.506, 444.771, 423.453, 454.711, 494.25, 447.91, 511.862],
    ]
    return (RAW_CURVES,)


@app.cell
def _(json, np):
    def parse_curves(raw):
        data = json.loads(raw) if isinstance(raw, str) else raw
        parsed = []
        for item in data:
            if isinstance(item, dict):
                pts = [item["P0"], item["P1"], item["P2"], item["P3"]]
            elif len(item) == 8 and not isinstance(item[0], (list, tuple)):
                pts = [item[0:2], item[2:4], item[4:6], item[6:8]]
            elif len(item) == 4 and len(item[0]) == 2:
                pts = item
            else:
                raise ValueError(f"unrecognized curve shape: {item}")
            parsed.append([tuple(p) for p in pts])
        return np.array(parsed, dtype=np.float64)  # (N, 4, 2)

    return (parse_curves,)


@app.cell
def _(RAW_CURVES, json, parse_curves):
    curves = parse_curves(RAW_CURVES)
    json_bytes = len(json.dumps(RAW_CURVES))
    f32_bytes = curves.size * 4
    return curves, f32_bytes, json_bytes


@app.cell
def _(curves, f32_bytes, json_bytes, mo):
    mo.hstack(
        [
            mo.stat(label="curves", value=f"{len(curves)}"),
            mo.stat(label="floats", value=f"{curves.size}"),
            mo.stat(label="raw f32", value=f"{f32_bytes} B"),
            mo.stat(label="JSON text", value=f"{json_bytes} B"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(Path, PathPatch):
    def plot_curves(ax, curve_pts, color="C0", alpha=0.7, lw=1.2, label=None):
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        for i, c in enumerate(curve_pts):
            ax.add_patch(
                PathPatch(
                    Path(list(c), codes),
                    fc="none",
                    ec=color,
                    alpha=alpha,
                    lw=lw,
                    label=label if i == 0 else None,
                )
            )
        pts = curve_pts.reshape(-1, 2)
        pad = 30.0
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    return (plot_curves,)


@app.cell
def _(curves, plot_curves, plt):
    fig_input, ax_input = plt.subplots(figsize=(8, 8))
    plot_curves(ax_input, curves, color="C0")
    ax_input.set_title(f"{len(curves)} cubic Bezier curves (original)")
    fig_input
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Empirical ranges

    Decompose into `(P0, d, h1, h2)` and measure the real span of each component.
    Ranges are padded by 1% so the extremes sit strictly inside the quantizer.
    """)
    return


@app.cell
def _(np):
    COMPONENT_NAMES = ("P0 anchor", "d chord", "h1 handle", "h2 handle")

    def decompose(curve_pts):
        p0, p1, p2, p3 = (curve_pts[:, k, :] for k in range(4))
        return np.stack([p0, p3 - p0, p1 - p0, p2 - p3])  # (4, N, 2)

    return COMPONENT_NAMES, decompose


@app.cell
def _(curves, decompose, np):
    components = decompose(curves)
    obs_lo = components.min(axis=1)  # (4, 2)
    obs_hi = components.max(axis=1)
    pad_amount = (obs_hi - obs_lo) * 0.01
    # The header stores f32 bounds, so the encoder must quantize against exactly
    # the numbers the decoder reads back. Round outward by one f32 ulp: nearest
    # rounding could otherwise pull a bound inside the data and clip a value by
    # up to an ulp, which the half-step bound does not account for. Outward
    # rounding also keeps a constant component's range non-empty (2 ulps wide).
    lo32 = np.nextafter((obs_lo - pad_amount).astype(np.float32), np.float32(-np.inf))
    hi32 = np.nextafter((obs_hi + pad_amount).astype(np.float32), np.float32(np.inf))
    lo = lo32.astype(np.float64)
    hi = hi32.astype(np.float64)
    widths = (hi - lo).max(axis=1)  # per component, worst axis
    return components, hi, lo, widths


@app.cell
def _(COMPONENT_NAMES, hi, lo, mo, widths):
    mo.md(
        "| component | x range | y range | width used |\n| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| `{name}` | [{lo[k, 0]:.2f}, {hi[k, 0]:.2f}] "
            f"| [{lo[k, 1]:.2f}, {hi[k, 1]:.2f}] | {widths[k]:.2f} |"
            for k, name in enumerate(COMPONENT_NAMES)
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bit budget

    Reconstruction stacks quantization errors. Write `e_p`, `e_d`, `e_1`, `e_2`
    for the half-step error of `P0`, `d`, `h1`, `h2`:

    - `P0 = P0q` — error `e_p`
    - `P1 = P0q + h1q` — error `e_p + e_1`
    - `P3 = P0q + dq` — error `e_p + e_d`
    - `P2 = P0q + dq + h2q` — error `e_p + e_d + e_2`

    With `e = width / (2**bits - 1) / 2`, half a quantization step, the worst-case
    L2 over the 8-D parameter vector is

    $$L_2^{\max} = \sqrt{2\,(e_p^2 + (e_p+e_1)^2 + (e_p+e_d)^2 + (e_p+e_d+e_2)^2)}$$

    The search takes the cheapest allocation whose bound stays inside the budget.
    """)
    return


@app.cell
def _(mo):
    target_l2 = mo.ui.slider(
        start=0.5,
        stop=30.0,
        step=0.5,
        value=10.0,
        label="worst-case L2 budget",
        show_value=True,
    )
    target_l2
    return (target_l2,)


@app.cell
def _(itertools, math):
    BIT_CANDIDATES = (range(6, 17), range(4, 17), range(4, 17), range(4, 17))

    def worst_case_l2(widths, bits):
        e_p0, e_d, e_h1, e_h2 = (
            w / ((1 << b) - 1) / 2 for w, b in zip(widths, bits, strict=True)
        )
        per_coord = (e_p0, e_p0 + e_h1, e_p0 + e_d + e_h2, e_p0 + e_d)
        return math.sqrt(2.0 * sum(e * e for e in per_coord))

    def choose_bits(widths, target):
        best = None
        for bits in itertools.product(*BIT_CANDIDATES):
            err = worst_case_l2(widths, bits)
            if err <= target and (best is None or (sum(bits), err) < best[0]):
                best = ((sum(bits), err), bits)
        if best is None:  # budget tighter than the finest allocation on offer
            finest = tuple(c[-1] for c in BIT_CANDIDATES)
            return finest, worst_case_l2(widths, finest)
        return best[1], best[0][1]

    return BIT_CANDIDATES, choose_bits, worst_case_l2


@app.cell
def _(choose_bits, target_l2, widths):
    bit_alloc, predicted_l2 = choose_bits(widths, target_l2.value)
    bits_per_curve = 2 * sum(bit_alloc)
    return bit_alloc, bits_per_curve, predicted_l2


@app.cell
def _(COMPONENT_NAMES, bit_alloc, bits_per_curve, mo, predicted_l2):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="bits / curve", value=f"{bits_per_curve}"),
                    mo.stat(label="bytes / curve", value=f"{bits_per_curve / 8:.2f}"),
                    mo.stat(label="worst-case L2", value=f"{predicted_l2:.3f}"),
                ],
                widths="equal",
            ),
            mo.md(
                "Bits per axis: "
                + ", ".join(
                    f"`{name}` = {b}"
                    for name, b in zip(COMPONENT_NAMES, bit_alloc, strict=True)
                )
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Codec

    `[u16 count][u8 bits x4][16 x f32 ranges][bit-packed payload]`: a 70-byte
    header, then `2 * sum(bits)` bits per curve. The decoder needs nothing but
    the blob.
    """)
    return


@app.cell
def _(decompose, np, struct):
    HEADER_BYTES = 2 + 4 + 64

    def quantize(values, lo, hi, bits):
        levels = (1 << bits) - 1
        t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        return np.round(t * levels).astype(np.int64)

    def dequantize(q, lo, hi, bits):
        return lo + (q / ((1 << bits) - 1)) * (hi - lo)

    class BitWriter:
        def __init__(self):
            self.buf = bytearray()
            self.acc = 0
            self.nbits = 0

        def write(self, value, bits):
            self.acc |= (int(value) & ((1 << bits) - 1)) << self.nbits
            self.nbits += bits
            while self.nbits >= 8:
                self.buf.append(self.acc & 0xFF)
                self.acc >>= 8
                self.nbits -= 8

        def flush(self):
            if self.nbits:
                self.buf.append(self.acc & 0xFF)
                self.acc = 0
                self.nbits = 0
            return bytes(self.buf)

    class BitReader:
        def __init__(self, data):
            self.data = data
            self.pos = 0
            self.acc = 0
            self.nbits = 0

        def read(self, bits):
            while self.nbits < bits:
                self.acc |= self.data[self.pos] << self.nbits
                self.pos += 1
                self.nbits += 8
            value = self.acc & ((1 << bits) - 1)
            self.acc >>= bits
            self.nbits -= bits
            return value

    def encode(curve_pts, lo, hi, bits):
        comps = decompose(curve_pts)
        header = struct.pack("<H", len(curve_pts)) + bytes(int(b) for b in bits)
        header += struct.pack(
            "<16f",
            *(v for k in range(4) for v in (lo[k, 0], lo[k, 1], hi[k, 0], hi[k, 1])),
        )
        q = [quantize(comps[k], lo[k], hi[k], bits[k]) for k in range(4)]
        writer = BitWriter()
        for i in range(len(curve_pts)):
            for k in range(4):
                writer.write(q[k][i, 0], bits[k])
                writer.write(q[k][i, 1], bits[k])
        return header + writer.flush()

    def decode(blob):
        count = struct.unpack_from("<H", blob, 0)[0]
        bits = list(blob[2:6])
        ranges = np.array(struct.unpack_from("<16f", blob, 6)).reshape(4, 2, 2)
        reader = BitReader(blob[HEADER_BYTES:])
        out = np.zeros((count, 4, 2), dtype=np.float64)
        for i in range(count):
            p0, d, h1, h2 = (
                np.array(
                    [
                        dequantize(
                            reader.read(bits[k]),
                            ranges[k, 0, a],
                            ranges[k, 1, a],
                            bits[k],
                        )
                        for a in range(2)
                    ]
                )
                for k in range(4)
            )
            p3 = p0 + d
            out[i] = [p0, p0 + h1, p3 + h2, p3]
        return out

    return BitReader, BitWriter, HEADER_BYTES, decode, dequantize, encode, quantize


@app.cell
def _(bit_alloc, curves, decode, encode, hi, lo):
    blob = encode(curves, lo, hi, bit_alloc)
    recovered = decode(blob)
    return blob, recovered


@app.cell
def _(HEADER_BYTES, blob, f32_bytes, json_bytes, mo):
    mo.hstack(
        [
            mo.stat(label="compressed", value=f"{len(blob)} B"),
            mo.stat(
                label="header / payload",
                value=f"{HEADER_BYTES} / {len(blob) - HEADER_BYTES} B",
            ),
            mo.stat(label="vs f32", value=f"{f32_bytes / len(blob):.2f}x"),
            mo.stat(label="vs JSON", value=f"{json_bytes / len(blob):.2f}x"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Reconstruction error
    """)
    return


@app.cell
def _(curves, np, recovered):
    l2 = np.sqrt(((curves - recovered).reshape(len(curves), -1) ** 2).sum(axis=1))
    coord_err = np.abs(curves - recovered)
    worst_index = int(np.argmax(l2))
    return coord_err, l2, worst_index


@app.cell
def _(coord_err, l2, mo, np, predicted_l2, target_l2):
    mo.hstack(
        [
            mo.stat(label="mean L2", value=f"{l2.mean():.3f}"),
            mo.stat(label="p95 L2", value=f"{np.percentile(l2, 95):.3f}"),
            mo.stat(label="max L2", value=f"{l2.max():.3f}"),
            mo.stat(
                label="bound / budget",
                value=f"{predicted_l2:.2f} / {target_l2.value:.1f}",
            ),
            mo.stat(label="max coord error", value=f"{coord_err.max():.3f}"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(curves, plot_curves, plt, recovered):
    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(18, 6))
    plot_curves(axes_cmp[0], curves, color="C0")
    axes_cmp[0].set_title("original")
    plot_curves(axes_cmp[1], recovered, color="C3")
    axes_cmp[1].set_title("decompressed")
    plot_curves(axes_cmp[2], curves, color="C0", alpha=0.5, lw=2.0, label="original")
    plot_curves(
        axes_cmp[2], recovered, color="C3", alpha=0.8, lw=0.8, label="decompressed"
    )
    axes_cmp[2].set_title("overlay")
    axes_cmp[2].legend(loc="upper right")
    fig_cmp.tight_layout()
    fig_cmp
    return


@app.cell
def _(curves, l2, mo, worst_index):
    curve_index = mo.ui.slider(
        start=0,
        stop=len(curves) - 1,
        step=1,
        value=worst_index,
        label=f"inspect curve (worst is {worst_index}, L2 = {l2[worst_index]:.2f})",
        show_value=True,
    )
    curve_index
    return (curve_index,)


@app.cell
def _(curve_index, curves, l2, plot_curves, plt, recovered):
    idx = curve_index.value
    fig_one, ax_one = plt.subplots(figsize=(7, 7))
    plot_curves(ax_one, curves[idx : idx + 1], color="C0", lw=2.5, label="original")
    plot_curves(ax_one, recovered[idx : idx + 1], color="C3", lw=1.5, label="decoded")
    for point, name in zip(curves[idx], ("P0", "P1", "P2", "P3"), strict=True):
        ax_one.plot(point[0], point[1], "o", color="C0")
        ax_one.annotate(
            name, point, textcoords="offset points", xytext=(8, 8), color="C0"
        )
    for point in recovered[idx]:
        ax_one.plot(point[0], point[1], "s", color="C3")
    ax_one.set_title(f"curve {idx} — L2 = {l2[idx]:.3f}")
    ax_one.legend()
    fig_one
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. TypeScript decoder

    Same layout, no dependencies, bit widths read from the header — so it keeps
    working when the budget slider picks a different allocation.
    """)
    return


@app.cell
def _():
    TS_DECODER = """// Decoder for compressed 2D cubic Bezier curves.
    // Layout: [u16 count][u8 bits x4][16 x f32 ranges][bit-packed payload]
    // Bit widths come from the header, so this decoder is universal.

    export function decodeBezierCurves(blob: ArrayBuffer): number[][] {
      const dv = new DataView(blob);
      const u8 = new Uint8Array(blob);
      const n = dv.getUint16(0, true);
      const bP0 = u8[2];
      const bD = u8[3];
      const bH1 = u8[4];
      const bH2 = u8[5];
      const f: number[] = [];
      for (let i = 0; i < 16; i++) f.push(dv.getFloat32(6 + i * 4, true));
      const p0Lo = [f[0], f[1]], p0Hi = [f[2], f[3]];
      const dLo = [f[4], f[5]], dHi = [f[6], f[7]];
      const h1Lo = [f[8], f[9]], h1Hi = [f[10], f[11]];
      const h2Lo = [f[12], f[13]], h2Hi = [f[14], f[15]];

      let pos = 6 + 64;
      let acc = 0, nb = 0;
      const readBits = (k: number): number => {
        while (nb < k) { acc |= u8[pos++] << nb; nb += 8; }
        const v = acc & ((1 << k) - 1);
        acc >>>= k; nb -= k;
        return v;
      };
      const dq = (q: number, lo: number, hi: number, bits: number): number =>
        lo + (q / ((1 << bits) - 1)) * (hi - lo);

      const out: number[][] = [];
      for (let i = 0; i < n; i++) {
        const p0x = dq(readBits(bP0), p0Lo[0], p0Hi[0], bP0);
        const p0y = dq(readBits(bP0), p0Lo[1], p0Hi[1], bP0);
        const dx = dq(readBits(bD), dLo[0], dHi[0], bD);
        const dy = dq(readBits(bD), dLo[1], dHi[1], bD);
        const h1x = dq(readBits(bH1), h1Lo[0], h1Hi[0], bH1);
        const h1y = dq(readBits(bH1), h1Lo[1], h1Hi[1], bH1);
        const h2x = dq(readBits(bH2), h2Lo[0], h2Hi[0], bH2);
        const h2y = dq(readBits(bH2), h2Lo[1], h2Hi[1], bH2);
        const p3x = p0x + dx, p3y = p0y + dy;
        const p1x = p0x + h1x, p1y = p0y + h1y;
        const p2x = p3x + h2x, p2y = p3y + h2y;
        out.push([p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]);
      }
      return out;
    }
    """
    return (TS_DECODER,)


@app.cell
def _(TS_DECODER, blob, mo):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.download(
                        data=TS_DECODER.encode(),
                        filename="decodeBezierCurves.ts",
                        label="decodeBezierCurves.ts",
                    ),
                    mo.download(data=blob, filename="curves.bin", label="curves.bin"),
                ],
                justify="start",
                gap=1,
            ),
            mo.md(f"```ts\n{TS_DECODER}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
