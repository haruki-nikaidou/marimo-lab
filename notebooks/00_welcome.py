import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from marimo_lab import PROJECT_ROOT, data_path

    return PROJECT_ROOT, alt, data_path, mo, np, pl


@app.cell
def _(PROJECT_ROOT, mo):
    mo.md(f"""
    # marimo-lab

    Reactive research workspace. Cells re-run when their inputs change, so
    there is no stale state to reason about.

    Project root: `{PROJECT_ROOT}`
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(start=50, stop=2000, step=50, value=400, label="samples")
    noise = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.0, label="noise sigma")
    slope = mo.ui.slider(start=-3.0, stop=3.0, step=0.1, value=1.5, label="true slope")
    mo.hstack([n_points, noise, slope], justify="start", gap=2)
    return n_points, noise, slope


@app.cell
def _(n_points, noise, np, pl, slope):
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(-5, 5, size=n_points.value)
    y = slope.value * x + rng.normal(0.0, noise.value, size=n_points.value)
    samples = pl.DataFrame({"x": x, "y": y})
    return samples, x, y


@app.cell
def _(np, x, y):
    fit_slope, fit_intercept = np.polyfit(x, y, deg=1)
    residuals = y - (fit_slope * x + fit_intercept)
    r_squared = 1.0 - residuals.var() / y.var()
    return fit_intercept, fit_slope, r_squared


@app.cell
def _(fit_intercept, fit_slope, mo, r_squared, slope):
    mo.hstack(
        [
            mo.stat(label="fitted slope", value=f"{fit_slope:.3f}"),
            mo.stat(label="bias", value=f"{fit_slope - slope.value:+.3f}"),
            mo.stat(label="intercept", value=f"{fit_intercept:.3f}"),
            mo.stat(label="R²", value=f"{r_squared:.3f}"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(alt, mo, samples):
    chart = mo.ui.altair_chart(
        alt.Chart(samples)
        .mark_circle(opacity=0.5)
        .encode(x="x:Q", y="y:Q")
        .properties(height=320)
    )
    chart
    return (chart,)


@app.cell
def _(chart, mo):
    mo.md(f"""
    Selected **{len(chart.value)}** points — drag on the chart to filter,
    downstream cells react.
    """)
    return


@app.cell
def _(data_path, mo, samples):
    mo.md(f"""
    ## Persisting results

    Use `data_path()` so paths resolve from the project root regardless of
    the notebook's working directory:

    ```python
    samples.write_parquet(data_path("processed", "samples.parquet"))
    ```

    Target: `{data_path("processed", "samples.parquet")}`
    ({samples.height} rows ready to write)
    """)
    return


@app.cell
def _(mo, samples):
    mo.ui.table(samples.head(10), selection=None)
    return


if __name__ == "__main__":
    app.run()
