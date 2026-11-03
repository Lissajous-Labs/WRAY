import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import offset_copy

THEMES = {
    "dark": {
        "ink": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d", "surface": "#161b22",
        "colors": {"libm": "#848d97", "vforce": "#79c0ff", "wray": "#ff6a57", "wray fast": "#ffb340",
                   "f32": "#ff6a57", "f64": "#bc9cff", "fast": "#ffb340", "contract": "#e6edf3"},
        "spectra": ["#ff6a57", "#ffb340", "#bc9cff", "#848d97"],
        "ramp": ["#161b22", "#3d2a6e", "#9b2f6d", "#ff5c4a", "#ffd166"],
    },
    "light": {
        "ink": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "surface": "#f6f8fa",
        "colors": {"libm": "#6e7781", "vforce": "#0969da", "wray": "#cf2e1c", "wray fast": "#9a6700",
                   "f32": "#cf2e1c", "f64": "#7c3aed", "fast": "#9a6700", "contract": "#1f2328"},
        "spectra": ["#cf2e1c", "#9a6700", "#7c3aed", "#6e7781"],
        "ramp": ["#f6f8fa", "#d9c8f5", "#c86aa8", "#e5432f", "#ffb02e"],
    },
}
SANS = ["Avenir Next", "Helvetica Neue", "Inter", "Arial", "DejaVu Sans"]
MONO = ["Menlo", "SF Mono", "JetBrains Mono", "DejaVu Sans Mono"]
LABELS = {"libm": "scalar libm", "vforce": "Apple vForce", "wray": "WRAY precise", "wray fast": "WRAY fast"}
FUNCTIONS = ["product_log", "omega", "ratio_from_cents", "hz_to_midi", "midi_to_hz", "erf", "cbrt", "asinh", "cosh", "sinh", "tanpi", "tan_turns", "cos_turns", "sin_turns", "log10", "log1p", "exp", "expm1", "exp2", "log", "log2", "sin", "cos", "tan", "atan", "atan2", "tanh", "sigmoid", "pow", "db_to_gain", "gain_to_db"]
BENCH_FUNCTIONS = ["product_log", "omega", "smoothing_coefficient", "pole_from_cutoff", "prewarp", "ratio_from_cents", "hz_to_midi", "midi_to_hz", "erf", "cbrt", "asinh", "cosh", "sinh", "tanpi", "tan_turns", "cos_turns", "sin_turns", "log10", "log1p", "exp", "expm1", "exp2", "log", "log2", "sin", "cos", "sincos", "tan", "atan", "atan2", "tanh", "sigmoid", "pow", "db_to_gain", "gain_to_db"]
SPEEDUP_FUNCTIONS = ["exp", "log", "sin", "cos", "sincos", "tanh", "pow", "db_to_gain"]
TIER_LABEL = {"f32": "float precise", "f64": "double precise", "fast": "float fast"}

T = {}
COLORS = {}


def style(theme):
    T.clear()
    T.update(THEMES[theme], name=theme)
    COLORS.clear()
    COLORS.update(T["colors"])
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.monospace": MONO, "font.size": 9.5,
        "text.color": T["ink"], "axes.labelcolor": T["muted"], "axes.titlecolor": T["ink"],
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9, "axes.edgecolor": T["grid"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.6, "grid.alpha": 0.9, "axes.axisbelow": True,
        "xtick.color": T["muted"], "ytick.color": T["muted"], "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 6, "ytick.major.pad": 6, "patch.linewidth": 0,
        "legend.frameon": False, "legend.fontsize": 9, "legend.handlelength": 1.0, "legend.handleheight": 0.8,
        "legend.labelcolor": T["ink"], "lines.solid_capstyle": "round",
    })


def save(fig, out, name):
    fig.savefig(out / f"{name}-{T['name']}.png", dpi=200, bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.close(fig)


def drawn_axes(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.axison]


def header(fig, title, subtitle=None):
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
        fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    axes = drawn_axes(fig)
    left = min(ax.get_position().x0 for ax in axes)
    top = max(ax.get_tightbbox(renderer).y1 for ax in axes) / fig.bbox.height
    lift = 26 if subtitle else 10
    fig.text(left, top, title, fontsize=15, fontweight="bold", color=T["ink"], ha="left", va="bottom",
             transform=offset_copy(fig.transFigure, fig=fig, y=lift, units="points"))
    if subtitle:
        fig.text(left, top, subtitle, fontsize=9.5, color=T["muted"], ha="left", va="bottom",
                 transform=offset_copy(fig.transFigure, fig=fig, y=10, units="points"))


def footer(fig, text):
    renderer = fig.canvas.get_renderer()
    axes = drawn_axes(fig)
    left = min(ax.get_position().x0 for ax in axes)
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes) / fig.bbox.height
    fig.text(left, bottom, text, fontsize=8.5, color=T["muted"], ha="left", va="top",
             transform=offset_copy(fig.transFigure, fig=fig, y=-8, units="points"))


def value_axis(ax, axis="y"):
    ax.grid(axis="x" if axis == "y" else "y", visible=False)
    if axis == "x":
        ax.spines["bottom"].set_visible(False)


def mono_ticks(labels):
    for label in labels:
        label.set_fontfamily("monospace")
        label.set_fontsize(8.5)
        label.set_color(T["ink"])


def grid(n, cols, w=3.6, h=2.6):
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w, rows * h))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def envelope(y, bins=1400):
    y = np.asarray(y)
    x = np.linspace(0, 0.5, len(y))
    if len(y) <= 2 * bins:
        return x, y
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    peaks = np.maximum.reduceat(y, edges[:-1])
    return 0.5 * (edges[:-1] + edges[1:] - 1) / (len(y) - 1) * 0.5, peaks


def panel_title(ax, name, note=None):
    ax.set_title(name, fontfamily="monospace", fontsize=10, fontweight="semibold", loc="left")
    if note:
        ax.text(1.0, 1.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=T["muted"])


def remez_chart(cores, out):
    entries = cores["cores"]
    fig, axes = grid(len(entries), 4, 3.6, 2.2)
    for ax, c in zip(axes, entries):
        t = np.array([p[0] for p in c["curve"]])
        e = np.array([p[1] for p in c["curve"]]) / c["error"]
        color = COLORS[c["tier"]]
        ax.fill_between(t, e, 0, color=color, alpha=0.12, linewidth=0)
        ax.plot(t, e, color=color, linewidth=1.3)
        for level in (-1, 1):
            ax.axhline(level, color=T["muted"], linewidth=0.7, linestyle=(0, (2, 3)))
        ax.set_ylim(-1.35, 1.35)
        ax.set_yticks([-1, 0, 1])
        ax.grid(False)
        ax.spines["bottom"].set_color(T["grid"])
        panel_title(ax, c["name"], f"{TIER_LABEL[c['tier']]} · [{c['m']}/{c['n']}] · E {c['error']:.1e}")
    fig.tight_layout(h_pad=2.2)
    header(fig, "Minimax cores", "weighted error ÷ E across each reduced interval; touching ±1 everywhere is equioscillation")
    save(fig, out, "remez")


def pareto_chart(cores, out):
    names = sorted({p["name"] for p in cores["pareto"]})
    cols = 3
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.4, rows_n * 3.4), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    plan = cores["plan"]
    cmap = LinearSegmentedColormap.from_list("wray", T["ramp"][::-1])
    cmap.set_bad((0, 0, 0, 0))
    im = None
    for ax, name in zip(axes, names):
        rows = [p for p in cores["pareto"] if p["name"] == name]
        mmax = max(p["m"] for p in rows)
        nmax = max(p["n"] for p in rows)
        z = np.full((nmax + 1, mmax + 1), np.nan)
        for p in rows:
            if p["error"]:
                z[p["n"], p["m"]] = math.log10(p["error"])
        im = ax.imshow(np.ma.masked_invalid(z), origin="lower", cmap=cmap, aspect="auto", vmin=-30, vmax=0)
        for p in rows:
            if p["error"]:
                v = math.log10(p["error"])
                r, g, b = to_rgb(cmap((v + 30) / 30))
                light = 0.299 * r + 0.587 * g + 0.114 * b > 0.55
                ax.text(p["m"], p["n"], f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="#1f2328" if light else "#f0f3f6")
        for m, n in plan.get(name, {}).values():
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=False, edgecolor=COLORS["contract"], linewidth=1.8))
        ax.set_xlabel("numerator degree m")
        ax.set_ylabel("denominator degree n")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_title(ax, name)
    for ax in axes[len(names):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=list(axes[: len(names)]), shrink=0.6, aspect=30)
    cbar.outline.set_visible(False)
    cbar.set_label("log10 minimax error")
    header(fig, "Rational vs polynomial", "log10 minimax error for every [m/n]; outlined cells ship, blank cells are degenerate")
    save(fig, out, "pareto")


def axis_values(x):
    finite = np.isfinite(x)
    span = np.max(np.abs(x[finite])) if finite.any() else 1.0
    if span > 1e4:
        return np.sign(x) * np.log10(1 + np.abs(x)), "sign(x) log10(1 + |x|)"
    return x, "x"


def accuracy_chart(acc, tier, out):
    rows = {r["function"]: r for r in acc if r["tier"] == tier}
    names = [f for f in FUNCTIONS if f in rows]
    fig, axes = grid(len(names), 4, 3.6, 2.4)
    for ax, name in zip(axes, names):
        r = rows[name]
        x = np.array(r["x"])
        y = np.array(r["rel"] if tier == "fast" else r["ulp"])
        if name == "pow":
            ys = np.array(r["y"])
            x = ys * np.log2(np.abs(x))
            if tier == "fast":
                y = y / (1 + np.abs(x))
        finite = np.isfinite(y) & np.isfinite(x)
        xv, label = axis_values(x[finite])
        ax.scatter(xv, y[finite], s=2.4, color=COLORS[tier], alpha=0.6, linewidths=0, rasterized=True)
        ax.axhline(r["contract"], color=COLORS["contract"], linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xlabel("y log2 x" if name == "pow" else label, fontsize=8.5)
        ax.set_ylim(0, r["contract"] * 1.25)
        if tier == "fast":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        panel_title(ax, name, f"{r['score']:.3g} / {r['contract']:.3g}")
    unit = "relative error, pow scaled by 1 + |y log2 x|" if tier == "fast" else "error in ulp against mpmath at 40 digits"
    fig.tight_layout(h_pad=2.0)
    header(fig, f"Accuracy · {TIER_LABEL[tier]}", f"{unit}; dashed line is the contract, corner shows worst / contract")
    save(fig, out, f"accuracy_{tier}")


def summary_chart(acc, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    idx = np.arange(len(FUNCTIONS))
    width = 0.36
    for k, tier in enumerate(["f32", "f64"]):
        rows = {r["function"]: r for r in acc if r["tier"] == tier}
        vals = [rows[f]["score"] for f in FUNCTIONS]
        caps = [rows[f]["contract"] for f in FUNCTIONS]
        x = idx + (k - 0.5) * width
        left.bar(x, vals, width * 0.92, color=COLORS[tier], label=TIER_LABEL[tier])
        left.scatter(x, caps, marker="_", s=70, linewidths=1.4, color=COLORS["contract"], zorder=3,
                     label="contract" if k == 0 else None)
    left.set_xticks(idx, FUNCTIONS, rotation=55, ha="right", rotation_mode="anchor")
    mono_ticks(left.get_xticklabels())
    left.set_ylabel("worst error (ulp)")
    left.set_xlim(-0.7, len(FUNCTIONS) - 0.3)
    left.set_title("Precise tier")
    left.legend(ncols=3, loc="upper right", bbox_to_anchor=(1, 1.14))
    value_axis(left)
    rows = {r["function"]: r for r in acc if r["tier"] == "fast"}
    fast = [f for f in FUNCTIONS if f in rows]
    fast_idx = np.arange(len(fast))
    right.barh(fast_idx, [rows[f]["score"] for f in fast], 0.62, color=COLORS["fast"])
    right.axvline(2.0 ** -16, color=COLORS["contract"], linestyle=(0, (4, 3)), linewidth=0.9)
    right.text(2.0 ** -16, -1.1, r" $2^{-16}$ contract", color=T["muted"], va="center", fontsize=8.5)
    right.set_xscale("log")
    right.set_yticks(fast_idx, fast)
    mono_ticks(right.get_yticklabels())
    right.set_ylim(len(fast) - 0.5, -1.6)
    right.set_xlabel("worst relative error")
    right.set_title("Fast tier · float")
    value_axis(right, "x")
    fig.tight_layout(w_pad=3)
    header(fig, "Worst measured error", "against mpmath at 40 digits, 50k samples per function and tier")
    save(fig, out, "accuracy_summary")


def trig_chart(acc, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, name in zip(axes, ["sin", "cos", "tan"]):
        for tier in ["f32", "f64"]:
            r = next(r for r in acc if r["function"] == name and r["tier"] == tier)
            x = np.abs(np.array(r["x"]))
            y = np.array(r["ulp"])
            keep = (x > 0) & np.isfinite(y)
            ax.scatter(np.log10(x[keep]), y[keep], s=2.4, alpha=0.6, color=COLORS[tier], label=TIER_LABEL[tier],
                       linewidths=0, rasterized=True)
        panel_title(ax, name)
        ax.set_xlabel("log10 |x|")
    axes[0].set_ylabel("ulp")
    axes[0].legend(markerscale=7, loc="upper left")
    fig.tight_layout()
    header(fig, "Trig error across the whole range", "three-part reduction for ordinary arguments, exact 2/π reduction beyond")
    save(fig, out, "trig_range")


def bench_table(bench, mode="throughput"):
    table = {}
    for row in bench["rows"]:
        if row["mode"] == mode:
            table.setdefault(row["type"], {}).setdefault(row["function"], {})[row["variant"]] = row["median"]
    return table


def grouped_bars(ax, data, names, variants):
    idx = np.arange(len(names))
    width = 0.82 / max(len(variants), 1)
    for k, v in enumerate(variants):
        vals = [data[f].get(v, np.nan) for f in names]
        ax.bar(idx + (k - (len(variants) - 1) / 2) * width, vals, width * 0.9, color=COLORS[v], label=LABELS[v])
    ax.set_xticks(idx, names, rotation=55, ha="right", rotation_mode="anchor")
    mono_ticks(ax.get_xticklabels())
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="minor", visible=False)
    value_axis(ax)


def latency_chart(bench, machine, out):
    table = bench_table(bench, "latency")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, kind in zip(axes, ["f32", "f64"]):
        data = table.get(kind, {})
        names = [f for f in BENCH_FUNCTIONS if f in data]
        variants = [v for v in ["libm", "wray", "wray fast"] if any(v in data[f] for f in names)]
        grouped_bars(ax, data, names, variants)
        ax.set_ylabel("ns per dependent call, log scale")
        ax.set_title("float" if kind == "f32" else "double")
    axes[1].legend(*axes[0].get_legend_handles_labels(), ncols=3, loc="upper right", bbox_to_anchor=(1, 1.14))
    fig.tight_layout(w_pad=3)
    header(fig, "Scalar latency · lower is better", f"median of dependent call chains · {machine}")
    save(fig, out, "latency")


def throughput_chart(bench, machine, out):
    table = bench_table(bench)
    for kind in ["f32", "f64"]:
        data = table[kind]
        names = [f for f in BENCH_FUNCTIONS if f in data]
        variants = [v for v in ["libm", "vforce", "wray", "wray fast"] if any(v in data[f] for f in names)]
        fig, ax = plt.subplots(figsize=(13, 4.6))
        grouped_bars(ax, data, names, variants)
        ax.set_ylabel("ns per input, log scale")
        ax.legend(ncols=len(variants), loc="upper right", bbox_to_anchor=(1, 1.12))
        fig.tight_layout()
        header(fig, f"{'Float' if kind == 'f32' else 'Double'} buffer throughput · lower is better",
               f"median paired samples over 4096 inputs; sincos writes both outputs · {machine}")
        save(fig, out, f"throughput_{kind}")


def speedup_chart(bench, machine, out):
    names = [f for f in SPEEDUP_FUNCTIONS if any(r["function"] == f for r in bench["rows"])]
    baselines = ["libm"]
    if any(r["variant"] == "vforce" for r in bench["rows"]):
        baselines.append("vforce")
    ratios = {(r["type"], r["function"], r["variant"], c["baseline"]): c
              for r in bench["rows"] if r["mode"] == "throughput" for c in r["comparisons"]}
    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 6.4), sharey=True, squeeze=False)
    series = [("f32", "wray", "float precise", COLORS["f32"]), ("f32", "wray fast", "float fast", COLORS["fast"]),
              ("f64", "wray", "double precise", COLORS["f64"])]
    idx = np.arange(len(names))
    width = 0.25
    for ax, baseline in zip(axes[0], baselines):
        top = 1.0
        for k, (kind, variant, label, color) in enumerate(series):
            points = [ratios.get((kind, f, variant, baseline)) for f in names]
            y = idx + (k - 1) * width
            vals = np.array([p["speedup"] if p else np.nan for p in points])
            ax.barh(y, vals, width * 0.86, color=color, label=label)
            for position, point in zip(y, points):
                if point:
                    ax.plot([point["ci_low"], point["ci_high"]], [position, position], color=T["ink"], linewidth=0.8, alpha=0.6)
                    ax.annotate(f"{point['speedup']:.2f}×", (point["ci_high"], position), xytext=(5, 0),
                                textcoords="offset points", va="center", fontsize=8.5, color=color, fontweight="semibold")
                    top = max(top, point["ci_high"])
        ax.axvline(1, color=T["muted"], linewidth=0.9, linestyle=(0, (2, 3)))
        ax.text(1, -0.75, " parity", color=T["muted"], fontsize=8.5, va="center")
        ax.set_yticks(idx, names)
        mono_ticks(ax.get_yticklabels())
        for label in ax.get_yticklabels():
            label.set_fontsize(10.5)
        ax.set_ylim(len(names) - 0.5, -1.0)
        ax.set_xlim(0, top * 1.22)
        ax.set_xlabel("paired throughput ratio · right of parity is faster")
        ax.set_title(f"vs {LABELS[baseline]}", fontsize=12)
        value_axis(ax, "x")
    axes[0, -1].legend(ncols=3, loc="lower right", bbox_to_anchor=(1, 1.06))
    fig.tight_layout(w_pad=3)
    header(fig, "Buffer throughput", machine)
    footer(fig, "Bars are medians of paired rounds; whiskers are 95% bootstrap intervals. Contracts and domains differ by library; scalar latency can be slower.")
    save(fig, out, "speedup")


def spectral_charts(spec, out):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    oscillators = spec["oscillators"]
    for k, (row, color) in enumerate(zip(oscillators, T["spectra"])):
        x, y = envelope(row["spectrum"])
        ax.plot(x, y, color=color, linewidth=1.0, zorder=len(oscillators) - k + 1, label=f"{row['name']} · SFDR {row['sfdr']:.0f} dB")
    ax.set_xlabel("frequency / sample rate")
    ax.set_ylabel("dB relative to carrier")
    ax.set_ylim(-340, 10)
    ax.set_xlim(0, 0.5)
    ax.legend(ncols=2, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8.5)
    value_axis(ax)
    fig.tight_layout()
    header(fig, "Oscillator purity", f"sin_turns spectra, {spec['size']}-point FFT, worst of four tones")
    save(fig, out, "oscillator_sfdr")

    rows = spec["aliasing"]
    fig, axes = grid(len(rows), 3, 4.2, 2.7)
    for ax, row in zip(axes, rows):
        x, naive = envelope(row["naive"], 700)
        _, adaa = envelope(row["adaa"], 700)
        ax.plot(x, naive, color=COLORS["libm"], linewidth=0.9, label="pointwise")
        ax.plot(x, adaa, color=COLORS["wray"], linewidth=0.9, label="first-order ADAA")
        ax.set_ylim(-160, 10)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel("frequency / sample rate")
        value_axis(ax)
        panel_title(ax, row["name"], f"−{row['reduction']:.1f} dB aliasing")
    axes[min(2, len(axes) - 1)].legend(ncols=2, loc="lower right", bbox_to_anchor=(1, 1.16), fontsize=8.5)
    fig.tight_layout(h_pad=2.4)
    header(fig, "Antialiased waveshaping", "4× overdriven sine at 0.092 fs, pointwise vs first-order ADAA")
    save(fig, out, "adaa_aliasing")

    sm = spec["smoothing"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.loglog(sm["samples"], np.maximum(sm["naive"], 1e-3), color=COLORS["libm"], linewidth=1.0, label="1 − exp(−1 / (t fs)) in float")
    ax.loglog(sm["samples"], np.maximum(sm["wray"], 1e-3), color=COLORS["wray"], linewidth=1.2, label="wray::smoothing_coefficient")
    ax.set_xlabel("time constant in samples (t fs)")
    ax.set_ylabel("error in ulp")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(which="minor", visible=False)
    fig.tight_layout()
    header(fig, "Smoothing coefficient precision", "float, error of the one-pole coefficient per time constant")
    save(fig, out, "smoothing_precision")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="docs/data")
    parser.add_argument("--out", default="docs/charts")
    parser.add_argument("--machine", default="")
    args = parser.parse_args()
    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cores = json.loads((data / "cores.json").read_text())
    acc = json.loads((data / "accuracy.json").read_text())
    bench = json.loads((data / "bench.json").read_text())
    machine = args.machine or f"{bench['meta']['cpu']} / {bench['meta']['backend']}"
    spectral = data / "spectral.json"
    spec = json.loads(spectral.read_text()) if spectral.exists() else None
    for theme in THEMES:
        style(theme)
        remez_chart(cores, out)
        pareto_chart(cores, out)
        for tier in ["f32", "f64", "fast"]:
            accuracy_chart(acc, tier, out)
        summary_chart(acc, out)
        trig_chart(acc, out)
        throughput_chart(bench, machine, out)
        latency_chart(bench, machine, out)
        speedup_chart(bench, machine, out)
        if spec:
            spectral_charts(spec, out)


if __name__ == "__main__":
    main()
