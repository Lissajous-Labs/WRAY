import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d0d7de"
COLORS = {"std": "#8c959f", "wray": "#0969da", "wray fast": "#1a7f37",
          "f32": "#0969da", "f64": "#8250df", "fast": "#1a7f37", "contract": "#cf222e"}
FUNCTIONS = ["product_log", "omega", "ratio_from_cents", "hz_to_midi", "midi_to_hz", "erf", "cbrt", "asinh", "cosh", "sinh", "tan_turns", "cos_turns", "sin_turns", "log10", "log1p", "exp", "expm1", "exp2", "log", "log2", "sin", "cos", "tan", "atan", "atan2", "tanh", "sigmoid", "pow", "db_to_gain", "gain_to_db"]
NO_STD = {"omega", "product_log"}
TIER_LABEL = {"f32": "float precise", "f64": "double precise", "fast": "float fast"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "figure.titlesize": 13, "figure.titleweight": "bold", "legend.frameon": False,
    })


def save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grid(n, cols):
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def remez_chart(cores, out):
    entries = cores["cores"]
    fig, axes = grid(len(entries), 4)
    for ax, c in zip(axes, entries):
        t = np.array([p[0] for p in c["curve"]])
        e = np.array([p[1] for p in c["curve"]])
        color = COLORS[c["tier"]]
        ax.plot(t, e / c["error"], color=color, linewidth=1.1)
        ax.axhline(1, color=MUTED, linewidth=0.7, linestyle="--")
        ax.axhline(-1, color=MUTED, linewidth=0.7, linestyle="--")
        ax.set_ylim(-1.35, 1.35)
        ax.set_title(f"{c['name']}  {TIER_LABEL[c['tier']]}  [{c['m']}/{c['n']}]")
        ax.text(0.98, 0.06, f"E = {c['error']:.2e}", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)
        ax.set_yticks([-1, 0, 1])
    fig.suptitle("Minimax cores: weighted error divided by E (equioscillation)", y=1.0)
    fig.tight_layout()
    save(fig, out, "remez.png")


def pareto_chart(cores, out):
    names = sorted({p["name"] for p in cores["pareto"]})
    cols = 3
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.4, rows_n * 3.6), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    plan = cores["plan"]
    im = None
    for ax, name in zip(axes, names):
        rows = [p for p in cores["pareto"] if p["name"] == name]
        tier = rows[0]["tier"]
        mmax = max(p["m"] for p in rows)
        nmax = max(p["n"] for p in rows)
        z = np.full((nmax + 1, mmax + 1), np.nan)
        for p in rows:
            if p["error"]:
                z[p["n"], p["m"]] = math.log10(p["error"])
        im = ax.imshow(z, origin="lower", cmap="viridis_r", aspect="auto", vmin=-30, vmax=0)
        for p in rows:
            if p["error"]:
                v = math.log10(p["error"])
                ax.text(p["m"], p["n"], f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=INK if v < -14 else "white")
        chosen = [mn for mn in plan.get(name, {}).values()]
        for m, n in chosen:
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=False, edgecolor=COLORS["contract"], linewidth=2))
        ax.set_xlabel("numerator degree m")
        ax.set_ylabel("denominator degree n")
        ax.set_title(name)
        ax.grid(False)
    for ax in axes[len(names):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=list(axes[: len(names)]), shrink=0.8)
    cbar.set_label("log10 minimax error")
    fig.suptitle("Rational vs polynomial: log10 minimax error for every [m/n]; red boxes are shipped cores; blank = degenerate")
    save(fig, out, "pareto.png")


def axis_values(x):
    finite = np.isfinite(x)
    span = np.max(np.abs(x[finite])) if finite.any() else 1.0
    if span > 1e4:
        return np.sign(x) * np.log10(1 + np.abs(x)), "sign(x) log10(1 + |x|)"
    return x, "x"


def accuracy_chart(acc, tier, out):
    rows = {r["function"]: r for r in acc if r["tier"] == tier}
    names = [f for f in FUNCTIONS if f in rows]
    fig, axes = grid(len(names), 4)
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
        ax.scatter(xv, y[finite], s=1.2, color=COLORS[tier], alpha=0.45, linewidths=0)
        ax.axhline(r["contract"], color=COLORS["contract"], linewidth=1.0, linestyle="--")
        ax.set_xlabel(label, color=MUTED, fontsize=7.5)
        ax.set_title(name)
        ax.set_ylim(0, r["contract"] * 1.25)
        if name == "pow":
            ax.set_xlabel("y log2 x")
        if tier == "fast":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    unit = "relative error (pow: / (1 + |y log2 x|))" if tier == "fast" else "error in ulp vs mpmath at 40 digits"
    fig.suptitle(f"{TIER_LABEL[tier]}: {unit}; dashed = contract", y=1.0)
    fig.tight_layout()
    save(fig, out, f"accuracy_{tier}.png")


def summary_chart(acc, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    idx = np.arange(len(FUNCTIONS))
    width = 0.38
    for k, tier in enumerate(["f32", "f64"]):
        rows = {r["function"]: r for r in acc if r["tier"] == tier}
        vals = [rows[f]["score"] for f in FUNCTIONS]
        caps = [rows[f]["contract"] for f in FUNCTIONS]
        left.bar(idx + (k - 0.5) * width, vals, width, color=COLORS[tier], label=TIER_LABEL[tier])
        left.scatter(idx + (k - 0.5) * width, caps, marker="_", s=180, color=COLORS["contract"], zorder=3,
                     label="contract" if k == 0 else None)
    left.set_xticks(idx, FUNCTIONS, rotation=40, ha="right")
    left.set_ylabel("max error (ulp)")
    left.set_title("Precise tier: worst error vs mpmath (50k samples per function)")
    left.legend(ncols=3, loc="upper left")
    rows = {r["function"]: r for r in acc if r["tier"] == "fast"}
    fast = [f for f in FUNCTIONS if f in rows]
    fast_idx = np.arange(len(fast))
    right.barh(fast_idx, [rows[f]["score"] for f in fast], color=COLORS["fast"])
    right.axvline(2.0 ** -16, color=COLORS["contract"], linestyle="--", linewidth=1)
    right.text(2.0 ** -16, len(fast) - 0.4, "  2^-16", color=COLORS["contract"], va="center")
    right.set_xscale("log")
    right.set_yticks(fast_idx, fast)
    right.invert_yaxis()
    right.set_xlabel("max relative error")
    right.set_title("Fast tier (float): worst relative error")
    fig.tight_layout()
    save(fig, out, "accuracy_summary.png")


def trig_chart(acc, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, name in zip(axes, ["sin", "cos", "tan"]):
        for tier in ["f32", "f64"]:
            r = next(r for r in acc if r["function"] == name and r["tier"] == tier)
            x = np.abs(np.array(r["x"]))
            y = np.array(r["ulp"])
            keep = (x > 0) & np.isfinite(y)
            ax.scatter(x[keep], y[keep], s=1.2, alpha=0.45, color=COLORS[tier], label=TIER_LABEL[tier], linewidths=0)
        ax.set_xscale("log")
        ax.set_title(f"{name}: error vs |x| up to the reduction limit")
        ax.set_xlabel("|x|")
        ax.set_ylabel("ulp")
        ax.legend(markerscale=6, loc="upper left")
    fig.tight_layout()
    save(fig, out, "trig_range.png")


def bench_table(bench, mode="throughput"):
    table = {}
    for row in bench:
        if row.get("mode", "throughput") == mode:
            table.setdefault(row["type"], {}).setdefault(row["function"], {})[row["variant"]] = row["ns"]
    return table


def latency_chart(bench, machine, out):
    table = bench_table(bench, "latency")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, kind in zip(axes, ["f32", "f64"]):
        data = table.get(kind, {})
        names = [f for f in FUNCTIONS if f in data]
        variants = [v for v in ["std", "wray", "wray fast"] if any(v in data[f] for f in names)]
        idx = np.arange(len(names))
        width = 0.8 / max(len(variants), 1)
        for k, v in enumerate(variants):
            vals = [data[f].get(v, np.nan) for f in names]
            ax.bar(idx + (k - (len(variants) - 1) / 2) * width, vals, width, color=COLORS[v], label=v)
        ax.set_xticks(idx, names, rotation=40, ha="right")
        ax.set_ylabel("ns per dependent call (lower is better)")
        ax.set_title(f"{'float' if kind == 'f32' else 'double'} single-value latency ({machine})")
        ax.legend(ncols=len(variants))
    fig.tight_layout()
    save(fig, out, "latency.png")


def throughput_chart(bench, machine, out):
    table = bench_table(bench)
    for kind in ["f32", "f64"]:
        data = table[kind]
        names = [f for f in FUNCTIONS if f in data]
        variants = [v for v in ["std", "wray", "wray fast"] if any(v in data[f] for f in names)]
        fig, ax = plt.subplots(figsize=(12, 4.4))
        idx = np.arange(len(names))
        width = 0.8 / len(variants)
        for k, v in enumerate(variants):
            vals = [data[f].get(v, np.nan) for f in names]
            ax.bar(idx + (k - (len(variants) - 1) / 2) * width, vals, width, color=COLORS[v], label=v)
        ax.set_xticks(idx, names, rotation=35, ha="right")
        ax.set_ylabel("ns per element (lower is better)")
        ax.set_title(f"{'float' if kind == 'f32' else 'double'} throughput, 4096-element arrays, fastest of 25 interleaved rounds ({machine})")
        ax.legend(ncols=len(variants))
        fig.tight_layout()
        save(fig, out, f"throughput_{kind}.png")


def speedup_chart(bench, machine, out):
    table = bench_table(bench)
    names = [f for f in FUNCTIONS if f not in NO_STD]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    idx = np.arange(len(names))
    series = [("f32", "wray", "float precise"), ("f32", "wray fast", "float fast"), ("f64", "wray", "double precise")]
    width = 0.8 / len(series)
    colors = [COLORS["f32"], COLORS["fast"], COLORS["f64"]]
    for k, (kind, variant, label) in enumerate(series):
        vals = [table[kind][f]["std"] / table[kind][f][variant] if variant in table[kind].get(f, {}) else np.nan for f in names]
        bars = ax.bar(idx + (k - 1) * width, vals, width, color=colors[k], label=label)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=6.5, color=MUTED)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_xticks(idx, names, rotation=35, ha="right")
    ax.set_ylabel("speedup over std:: (x)")
    ax.set_title(f"Speedup over the C++ standard library ({machine})")
    ax.legend(ncols=3, loc="upper left")
    fig.tight_layout()
    save(fig, out, "speedup.png")


def spectral_charts(spec, out):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for row, color in zip(spec["oscillators"], ["#0969da", "#1a7f37", "#8250df", "#8c959f"]):
        y = np.array(row["spectrum"])
        x = np.linspace(0, 0.5, len(y))
        ax.plot(x, y, color=color, linewidth=0.7, label=f"{row['name']}  SFDR {row['sfdr']:.0f} dB")
    ax.set_xlabel("frequency / sample rate")
    ax.set_ylabel("dB relative to the carrier")
    ax.set_title(f"sin_turns oscillator spectra, {spec['size']}-point FFT, worst of four tones")
    ax.set_ylim(-340, 10)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "oscillator_sfdr.png")

    rows = spec["aliasing"]
    fig, axes = grid(len(rows), 3)
    for ax, row in zip(axes, rows):
        naive = np.array(row["naive"])
        adaa = np.array(row["adaa"])
        x = np.linspace(0, 0.5, len(naive))
        ax.plot(x, naive, color=COLORS["std"], linewidth=0.6, label="pointwise")
        ax.plot(x, adaa, color=COLORS["wray"], linewidth=0.6, label="first-order ADAA")
        ax.set_title(f"{row['name']}: aliasing -{row['reduction']:.1f} dB")
        ax.set_ylim(-160, 10)
        ax.set_xlabel("frequency / sample rate")
    axes[0].legend(fontsize=7)
    fig.suptitle("Aliasing of a 4x overdriven sine at 0.092 fs")
    fig.subplots_adjust(hspace=0.5)
    save(fig, out, "adaa_aliasing.png")

    sm = spec["smoothing"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.loglog(sm["samples"], np.maximum(sm["naive"], 1e-3), color=COLORS["std"], linewidth=0.9, label="1 - exp(-1 / (t fs)) in float")
    ax.loglog(sm["samples"], np.maximum(sm["wray"], 1e-3), color=COLORS["wray"], linewidth=0.9, label="wray::smoothing_coefficient")
    ax.set_xlabel("time constant in samples (t fs)")
    ax.set_ylabel("error in ulp")
    ax.set_title("Smoothing coefficient precision, float")
    ax.legend(fontsize=8)
    save(fig, out, "smoothing_precision.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="docs/data")
    parser.add_argument("--out", default="docs/charts")
    parser.add_argument("--machine", default="")
    args = parser.parse_args()
    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    cores = json.loads((data / "cores.json").read_text())
    acc = json.loads((data / "accuracy.json").read_text())
    bench = json.loads((data / "bench.json").read_text())
    remez_chart(cores, out)
    pareto_chart(cores, out)
    for tier in ["f32", "f64", "fast"]:
        accuracy_chart(acc, tier, out)
    summary_chart(acc, out)
    trig_chart(acc, out)
    throughput_chart(bench, args.machine, out)
    latency_chart(bench, args.machine, out)
    speedup_chart(bench, args.machine, out)
    spectral = data / "spectral.json"
    if spectral.exists():
        spectral_charts(json.loads(spectral.read_text()), out)


if __name__ == "__main__":
    main()
