import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cores
import remez

DPS = 60

TIERS = {"f32": ("float", 24), "f64": ("double", 53), "fast": ("float", 24)}

PLAN = {
    "exp": {"f32": (4, 0), "f64": (9, 0), "fast": (3, 0)},
    "log": {"f32": (2, 0), "f64": (6, 0), "fast": (1, 0)},
    "sin": {"f32": (2, 0), "f64": (5, 0), "fast": (1, 0)},
    "cos": {"f32": (2, 0), "f64": (5, 0), "fast": (1, 0)},
    "tan": {"f32": (6, 0), "f64": (3, 3)},
    "tan_unit": {"fast": (1, 1)},
    "atan": {"fast": (6, 0)},
    "atan_rest": {"f32": (8, 0), "f64": (5, 6)},
    "tanh": {"f32": (4, 0), "f64": (3, 3)},
    "tanh_range": {"fast": (3, 2)},
}

PARETO = {"exp": "f64", "log": "f64", "tan": "f64", "atan_rest": "f64", "tanh": "f64", "tanh_range": "f32"}
PARETO_MAX_TOTAL = 12
PARETO_MAX_DEN = 6


def round_to(x, bits):
    with mp.workprec(bits):
        return +mp.mpf(x)


def literal(x, ctype):
    bits = 24 if ctype == "float" else 53
    v = float(round_to(x, bits))
    text = v.hex()
    if "." in text:
        mantissa, exponent = text.split("p")
        text = mantissa.rstrip("0").rstrip(".") + "p" + exponent
    return text + "f" if ctype == "float" else text


def interval(name, tier):
    with mp.workdps(DPS):
        a, b = cores.CORES[name]["interval"](tier)
        return mp.mpf(a), mp.mpf(b)


def fit(job):
    name, tier, m, n = job
    with mp.workdps(DPS):
        a, b = interval(name, tier)
        g = cores.CORES[name]["g"]
        weight = cores.CORES[name].get("weight", lambda tier: "rel")(tier)
        try:
            r, err = remez.remez(g, a, b, m, n, weight=weight, dps=DPS)
        except remez.RemezFailure:
            return job, None
        curve = remez.weighted_error_curve(g, r, a, b, weight=weight, samples=1200, dps=DPS)
        return job, dict(p=[str(c) for c in r.p], q=[str(c) for c in r.q], error=float(err),
                         interval=[float(a), float(b)], curve=curve)


def split(value, bits_hi, total_bits):
    with mp.workdps(DPS):
        hi = round_to(value, bits_hi)
        lo = round_to(value - hi, total_bits)
        return hi, lo


def split_n(value, bits, total_bits, n):
    with mp.workdps(DPS):
        parts = []
        rest = mp.mpf(value)
        for _ in range(n - 1):
            parts.append(round_to(rest, bits))
            rest -= parts[-1]
        parts.append(round_to(rest, total_bits))
        return parts


def constants(ctype):
    total = 24 if ctype == "float" else 53
    with mp.workdps(DPS):
        ln2 = mp.log(2)
        ln2_hi, ln2_lo = split(ln2, 16 if ctype == "float" else 42, total)
        pio2 = mp.pi / 2
        pio2_parts = split_n(pio2, 12, total, 4) if ctype == "float" else split_n(pio2, 31, total, 3)
        pi_hi, pi_lo = split(mp.pi, total, total)
        pio2_hi, pio2_lo = split(pio2, total, total)
        if ctype == "float":
            max_arg = mp.log(mp.mpf(2) ** 128 * (1 - mp.mpf(2) ** -24))
            min_arg = mp.log(mp.mpf(2) ** -150)
            exp2_max = mp.mpf(128)
            exp2_min = mp.mpf(-150)
            trig_k_max = mp.mpf(2) ** 12
            tanh_clamp = mp.mpf(cores.TANH_CLAMP["f32"])
            tanh_clamp_fast = mp.mpf(cores.TANH_CLAMP["fast"])
        else:
            max_arg = mp.log(mp.mpf(2) ** 1024 * (1 - mp.mpf(2) ** -53))
            min_arg = mp.log(mp.mpf(2) ** -1075)
            exp2_max = mp.mpf(1024)
            exp2_min = mp.mpf(-1075)
            trig_k_max = mp.mpf(2) ** 22
            tanh_clamp = mp.mpf(cores.TANH_CLAMP["f64"])
            tanh_clamp_fast = tanh_clamp
        pio2_terms = {f"pio2_{i + 1}": p for i, p in enumerate(pio2_parts)}
        return {
            "ln2": ln2, "ln2_hi": ln2_hi, "ln2_lo": ln2_lo, "inv_ln2": 1 / ln2,
            "inv_ln2_hi": split(1 / ln2, total, total)[0], "inv_ln2_lo": split(1 / ln2, total, total)[1],
            "veltkamp": mp.mpf(2) ** (12 if ctype == "float" else 27) + 1,
            "sqrt2": mp.sqrt(2), "sqrt_half": mp.sqrt(mp.mpf(1) / 2),
            **pio2_terms, "two_over_pi": 2 / mp.pi,
            "pi_hi": pi_hi, "pi_lo": pi_lo, "pio2_hi": pio2_hi, "pio2_lo": pio2_lo,
            "third_hi": split(mp.mpf(1) / 3, total, total)[0], "third_lo": split(mp.mpf(1) / 3, total, total)[1],
            "max_arg": round_to(max_arg, total), "min_arg": round_to(min_arg, total),
            "exp2_max": exp2_max, "exp2_min": exp2_min, "trig_arg_max": trig_k_max * pio2,
            "tanh_clamp": tanh_clamp, "tanh_clamp_fast": tanh_clamp_fast, "tanh_switch": mp.mpf("0.625"),
            "db_per_log2": 20 * mp.log10(2), "log2_per_db": mp.log(10, 2) / 20,
            "log2_per_db_hi": split(mp.log(10, 2) / 20, total, total)[0],
            "log2_per_db_lo": split(mp.log(10, 2) / 20, total, total)[1],
            "db_per_log2_hi": split(20 * mp.log10(2), 16 if ctype == "float" else 42, total)[0],
            "db_per_log2_lo": split(20 * mp.log10(2), 16 if ctype == "float" else 42, total)[1],
            "db_per_ln_hi": split(20 / mp.log(10), total, total)[0],
            "db_per_ln_lo": split(20 / mp.log(10), total, total)[1],
        }


def emit_header(results, out):
    lines = ["#pragma once", "", "#include <array>", "", "namespace wray::math::coeff", "{"]
    for (name, tier), data in sorted(results.items()):
        ctype = TIERS[tier][0]
        p = ", ".join(literal(mp.mpf(c), ctype) for c in data["p"])
        q = ", ".join(literal(mp.mpf(c), ctype) for c in data["q"])
        lines += [
            f"    struct {name}_{tier}",
            "    {",
            f"        using value_type = {ctype};",
            f"        static constexpr std::array<{ctype}, {len(data['p'])}> p{{ {p} }};",
            f"        static constexpr std::array<{ctype}, {len(data['q'])}> q{{ {q} }};",
            "    };",
            "",
        ]
    lines[-1:] = ["}", ""]
    lines += ["namespace wray::math", "{", "    template <class T>", "    struct constants;", ""]
    for ctype in ("float", "double"):
        lines += ["    template <>", f"    struct constants<{ctype}>", "    {"]
        for key, value in constants(ctype).items():
            lines.append(f"        static constexpr {ctype} {key} = {literal(value, ctype)};")
        lines += ["    };", ""]
    lines[-1:] = ["}", ""]
    out.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", required=True)
    parser.add_argument("--report")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    plan_jobs = [(name, tier, m, n) for name, tiers in PLAN.items() for tier, (m, n) in tiers.items()]
    pareto_jobs = [(name, tier, m, n) for name, tier in PARETO.items()
                   for n in range(PARETO_MAX_DEN + 1) for m in range(0, PARETO_MAX_TOTAL + 1 - n)] if args.report else []
    with ProcessPoolExecutor(args.jobs) as pool:
        plan_out = dict(pool.map(fit, plan_jobs))
        pareto_out = dict(pool.map(fit, pareto_jobs))

    results = {}
    for (name, tier, m, n), data in plan_out.items():
        if data is None:
            raise SystemExit(f"fit failed: {name} {tier} [{m}/{n}]")
        results[(name, tier)] = data

    pathlib.Path(args.header).parent.mkdir(parents=True, exist_ok=True)
    emit_header(results, pathlib.Path(args.header))
    if not args.report:
        return

    report = {
        "cores": [dict(name=name, tier=tier, m=len(d["p"]) - 1, n=len(d["q"]) - 1, error=d["error"],
                       interval=d["interval"], curve=[[float(f"{x:.6g}"), float(f"{e:.5g}")] for x, e in d["curve"]])
                  for (name, tier), d in sorted(results.items())],
        "pareto": [dict(name=name, tier=tier, m=m, n=n, error=None if d is None else d["error"])
                   for (name, tier, m, n), d in sorted(pareto_out.items())],
        "plan": {name: {tier: list(mn) for tier, mn in tiers.items()} for name, tiers in PLAN.items()},
    }
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
