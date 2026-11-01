import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
import oracle

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)


def section(text, title):
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def table_rows(block):
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and not set(cells[0]) <= set("-") and cells[0] not in ("function", "shape"):
            rows += [cells[0:3], cells[3:6]]
    return [r for r in rows if r[0]]


def hex_constant(text, name, kind):
    m = re.search(rf"static constexpr {kind} {name} = ([-0-9a-fx.p+]+);", text)
    value = m.group(1)
    value = value[:-1] if kind == "float" and value.endswith("f") else value
    return float.fromhex(value) if "x" in value else float(value)


class Dump:
    def __init__(self, path):
        self.path = path
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def __call__(self, fn, tier, *args):
        dt = np.float64 if tier == "f64" else np.float32
        data = np.concatenate([np.atleast_1d(np.array(a, dtype=dt)) for a in args])
        data.tofile(self.dir / "in.bin")
        subprocess.run([self.path, fn, tier, "vector", str(self.dir / "in.bin"), str(self.dir / "out.bin")], check=True)
        return np.fromfile(self.dir / "out.bin", dtype=dt)


def correctly_rounded(value, dt):
    import mpmath as mp
    f = dt(float(value))
    if dt is np.float64:
        return f
    candidates = [np.nextafter(f, dt(-np.inf)), f, np.nextafter(f, dt(np.inf))]
    return min(candidates, key=lambda c: abs(mp.mpf(float(c)) - value))


def threshold(dump, fn, tier, lo, hi, keep):
    dt = np.float64 if tier == "f64" else np.float32
    lo, hi = dt(lo), dt(hi)
    while True:
        mid = dt((float(lo) + float(hi)) / 2)
        if mid in (lo, hi):
            return float(lo), float(hi)
        if keep(dump(fn, tier, mid)[0]):
            lo = mid
        else:
            hi = mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    args = parser.parse_args()
    text = pathlib.Path(args.readme).read_text()
    dump = Dump(args.dump)
    coeff = (ROOT / "include/wray/math/coefficients.h").read_text()
    kernels = (ROOT / "include/wray/math/kernels.h").read_text()
    common = (ROOT / "include/wray/filters/common.h").read_text()

    for name, f32, f64 in table_rows(section(text, "Accuracy contracts")):
        check(float(f32) == oracle.CONTRACTS["f32"].get(name) and float(f64) == oracle.CONTRACTS["f64"].get(name), f"contract row {name}")
    listed = {r[0] for r in table_rows(section(text, "Accuracy contracts"))}
    check(listed == {k for k in oracle.CONTRACTS["f32"] if not k.startswith("adaa_")}, "contract table lists every function")
    for name, f1, first in table_rows(section(text, "ADAA")):
        check(oracle.ADAA_CONTRACTS.get(name) == (float(f1), float(first)), f"adaa row {name}")
    check(len(table_rows(section(text, "ADAA"))) == len(oracle.ADAA_CONTRACTS), "adaa table lists every shape")

    limits = section(text, "Where it stops working")

    def claims(*numbers):
        for n in numbers:
            check(n in limits, f"README states {n}")

    claims("add 0.5 ulp")
    contracts = section(text, "Accuracy contracts")
    check(f"|y·log₂|x|| ≤ {oracle.POW_SCALE:g}" in contracts and f"|y·log₂|x|| / {oracle.POW_SCALE:g} ulp" in contracts, "pow scaling threshold")
    neg = dump("pow", "f64", [-2.0, -2.0, -2.0], [3.0, 2.0, 0.5])
    check(neg[0] == -8.0 and neg[1] == 4.0 and np.isnan(neg[2]), "pow negative bases follow C")
    check(oracle.NO_FMA_MARGIN == 0.5, "no-FMA margin")

    for tier, kind, shown in (("f32", "float", "6434"), ("f64", "double", "6.59e6")):
        limit = hex_constant(coeff, "trig_arg_max", kind)
        claims(shown)
        check(float(shown) == float(f"{limit:.3g}") if tier == "f64" else round(limit) == int(shown), f"trig limit {tier}")
        dt = np.float64 if tier == "f64" else np.float32
        above = np.nextafter(dt(limit), dt(np.inf))
        for fn in ("sin", "cos", "tan"):
            values = dump(fn, tier, [limit, above])
            check(np.isfinite(values[0]) and np.isnan(values[1]), f"{fn} {tier} returns NaN past the reduction range")
    check(not np.isnan(dump("sin", "fast", 1.0e6)[0]), "fast sin has no range check")

    def edge(fn, tier, lo, hi, kind):
        return threshold(dump, fn, tier, lo, hi, np.isfinite if kind == "over" else (lambda v: v == 0))

    for fn, tier, kind, lo, hi, shown in (("exp", "f32", "over", 1, 100, "88.72"), ("exp", "f64", "over", 1, 800, "709.78"),
                                          ("exp", "f32", "under", -110, 0, "-103.97"), ("exp", "f64", "under", -800, 0, "-745.13"),
                                          ("db_to_gain", "f32", "over", 1, 900, "770.64"), ("db_to_gain", "f64", "over", 1, 7000, "6165.09"),
                                          ("db_to_gain", "f32", "under", -1000, 0, "-903.09"), ("db_to_gain", "f64", "under", -7000, 0, "-6472.14")):
        a, b = edge(fn, tier, lo, hi, kind)
        claims(shown)
        edge_value = a if kind == "over" else b
        check(f"{edge_value:.2f}" == shown, f"{fn} {tier} {kind}flow at {edge_value}")
    for tier, top, bottom in (("f32", 128.0, -150.0), ("f64", 1024.0, -1075.0)):
        dt = np.float64 if tier == "f64" else np.float32
        claims(f"{top:g}", f"{bottom:g}")
        e = dump("exp2", tier, [np.nextafter(dt(top), dt(0)), top, np.nextafter(dt(bottom), dt(0)), bottom])
        check(np.isfinite(e[0]) and np.isinf(e[1]) and e[2] > 0 and e[3] == 0, f"exp2 {tier} limits")
        p = dump("pow", tier, [2, 2], [np.nextafter(dt(top), dt(0)), top])
        check(np.isfinite(p[0]) and np.isinf(p[1]), f"pow {tier} overflow")

    claims("[-87, 88]", "[-126, 127]", "from 5.9")
    check(hex_constant(kernels, "fast_exp_min", "float") == -87 and hex_constant(kernels, "fast_exp_max", "float") == 88, "fast exp clamp")
    check(hex_constant(kernels, "fast_exp2_min", "float") == -126 and hex_constant(kernels, "fast_exp2_max", "float") == 127, "fast exp2 clamp")
    check(f"{hex_constant(coeff, 'tanh_clamp_fast', 'float'):.2g}" == "5.9", "fast tanh clamp")
    t = dump("tanh", "fast", [5.9, -5.9, 5.8])
    check(t[0] == 1 and t[1] == -1 and t[2] < 1, "fast tanh saturates at 5.9")
    fast = dump("exp", "fast", [-1e30, 1e30, -np.inf, np.inf])
    check(bool(np.all(np.isfinite(fast)) and np.all(fast >= np.finfo(np.float32).tiny)), "fast exp stays finite and normal")
    fp = dump("pow", "fast", [2, 2], [-1e3, 1e3])
    check(bool(np.all(np.isfinite(fp)) and np.all(fp >= np.finfo(np.float32).tiny)), "fast pow stays finite and normal")

    claims("in double, `sin_turns(2^40 + 0.25)` is exactly 1", "not representable in float")
    check(dump("sin_turns", "f64", 2.0 ** 40 + 0.25)[0] == 1.0, "sin_turns exact for huge turns")
    check(float(np.float32(2.0 ** 40 + 0.25)) != 2.0 ** 40 + 0.25, "the turns example is not a float")
    for tier, dt in (("f32", np.float32), ("f64", np.float64)):
        import mpmath as mp
        eighths = np.arange(-16, 17, dtype=np.float64) / 8
        for fn, ref in (("sin_turns", mp.sinpi), ("cos_turns", mp.cospi)):
            got = dump(fn, tier, eighths)
            with mp.workdps(50):
                want = [correctly_rounded(ref(2 * mp.mpf(e)), dt) for e in eighths]
            check(all(g == w for g, w in zip(got, want)), f"{fn} {tier} correctly rounded at eighths")
    claims("fast `log` of 0 returns -88.03", "fast `sigmoid` can return subnormals")
    check(f"{dump('log', 'fast', 0.0)[0]:.2f}" == "-88.03", "fast log of 0")
    check(0 < dump("sigmoid", "fast", -1e9)[0] < np.finfo(np.float32).tiny, "fast sigmoid subnormal")
    fx = dump("db_to_gain", "fast", [-1e9, 1e9])
    check(bool(np.all(np.isfinite(fx)) and np.all(fx >= np.finfo(np.float32).tiny)), "fast db_to_gain stays finite and normal")
    claims("The fast tier exists for `float` only")
    fast_api = (ROOT / "include/wray/math.h").read_text().split("namespace wray::fast")[1]
    check("concept single = std::is_same_v<simd::scalar_t<V>, float>;" in fast_api and "template<simd::real V>" not in fast_api, "fast tier is float only")

    filters = section(text, "Filters")
    for number in ("[1e-6, 0.4999]", "[1e-3, 1e3]", "±120 dB", "48 kHz", "2^-80", "2^-200"):
        check(number in filters, f"filters section states {number}")
    for name, value in (("lowest_frequency", 1.0e-6), ("highest_frequency", 0.4999), ("lowest_q", 1.0e-3), ("highest_q", 1.0e3),
                        ("default_sample_rate", 48000.0)):
        m = re.search(rf"inline constexpr double {name} = ([0-9.e+-]+);", common)
        check(m is not None and float(m.group(1)) == value, f"filter constant {name}")
    check("gain_db < -120.0 ? -120.0 : (gain_db > 120.0 ? 120.0" in common and "std::isnan(gain_db) ? 0.0" in common, "gain clamp")
    check("S(0x1p-80) : S(0x1p-200)" in common, "flush thresholds")

    config = (ROOT / "include/wray/simd/config.h").read_text()
    for macro in ("__FAST_MATH__", "_M_FP_FAST", "__FINITE_MATH_ONLY__", "__ASSOCIATIVE_MATH__", "__RECIPROCAL_MATH__", "__NO_SIGNED_ZEROS__", "WRAY_ALLOW_FAST_MATH"):
        check(macro in config, f"config.h checks {macro}")
    for text in ("`__FAST_MATH__`", "`_M_FP_FAST`", "`-ffinite-math-only`", "`-fassociative-math`", "`-freciprocal-math`", "`-fno-signed-zeros`", "`wray::floating_point_ok()`", "WRAY_ALLOW_FAST_MATH"):
        claims(text)

    for f in failures:
        print("FAIL", f)
    print(f"readme: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
