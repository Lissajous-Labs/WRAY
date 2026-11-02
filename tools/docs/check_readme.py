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
    heading = f"## {title}\n"
    parts = []
    start = text.index(heading)
    while start >= 0:
        end = text.find("\n## ", start + 1)
        parts.append(text[start:] if end < 0 else text[start:end])
        start = text.find(heading, start + 1)
    return "\n".join(parts)


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


def ulp_of(value, dt):
    import mpmath as mp
    fi = np.finfo(dt)
    e = max(int(mp.floor(mp.log(abs(value), 2))), int(np.log2(fi.tiny))) if value != 0 else int(np.log2(fi.tiny))
    return mp.ldexp(1, e - fi.nmant)


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
    import mpmath as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--details", default=str(ROOT / "docs" / "details.md"))
    args = parser.parse_args()
    text = pathlib.Path(args.readme).read_text(encoding="utf-8") + "\n" + pathlib.Path(args.details).read_text(encoding="utf-8")
    dump = Dump(args.dump)
    coeff = (ROOT / "include/wray/math/coefficients.h").read_text(encoding="utf-8")
    kernels = (ROOT / "include/wray/math/kernels.h").read_text(encoding="utf-8")
    common = (ROOT / "include/wray/filters/common.h").read_text(encoding="utf-8")

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

    claims("add 0.5 ulp", "arm64 always has FMA")
    check("bit-identical apart from the sign and payload of NaN results" in text, "README qualifies bit-identity for NaN")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    check("inputs.dense_samples || '100000000'" in ci and "10^8 inputs per `double` function" in text, "dense double sample count")
    check(all(f"simd: {s} }}" in ci.split("  exhaustive:")[1].split("  double-dense:")[0] for s in ("sse2", "avx2", "scalar", "avx512")), "exhaustive backends")
    hostile = (ROOT / "tools/ci/hostile_flags.sh").read_text(encoding="utf-8")
    for flag in re.findall(r'"[a-z/]+\|(-[a-z0-9=+-]+)', hostile):
        check(f"`{flag}`" in text, f"README states the outcome of {flag}")
    check("tools/ci/hostile_flags.sh" in ci and "tools/ci/header_only.sh" in ci and "tools/ci/mixed_link.sh" in ci, "hostility scripts run in CI")
    check("windows-11-arm" in ci and "MSVC (Windows on ARM)" in text, "Windows on ARM is claimed and tested")
    strict = "-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wdouble-promotion -Wold-style-cast -Wshadow"
    check(strict in text and strict in ci and '"/W4", "/WX"' in ci, "strict warning set is claimed and tested")
    check("__aarch64__" in (ROOT / "include/wray/simd/config.h").read_text(encoding="utf-8").split("#define WRAY_HAS_FMA 1")[0].split("\n")[-2], "arm64 builds have FMA")
    contracts = section(text, "Accuracy contracts")
    check(f"|y·log₂|x|| ≤ {oracle.POW_SCALE:g}" in contracts and f"|y·log₂|x|| / {oracle.POW_SCALE:g} ulp" in contracts, "pow scaling threshold")
    neg = dump("pow", "f64", [-2.0, -2.0, -2.0], [3.0, 2.0, 0.5])
    check(neg[0] == -8.0 and neg[1] == 4.0 and np.isnan(neg[2]), "pow negative bases follow C")
    check(oracle.NO_FMA_MARGIN == 0.5, "no-FMA margin")

    for tier, kind, shown in (("f32", "float", "6433.98"), ("f64", "double", "6588397.3")):
        limit = hex_constant(coeff, "trig_arg_max", kind)
        claims(shown)
        with mp.workdps(50):
            exact = mp.pi * (2048 if tier == "f32" else 2 ** 21)
        dt = np.float64 if tier == "f64" else np.float32
        check(float(dt(float(exact))) == limit and f"{limit:.{len(shown.split('.')[1])}f}" == shown, f"trig limit {tier} is {shown}")
        dt = np.float64 if tier == "f64" else np.float32
        probes = [float(limit), float(np.nextafter(dt(limit), dt(np.inf))), -2.0e7, float(np.finfo(dt).max), -float(np.finfo(dt).max)]
        for fn, ref in (("sin", mp.sin), ("cos", mp.cos), ("tan", mp.tan)):
            for t in ((tier, "fast") if tier == "f32" else (tier,)):
                values = dump(fn, t, probes)
                with mp.workdps(400):
                    want = [ref(mp.mpf(float(dt(v)))) for v in probes]
                ok = all(np.isfinite(g) and abs(mp.mpf(float(g)) - w) <= (abs(w) * 2.0 ** -16 if t == "fast" else ulp_of(w, dt)) for g, w in zip(values, want))
                check(ok, f"{fn} {t} accurate past the fast reduction range")
            check(np.isnan(dump(fn, tier, [np.inf])[0]), f"{fn} {tier} of infinity is NaN")
    claims("1280-bit table of 2/π")
    generate = (ROOT / "tools/remez/generate.py").read_text(encoding="utf-8")
    check(re.search(r"TWO_OVER_PI_WORDS = (\d+)", generate) and 32 * int(re.search(r"TWO_OVER_PI_WORDS = (\d+)", generate).group(1)) == 1280, "2/pi table size")

    def edge(fn, tier, lo, hi, kind):
        return threshold(dump, fn, tier, lo, hi, np.isfinite if kind == "over" else (lambda v: v == 0))

    for fn, tier, kind, inside, outside in (("exp", "f32", "over", "88.72", "88.73"), ("exp", "f64", "over", "709.78", "709.79"),
                                            ("exp", "f32", "under", "-103.97", "-103.98"), ("exp", "f64", "under", "-745.13", "-745.14"),
                                            ("db_to_gain", "f32", "over", "770.63", "770.64"), ("db_to_gain", "f64", "over", "6165.09", "6165.10"),
                                            ("db_to_gain", "f32", "under", "-903.08", "-903.09"), ("db_to_gain", "f64", "under", "-6472.14", "-6472.15")):
        claims(inside, outside)
        dt = np.float64 if tier == "f64" else np.float32
        lo, hi = (inside, outside) if kind == "over" else (outside, inside)
        a, b = edge(fn, tier, float(lo), float(hi), "under" if kind == "under" else "over")
        values = dump(fn, tier, [float(dt(float(inside))), float(dt(float(outside)))])
        if kind == "over":
            ok = np.isfinite(values[0]) and np.isinf(values[1]) and dt(float(inside)) <= dt(a) and dt(b) <= dt(float(outside))
        else:
            ok = values[0] > 0 and values[1] == 0 and dt(float(outside)) <= dt(a) and dt(b) <= dt(float(inside))
        check(bool(ok), f"{fn} {tier} {kind}flows between {inside} and {outside}")
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
    fast_api = (ROOT / "include/wray/math.h").read_text(encoding="utf-8").split("namespace wray::inline WRAY_ABI::fast")[1]
    check("concept single = std::is_same_v<simd::scalar_t<V>, float>;" in fast_api and "template<simd::real V>" not in fast_api, "fast tier is float only")

    filters = section(text, "Filters")
    for dt, stated in ((np.float32, 1.07e7), (np.float64, 5.73e15)):
        with mp.workdps(60):
            edge = mp.tan(mp.pi * mp.mpf(float(np.nextafter(dt(0.5), dt(0)))))
        check(abs(float(edge) / stated - 1) < 5e-3, f"prewarp clamp {stated} matches tan(pi * prev(0.5)) = {float(edge):.4g}")
    for text in ("at least fs/1000", "DC gain of 0.22", "`biquad<float, double>`", "within 1e-5 down to 1e-6 of the sample rate"):
        check(text in filters, f"filters section states {text}")
    tests_text = (ROOT / "tests/filters_test.cpp").read_text(encoding="utf-8")
    check("std::size_t deep_cases = 4000;" in tests_text and "4000 random parameter sets" in filters, "deep sweep case count")
    check("one_pole_kind> ? 4.0 : (std::is_same_v<K, biquad_kind> ? 1.0e4 : 1.0e8)" in tests_text and "below 1e8" in filters and "below 4 for inputs" in filters, "modulation bounds match the test")
    check("std::numeric_limits<T>::max() / T(8)" in tests_text and "an eighth of the format's maximum" in filters, "full-scale headroom matches the test")
    check("if (ratio >= 1.0e-3)" in tests_text and "passband_ok(narrow, f0, 5e-3)" in tests_text and "passband_ok(mixed, f0, 1e-5)" in tests_text, "filter limits match the tests")
    for number in ("[1e-6, 0.4999]", "[1e-3, 1e3]", "±120 dB", "48 kHz", "2^-80", "2^-200"):
        check(number in filters, f"filters section states {number}")
    for name, value in (("lowest_frequency", 1.0e-6), ("highest_frequency", 0.4999), ("lowest_q", 1.0e-3), ("highest_q", 1.0e3),
                        ("default_sample_rate", 48000.0)):
        m = re.search(rf"inline constexpr double {name} = ([0-9.e+-]+);", common)
        check(m is not None and float(m.group(1)) == value, f"filter constant {name}")
    check("gain_db < -120.0 ? -120.0 : (gain_db > 120.0 ? 120.0" in common and "std::isnan(gain_db) ? 0.0" in common, "gain clamp")
    check("S(0x1p-80) : S(0x1p-200)" in common, "flush thresholds")

    config = (ROOT / "include/wray/simd/config.h").read_text(encoding="utf-8")
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
