import argparse
import json
import pathlib
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
        if len(cells) in (3, 6) and not set(cells[0]) <= set("-") and cells[0] not in ("function", "shape"):
            rows.extend(cells[i:i + 3] for i in range(0, len(cells), 3))
    return [r for r in rows if r[0]]


class Dump:
    def __init__(self, path):
        self.path = path
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def __call__(self, fn, tier, *args):
        dt = np.float64 if tier.endswith("f64") else np.float32
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
    dt = np.float64 if tier.endswith("f64") else np.float32
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
    info = json.loads(subprocess.run([args.dump, "info"], check=True, capture_output=True, text=True).stdout)
    margin = 0.0 if info["fma"] else oracle.NO_FMA_MARGIN

    for name, f32, f64 in table_rows(section(text, "Accuracy contracts")):
        check(float(f32) == oracle.CONTRACTS["f32"].get(name) and float(f64) == oracle.CONTRACTS["f64"].get(name), f"contract row {name}")
    listed = {r[0] for r in table_rows(section(text, "Accuracy contracts"))}
    check(listed == {k for k in oracle.CONTRACTS["f32"] if not k.startswith("adaa_")}, "contract table lists every function")
    for name, f1, first in table_rows(section(text, "ADAA")):
        check(oracle.ADAA_CONTRACTS.get(name) == (float(f1), float(first)), f"adaa row {name}")
    check(len(table_rows(section(text, "ADAA"))) == len(oracle.ADAA_CONTRACTS), "adaa table lists every shape")
    for name, f32, f64 in table_rows(section(text, "Balanced tier")):
        functions = ("sincos_sin", "sincos_cos") if name == "sincos" else (name,)
        for fn in functions:
            check(float(f32) == oracle.CONTRACTS["balanced_f32"].get(fn)
                  and float(f64) == oracle.CONTRACTS["balanced_f64"].get(fn), f"balanced contract row {fn}")
    check({r[0] for r in table_rows(section(text, "Balanced tier"))} == {"sin", "cos", "sincos", "tanh"},
          "balanced contract table lists every function")
    for tier, dt in (("balanced_f32", np.float32), ("balanced_f64", np.float64)):
        probes = [0.0, -0.0, 0.5, -100.0, 2.0e7, -float(np.finfo(dt).max), float(np.finfo(dt).max)]
        for fn, ref in (("sin", mp.sin), ("cos", mp.cos), ("sincos_sin", mp.sin), ("sincos_cos", mp.cos)):
            values = dump(fn, tier, probes)
            with mp.workdps(400):
                want = [ref(mp.mpf(float(dt(v)))) for v in probes]
                bound = oracle.CONTRACTS[tier][fn] + margin
                check(all(np.isfinite(g) and abs(mp.mpf(float(g)) - w) <= bound * ulp_of(w, dt)
                          for g, w in zip(values, want)), f"{fn} {tier} finite-domain contract probes")
            check(bool(np.all(np.isnan(dump(fn, tier, [np.inf, -np.inf, np.nan])))),
                  f"{fn} {tier} non-finite inputs produce NaN")
            zeros = dump(fn, tier, [0.0, -0.0])
            if fn in ("sin", "sincos_sin"):
                check(bool(np.all(zeros == 0) and not np.signbit(zeros[0]) and np.signbit(zeros[1])),
                      f"{fn} {tier} preserves signed zero")
            else:
                check(bool(np.all(zeros == 1)), f"{fn} {tier} of zero is one")
        probes = [0.0, -0.0, float(np.finfo(dt).tiny), -float(np.finfo(dt).tiny), 0.25, -0.25, 1.0, -1.0, 8.0, -8.0]
        values = dump("tanh", tier, probes)
        with mp.workdps(400):
            want = [mp.tanh(mp.mpf(float(dt(v)))) for v in probes]
            bound = oracle.CONTRACTS[tier]["tanh"] + margin
            check(all(np.isfinite(g) and abs(mp.mpf(float(g)) - w) <= bound * ulp_of(w, dt)
                      for g, w in zip(values, want)), f"tanh {tier} contract probes")
        zeros = dump("tanh", tier, [0.0, -0.0])
        check(bool(np.all(zeros == 0) and not np.signbit(zeros[0]) and np.signbit(zeros[1])),
              f"tanh {tier} preserves signed zero")
        special = dump("tanh", tier, [np.inf, -np.inf, np.nan, np.finfo(dt).max, -np.finfo(dt).max])
        check(bool(special[0] == 1 and special[1] == -1 and np.isnan(special[2])
                   and special[3] == 1 and special[4] == -1), f"tanh {tier} infinity, NaN and saturation")

    neg = dump("pow", "f64", [-2.0, -2.0, -2.0], [3.0, 2.0, 0.5])
    check(neg[0] == -8.0 and neg[1] == 4.0 and np.isnan(neg[2]), "pow negative bases follow C")

    for tier, dt in (("f32", np.float32), ("f64", np.float64)):
        with mp.workdps(50):
            limit = float(dt(float(mp.pi * (2048 if tier == "f32" else 2 ** 21))))
        probes = [float(limit), float(np.nextafter(dt(limit), dt(np.inf))), -2.0e7, float(np.finfo(dt).max), -float(np.finfo(dt).max)]
        for fn, ref in (("sin", mp.sin), ("cos", mp.cos), ("tan", mp.tan)):
            for t in ((tier, "fast") if tier == "f32" else (tier,)):
                values = dump(fn, t, probes)
                with mp.workdps(400):
                    want = [ref(mp.mpf(float(dt(v)))) for v in probes]
                ok = all(np.isfinite(g) and abs(mp.mpf(float(g)) - w) <= (abs(w) * 2.0 ** -16 if t == "fast" else ulp_of(w, dt)) for g, w in zip(values, want))
                check(ok, f"{fn} {t} accurate past the fast reduction range")
            check(np.isnan(dump(fn, tier, [np.inf])[0]), f"{fn} {tier} of infinity is NaN")

    def edge(fn, tier, lo, hi, kind):
        return threshold(dump, fn, tier, lo, hi, np.isfinite if kind == "over" else (lambda v: v == 0))

    for fn, tier, kind, inside, outside in (("exp", "f32", "over", "88.72", "88.73"), ("exp", "f64", "over", "709.78", "709.79"),
                                            ("exp", "f32", "under", "-103.97", "-103.98"), ("exp", "f64", "under", "-745.13", "-745.14"),
                                            ("db_to_gain", "f32", "over", "770.63", "770.64"), ("db_to_gain", "f64", "over", "6165.09", "6165.10"),
                                            ("db_to_gain", "f32", "under", "-903.08", "-903.09"), ("db_to_gain", "f64", "under", "-6472.14", "-6472.15")):
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
        e = dump("exp2", tier, [np.nextafter(dt(top), dt(0)), top, np.nextafter(dt(bottom), dt(0)), bottom])
        check(np.isfinite(e[0]) and np.isinf(e[1]) and e[2] > 0 and e[3] == 0, f"exp2 {tier} limits")
        p = dump("pow", tier, [2, 2], [np.nextafter(dt(top), dt(0)), top])
        check(np.isfinite(p[0]) and np.isinf(p[1]), f"pow {tier} overflow")

    t = dump("tanh", "fast", [5.9, -5.9, 5.8])
    check(t[0] == 1 and t[1] == -1 and t[2] < 1, "fast tanh saturates at 5.9")
    fast = dump("exp", "fast", [-1e30, 1e30, -np.inf, np.inf])
    check(bool(np.all(np.isfinite(fast)) and np.all(fast >= np.finfo(np.float32).tiny)), "fast exp stays finite and normal")
    fp = dump("pow", "fast", [2, 2], [-1e3, 1e3])
    check(bool(np.all(np.isfinite(fp)) and np.all(fp >= np.finfo(np.float32).tiny)), "fast pow stays finite and normal")

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
    check(f"{dump('log', 'fast', 0.0)[0]:.2f}" == "-88.03", "fast log of 0")
    check(0 < dump("sigmoid", "fast", -1e9)[0] < np.finfo(np.float32).tiny, "fast sigmoid subnormal")
    fx = dump("db_to_gain", "fast", [-1e9, 1e9])
    check(bool(np.all(np.isfinite(fx)) and np.all(fx >= np.finfo(np.float32).tiny)), "fast db_to_gain stays finite and normal")

    for f in failures:
        print("FAIL", f)
    print(f"readme: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
