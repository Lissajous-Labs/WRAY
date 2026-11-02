import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "oracle"))
import mpmath as mp
import oracle


def verify(dense, seed, count):
    out = subprocess.run([dense, "--references", "--samples", str(count), "--seed", str(seed)], check=True, capture_output=True, text=True).stdout
    mp.mp.dps = 60
    worst = {}
    for line in out.splitlines()[1:]:
        name, x, y, ref = line.split()
        args = (mp.mpf(float.fromhex(x)),) if name in oracle.UNARY else (mp.mpf(float.fromhex(x)), mp.mpf(float.fromhex(y)))
        want = oracle.REFERENCE[name](*args)
        got = mp.mpf(ref)
        if not mp.isfinite(want) or want == 0:
            rel = 0 if got == want else mp.inf
        elif abs(want) < mp.ldexp(1, -1100):
            rel = 0 if abs(got) < mp.ldexp(1, -1100) else mp.inf
        else:
            rel = abs(got - want) / abs(want)
        worst[name] = max(worst.get(name, 0), float(rel))
    failures = 0
    for name, rel in sorted(worst.items()):
        ok = rel <= 2.0 ** -70
        failures += 0 if ok else 1
        print(f"reference {name:>22} {rel:.3g} relative to mpmath  {'ok' if ok else 'FAIL'}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verify", type=int, default=200)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("only", nargs="*")
    args = parser.parse_args()
    index, count = (int(v) for v in args.shard.split("/"))
    names = args.only or [n for n in oracle.UNARY + oracle.BINARY if n in oracle.CONTRACTS["f64"] and not n.startswith("adaa_")]
    names = names[index::count]
    failures = verify(args.dense, args.seed, args.verify) if args.verify else 0
    run = subprocess.run([args.dense, "--samples", str(args.samples), "--seed", str(args.seed), "--pow-scale", str(oracle.POW_SCALE)] + names,
                         check=True, capture_output=True, text=True).stdout.splitlines()
    print(run[0])
    fma = run[0].split()[-1] == "1"
    margin = 0.0 if fma else oracle.NO_FMA_MARGIN
    for line in run[1:]:
        name, value, x, y = line.split()
        bound = oracle.CONTRACTS["f64"][name] + margin
        ok = float(value) <= bound
        failures += 0 if ok else 1
        print(f"{name:>22} {float(value):10.4g} ulp (contract {bound:.4g})  {'ok' if ok else 'FAIL'}  worst x {float.fromhex(x)!r} y {float.fromhex(y)!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
