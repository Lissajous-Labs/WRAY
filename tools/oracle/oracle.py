import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np

FORMATS = {
    "f32": dict(dtype=np.float32, bits=np.uint32, precision=24, emin=-126, trig=float.fromhex("0x1.921fb6p+12")),
    "fast": dict(dtype=np.float32, bits=np.uint32, precision=24, emin=-126, trig=float.fromhex("0x1.921fb6p+12")),
    "f64": dict(dtype=np.float64, bits=np.uint64, precision=53, emin=-1022, trig=float.fromhex("0x1.921fb54442d18p+22")),
    "balanced_f32": dict(dtype=np.float32, bits=np.uint32, precision=24, emin=-126, trig=float.fromhex("0x1.921fb6p+12")),
    "balanced_f64": dict(dtype=np.float64, bits=np.uint64, precision=53, emin=-1022, trig=float.fromhex("0x1.921fb54442d18p+22")),
}

def _omega_ref(x):
    if x == mp.inf:
        return mp.inf
    y = x - mp.log(x) if x > 1 else mp.exp(x)
    if y == 0:
        return y
    with mp.workdps(mp.mp.dps + 20):
        for _ in range(200):
            step = (y + mp.log(y) - x) * y / (1 + y)
            y -= step
            if abs(step) <= mp.mpf(10) ** (5 - mp.mp.dps) * abs(y):
                break
    return +y


def _product_log_ref(x):
    if x == 0:
        return x
    if x > 0:
        return _omega_ref(mp.log(x))
    if x < -1 / mp.e:
        return mp.nan
    lo, hi = mp.mpf(-1), mp.mpf(0)
    with mp.workdps(mp.mp.dps + 20):
        for _ in range(400):
            mid = (lo + hi) / 2
            if mid * mp.exp(mid) > x:
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2


def _wide_trig(f, x):
    with mp.workdps(mp.mp.dps + max(0, int(mp.log10(abs(x) + 1))) + 20):
        return +f(x)


REFERENCE = {
    "exp": lambda x: mp.exp(x),
    "expm1": lambda x: mp.expm1(x),
    "exp2": lambda x: mp.power(2, x),
    "log": lambda x: mp.log(x),
    "log1p": lambda x: mp.log1p(x),
    "log10": lambda x: mp.log10(x),
    "log2": lambda x: mp.log(x, 2),
    "sin_turns": lambda x: mp.sinpi(2 * x),
    "sin": lambda x: _wide_trig(mp.sin, x),
    "cos_turns": lambda x: mp.cospi(2 * x),
    "cos": lambda x: _wide_trig(mp.cos, x),
    "sincos_sin": lambda x: _wide_trig(mp.sin, x),
    "sincos_cos": lambda x: _wide_trig(mp.cos, x),
    "tanpi": lambda x: mp.sinpi(x) / mp.cospi(x) if mp.cospi(x) != 0 else mp.inf * mp.sign(x),
    "tan_turns": lambda x: mp.sinpi(2 * x) / mp.cospi(2 * x) if mp.cospi(2 * x) != 0 else mp.inf * mp.sign(x),
    "tan": lambda x: _wide_trig(mp.tan, x),
    "atan": lambda x: mp.atan(x),
    "sinh": lambda x: mp.sinh(x),
    "cosh": lambda x: mp.cosh(x),
    "asinh": lambda x: mp.asinh(x),
    "cbrt": lambda x: mp.sign(x) * mp.cbrt(abs(x)),
    "erf": lambda x: mp.erf(x),
    "midi_to_hz": lambda x: 440 * mp.power(2, (x - 69) / 12),
    "hz_to_midi": lambda x: 69 + 12 * mp.log(x / 440, 2),
    "ratio_from_cents": lambda x: mp.power(2, x / 1200),
    "omega": _omega_ref,
    "product_log": _product_log_ref,
    "tanh": lambda x: mp.tanh(x),
    "sigmoid": lambda x: 1 / (1 + mp.exp(-x)),
    "db_to_gain": lambda x: mp.power(10, x / 20),
    "gain_to_db": lambda x: 20 * mp.log10(x),
    "atan2": lambda y, x: mp.atan2(y, x),
    "smoothing_coefficient": lambda s, fs: -mp.expm1(-1 / (s * fs)),
    "pole_from_cutoff": lambda hz, fs: mp.exp(-2 * mp.pi * hz / fs),
    "prewarp": lambda hz, fs: mp.tan(mp.pi * hz / fs),
    "pow": lambda x, y: mp.power(x, y),
}


def _series_or(x, small, exact, cut=mp.mpf("1e-6")):
    return small(x) if abs(x) < cut else exact(x)


ADAA = {
    "hard_clip": (lambda x: max(-1, min(1, x)), lambda x: x * x / 2 if abs(x) <= 1 else abs(x) - mp.mpf(1) / 2),
    "soft_clip": (lambda x: (lambda c: c - c ** 3 / 3)(max(-1, min(1, x))), lambda x: x * x / 2 - x ** 4 / 12 if abs(x) <= 1 else 2 * abs(x) / 3 - mp.mpf(1) / 4),
    "tanh": (mp.tanh, lambda x: mp.log1p(2 * mp.sinh(x / 2) ** 2)),
    "atan": (mp.atan, lambda x: _series_or(x, lambda y: y ** 2 / 2 - y ** 4 / 12 + y ** 6 / 30, lambda y: y * mp.atan(y) - mp.log1p(y * y) / 2)),
    "algebraic": (lambda x: x / (1 + abs(x)), lambda x: _series_or(abs(x), lambda a: a ** 2 / 2 - a ** 3 / 3 + a ** 4 / 4 - a ** 5 / 5, lambda a: a - mp.log1p(a))),
    "radical": (lambda x: x / mp.sqrt(1 + x * x), lambda x: x * x / (1 + mp.sqrt(1 + x * x))),
}


def _mean(f, F):
    def g(a, b):
        if a == b:
            return f(a)
        with mp.workdps(120):
            return (F(b) - F(a)) / (b - a)
    return g


for _name, (_f, _F) in ADAA.items():
    REFERENCE[f"adaa_{_name}_F1"] = _F
    REFERENCE[f"adaa_{_name}_first"] = _mean(_f, _F)

UNARY = ["product_log", "omega", "ratio_from_cents", "hz_to_midi", "midi_to_hz", "erf", "cbrt", "asinh", "cosh", "sinh", "tanpi", "tan_turns", "cos_turns", "sin_turns", "log10", "log1p", "exp", "expm1", "exp2", "log", "log2", "sin", "cos", "tan", "atan", "tanh", "sigmoid", "db_to_gain", "gain_to_db"] + [f"adaa_{n}_F1" for n in ADAA]

CONTRACTS = {
    "f32": {"product_log": 2.05, "omega": 1.65, "smoothing_coefficient": 2.0, "pole_from_cutoff": 2.0, "prewarp": 2.0, "ratio_from_cents": 0.85, "hz_to_midi": 1.25, "midi_to_hz": 1.4, "erf": 0.9, "cbrt": 0.55, "asinh": 0.9, "cosh": 0.9, "sinh": 0.85, "tanpi": 1.2, "tan_turns": 1.2, "cos_turns": 0.8, "sin_turns": 1.0, "log10": 1.4, "log1p": 1.0, "exp": 1.0, "expm1": 2.0, "exp2": 1.0, "log": 1.0, "log2": 1.5, "sin": 1.0, "cos": 1.0, "tan": 1.75, "atan": 1.0, "tanh": 1.35,
            "sigmoid": 2.0, "db_to_gain": 1.5, "gain_to_db": 1.5, "atan2": 1.0, "pow": 2.0},
    "f64": {"product_log": 1.75, "omega": 1.4, "smoothing_coefficient": 2.0, "pole_from_cutoff": 2.0, "prewarp": 2.0, "ratio_from_cents": 0.9, "hz_to_midi": 1.2, "midi_to_hz": 0.9, "erf": 0.95, "cbrt": 0.55, "asinh": 0.95, "cosh": 0.95, "sinh": 0.9, "tanpi": 1.4, "tan_turns": 1.4, "cos_turns": 0.85, "sin_turns": 0.9, "log10": 1.1, "log1p": 1.0, "exp": 1.0, "expm1": 1.85, "exp2": 1.0, "log": 1.0, "log2": 1.15, "sin": 1.0, "cos": 1.0, "tan": 1.85, "atan": 1.3, "tanh": 1.4,
            "sigmoid": 1.75, "db_to_gain": 1.5, "gain_to_db": 1.5, "atan2": 1.35, "pow": 2.0},
    "fast": {name: 2.0 ** -16 for name in ["product_log", "omega", "tanpi", "tan_turns", "cos_turns", "sin_turns", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "atan", "tanh", "sigmoid",
                                           "db_to_gain", "gain_to_db", "atan2", "pow"]},
    "balanced_f32": {**{name: 2.5 for name in ("sin", "cos", "sincos_sin", "sincos_cos")}, "tanh": 3.0, "pow": 4.0},
    "balanced_f64": {**{name: 2.5 for name in ("sin", "cos", "sincos_sin", "sincos_cos")}, "tanh": 3.0, "pow": 4.0},
}
ADAA_CONTRACTS = {"hard_clip": (0.8, 1.0), "soft_clip": (1.5, 3.0), "tanh": (3.0, 10.0), "atan": (3.5, 15.0), "algebraic": (6.0, 62.0),
                  "radical": (3.0, 5.0)}
for _tier in ("f32", "f64"):
    for _name, (_f1, _first) in ADAA_CONTRACTS.items():
        CONTRACTS[_tier][f"adaa_{_name}_F1"] = _f1
        CONTRACTS[_tier][f"adaa_{_name}_first"] = _first
BINARY = ["atan2", "pow", "smoothing_coefficient", "pole_from_cutoff", "prewarp"] + [f"adaa_{n}_first" for n in ADAA]
NO_FMA_MARGIN = 0.5
POW_SCALE = 4.0


def bit_uniform(rng, fmt, lo, hi, n, signed=False):
    dtype, bits = fmt["dtype"], fmt["bits"]
    lo_b = np.array([lo], dtype=dtype).view(bits)[0]
    hi_b = np.array([hi], dtype=dtype).view(bits)[0]
    draw = rng.integers(int(lo_b), int(hi_b), size=n, dtype=np.uint64, endpoint=True).astype(bits)
    v = draw.view(dtype)
    if signed:
        v = np.where(rng.random(n) < 0.5, -v, v).astype(dtype)
    return v


def value_uniform(rng, fmt, lo, hi, n):
    return rng.uniform(lo, hi, size=n).astype(fmt["dtype"])


def domain(fn, tier, rng, n):
    fmt = FORMATS[tier]
    f32 = fmt["dtype"] == np.float32
    tiny = np.finfo(fmt["dtype"]).smallest_subnormal
    huge = np.finfo(fmt["dtype"]).max
    fast = tier == "fast"
    half = n // 2
    if fn == "exp":
        lo, hi = ((-87.0, 88.0) if fast else (-103.9, 88.72)) if f32 else (-745.0, 709.78)
        return np.concatenate([value_uniform(rng, fmt, lo, hi, half), bit_uniform(rng, fmt, 1e-6, min(-lo, hi), n - half, True)])
    if fn == "expm1":
        lo, hi = (-103.9, 88.72) if f32 else (-745.0, 709.78)
        return np.concatenate([value_uniform(rng, fmt, -2.0, 2.0, half), bit_uniform(rng, fmt, tiny, -lo, n - half, True)])
    if fn == "log1p":
        q = n // 4
        return np.concatenate([value_uniform(rng, fmt, -0.75, 2.0, half), bit_uniform(rng, fmt, tiny, huge, q), -bit_uniform(rng, fmt, tiny, 0.999, n - half - q)])
    if fn == "log10":
        return np.concatenate([bit_uniform(rng, fmt, tiny, huge, half), value_uniform(rng, fmt, 0.5, 20.0, n - half)])
    if fn in ("sin_turns", "cos_turns", "tanpi", "tan_turns"):
        q = n // 4
        r = n - half - q
        k = rng.integers(-64, 64, size=r).astype(np.float64) / 8
        near = np.nextafter(k.astype(fmt["dtype"]), (k + rng.choice([-1.0, 1.0], size=r)).astype(fmt["dtype"]))
        return np.concatenate([value_uniform(rng, fmt, -1.0, 1.0, half), bit_uniform(rng, fmt, tiny, huge, q, True), near])
    if fn in ("sinh", "cosh"):
        lim = 89.5 if f32 else 710.5
        return np.concatenate([value_uniform(rng, fmt, -3.0, 3.0, half), value_uniform(rng, fmt, -lim, lim, n // 4), bit_uniform(rng, fmt, tiny, lim, n - half - n // 4, True)])
    if fn == "asinh":
        return np.concatenate([value_uniform(rng, fmt, -4.0, 4.0, half), bit_uniform(rng, fmt, tiny, huge, n - half, True)])
    if fn == "cbrt":
        return np.concatenate([value_uniform(rng, fmt, -10.0, 10.0, half), bit_uniform(rng, fmt, tiny, huge, n - half, True)])
    if fn == "erf":
        return np.concatenate([value_uniform(rng, fmt, -6.5, 6.5, half), bit_uniform(rng, fmt, tiny, huge, n - half, True)])
    if fn in ("midi_to_hz", "hz_to_midi", "ratio_from_cents"):
        if fn == "midi_to_hz":
            return np.concatenate([value_uniform(rng, fmt, -100.0, 250.0, half), value_uniform(rng, fmt, -1700.0 if f32 else -12900.0, 1600.0 if f32 else 12300.0, n - half)])
        if fn == "ratio_from_cents":
            return np.concatenate([value_uniform(rng, fmt, -2400.0, 2400.0, half), value_uniform(rng, fmt, -170000.0 if f32 else -1290000.0, 150000.0 if f32 else 1220000.0, n - half)])
        return np.concatenate([np.exp(rng.uniform(np.log(1e-3), np.log(1e6), half)).astype(fmt["dtype"]), bit_uniform(rng, fmt, tiny, huge, n - half)])
    if fn in ("omega", "product_log"):
        if fn == "omega":
            wide = bit_uniform(rng, fmt, tiny, huge, n - half, True)
            return np.concatenate([value_uniform(rng, fmt, -10.0, 30.0, half), wide[wide >= -80.0] if fast else wide])
        q = n // 4
        return np.concatenate([value_uniform(rng, fmt, -0.36787944, 10.0, half), bit_uniform(rng, fmt, tiny, huge, q), value_uniform(rng, fmt, -0.36787944, -0.36, n - half - q)])
    if fn == "exp2":
        lo, hi = ((-126.0, 127.0) if fast else (-149.9, 127.99)) if f32 else (-1074.9, 1023.99)
        return np.concatenate([value_uniform(rng, fmt, lo, hi, half), bit_uniform(rng, fmt, 1e-6, min(-lo, hi), n - half, True)])
    if fn in ("log", "log2", "gain_to_db"):
        lo = np.finfo(fmt["dtype"]).tiny if fast else tiny
        return np.concatenate([bit_uniform(rng, fmt, lo, huge, half), value_uniform(rng, fmt, 0.5, 2.0, n - half)])
    if fn in ("sin", "cos", "sincos_sin", "sincos_cos", "tan"):
        trig = fmt["trig"]
        q = n // 4
        k = rng.integers(-int(trig / (np.pi / 2)), int(trig / (np.pi / 2)), size=q)
        near = (k * (np.pi / 2)).astype(fmt["dtype"])
        near = np.nextafter(near, near + rng.choice([-1, 1], size=q)).astype(fmt["dtype"])
        wide = q // 2
        if tier.startswith("balanced_"):
            dtype = fmt["dtype"]
            anchors = np.array([0.0, -0.0, tiny, -tiny, np.finfo(dtype).tiny, huge, -huge,
                                np.pi / 4, np.pi / 2, np.pi, trig, float.fromhex("0x1.a9adcc7f96cf0p+20")], dtype=dtype)
            boundaries = anchors[7:]
            edges = np.concatenate([anchors, -boundaries, np.nextafter(boundaries, dtype(0)),
                                    np.nextafter(boundaries, dtype(huge)), -np.nextafter(boundaries, dtype(0)),
                                    -np.nextafter(boundaries, dtype(huge))])
            if not f32:
                root_bits = np.array([float.fromhex("0x1.a9adcc7f96cf0p+20")], dtype=dtype).view(fmt["bits"])[0]
                neighborhood = np.arange(int(root_bits) - 8, int(root_bits) + 9, dtype=fmt["bits"]).view(dtype)
                edges = np.concatenate([edges, neighborhood, -neighborhood])
            return np.concatenate([value_uniform(rng, fmt, -2 * np.pi, 2 * np.pi, q),
                                   value_uniform(rng, fmt, -trig, trig, q), near,
                                   bit_uniform(rng, fmt, tiny, huge, n - 3 * q, True), edges])
        return np.concatenate([value_uniform(rng, fmt, -2 * np.pi, 2 * np.pi, q), value_uniform(rng, fmt, -trig, trig, q),
                               bit_uniform(rng, fmt, 1e-6, trig, n - 3 * q - wide, True), near[: q], bit_uniform(rng, fmt, trig, huge, wide, True)])
    if fn == "atan":
        return np.concatenate([bit_uniform(rng, fmt, tiny, huge, half, True), value_uniform(rng, fmt, -4.0, 4.0, n - half)])
    if fn == "tanh":
        if tier.startswith("balanced_"):
            dtype = fmt["dtype"]
            boundaries = np.array([np.finfo(dtype).tiny, 0.25, 0.5, 1.0, 8.0, 9.0, 10.0, 18.0, 19.0, 20.0], dtype=dtype)
            edges = np.concatenate([np.array([0.0, -0.0, tiny, -tiny, huge, -huge], dtype=dtype),
                                    boundaries, -boundaries, np.nextafter(boundaries, dtype(0)),
                                    np.nextafter(boundaries, dtype(huge)), -np.nextafter(boundaries, dtype(0)),
                                    -np.nextafter(boundaries, dtype(huge))])
            q = n // 4
            return np.concatenate([value_uniform(rng, fmt, -20.0, 20.0, half),
                                   bit_uniform(rng, fmt, tiny, 1.0, q, True),
                                   bit_uniform(rng, fmt, tiny, huge, n - half - q, True), edges])
        return np.concatenate([value_uniform(rng, fmt, -20.0, 20.0, half), bit_uniform(rng, fmt, tiny, 30.0, n - half, True)])
    if fn == "sigmoid":
        lim = (87.0 if fast else 103.0) if f32 else 744.0
        return value_uniform(rng, fmt, -lim, lim, n)
    if fn == "db_to_gain":
        lim = 700.0 if f32 else 6000.0
        return value_uniform(rng, fmt, -lim, lim, n)
    if fn == "atan2":
        return tuple(np.concatenate([bit_uniform(rng, fmt, tiny, huge, half, True), value_uniform(rng, fmt, -4.0, 4.0, n - half)])
                     for _ in range(2))
    if fn in ("smoothing_coefficient", "pole_from_cutoff", "prewarp"):
        r = np.exp(rng.uniform(np.log(1e-9), np.log(0.5), n))
        fs = np.where(rng.random(n) < 0.5, rng.choice([44100.0, 48000.0, 96000.0, 192000.0], n), rng.uniform(8000.0, 768000.0, n)).astype(fmt["dtype"])
        a = (1 / (r * fs.astype(np.float64))) if fn == "smoothing_coefficient" else r * fs.astype(np.float64)
        return (a.astype(fmt["dtype"]), fs)
    if fn.startswith("adaa_") and fn.endswith("_F1"):
        return np.concatenate([value_uniform(rng, fmt, -8.0, 8.0, half), bit_uniform(rng, fmt, tiny, huge, n - half, True)])
    if fn.startswith("adaa_"):
        q = n // 4
        r = n - 3 * q
        wide = bit_uniform(rng, fmt, 1e-3, huge, r, True)
        mirror = -(wide[: r // 2].astype(np.float64) * (1 + rng.uniform(-1e-3, 1e-3, r // 2)))
        mirror = np.clip(mirror, -float(huge), float(huge)).astype(fmt["dtype"])
        x0 = np.concatenate([value_uniform(rng, fmt, -8.0, 8.0, q), value_uniform(rng, fmt, -8.0, 8.0, q), value_uniform(rng, fmt, -8.0, 8.0, q), wide,
                             np.array([0.0, -huge, huge], dtype=fmt["dtype"])])
        near = (x0[q:2 * q].astype(np.float64) * (1 + rng.uniform(-1e-6, 1e-6, q))).astype(fmt["dtype"])
        x1 = np.concatenate([value_uniform(rng, fmt, -8.0, 8.0, q), near, x0[2 * q:3 * q], mirror, bit_uniform(rng, fmt, 1e-3, huge, r - r // 2, True),
                             np.array([huge, huge, -huge], dtype=fmt["dtype"])])
        e = max(n // 16, 8)
        far = bit_uniform(rng, fmt, 1e10, huge, e)
        opposite = (far, -bit_uniform(rng, fmt, 1e10, huge, e))
        lopsided = (bit_uniform(rng, fmt, 1e10, huge, e, True), value_uniform(rng, fmt, -8.0, 8.0, e))
        start = bit_uniform(rng, fmt, 1e3, huge, e, True)
        adjacent = (start, np.nextafter(start, np.where(rng.random(e) < 0.5, -np.inf, np.inf).astype(fmt["dtype"])))
        sub = (bit_uniform(rng, fmt, float(np.finfo(fmt["dtype"]).smallest_subnormal), float(np.finfo(fmt["dtype"]).tiny), e, True),
               bit_uniform(rng, fmt, float(np.finfo(fmt["dtype"]).smallest_subnormal), float(np.finfo(fmt["dtype"]).tiny), e, True))
        zeros = (np.array([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, tiny, -tiny], dtype=fmt["dtype"]),
                 np.array([0.0, 0.0, -0.0, -0.0, tiny, -tiny, 0.0, -0.0], dtype=fmt["dtype"]))
        extra = [opposite, lopsided, adjacent, sub, zeros]
        x0 = np.concatenate([x0] + [p[0] for p in extra]).astype(fmt["dtype"])
        x1 = np.concatenate([x1] + [p[1] for p in extra]).astype(fmt["dtype"])
        return (x0, x1)
    if fn == "pow":
        q = n // 4
        x = bit_uniform(rng, fmt, 1e-10, 1e10, n - q)
        y = value_uniform(rng, fmt, -3.0, 3.0, n - q)
        nx = -bit_uniform(rng, fmt, 1e-3, 1e3, q)
        ny = rng.integers(-12, 13, size=q).astype(fmt["dtype"])
        w = q // 2
        top = 120.0 if f32 else 1000.0
        wx = np.exp2(rng.uniform(-1.0, 1.0, w)).astype(fmt["dtype"])
        wy = (rng.uniform(-top, top, w) / np.log2(wx.astype(np.float64))).astype(fmt["dtype"])
        x, y = x[: n - q - w], y[: n - q - w]
        if tier.startswith("balanced_"):
            dtype = fmt["dtype"]
            anchors = np.array([tiny, np.finfo(dtype).tiny, 0.5, 1.0, np.sqrt(2.0), 2.0, huge], dtype=dtype)
            bases = np.concatenate([anchors, np.nextafter(anchors, dtype(0)), np.nextafter(anchors, dtype(huge))])
            bases = bases[(bases > 0) & np.isfinite(bases)]
            broad = bit_uniform(rng, fmt, tiny, huge, q)
            targets = [-150.0, -149.0, -126.0, -4.0, 0.0, 4.0, 127.0, 128.0, 129.0] if f32 else [-1075.0, -1074.0, -1022.0, -4.0, 0.0, 4.0, 1023.0, 1024.0, 1025.0]
            bx = np.concatenate([broad, np.repeat(bases, len(targets))])
            z = np.concatenate([rng.uniform(-top, top, q), np.tile(targets, len(bases))])
            by = (z / np.maximum(np.abs(np.log2(bx.astype(np.float64))), np.finfo(dtype).eps)).astype(dtype)
            return (np.concatenate([x, nx, wx, bx]).astype(dtype), np.concatenate([y, ny, wy, by]).astype(dtype))
        return (np.concatenate([x, nx, wx]).astype(fmt["dtype"]), np.concatenate([y, ny, wy]).astype(fmt["dtype"]))
    raise KeyError(fn)


def ulp(ref, fmt):
    if ref == 0 or mp.isinf(ref):
        return mp.ldexp(1, fmt["emin"] - fmt["precision"] + 1)
    e = max(int(mp.floor(mp.log(abs(ref), 2))), fmt["emin"])
    return mp.ldexp(1, e - fmt["precision"] + 1)


def rounded_limit(ref, fmt):
    top = mp.ldexp(1, 128 if fmt["precision"] == 24 else 1024)
    return abs(ref) >= top * (1 - mp.ldexp(1, -fmt["precision"] - 1))


def measure(job):
    fn, tier, dump, n, seed = job
    fmt = FORMATS[tier]
    rng = np.random.default_rng(seed)
    data = domain(fn, tier, rng, n)
    two = isinstance(data, tuple)
    if tier == "fast":
        normal = np.finfo(np.float32).tiny
        flush = lambda v: np.where((v != 0) & (np.abs(v) < normal), np.float32(0), v).astype(np.float32)
        data = tuple(flush(d) for d in data) if two else flush(data)
    with tempfile.TemporaryDirectory() as tmp:
        inp = pathlib.Path(tmp) / "in.bin"
        out = pathlib.Path(tmp) / "out.bin"
        (np.concatenate(data) if two else data).astype(fmt["dtype"]).tofile(inp)
        subprocess.run([dump, fn, tier, "vector", str(inp), str(out)], check=True)
        got = np.fromfile(out, dtype=fmt["dtype"])
    mp.mp.dps = 40
    ref_fn = REFERENCE[fn]
    xs, ys, ulps, rels, mags = [], [], [], [], []
    for i in range(len(got)):
        args = (mp.mpf(float(data[0][i])), mp.mpf(float(data[1][i]))) if two else (mp.mpf(float(data[i])),)
        ref = ref_fn(*args)
        y = float(got[i])
        if np.isnan(y):
            err_ulp, err_rel = float("inf"), float("inf")
        elif np.isinf(y):
            ok = rounded_limit(ref, fmt) and (y > 0) == (ref > 0)
            err_ulp, err_rel = (0.0, 0.0) if ok else (float("inf"), float("inf"))
        else:
            diff = abs(mp.mpf(y) - ref)
            err_ulp = float(diff / ulp(ref, fmt))
            err_rel = float(diff / abs(ref)) if ref != 0 else (0.0 if diff == 0 else float("inf"))
        xs.append(float(data[0][i]) if two else float(data[i]))
        ys.append(float(data[1][i]) if two else 0.0)
        ulps.append(err_ulp)
        rels.append(err_rel)
        mags.append(float(abs(ref)) if mp.isfinite(ref) else float("inf"))
    xs, ys, ulps, rels, mags = np.array(xs), np.array(ys), np.array(ulps), np.array(rels), np.array(mags)
    precise = ulps
    if fn == "pow":
        z = np.abs(ys * np.log2(np.abs(xs)))
        scaled = rels / (1 + z)
        precise = ulps / np.maximum(1.0, z / POW_SCALE)
    else:
        scaled = rels
    normal_out = (mags >= np.finfo(np.float32).tiny) | (mags == 0)
    score = float(scaled[normal_out].max()) if tier == "fast" else float(precise.max())
    order = np.argsort(xs)
    keep = order[np.linspace(0, len(order) - 1, min(len(order), 6000)).astype(int)]
    finite = np.isfinite(ulps)
    return dict(function=fn, tier=tier, samples=int(len(xs)), max_ulp=float(ulps.max()), p999_ulp=float(np.quantile(ulps, 0.999)),
                mean_ulp=float(ulps[finite].mean()), max_rel=float(rels.max()), worst_x=float(xs[int(np.argmax(ulps))]),
                score=score, contract=CONTRACTS.get(tier, {}).get(fn, float("inf")),
                x=xs[keep].tolist(), y=ys[keep].tolist(), ulp=ulps[keep].tolist(), rel=rels[keep].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tiers", nargs="*", choices=FORMATS, default=["f32", "f64", "fast", "balanced_f32", "balanced_f64"])
    args = parser.parse_args()
    info = json.loads(subprocess.run([args.dump, "info"], check=True, capture_output=True, text=True).stdout)
    margin = 0.0 if info["fma"] else NO_FMA_MARGIN
    jobs = []
    for tier in args.tiers:
        for i, fn in enumerate(UNARY + BINARY + ["sincos_sin", "sincos_cos"]):
            if (args.only and fn not in args.only) or fn not in CONTRACTS.get(tier, {}):
                continue
            jobs.append((fn, tier, args.dump, args.samples, 1000 + i))
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(measure, jobs))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results))
    failures = 0
    for r in results:
        if r["tier"] != "fast":
            r["contract"] += margin
        ok = r["score"] <= r["contract"]
        failures += 0 if ok else 1
        unit = "rel" if r["tier"] == "fast" else "ulp"
        print(f"{r['function']:>11} {r['tier']:>4} {r['score']:10.4g} {unit} (contract {r['contract']:.4g})  {'ok' if ok else 'FAIL'}  worst x {r['worst_x']:.9g}")
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
