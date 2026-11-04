import mpmath as mp


def _limit(fn, at_zero):
    def g(x):
        x = mp.mpf(x)
        return at_zero() if x == 0 else fn(x)
    return g


def exp_tail(r):
    with mp.workprec(mp.mp.prec + 2 * max(0, -int(mp.floor(mp.log(abs(r), 2)))) + 16):
        return (mp.expm1(r) - r) / (r * r)


def exp_weight(tier):
    return (lambda r: r * r / mp.exp(r)) if tier == "fast" else "rel"


def log_core(t):
    s = mp.sqrt(t)
    return 2 * mp.atanh(s) / s


def log2_core(t):
    return log_core(t) / mp.log(2)


def sin_core(t):
    s = mp.sqrt(t)
    return (mp.sin(s) - s) / (s * t)


def cos_core(t):
    return (mp.cos(mp.sqrt(t)) - 1 + t / 2) / (t * t)


def tan_core(t):
    s = mp.sqrt(t)
    return mp.tan(s) / s


def tanh_core(t):
    s = mp.sqrt(t)
    return mp.tanh(s) / s


def atan_core(t):
    s = mp.sqrt(t)
    return mp.atan(s) / s
def log_tail(t):
    return (log_core(t) - 2) / t


def log_direct_tail(f):
    with mp.workprec(mp.mp.prec + 2 * max(0, -int(mp.floor(mp.log(abs(f), 2)))) + 16):
        return (mp.log1p(f) - f) / (f * f)


def tan_tail(t):
    return (tan_core(t) - 1) / t


def atan_tail(t):
    return (atan_core(t) - 1) / t


def atan_rest(t):
    return (atan_tail(t) + mp.mpf(1) / 3) / t


def tanh_tail(t):
    return (tanh_core(t) - 1) / t




def ln2_half():
    return mp.log(2) / 2


def log_t():
    return ((mp.sqrt(2) - 1) / (mp.sqrt(2) + 1)) ** 2


def pi2_sq():
    return (mp.pi / 2) ** 2


def pi4_sq():
    return (mp.pi / 4) ** 2


TANH_CLAMP = {"f32": "9.25", "f64": "19.25", "fast": "5.9"}


def fast_pi4(tier):
    return [pi4_sq()] if tier == "fast" else []


def precise(fn):
    def g(t):
        with mp.workprec(mp.mp.prec + max(0, -int(mp.floor(mp.log(abs(t), 2)))) * 3 + 16):
            return fn(t)
    return g


def sin_turns(t):
    s = mp.sqrt(t)
    return (mp.sin(2 * mp.pi * s) - 2 * mp.pi * s) / (s * t)


def cos_turns(t):
    return (mp.cos(2 * mp.pi * mp.sqrt(t)) - 1 + 2 * mp.pi ** 2 * t) / (t * t)


def sinh_tail(t):
    s = mp.sqrt(t)
    return (mp.sinh(s) - s) / (s * t)


def cosh_tail(t):
    return (mp.cosh(mp.sqrt(t)) - 1 - t / 2) / (t * t)


def asinh_tail(t):
    s = mp.sqrt(t)
    return (mp.asinh(s) - s) / (s * t)


def erf_tail(t):
    s = mp.sqrt(t)
    return (mp.erf(s) / s - 2 / mp.sqrt(mp.pi)) / t


def erf_mid(d):
    return (mp.erf(mp.mpf(3) / 4 + d) - mp.erf(mp.mpf(3) / 4)) / d


def erf_mid2(d):
    return (mp.erf(mp.mpf(5) / 4 + d) - mp.erf(mp.mpf(5) / 4)) / d


def erfc_scaled(a):
    return mp.erfc(a) * mp.exp(a * a)


def omega(x):
    x = mp.mpf(x)
    y = x - mp.log(x) if x > 1 else mp.exp(x)
    for _ in range(200):
        step = (y + mp.log(y) - x) * y / (1 + y)
        y -= step
        if abs(step) <= mp.mpf(10) ** (5 - mp.mp.dps) * abs(y):
            break
    return y


def tanpi(t):
    s = mp.sqrt(t)
    return (mp.tan(mp.pi * s) - mp.pi * s) / (s * t)



CORES = {
    "exp": dict(g=_limit(exp_tail, lambda: mp.mpf(1) / 2), interval=lambda tier: (-ln2_half(), ln2_half()), weight=exp_weight,
                pin=lambda tier: [-ln2_half(), ln2_half()] if tier == "fast" else []),
    "log": dict(g=_limit(log_tail, lambda: mp.mpf(2) / 3), interval=lambda tier: (0, log_t())),
    "log_direct": dict(g=_limit(log_direct_tail, lambda: -mp.mpf(1) / 2), interval=lambda tier: (mp.sqrt(mp.mpf("0.5")) - 1, mp.sqrt(2) - 1)),
    "sin": dict(g=_limit(sin_core, lambda: mp.mpf(-1) / 6), interval=lambda tier: (0, pi4_sq()), pin=fast_pi4),
    "sin_half": dict(g=_limit(sin_core, lambda: mp.mpf(-1) / 6), interval=lambda tier: (0, pi2_sq())),
    "cos": dict(g=_limit(cos_core, lambda: mp.mpf(1) / 24), interval=lambda tier: (0, pi4_sq()), pin=fast_pi4),
    "tan": dict(g=_limit(tan_tail, lambda: mp.mpf(1) / 3), interval=lambda tier: (0, pi4_sq()), pin=fast_pi4),
    "atan": dict(g=_limit(atan_tail, lambda: mp.mpf(-1) / 3), interval=lambda tier: (0, 1)),
    "atan_rest": dict(g=_limit(atan_rest, lambda: mp.mpf(1) / 5), interval=lambda tier: (0, 1)),
    "tanh": dict(g=_limit(tanh_tail, lambda: mp.mpf(-1) / 3), interval=lambda tier: (0, mp.mpf("0.390625"))),
    "tanh_range": dict(g=_limit(tanh_core, lambda: mp.mpf(1)), interval=lambda tier: (0, mp.mpf(TANH_CLAMP[tier]) ** 2)),
    "sinh": dict(g=_limit(precise(sinh_tail), lambda: mp.mpf(1) / 6), interval=lambda tier: (0, 1)),
    "cosh": dict(g=_limit(precise(cosh_tail), lambda: mp.mpf(1) / 24), interval=lambda tier: (0, 1)),
    "asinh": dict(g=_limit(precise(asinh_tail), lambda: mp.mpf(-1) / 6), interval=lambda tier: (0, mp.mpf(1) / 4)),
    "cbrt": dict(g=lambda y: mp.mpf(y) ** (-mp.mpf(1) / 3), interval=lambda tier: (1, 8)),
    "erf": dict(g=_limit(precise(erf_tail), lambda: -2 / (3 * mp.sqrt(mp.pi))), interval=lambda tier: (0, mp.mpf(1) / 4)),
    "erf_mid": dict(g=_limit(precise(erf_mid), lambda: 2 / mp.sqrt(mp.pi) * mp.exp(-mp.mpf(9) / 16)), interval=lambda tier: (-mp.mpf(1) / 4, mp.mpf(1) / 4)),
    "erf_mid2": dict(g=_limit(precise(erf_mid2), lambda: 2 / mp.sqrt(mp.pi) * mp.exp(-mp.mpf(25) / 16)), interval=lambda tier: (-mp.mpf(1) / 4, mp.mpf(1) / 4)),
    "erfc": dict(g=erfc_scaled, interval=lambda tier: (mp.mpf(3) / 2, 4 if tier == "f32" else 6)),
    "omega": dict(g=omega, interval=lambda tier: (-2, 3)),
    "sin_turns": dict(g=_limit(precise(sin_turns), lambda: -(2 * mp.pi) ** 3 / 6), interval=lambda tier: (0, mp.mpf(1) / 64)),
    "cos_turns": dict(g=_limit(precise(cos_turns), lambda: (2 * mp.pi) ** 4 / 24), interval=lambda tier: (0, mp.mpf(1) / 64)),
    "tanpi": dict(g=_limit(precise(tanpi), lambda: mp.pi ** 3 / 3), interval=lambda tier: (0, mp.mpf(1) / 16)),
}
