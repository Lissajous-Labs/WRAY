import mpmath as mp


class RemezFailure(Exception):
    pass


def chebyshev_nodes(a, b, count):
    return [mp.mpf(a) + (mp.mpf(b) - a) * (1 - mp.cos(mp.pi * i / (count - 1))) / 2 for i in range(count)]


def horner(coeffs, x):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


class Rational:
    def __init__(self, p, q):
        self.p = list(p)
        self.q = list(q)

    def __call__(self, x):
        return horner(self.p, x) / horner(self.q, x)

    def denominator(self, x):
        return horner(self.q, x)


def solve_reference(f, w, nodes, m, n, q_prev, inner=12):
    count = m + n + 2
    fx = [f(x) for x in nodes]
    wx = [w(x) for x in nodes]
    q = list(q_prev)
    E = mp.mpf(0)
    for _ in range(inner):
        A = mp.matrix(count, count)
        rhs = mp.matrix(count, 1)
        for i, x in enumerate(nodes):
            xp = mp.mpf(1)
            for j in range(m + 1):
                A[i, j] = xp
                xp *= x
            xp = x
            for k in range(1, n + 1):
                A[i, m + k] = -fx[i] * xp
                xp *= x
            A[i, count - 1] = -((-1) ** i) * horner(q, x) / wx[i]
            rhs[i] = fx[i]
        sol = mp.lu_solve(A, rhs)
        p_new = [sol[j] for j in range(m + 1)]
        q_new = [mp.mpf(1)] + [sol[m + k] for k in range(1, n + 1)]
        E_new = sol[count - 1]
        converged = abs(E_new - E) <= abs(E_new) * mp.mpf(10) ** (-mp.mp.dps // 2)
        q, E = q_new, E_new
        if n == 0 or converged:
            return Rational(p_new, q), E
    return Rational(p_new, q), E


def golden_max(g, lo, hi, iters=60):
    phi = (mp.sqrt(5) - 1) / 2
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    g1, g2 = g(x1), g(x2)
    for _ in range(iters):
        if g1 > g2:
            hi, x2, g2 = x2, x1, g1
            x1 = hi - phi * (hi - lo)
            g1 = g(x1)
        else:
            lo, x1, g1 = x1, x2, g2
            x2 = lo + phi * (hi - lo)
            g2 = g(x2)
    return (x1, g1) if g1 > g2 else (x2, g2)


def local_extrema(err, a, b, grid):
    xs = [mp.mpf(a) + (mp.mpf(b) - a) * (1 - mp.cos(mp.pi * i / (grid - 1))) / 2 for i in range(grid)]
    es = [err(x) for x in xs]
    cands = []
    for i in range(grid):
        left = abs(es[i - 1]) if i > 0 else mp.mpf(-1)
        right = abs(es[i + 1]) if i < grid - 1 else mp.mpf(-1)
        if abs(es[i]) >= left and abs(es[i]) >= right:
            if i == 0 or i == grid - 1:
                cands.append((xs[i], es[i]))
            else:
                sign = 1 if es[i] > 0 else -1
                x, _ = golden_max(lambda t: sign * err(t), xs[i - 1], xs[i + 1])
                cands.append((x, err(x)))
    return cands


def alternate(cands, want):
    merged = []
    for x, e in cands:
        if merged and (merged[-1][1] > 0) == (e > 0):
            if abs(e) > abs(merged[-1][1]):
                merged[-1] = (x, e)
        else:
            merged.append((x, e))
    while len(merged) > want:
        if abs(merged[0][1]) < abs(merged[-1][1]):
            merged.pop(0)
        else:
            merged.pop()
    return merged


def remez(f, a, b, m, n, weight="rel", dps=60, maxiter=80, tol=mp.mpf("1e-9"), grid=None):
    grid = grid or max(1000, 60 * (m + n + 2))
    for attempt in range(3):
        try:
            return _remez(f, a, b, m, n, weight, dps, maxiter, tol, grid * (4 ** attempt))
        except RemezFailure:
            if attempt == 2:
                raise


def _remez(f, a, b, m, n, weight, dps, maxiter, tol, grid):
    with mp.workdps(dps):
        a = mp.mpf(a() if callable(a) else a)
        b = mp.mpf(b() if callable(b) else b)
        if callable(weight):
            w = weight
        elif weight == "rel":
            w = lambda x: 1 / abs(f(x))
        else:
            w = lambda x: mp.mpf(1)
        count = m + n + 2
        nodes = chebyshev_nodes(a, b, count)
        if callable(weight):
            spare = chebyshev_nodes(a, b, count + 1)
            spare.remove(min(spare, key=lambda x: abs(w(x))))
            nodes = spare
        q = [mp.mpf(1)] + [mp.mpf(0)] * n
        best = None
        for _ in range(maxiter):
            r, E = solve_reference(f, w, nodes, m, n, q)
            q = r.q
            err = lambda x, r=r: w(x) * (r(x) - f(x))
            dens = [r.denominator(x) for x in chebyshev_nodes(a, b, grid)]
            if min(dens) <= 0:
                raise RemezFailure("denominator vanishes")
            cands = local_extrema(err, a, b, grid)
            alt = alternate(cands, count)
            peak = max(abs(e) for _, e in cands)
            if best is None or peak < best[1]:
                best = (r, peak, abs(E))
            if len(alt) < count:
                raise RemezFailure("lost alternation")
            if peak - abs(E) <= tol * peak:
                return r, peak
            nodes = [x for x, _ in alt]
        return best[0], best[1]


def weighted_error_curve(f, r, a, b, weight="rel", samples=2000, dps=60):
    with mp.workdps(dps):
        xs = [mp.mpf(a) + (mp.mpf(b) - a) * i / (samples - 1) for i in range(samples)]
        out = []
        for x in xs:
            fx = f(x)
            e = r(x) - fx
            scale = weight(x) if callable(weight) else 1 / abs(fx) if weight == "rel" else 1
            out.append((float(x), float(e * scale)))
        return out
