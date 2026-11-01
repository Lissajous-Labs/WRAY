#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../numerics/polynomial.h"
#include "../simd/vec.h"
#include "coefficients.h"

namespace wray::math
{
    struct precise
    {
    };

    struct fast
    {
    };

    template<class Tier, class T>
    struct table;

    template<>
    struct table<precise, float>
    {
        using exp = coeff::exp_f32;
        using log = coeff::log_f32;
        using sin = coeff::sin_f32;
        using cos = coeff::cos_f32;
        using tan = coeff::tan_f32;
        using atan_rest = coeff::atan_rest_f32;
        using tanh = coeff::tanh_f32;
    };

    template<>
    struct table<precise, double>
    {
        using exp = coeff::exp_f64;
        using log = coeff::log_f64;
        using sin = coeff::sin_f64;
        using cos = coeff::cos_f64;
        using tan = coeff::tan_f64;
        using atan_rest = coeff::atan_rest_f64;
        using tanh = coeff::tanh_f64;
    };

    template<>
    struct table<fast, float>
    {
        using exp = coeff::exp_fast;
        using log = coeff::log_fast;
        using sin = coeff::sin_fast;
        using cos = coeff::cos_fast;
        using tan = coeff::tan_fast;
        using atan = coeff::atan_fast;
        using tanh = coeff::tanh_range_fast;
    };

    template<class V>
    struct sincos_result
    {
        V sin;
        V cos;
    };
}

namespace wray::math::detail
{
    using simd::abs;
    using simd::as_bits;
    using simd::both;
    using simd::either;
    using simd::fma;
    using simd::from_bits;
    using simd::integer_t;
    using simd::nonzero;
    using simd::scalar_t;
    using simd::select;
    using simd::shift_left;
    using simd::shift_right_logical;

    template<class S>
    struct format;

    template<>
    struct format<float>
    {
        using integer = std::int32_t;
        static constexpr int width = 32;
        static constexpr int mantissa = 23;
        static constexpr integer bias = 127;
        static constexpr float magic = 0x1.8p23f;
        static constexpr float two_mantissa = 0x1p23f;
        static constexpr float exponent_offset = 0x1p23f + 127.0f;
        static constexpr integer mantissa_mask = 0x007FFFFF;
        static constexpr integer one_bits = 0x3F800000;
        static constexpr integer sign_bit = std::numeric_limits<integer>::min();
        static constexpr float min_normal = 0x1p-126f;
        static constexpr float subnormal_scale = 0x1p24f;
        static constexpr float subnormal_exponent = 24.0f;
        static constexpr float fast_exp_max = 88.0f;
        static constexpr float fast_exp_min = -87.0f;
        static constexpr float fast_exp2_max = 127.0f;
        static constexpr float fast_exp2_min = -126.0f;
        static constexpr float exp_safe = 86.0f;
        static constexpr float exp2_safe = 124.0f;
        static constexpr float pow_y_safe = 0x1p32f;
    };

    template<>
    struct format<double>
    {
        using integer = std::int64_t;
        static constexpr int width = 64;
        static constexpr int mantissa = 52;
        static constexpr integer bias = 1023;
        static constexpr double magic = 0x1.8p52;
        static constexpr double two_mantissa = 0x1p52;
        static constexpr double exponent_offset = 0x1p52 + 1023.0;
        static constexpr integer mantissa_mask = 0x000FFFFFFFFFFFFFll;
        static constexpr integer one_bits = 0x3FF0000000000000ll;
        static constexpr integer sign_bit = std::numeric_limits<integer>::min();
        static constexpr double min_normal = 0x1p-1022;
        static constexpr double subnormal_scale = 0x1p54;
        static constexpr double subnormal_exponent = 54.0;
        static constexpr double exp_safe = 707.0;
        static constexpr double exp2_safe = 1020.0;
        static constexpr double pow_y_safe = 0x1p64;
    };

    template<class Tier>
    inline constexpr bool full = std::is_same_v<Tier, precise>;

    template<class V>
    struct pair
    {
        V hi;
        V lo;
    };

    template<class V>
    WRAY_INLINE auto isnan(const V &x) noexcept
    {
        return x != x;
    }

    template<class V>
    WRAY_INLINE V nan() noexcept
    {
        return V(std::numeric_limits<scalar_t<V> >::quiet_NaN());
    }

    template<class V>
    WRAY_INLINE V inf() noexcept
    {
        return V(std::numeric_limits<scalar_t<V> >::infinity());
    }

    template<class V>
    WRAY_INLINE V round_nearest(const V &x) noexcept
    {
        return simd::nearest(x);
    }

    template<class V>
    WRAY_INLINE integer_t<V> to_integer(const V &integral) noexcept
    {
        const V m(format<scalar_t<V> >::magic);
        const V y = integral + m;
        return as_bits(y) - as_bits(m);
    }

    template<class V>
    WRAY_INLINE V pow2(const V &integral) noexcept
    {
        using F = format<scalar_t<V> >;
        using I = integer_t<V>;
        const I e = to_integer(integral) + I(F::bias);
        return from_bits<V>(shift_left(e, F::mantissa));
    }

    template<class V>
    WRAY_INLINE V copysign_from(const V &magnitude, const V &sign) noexcept
    {
        using F = format<scalar_t<V> >;
        using I = integer_t<V>;
        return from_bits<V>(as_bits(magnitude) | (as_bits(sign) & I(F::sign_bit)));
    }

    template<class V>
    WRAY_INLINE V clamp(const V &x, const V &lo, const V &hi) noexcept
    {
        return select(x > hi, hi, select(x < lo, lo, x));
    }

    template<class V>
    WRAY_INLINE pair<V> two_product(const V &a, const V &b) noexcept
    {
        const V p = a * b;
#if WRAY_HAS_FMA
        return {p, fma(a, b, -p)};
#else
        using K = constants<scalar_t<V> >;
        const V s(K::veltkamp);
        const V ca = a * s;
        const V cb = b * s;
        const V ah = ca - (ca - a);
        const V bh = cb - (cb - b);
        const V al = a - ah;
        const V bl = b - bh;
        const V e1 = ah * bh - p;
        const V e2 = e1 + ah * bl;
        const V e3 = e2 + al * bh;
        return {p, e3 + al * bl};
#endif
    }

    template<class V>
    WRAY_INLINE pair<V> two_sum(const V &a, const V &b) noexcept
    {
        const V s = a + b;
        const V bb = s - a;
        const V e = (a - (s - bb)) + (b - bb);
        return {s, e};
    }

    template<class V>
    WRAY_INLINE V scale_split(const V &p, const V &kf) noexcept
    {
        using S = scalar_t<V>;
        const V k1 = round_nearest(fma(kf, V(S(0.5)), V(S(-0.25))));
        const V k2 = kf - k1;
        const V y = p * pow2(k1);
        return y * pow2(k2);
    }

    template<class Tier, class V>
    WRAY_INLINE V exp_poly(const V &r) noexcept
    {
        using S = scalar_t<V>;
        using C = typename table<Tier, S>::exp;
        const V h = numerics::rational<C>(r);
        const V r2 = r * r;
        const V one(S(1));
        const V hi = one + r;
        const V lo = r - (hi - one);
        return hi + fma(r2, h, lo);
    }

    template<class V>
    WRAY_INLINE pair<V> exp_reduce(const V &x) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V y = x * V(K::inv_ln2);
        const V kf = round_nearest(y);
        V r = fma(-kf, V(K::ln2_hi), x);
        r = fma(-kf, V(K::ln2_lo), r);
        return {r, kf};
    }

    template<class Tier, class V>
    WRAY_INLINE V exp_regular(const V &x) noexcept
    {
        const auto red = exp_reduce(x);
        return exp_poly<Tier>(red.hi) * pow2(red.lo);
    }

    template<class Tier, class V>
    V exp_edge(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V xc = clamp(x, V(K::min_arg - S(1)), V(K::max_arg + S(1)));
        const auto red = exp_reduce(xc);
        const V e = scale_split(exp_poly<Tier>(red.hi), red.lo);
        return select(isnan(x), x, e);
    }

    template<class Tier, class V>
    WRAY_INLINE V exp(const V &x) noexcept
    {
        using S = scalar_t<V>;
        if constexpr (full<Tier>)
        {
            if (simd::any(!(abs(x) < V(format<S>::exp_safe))))
                return exp_edge<Tier>(x);
            return exp_regular<Tier>(x);
        } else
        {
            const V xc = clamp(x, V(format<S>::fast_exp_min), V(format<S>::fast_exp_max));
            const auto red = exp_reduce(xc);
            return exp_poly<Tier>(red.hi) * pow2(red.lo);
        }
    }

    template<class Tier, class V>
    WRAY_INLINE pair<V> exp2_reduce(const V &hi, const V &lo) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V kf = round_nearest(hi);
        const V f0 = hi - kf;
        const V f = f0 + lo;
        return {f * V(K::ln2), kf};
    }

    template<class Tier, class V>
    V exp2_edge(const V &hi, const V &lo) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V xc = clamp(hi, V(K::exp2_min - S(1)), V(K::exp2_max + S(1)));
        const V lc = select(xc == hi, lo, V(S(0)));
        const auto red = exp2_reduce<Tier>(xc, lc);
        const V e = scale_split(exp_poly<Tier>(red.hi), red.lo);
        return select(isnan(hi), hi, e);
    }

    template<class Tier, class V>
    WRAY_INLINE V exp2_pair(const V &hi, const V &lo) noexcept
    {
        using S = scalar_t<V>;
        if constexpr (full<Tier>)
        {
            if (simd::any(!(abs(hi) < V(format<S>::exp2_safe))))
                return exp2_edge<Tier>(hi, lo);
            const auto red = exp2_reduce<Tier>(hi, lo);
            return exp_poly<Tier>(red.hi) * pow2(red.lo);
        } else
        {
            const V xc = clamp(hi, V(format<S>::fast_exp2_min), V(format<S>::fast_exp2_max));
            const auto red = exp2_reduce<Tier>(xc, lo);
            return exp_poly<Tier>(red.hi) * pow2(red.lo);
        }
    }

    template<class Tier, class V>
    WRAY_INLINE V exp2(const V &x) noexcept
    {
        return exp2_pair<Tier>(x, V(scalar_t<V>(0)));
    }

    template<class V>
    struct log_parts
    {
        V e;
        V hi;
        V lo;
    };

    template<class V>
    WRAY_INLINE auto log_regular_mask(const V &x) noexcept
    {
        return both(x >= V(format<scalar_t<V> >::min_normal), x < inf<V>());
    }

    template<class Tier, bool Subnormal, class V>
    WRAY_INLINE log_parts<V> log_reduce(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using F = format<S>;
        using I = integer_t<V>;
        using C = typename table<Tier, S>::log;
        V xs = x;
        V adjust(S(0));
        if constexpr (Subnormal)
        {
            const auto sub = x < V(F::min_normal);
            const V scaled = x * V(F::subnormal_scale);
            xs = select(sub, scaled, x);
            adjust = select(sub, V(-F::subnormal_exponent), V(S(0)));
        }
        const I bits = as_bits(xs);
        const I biased = shift_right_logical(bits, F::mantissa) | as_bits(V(F::two_mantissa));
        V e = from_bits<V>(biased) - V(F::exponent_offset);
        V m = from_bits<V>((bits & I(F::mantissa_mask)) | I(F::one_bits));
        const auto big = m > V(K::sqrt2);

        const V half = m * V(S(0.5));
        m = select(big, half, m);
        const V e1 = e + V(S(1));
        e = select(big, e1, e);

        if constexpr (Subnormal) e = e + adjust;

        const V f = m - V(S(1));
        const V d = m + V(S(1));
        const V s = f / d;
        const V t = s * s;
        const V tail = numerics::rational<C>(t);
        const V w = t * tail;
        const V g = f - w;
        const V a = s * g;
        const V hi = f - a;
        const V lo = (f - hi) - a;
        return {e, hi, lo};
    }

    template<class V>
    WRAY_INLINE V log_specials(const V &x, const V &res) noexcept
    {
        using S = scalar_t<V>;
        V r = select(x < V(S(0)), nan<V>(), res);
        r = select(x == V(S(0)), -inf<V>(), r);
        r = select(x == inf<V>(), inf<V>(), r);
        return select(isnan(x), x, r);
    }

    template<class V>
    WRAY_INLINE V log_combine(const log_parts<V> &p) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V small = fma(p.e, V(K::ln2_lo), p.lo);
        const V body = p.hi + small;
        return fma(p.e, V(K::ln2_hi), body);
    }

    template<class V>
    WRAY_INLINE V log2_combine(const log_parts<V> &p) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V inner = fma(p.hi, V(K::inv_ln2_lo), p.lo * V(K::inv_ln2_hi));
        const V v = fma(p.hi, V(K::inv_ln2_hi), inner);
        return p.e + v;
    }

    template<class V>
    WRAY_INLINE pair<V> log2_pair_combine(const log_parts<V> &p) noexcept
    {
        using K = constants<scalar_t<V> >;
        const auto prod = two_product(p.hi, V(K::inv_ln2_hi));
        const V cross = fma(p.hi, V(K::inv_ln2_lo), p.lo * V(K::inv_ln2_hi));
        const V plo = prod.lo + cross;
        const auto sum = two_sum(p.e, prod.hi);
        const V lo = sum.lo + plo;
        const V hi = sum.hi + lo;
        const V rest = lo - (hi - sum.hi);
        return {hi, rest};
    }

    template<class Tier, class V>
    V log_edge(const V &x) noexcept
    {
        return log_specials(x, log_combine(log_reduce<Tier, true>(x)));
    }

    template<class Tier, class V>
    WRAY_INLINE V log(const V &x) noexcept
    {
        if constexpr (full<Tier>)
        {
            if (simd::any(!log_regular_mask(x)))
                return log_edge<Tier>(x);
        }
        return log_combine(log_reduce<Tier, false>(x));
    }

    template<class Tier, class V>
    V log2_edge(const V &x) noexcept
    {
        return log_specials(x, log2_combine(log_reduce<Tier, true>(x)));
    }

    template<class Tier, class V>
    WRAY_INLINE V log2(const V &x) noexcept
    {
        if constexpr (full<Tier>)
        {
            if (simd::any(!log_regular_mask(x)))
                return log2_edge<Tier>(x);
            return log2_combine(log_reduce<Tier, false>(x));
        } else
        {
            using K = constants<scalar_t<V> >;
            const auto p = log_reduce<Tier, false>(x);
            const V v = p.hi * V(K::inv_ln2);
            return p.e + v;
        }
    }

    template<class V>
    struct reduced
    {
        V hi;
        V lo;
        integer_t<V> q;
    };

    template<class Tier, class V>
    WRAY_INLINE reduced<V> reduce_pio2(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V y = x * V(K::two_over_pi);
        const V kf = round_nearest(y);
        constexpr bool four = std::is_same_v<S, float>;
        const V r1 = fma(-kf, V(K::pio2_1), x);
        const V c1 = kf * V(K::pio2_2);
        const V c2 = kf * V(K::pio2_3);
        const V mid = r1 - c1;
        const V hi = mid - c2;
        if constexpr (full<Tier>)
        {
            V e1 = (r1 - mid) - c1;
            if constexpr (four)
                e1 = fma(-kf, V(K::pio2_4), e1);
            const V back = hi - mid;
            const V e2 = (mid - (hi - back)) - (c2 + back);
            return {hi, e1 + e2, to_integer(kf)};
        } else if constexpr (four)
            return {fma(-kf, V(K::pio2_4), hi), V(S(0)), to_integer(kf)};
        else
            return {hi, V(S(0)), to_integer(kf)};
    }

    template<class Tier, class V>
    WRAY_INLINE sincos_result<V> trig_core(const reduced<V> &red) noexcept
    {
        using S = scalar_t<V>;
        using Cs = typename table<Tier, S>::sin;
        using Cc = typename table<Tier, S>::cos;
        const V r = red.hi;
        const V t = r * r;
        const V rt = r * t;
        const V ps = numerics::rational<Cs>(t);
        const V pc = numerics::rational<Cc>(t);
        if constexpr (full<Tier>)
        {
            const V h = t * V(S(0.5));
            const V tt = t * t;
            const V s = r + (fma(rt, ps, red.lo) - h * red.lo);
            const V w = fma(t, V(S(-0.5)), V(S(1)));
            const V werr = (V(S(1)) - w) - h;
            const V c = w + (fma(tt, pc, werr) - r * red.lo);
            return {s, c};
        } else
        {
            const V s = fma(rt, ps, r);
            const V c = fma(t, fma(t, pc, V(S(-0.5))), V(S(1)));
            return {s, c};
        }
    }

    template<class Tier, class V>
    WRAY_INLINE V trig_unit(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        if constexpr (full<Tier>)
            return select(abs(x) <= V(K::trig_arg_max), V(S(1)), nan<V>());
        else
            return V(S(1));
    }

    template<class V>
    WRAY_INLINE V orient(const sincos_result<V> &p, const integer_t<V> &q, const V &unit) noexcept
    {
        using I = integer_t<V>;
        const V v = select(nonzero(q & I(1)), p.cos, p.sin);
        const V sign = select(nonzero(q & I(2)), -unit, unit);
        return v * sign;
    }

    template<class V>
    WRAY_INLINE V keep_zero(const V &x, const V &v) noexcept
    {
        return select(x == V(scalar_t<V>(0)), x, v);
    }

    template<class Tier, class V>
    WRAY_INLINE V sin(const V &x) noexcept
    {
        const auto red = reduce_pio2<Tier>(x);
        return keep_zero(x, orient(trig_core<Tier>(red), red.q, trig_unit<Tier>(x)));
    }

    template<class Tier, class V>
    WRAY_INLINE V cos(const V &x) noexcept
    {
        using I = integer_t<V>;
        const auto red = reduce_pio2<Tier>(x);
        return orient(trig_core<Tier>(red), red.q + I(1), trig_unit<Tier>(x));
    }

    template<class Tier, class V>
    WRAY_INLINE sincos_result<V> sincos(const V &x) noexcept
    {
        using I = integer_t<V>;
        const auto red = reduce_pio2<Tier>(x);
        const auto p = trig_core<Tier>(red);
        const V unit = trig_unit<Tier>(x);
        return {keep_zero(x, orient(p, red.q, unit)), orient(p, red.q + I(1), unit)};
    }

    template<class Tier, class V>
    WRAY_INLINE V tan(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using I = integer_t<V>;
        using C = typename table<Tier, S>::tan;
        const auto red = reduce_pio2<Tier>(x);
        const V r = red.hi;
        const V t = r * r;
        const auto odd = nonzero(red.q & I(1));
        const V rt = r * t;
        if constexpr (full<Tier>)
        {
            const V tail = fma(rt, numerics::rational<C>(t), fma(red.lo, t, red.lo));
            const V v = r + tail;
            const V e = (r - v) + tail;
            const V q = V(S(-1)) / v;
            const V c = fma(q, fma(q, e, fma(q, v, V(S(1)))), q);
            return keep_zero(x, select(odd, c, v) * trig_unit<Tier>(x));
        } else
        {
            const V v = fma(rt, numerics::rational<C>(t), r);
            const V c = V(S(-1)) / v;
            return keep_zero(x, select(odd, c, v));
        }
    }

    template<class Tier, class V>
    WRAY_INLINE pair<V> atan_unit(const V &u, const V &delta) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V t = u * u;
        const V ut = u * t;
        if constexpr (full<Tier>)
        {
            using C = typename table<Tier, S>::atan_rest;
            const V third = ut * V(K::third_hi);
            const V hi = u - third;
            const V rounding = fma(ut, V(K::third_hi), -third);
            const V small = fma(ut, V(K::third_lo), fma(fma(u, t, -ut), V(K::third_hi), rounding));
            const V slope = fma(t, V(S(-0.5)), V(S(0.25) + S(0.5) * K::sqrt2));
            const V lo = fma(delta, slope, ((u - hi) - third) - small);
            return {hi, fma(ut * t, numerics::rational<C>(t), lo)};
        }
        else
        {
            using C = typename table<Tier, S>::atan;
            return {fma(ut, numerics::rational<C>(t), u), V(S(0))};
        }
    }

    template<class Tier, class V>
    WRAY_INLINE pair<V> complement(const pair<V> &v, scalar_t<V> hi, scalar_t<V> lo) noexcept
    {
        const V d = V(hi) - v.hi;
        if constexpr (full<Tier>)
            return {d, ((V(hi) - d) - v.hi) + (V(lo) - v.lo)};
        else
            return {d, V(scalar_t<V>(0))};
    }

    template<class V, class M>
    WRAY_INLINE pair<V> choose(const M &m, const pair<V> &a, const pair<V> &b) noexcept
    {
        return {select(m, a.hi, b.hi), select(m, a.lo, b.lo)};
    }

    template<class Tier, class V>
    WRAY_INLINE V atan(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V a = abs(x);
        const V one(S(1));
        const auto big = a > one;
        const V n = select(big, one, a);
        const V d = select(big, a, one);
        const V u = n / d;
        V delta(S(0));
        if constexpr (full<Tier>)
            delta = fma(-u, simd::min(d, V(std::numeric_limits<S>::max())), n) * u;
        const auto v = atan_unit<Tier>(u, delta);
        const auto w = choose(big, complement<Tier>(v, K::pio2_hi, K::pio2_lo), v);
        if constexpr (full<Tier>)
            return copysign_from(w.hi + w.lo, x);
        else
            return copysign_from(w.hi, x);
    }

    template<class V>
    WRAY_INLINE pair<V> ratio(const V &n, const V &d) noexcept
    {
        const V inv = V(scalar_t<V>(1)) / d;
        const V u = n * inv;
        return {u, fma(-u, d, n) * inv};
    }

    template<class S>
    inline constexpr S residual_floor = format<S>::min_normal * format<S>::two_mantissa * S(512);

    template<class S>
    inline constexpr S ratio_ceiling = S(1) / format<S>::min_normal;

    template<class V>
    pair<V> ratio_edge(const V &n, const V &d) noexcept
    {
        using S = scalar_t<V>;
        using F = format<S>;
        const V up(residual_floor<S> / std::numeric_limits<S>::denorm_min());
        const V down(S(1) / F::subnormal_scale);
        const V s = select(n < V(residual_floor<S>), up, select(d < V(ratio_ceiling<S>), V(S(1)), down));
        const V ns = n * s;
        const V ds = d * s;
        const V hi = ns / ds;
        const auto exact = !(n >= d * V(residual_floor<S>));
        const auto r = ratio(ns, ds);
        const pair<V> q{select(exact, hi, r.hi), select(exact, fma(-hi, ds, ns) / ds, r.lo)};
        const auto finite = both(ds > V(S(0)), ds < inf<V>());
        const V u = select(finite, q.hi, select(ns == inf<V>(), V(S(1)), V(S(0))));
        return {u, select(finite, q.lo, V(S(0)))};
    }

    template<class Tier, class V>
    WRAY_INLINE V atan2(const V &y, const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using F = format<S>;
        using I = integer_t<V>;
        const V ax = abs(x);
        const V ay = abs(y);
        const auto swap = ay > ax;
        const V n = select(swap, ax, ay);
        const V d = select(swap, ay, ax);
        pair<V> q;
        if constexpr (full<Tier>)
        {
            if (simd::any(!both(both(n >= V(residual_floor<S>), d < V(ratio_ceiling<S>)), n >= d * V(residual_floor<S>))))
                q = ratio_edge(n, d);
            else
                q = ratio(n, d);
        }
        else
        {
            V u = n / d;
            u = select(d == V(S(0)), V(S(0)), u);
            q = {select(both(ax == inf<V>(), ay == inf<V>()), V(S(1)), u), V(S(0))};
        }
        auto v = atan_unit<Tier>(q.hi, q.lo);
        v = choose(swap, complement<Tier>(v, K::pio2_hi, K::pio2_lo), v);
        const auto negative_x = nonzero(as_bits(x) & I(F::sign_bit));
        v = choose(negative_x, complement<Tier>(v, K::pi_hi, K::pi_lo), v);
        if constexpr (full<Tier>)
            return select(either(isnan(x), isnan(y)), nan<V>(), copysign_from(v.hi + v.lo, y));
        else
            return copysign_from(v.hi, y);
    }

    template<class Tier, class V>
    WRAY_INLINE V tanh(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using C = typename table<Tier, S>::tanh;
        const V a = abs(x);
        if constexpr (full<Tier>)
        {
            const V c(K::tanh_clamp);
            const auto small = a < V(K::tanh_switch);
            const V t = a * a;
            const V at = a * t;
            const auto parts = numerics::rational_parts<C>(t);
            const V ac = select(a < c, a, c);
            const V e = exp_regular<Tier>(ac + ac);
            const V dl = e + V(S(1));
            const V q = select(small, parts.num, V(S(2))) / select(small, parts.den, dl);
            const V v = select(small, fma(at, q, a), V(S(1)) - q);
            return select(isnan(x), x, copysign_from(v, x));
        } else
        {
            const V c(K::tanh_clamp_fast);
            const auto inside = a < c;
            const V ac = select(inside, a, c);
            const V t = ac * ac;
            const V vs = ac * numerics::rational<C>(t);
            return copysign_from(select(inside, vs, V(S(1))), x);
        }
    }

    template<class Tier, class V>
    WRAY_INLINE V sigmoid(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        if constexpr (full<Tier>)
        {
            const V e = exp<Tier>(-abs(x));
            const V num = select(x < V(S(0)), e, one);
            const V d = one + e;
            const V dl = e - (d - one);
            const V inv = one / d;
            const V q = num * inv;
            const V rest = fma(-q, d, num);
            return fma(fma(-q, dl, rest), inv, q);
        } else
            return one / (one + exp<Tier>(-x));
    }

    template<class V>
    WRAY_INLINE V finite_or_zero(const V &hi, const V &lo) noexcept
    {
        using S = scalar_t<V>;
        return select(abs(hi) < inf<V>(), lo, V(S(0)));
    }

    template<class Tier, class V>
    WRAY_INLINE V exp2_pair_regular(const V &hi, const V &lo) noexcept
    {
        const auto red = exp2_reduce<Tier>(hi, lo);
        return exp_poly<Tier>(red.hi) * pow2(red.lo);
    }

    template<class Tier, class V>
    V pow_edge(const V &x, const V &y) noexcept
    {
        using S = scalar_t<V>;
        const auto l = log2_pair_combine(log_reduce<Tier, true>(x));
        const V lh = log_specials(x, l.hi);
        const auto z = two_product(y, lh);
        const V cross = fma(y, l.lo, z.lo);
        const V r = exp2_edge<Tier>(z.hi, finite_or_zero(z.hi, cross));
        return select(either(y == V(S(0)), x == V(S(1))), V(S(1)), r);
    }

    template<class Tier, class V>
    WRAY_INLINE V pow(const V &x, const V &y) noexcept
    {
        using S = scalar_t<V>;
        if constexpr (full<Tier>)
        {
            if (simd::any(!both(log_regular_mask(x), abs(y) < V(format<S>::pow_y_safe))))
                return pow_edge<Tier>(x, y);
            const auto l = log2_pair_combine(log_reduce<Tier, false>(x));
            const auto z = two_product(y, l.hi);
            const V lo = fma(y, l.lo, z.lo);
            if (simd::any(!(abs(z.hi) < V(format<S>::exp2_safe))))
                return pow_edge<Tier>(x, y);
            return exp2_pair_regular<Tier>(z.hi, lo);
        } else
        {
            const V l = log2<Tier>(x);
            const V z = y * l;
            return exp2<Tier>(z);
        }
    }

    template<class Tier, class V>
    WRAY_INLINE V db_to_gain(const V &db) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        if constexpr (full<Tier>)
        {
            const auto z = two_product(db, V(K::log2_per_db_hi));
            const V lo = fma(db, V(K::log2_per_db_lo), z.lo);
            if (simd::any(!(abs(z.hi) < V(format<S>::exp2_safe))))
                return exp2_edge<Tier>(z.hi, finite_or_zero(z.hi, lo));
            return exp2_pair_regular<Tier>(z.hi, lo);
        } else
        {
            const V z = db * V(K::log2_per_db);
            return exp2<Tier>(z);
        }
    }

    template<class V>
    WRAY_INLINE V db_combine(const log_parts<V> &p) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V inner = fma(p.hi, V(K::db_per_ln_lo), p.lo * V(K::db_per_ln_hi));
        const V v = fma(p.hi, V(K::db_per_ln_hi), inner);
        const V small = fma(p.e, V(K::db_per_log2_lo), v);
        return fma(p.e, V(K::db_per_log2_hi), small);
    }

    template<class Tier, class V>
    V gain_to_db_edge(const V &gain) noexcept
    {
        return log_specials(gain, db_combine(log_reduce<Tier, true>(gain)));
    }

    template<class Tier, class V>
    WRAY_INLINE V gain_to_db(const V &gain) noexcept
    {
        using K = constants<scalar_t<V> >;
        if constexpr (full<Tier>)
        {
            if (simd::any(!log_regular_mask(gain)))
                return gain_to_db_edge<Tier>(gain);
            return db_combine(log_reduce<Tier, false>(gain));
        } else
        {
            const V l = log2<Tier>(gain);
            return l * V(K::db_per_log2);
        }
    }
}
