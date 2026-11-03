#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../numerics/polynomial.h"
#include "../simd/vec.h"
#include "coefficients.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI::math
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
        using sin_turns = coeff::sin_turns_f32;
        using cos_turns = coeff::cos_turns_f32;
        using tanpi = coeff::tanpi_f32;
        using sinh = coeff::sinh_f32;
        using cosh = coeff::cosh_f32;
        using asinh = coeff::asinh_f32;
        using cbrt = coeff::cbrt_f32;
        using erf = coeff::erf_f32;
        using erfc = coeff::erfc_f32;
        using erf_mid = coeff::erf_mid_f32;
        using erf_mid2 = coeff::erf_mid2_f32;
        using omega = coeff::omega_f32;
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
        using sin_turns = coeff::sin_turns_f64;
        using cos_turns = coeff::cos_turns_f64;
        using tanpi = coeff::tanpi_f64;
        using sinh = coeff::sinh_f64;
        using cosh = coeff::cosh_f64;
        using asinh = coeff::asinh_f64;
        using cbrt = coeff::cbrt_f64;
        using erf = coeff::erf_f64;
        using erfc = coeff::erfc_f64;
        using erf_mid = coeff::erf_mid_f64;
        using erf_mid2 = coeff::erf_mid2_f64;
        using omega = coeff::omega_f64;
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
        using omega = coeff::omega_fast;
    };

    template<class V>
    struct sincos_result
    {
        V sin;
        V cos;
    };
}

namespace wray::inline WRAY_ABI::math::detail
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
        if constexpr (std::is_arithmetic_v<V>)
        {
            using I = integer_t<V>;
            using U = std::make_unsigned_t<I>;
            return static_cast<I>(static_cast<U>(as_bits(y)) - static_cast<U>(as_bits(m)));
        } else
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
    WRAY_INLINE V keep_zero(const V &x, const V &v) noexcept
    {
        return select(x == V(scalar_t<V>(0)), x, v);
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
    WRAY_INLINE V residual(const V &q, const V &d, const V &n) noexcept
    {
#if WRAY_HAS_FMA
        return fma(-q, d, n);
#else
        const auto p = two_product(q, d);
        return (n - p.hi) - p.lo;
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
    WRAY_INLINE V expm1_regular(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using C = typename table<Tier, S>::exp;
        const auto red = exp_reduce(x);
        const V r = red.hi;
        const V h = numerics::rational<C>(r);
        const V r2 = r * r;
        const V p = fma(r2, h, fma(r, r, -r2) * h);
        const V s = r + p;
        const V sl = (r - s) + p;
        const V t = pow2(red.lo);
        const auto a = two_sum(t - V(S(1)), t * s);
        return keep_zero(x, a.hi + (a.lo + t * sl));
    }

    template<class Tier, class V>
    V expm1_edge(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const auto inside = abs(x) < V(format<S>::exp_safe);
        const V regular = expm1_regular<Tier>(select(inside, x, V(S(0))));
        const V e = exp_edge<Tier>(x);
        return select(inside, regular, select(x > V(S(0)), e, e - V(S(1))));
    }

    template<class Tier, class V>
    WRAY_INLINE V expm1(const V &x) noexcept
    {
        using S = scalar_t<V>;
        if (simd::any(!(abs(x) < V(format<S>::exp_safe))))
            return expm1_edge<Tier>(x);
        return expm1_regular<Tier>(x);
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

    template<class V>
    WRAY_INLINE V fast_log_domain(const V &x, const V &res) noexcept
    {
        return select(x >= V(scalar_t<V>(0)), res, nan<V>());
    }

    template<class Tier, class V>
    WRAY_INLINE V log(const V &x) noexcept
    {
        if constexpr (full<Tier>)
        {
            if (simd::any(!log_regular_mask(x)))
                return log_edge<Tier>(x);
            return log_combine(log_reduce<Tier, false>(x));
        } else
        {
            return fast_log_domain(x, log_combine(log_reduce<Tier, false>(abs(x))));
        }
    }

    template<class Tier, bool Subnormal, class V>
    WRAY_INLINE V log1p_combine(const V &x, const V &xl, const V &u) noexcept
    {
        using S = scalar_t<V>;
        auto p = log_reduce<Tier, Subnormal>(u);
        const V f = u - V(S(1));
        const V c = (x - f) + xl;
        const V q = c / u;
        const V near = x - (((((f - p.hi) - p.lo) + c * (f / u)) + q * q * V(S(0.5))) - xl);
        p.lo = p.lo + q;
        const V tiny(S(1) / (format<S>::two_mantissa * S(4)));
        return select(abs(x) <= tiny, select(xl == V(S(0)), x, x + xl), select(p.e == V(S(0)), near, log_combine(p)));
    }

    template<class Tier, class V>
    V log1p_edge(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V u = V(S(1)) + x;
        return log_specials(u, log1p_combine<Tier, true>(x, V(S(0)), u));
    }

    template<class Tier, class V>
    WRAY_INLINE V log1p(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V u = V(S(1)) + x;
        if (simd::any(!log_regular_mask(u)))
            return log1p_edge<Tier>(x);
        return log1p_combine<Tier, false>(x, V(S(0)), u);
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
            const auto p = log_reduce<Tier, false>(abs(x));
            const V v = p.hi * V(K::inv_ln2);
            return fast_log_domain(x, p.e + v);
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
    WRAY_INLINE reduced<V> reduce_pio2_inside(const V &x) noexcept;

    struct huge_reduction
    {
        double hi;
        double lo;
        int q;
    };

    inline std::uint64_t bits_at(const std::uint32_t *limbs, int count, int position, int width) noexcept
    {
        std::uint64_t v = 0;
        for (int b = width - 1; b >= 0; --b)
        {
            const int at = position + b;
            const std::uint64_t bit = at >= 0 && at < 32 * count ? (limbs[at / 32] >> (at % 32)) & 1u : 0u;
            v = (v << 1) | bit;
        }
        return v;
    }

    inline huge_reduction reduce_huge(double x) noexcept
    {
        using K = constants<double>;
        constexpr int window = 6;
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
        const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
        const std::uint64_t m = (bits & ((std::uint64_t(1) << 52) - 1)) | (biased != 0 ? std::uint64_t(1) << 52 : 0);
        const int e = (biased != 0 ? biased : 1) - 1075;
        const int k0 = e > 2 ? (e - 2) / 32 : 0;
        std::uint32_t w[window];
        for (int i = 0; i < window; ++i)
            w[i] = coeff::two_over_pi_words[static_cast<std::size_t>(k0 + window - 1 - i)];
        std::uint32_t product[window + 2] = {};
        const std::uint32_t half[2] = {static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(m >> 32)};
        for (int a = 0; a < 2; ++a)
        {
            std::uint64_t carry = 0;
            for (int j = 0; j < window; ++j)
            {
                const std::uint64_t t = std::uint64_t(w[j]) * half[a] + product[j + a] + carry;
                product[j + a] = static_cast<std::uint32_t>(t);
                carry = t >> 32;
            }
            for (int j = window + a; j < window + 2 && carry != 0; ++j)
            {
                const std::uint64_t t = std::uint64_t(product[j]) + carry;
                product[j] = static_cast<std::uint32_t>(t);
                carry = t >> 32;
            }
        }
        const int s = 32 * (k0 + window) - e;
        int q = static_cast<int>(bits_at(product, window + 2, s, 2));
        std::uint64_t f_hi = bits_at(product, window + 2, s - 64, 64);
        std::uint64_t f_lo = bits_at(product, window + 2, s - 128, 64);
        double sign = 1.0;
        if ((f_hi >> 63) != 0)
        {
            q = (q + 1) & 3;
            f_lo = ~f_lo + 1;
            f_hi = ~f_hi + (f_lo == 0 ? 1u : 0u);
            sign = -1.0;
        }
        const double top = static_cast<double>(f_hi & ~std::uint64_t(0x7ff)) * 0x1p-64;
        const double rest = (static_cast<double>(f_hi & 0x7ffu) + static_cast<double>(f_lo) * 0x1p-64) * 0x1p-64;
        const auto f = two_sum(top, rest);
        const auto p = two_product(f.hi, K::pio2_hi);
        const double lo = p.lo + (f.hi * K::pio2_lo + f.lo * K::pio2_hi);
        const auto r = two_sum(p.hi, lo);
        const bool negative = (bits >> 63) != 0;
        const double out = negative ? -sign : sign;
        return {r.hi * out, r.lo * out, negative ? (4 - q) & 3 : q};
    }

    template<class T, bool = std::is_arithmetic_v<T> >
    struct lane_of
    {
        using type = T;
    };

    template<class T>
    struct lane_of<T, false>
    {
        using type = typename T::value_type;
    };

    template<class V>
    void patch_huge(const V &x, V &hi, V &lo, integer_t<V> &q) noexcept
    {
        using S = scalar_t<V>;
        using I = typename lane_of<integer_t<V> >::type;
        using K = constants<S>;
        constexpr int n = simd::traits<V>::width;
        constexpr auto lanes = static_cast<std::size_t>(n);
        S xa[lanes], ha[lanes], la[lanes];
        I qa[lanes];
        if constexpr (n == 1)
        {
            xa[0] = x;
            ha[0] = hi;
            la[0] = lo;
            qa[0] = q;
        } else
        {
            x.store(xa);
            hi.store(ha);
            lo.store(la);
            q.store(qa);
        }
        for (int i = 0; i < n; ++i)
        {
            const S a = xa[i] < S(0) ? -xa[i] : xa[i];
            if (!(a > K::trig_arg_max) || !(a < std::numeric_limits<S>::infinity()))
                continue;
            const auto r = reduce_huge(static_cast<double>(xa[i]));
            const S h = static_cast<S>(r.hi);
            ha[i] = h;
            la[i] = static_cast<S>((r.hi - static_cast<double>(h)) + r.lo);
            qa[i] = static_cast<I>(r.q);
        }
        if constexpr (n == 1)
        {
            hi = ha[0];
            lo = la[0];
            q = qa[0];
        } else
        {
            hi = V::load(ha);
            lo = V::load(la);
            q = integer_t<V>::load(qa);
        }
    }

    template<class Tier, class V>
    WRAY_INLINE reduced<V> reduce_pio2(const V &input) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const auto inside = abs(input) <= V(K::trig_arg_max);
        const V x = select(inside, input, V(S(0)));
        reduced<V> r = reduce_pio2_inside<Tier>(x);
        if (simd::any(both(!inside, abs(input) < inf<V>())))
            patch_huge(input, r.hi, r.lo, r.q);
        return r;
    }

    template<class Tier, class V>
    WRAY_INLINE reduced<V> reduce_pio2_inside(const V &x) noexcept
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
        return select(abs(x) < inf<V>(), V(S(1)), nan<V>());
    }

    template<class V>
    WRAY_INLINE V orient(const sincos_result<V> &p, const integer_t<V> &q, const V &unit) noexcept
    {
        using I = integer_t<V>;
        const V v = select(nonzero(q & I(1)), p.cos, p.sin);
        const V sign = select(nonzero(q & I(2)), -unit, unit);
        return v * sign;
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
            return keep_zero(x, select(odd, c, v) * trig_unit<Tier>(x));
        }
    }

    template<class V>
    WRAY_INLINE V wrap_turns(const V &x) noexcept
    {
        return x - round_nearest(x);
    }

    template<class V>
    struct turns
    {
        V f;
        integer_t<V> q;
    };

    template<class V>
    WRAY_INLINE turns<V> reduce_turns(const V &x, scalar_t<V> parts) noexcept
    {
        const V w = wrap_turns(x);
        const V k = round_nearest(w * V(parts));
        return {w - k / V(parts), to_integer(k)};
    }

    template<class V>
    WRAY_INLINE V odd_zero(const V &x, const V &v) noexcept
    {
        return select(v == V(scalar_t<V>(0)), copysign_from(V(scalar_t<V>(0)), x), v);
    }

    template<class V>
    WRAY_INLINE V even_zero(const V &v) noexcept
    {
        using S = scalar_t<V>;
        using I = integer_t<V>;
        const auto zero = (as_bits(v) & I(~format<S>::sign_bit)) == I(0);
        return select(zero, V(S(0)), v);
    }

    template<class S>
    inline constexpr S tiny_turn = format<S>::min_normal * format<S>::two_mantissa * S(4);

    template<class Tier, class V>
    WRAY_INLINE sincos_result<V> turns_core(const V &f) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V two_pi(S(2) * K::pi_hi);
        V s;
        V c;
        if constexpr (full<Tier>)
        {
            using Cs = typename table<Tier, S>::sin_turns;
            using Cc = typename table<Tier, S>::cos_turns;
            const V t = f * f;
            const auto p = two_product(f, two_pi);
            s = p.hi + (fma(f, V(S(2) * K::pi_lo), p.lo) + (f * t) * numerics::rational<Cs>(t));
            s = select(abs(f) < V(tiny_turn<S>), f * two_pi, s);
            const auto sq = two_product(f, f);
            const auto a = two_product(sq.hi, V(-K::two_pi_sq_hi));
            const V w = V(S(1)) + a.hi;
            const V werr = (V(S(1)) - w) + a.hi;
            const V cross = fma(sq.hi, V(-K::two_pi_sq_lo), sq.lo * V(-K::two_pi_sq_hi));
            c = w + (werr + ((a.lo + cross) + (sq.hi * sq.hi) * numerics::rational<Cc>(t)));
        } else
        {
            using Cs = typename table<Tier, S>::sin;
            using Cc = typename table<Tier, S>::cos;
            const V theta = f * two_pi;
            const V t = theta * theta;
            s = fma(theta * t, numerics::rational<Cs>(t), theta);
            c = fma(t, fma(t, numerics::rational<Cc>(t), V(S(-0.5))), V(S(1)));
        }
        const auto eighth = abs(f) == V(S(0.125));
        return {select(eighth, copysign_from(V(K::sqrt_half), f), s), select(eighth, V(K::sqrt_half), c)};
    }

    template<class Tier, class V>
    WRAY_INLINE V sin_turns(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const auto r = reduce_turns(x, S(4));
        return odd_zero(x, orient(turns_core<Tier>(r.f), r.q, V(S(1))));
    }

    template<class Tier, class V>
    WRAY_INLINE V cos_turns(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using I = integer_t<V>;
        const auto r = reduce_turns(x, S(4));
        return even_zero(orient(turns_core<Tier>(r.f), r.q + I(1), V(S(1))));
    }

    template<class Tier, class V>
    WRAY_INLINE sincos_result<V> sincos_turns(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using I = integer_t<V>;
        const auto r = reduce_turns(x, S(4));
        const auto p = turns_core<Tier>(r.f);
        return {odd_zero(x, orient(p, r.q, V(S(1)))), even_zero(orient(p, r.q + I(1), V(S(1))))};
    }

    template<class Tier, class V>
    WRAY_INLINE V tanpi(const V &x, const V &xl = V(scalar_t<V>(0))) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using I = integer_t<V>;
        const auto r = reduce_turns(x, S(2));
        const V f = r.f;
        V v;
        V c;
        if constexpr (full<Tier>)
        {
            using C = typename table<Tier, S>::tanpi;
            const V t = f * f;
            const auto p = two_product(f, V(K::pi_hi));
            const V low = fma(xl, V(K::pi_hi), fma(f, V(K::pi_lo), p.lo) + (f * t) * numerics::rational<C>(t));
            v = p.hi + low;
            const V e = (p.hi - v) + low;
            const V q = V(S(-1)) / v;
            c = fma(q, fma(q, e, fma(q, v, V(S(1)))), q);
            v = select(abs(f) < V(tiny_turn<S>), f * V(K::pi_hi), v);
        } else
        {
            using C = typename table<Tier, S>::tan;
            const V theta = f * V(K::pi_hi);
            const V t = theta * theta;
            v = fma(theta * t, numerics::rational<C>(t), theta);
            c = V(S(-1)) / v;
        }
        const auto quarter = abs(f) == V(S(0.25));
        v = select(quarter, copysign_from(V(S(1)), f), v);
        c = select(quarter, -copysign_from(V(S(1)), f), select(v == V(S(0)), copysign_from(inf<V>(), x), c));
        return odd_zero(x, select(nonzero(r.q & I(1)), c, v));
    }

    template<class Tier, class V>
    WRAY_INLINE V tan_turns(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V whole = select(abs(x) < V(S(1) / std::numeric_limits<S>::epsilon()), x, x * V(S(0)));
        return tanpi<Tier>(whole + whole);
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
            u = select(d == V(S(0)), n, u);
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
            const auto inside = !(a >= c);
            const V ac = select(inside, a, c);
            const V t = ac * ac;
            const V vs = select(ac < V(S(0x1p-9)), ac, ac * numerics::rational<C>(t));
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
            const V rest = residual(q, d, num);
            return fma(fma(-q, dl, rest), inv, q);
        } else
            return one / (one + exp<Tier>(-x));
    }

    template<class Tier, class V>
    WRAY_INLINE pair<V> exp_half(const V &a) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using C = typename table<Tier, S>::exp;
        const auto red = exp_reduce(clamp(a, V(K::min_arg - S(1)), V(K::max_arg + S(2))));
        const V r = red.hi;
        const V one(S(1));
        const V hi = one + r;
        const V small = fma(r * r, numerics::rational<C>(r), r - (hi - one));
        const V p = hi + small;
        const V k = red.lo - one;
        return {scale_split(p, k), scale_split((hi - p) + small, k)};
    }

    template<class Tier, class V>
    WRAY_INLINE V sinh(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using C = typename table<Tier, S>::sinh;
        const V a = abs(x);
        const V t = a * a;
        const V small = fma(a * t, numerics::rational<C>(t), a);
        const auto e = exp_half<Tier>(a);
        const V quarter(S(0.25));
        const V inv = quarter / e.hi;
        const V d = e.hi - inv;
        const V rest = ((e.hi - d) - inv) + (fma(e.lo, inv / e.hi, e.lo) - select(a < V(S(16)), residual(inv, e.hi, quarter) / e.hi, V(S(0))));
        const V large = select(a > V(K::max_arg + K::ln2), inf<V>(), d + rest);
        return copysign_from(select(a < V(S(1)), small, large), x);
    }

    template<class Tier, class V>
    WRAY_INLINE V cosh(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using C = typename table<Tier, S>::cosh;
        const V a = abs(x);
        const V t = a * a;
        const V one(S(1));
        const V y = fma(a, a * V(S(0.5)), (t * t) * numerics::rational<C>(t));
        const V small = one + y;
        const auto e = exp_half<Tier>(a);
        const V quarter(S(0.25));
        const V inv = quarter / e.hi;
        const V d = e.hi + inv;
        const V rest = ((e.hi - d) + inv) + (fma(-e.lo, inv / e.hi, e.lo) + select(a < V(S(16)), residual(inv, e.hi, quarter) / e.hi, V(S(0))));
        const V large = select(a > V(K::max_arg + K::ln2), inf<V>(), d + rest);
        return select(a < one, small, large);
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
    WRAY_INLINE V pow_magnitude(const V &x, const V &y) noexcept
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
    WRAY_INLINE V pow(const V &x, const V &y) noexcept
    {
        using S = scalar_t<V>;
        using F = format<S>;
        using I = integer_t<V>;
        const V r = pow_magnitude<Tier>(abs(x), y);
        const V half = y * V(S(0.5));
        const auto whole = round_nearest(y) == y;
        const auto odd = both(whole, round_nearest(half) != half);
        const auto negative = nonzero(as_bits(x) & I(F::sign_bit));
        const V signed_r = select(both(negative, odd), -r, r);
        const auto invalid = both(both(x < V(S(0)), abs(x) < inf<V>()), !whole);
        return select(invalid, nan<V>(), signed_r);
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

    template<class Tier, class V>
    WRAY_INLINE V asinh(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using C = typename table<Tier, S>::asinh;
        const V one(S(1));
        const V a = abs(x);
        const V t = a * a;
        const V small = fma(a * t, numerics::rational<C>(t), a);
        const V big(std::numeric_limits<S>::digits == 24 ? S(0x1p13) : S(0x1p27));
        const V m = select(a < big, a, one);
        const auto mm = two_product(m, m);
        const V q = one + mm.hi;
        const V qe = ((one - q) + mm.hi) + mm.lo;
        const V root = simd::sqrt(q);
        const V re = (residual(root, root, q) + qe) / (root + root);
        const V d = root + one;
        const V de = ((root - d) + one) + re;
        const V fh = mm.hi / d;
        const V fl = (residual(fh, d, mm.hi) + mm.lo - fh * de) / d;
        const auto u = two_sum(m, fh);
        const V ul = u.lo + fl;
        const V medium = log1p_combine<Tier, false>(u.hi, ul, one + u.hi);
        auto p = log_reduce<Tier, false>(select(a < inf<V>(), a, one));
        p.e = p.e + one;
        const V large = select(a < inf<V>(), log_combine(p), a);
        const V v = select(a < V(S(0.5)), small, select(a < big, medium, large));
        return select(isnan(x), x, copysign_from(v, x));
    }

    template<class Tier, class V>
    WRAY_INLINE V cbrt(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using F = format<S>;
        using I = integer_t<V>;
        using C = typename table<Tier, S>::cbrt;
        const V one(S(1));
        const V third(S(1) / S(3));
        const V a = abs(x);
        const auto sub = a < V(F::min_normal);
        const V as = select(sub, a * V(F::subnormal_scale), a);
        const I bits = as_bits(as);
        const V e = from_bits<V>(shift_right_logical(bits, F::mantissa) | as_bits(V(F::two_mantissa))) - V(F::exponent_offset);
        const V m = from_bits<V>((bits & I(F::mantissa_mask)) | I(F::one_bits));
        const V q = round_nearest((e - one) * third);
        const V y = m * pow2(e - V(S(3)) * q);
        V g = numerics::rational<C>(y);
        constexpr int steps = std::is_same_v<S, float> ? 2 : 3;
        for (int i = 0; i < steps; ++i)
            g = (g * third) * fma(-y, (g * g) * g, V(S(4)));
        const V gg = g * g;
        const V c = y * gg;
        const auto c2 = two_product(c, c);
        const auto c3 = two_product(c2.hi, c);
        const V rest = ((y - c3.hi) - c3.lo) - c2.lo * c;
        const V root = fma(rest, gg * third, c);
        const V k = q - select(sub, V(F::subnormal_exponent / S(3)), V(S(0)));
        const V v = copysign_from(root * pow2(k), x);
        return select(either(either(a == V(S(0)), a == inf<V>()), isnan(x)), x, v);
    }

    template<class Tier, class V>
    WRAY_INLINE V erf(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        using Cs = typename table<Tier, S>::erf;
        using Cl = typename table<Tier, S>::erfc;
        const V one(S(1));
        const V a = abs(x);
        const V t = x * x;
        const auto lead = two_product(x, V(K::two_over_sqrt_pi_hi));
        const V small = lead.hi + (lead.lo + x * fma(t, numerics::rational<Cs>(t), V(K::two_over_sqrt_pi_lo)));
        const V cut(std::numeric_limits<S>::digits == 24 ? S(4) : S(6));
        const V ac = simd::max(simd::min(a, cut), V(S(1.5)));
        const auto sq = two_product(ac, ac);
        const V e = exp_regular<Tier>(-sq.hi) * (one - sq.lo);
        const V large = one - e * numerics::rational<Cl>(ac);
        using Cm = typename table<Tier, S>::erf_mid;
        const V d = a - V(S(0.75));
        const V mid = V(K::erf_mid_hi) + fma(d, numerics::rational<Cm>(d), V(K::erf_mid_lo));
        using Cm2 = typename table<Tier, S>::erf_mid2;
        const V d2 = a - V(S(1.25));
        const V mid2 = V(K::erf_mid2_hi) + fma(d2, numerics::rational<Cm2>(d2), V(K::erf_mid2_lo));
        const V outer = copysign_from(select(a < one, mid, select(a < V(S(1.5)), mid2, select(a < cut, large, one))), x);
        const V up(S(0x1p64));
        const V xs = x * up;
        const auto lead_s = two_product(xs, V(K::two_over_sqrt_pi_hi));
        const V scaled = (lead_s.hi + (lead_s.lo + xs * V(K::two_over_sqrt_pi_lo))) * V(S(0x1p-64));
        const V tiny = select(a < V(tiny_turn<S>), scaled, small);
        const V v = select(a < V(S(0.5)), keep_zero(x, tiny), outer);
        return select(isnan(x), x, v);
    }

    template<class Tier, class V>
    WRAY_INLINE V octaves(const V &d, const V &dl, scalar_t<V> steps, scalar_t<V> c_hi, scalar_t<V> c_lo, scalar_t<V> base) noexcept
    {
        using S = scalar_t<V>;
        using C = typename table<Tier, S>::exp;
        const V one(S(1));
        const V limit(steps * (std::numeric_limits<S>::digits == 24 ? S(160) : S(1100)));
        const V dc = simd::min(simd::max(d, -limit), limit);
        const V dlc = select(dc == d, dl, V(S(0)));
        const V q = round_nearest(dc * V(S(1) / steps));
        const V r = dc - q * V(steps);
        const auto t = two_product(r, V(c_hi));
        const V tl = fma(dlc, V(c_hi), fma(r, V(c_lo), t.lo));
        const V hi = one + t.hi;
        const V small = fma(t.hi * t.hi, numerics::rational<C>(t.hi), t.hi - (hi - one));
        const V p = hi + small;
        const V pl = fma(p, tl, (hi - p) + small);
        const V v = scale_split(fma(V(base), p, V(base) * pl), q);
        return select(either(isnan(d), d == inf<V>()), d, select(d == -inf<V>(), V(S(0)), v));
    }

    template<class Tier, class V>
    WRAY_INLINE V midi_to_hz(const V &m) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const auto d = two_sum(m, V(S(-69)));
        return octaves<Tier>(d.hi, d.lo, S(12), K::ln2_per_semitone_hi, K::ln2_per_semitone_lo, S(440));
    }

    template<class Tier, class V>
    WRAY_INLINE V ratio_from_cents(const V &c) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        return octaves<Tier>(c, V(S(0)), S(1200), K::ln2_per_cent_hi, K::ln2_per_cent_lo, S(1));
    }

    template<class V>
    WRAY_INLINE V hz_to_midi_far(const log_parts<V> &p) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const auto a = two_product(p.hi, V(K::semitones_per_ln_hi));
        const V cross = fma(p.hi, V(K::semitones_per_ln_lo), p.lo * V(K::semitones_per_ln_hi));
        const V b = p.e * V(S(12)) + V(K::midi_offset_hi);
        const auto t = two_sum(b, a.hi);
        return t.hi + (t.lo + (a.lo + (cross + V(K::midi_offset_lo))));
    }

    template<class Tier, class V>
    V hz_to_midi_edge(const V &f) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V one(S(1));
        const V far = hz_to_midi_far(log_reduce<Tier, true>(f));
        const V g = select(both(f > V(S(4)), f < V(S(16))), f, V(S(8)));
        const V rh = g / V(K::midi_zero_hz_hi);
        const V rl = fma(-rh, V(K::midi_zero_hz_lo), residual(rh, V(K::midi_zero_hz_hi), g)) / V(K::midi_zero_hz_hi);
        const V ln = log1p_combine<Tier, false>(rh - one, rl, rh);
        const auto z = two_product(ln, V(K::semitones_per_ln_hi));
        const V near = z.hi + fma(ln, V(K::semitones_per_ln_lo), z.lo);
        return log_specials(f, select(abs(far) < V(S(2)), near, far));
    }

    template<class Tier, class V>
    WRAY_INLINE V hz_to_midi(const V &f) noexcept
    {
        using S = scalar_t<V>;
        if (simd::any(either(!log_regular_mask(f), both(f > V(S(4)), f < V(S(16))))))
            return hz_to_midi_edge<Tier>(f);
        return hz_to_midi_far(log_reduce<Tier, false>(f));
    }

    template<class V>
    WRAY_INLINE pair<V> quotient(const V &n, const V &d) noexcept
    {
        const V h = n / d;
        return {h, residual(h, d, n) / d};
    }

    template<class V>
    WRAY_INLINE V either_nan(const V &a, const V &b, const V &r) noexcept
    {
        return select(either(isnan(a), isnan(b)), nan<V>(), r);
    }

    template<class Tier, class V>
    V smoothing_coefficient_edge(const V &seconds, const V &fs) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const auto p = two_product(seconds, fs);
        const auto x = quotient(one, p.hi);
        const auto overflow = both(p.hi == inf<V>(), both(abs(seconds) < inf<V>(), abs(fs) < inf<V>()));
        const V u = select(overflow, (one / simd::min(seconds, fs)) / simd::max(seconds, fs), x.hi);
        const V correction = fma(-x.hi, p.lo / p.hi, x.lo);
        const V xl = select(both(abs(correction) < inf<V>(), both(abs(x.hi) < inf<V>(), !overflow)), correction, V(S(0)));
        const V e = expm1<Tier>(-u);
        const V r = fma(xl, one + e, -e);
        return either_nan(seconds, fs, select(either(seconds <= V(S(0)), fs <= V(S(0))), one, r));
    }

    template<class Tier, class V>
    WRAY_INLINE V smoothing_coefficient(const V &seconds, const V &fs) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const auto p = two_product(seconds, fs);
        if (simd::any(!both(both(seconds > V(S(0)), fs > V(S(0))), both(p.hi >= V(format<S>::min_normal), p.hi < inf<V>()))))
            return smoothing_coefficient_edge<Tier>(seconds, fs);
        const auto x = quotient(one, p.hi);
        const V xl = fma(-x.hi, p.lo / p.hi, x.lo);
#if !WRAY_HAS_FMA
        if (simd::any(!(abs(xl) < inf<V>())))
            return smoothing_coefficient_edge<Tier>(seconds, fs);
#endif
        const V e = expm1<Tier>(-x.hi);
        return fma(xl, one + e, -e);
    }

    template<class V>
    struct cutoff_ratio
    {
        V hi;
        V lo;
    };

    template<class V>
    WRAY_INLINE cutoff_ratio<V> ratio_of(const V &hz, const V &fs) noexcept
    {
        using S = scalar_t<V>;
        const auto x = quotient(hz, fs);
        V hi = select(hz <= V(S(0)), V(S(0)), x.hi);
        hi = select(fs <= V(S(0)), inf<V>(), hi);
        const V lo = select(both(hi == x.hi, abs(x.lo) < inf<V>()), x.lo, V(S(0)));
        return {hi, lo};
    }

    template<class Tier, class V>
    WRAY_INLINE V pole_from_cutoff(const V &hz, const V &fs) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const auto x = ratio_of(hz, fs);
        const auto finite = x.hi < V(S(0x1p32));
        const V xh = select(finite, x.hi, V(S(0x1p32)));
        const V xl = select(finite, x.lo, V(S(0)));
        const V two_pi(S(2) * K::pi_hi);
        const auto t = two_product(xh, two_pi);
        const V tl = fma(xl, two_pi, fma(xh, V(S(2) * K::pi_lo), t.lo));
        const V e = exp<Tier>(-t.hi);
        return either_nan(hz, fs, fma(-tl, e, e));
    }

    template<class Tier, class V>
    WRAY_INLINE V prewarp(const V &hz, const V &fs) noexcept
    {
        using S = scalar_t<V>;
        const auto x = ratio_of(hz, fs);
        const auto inside = x.hi < V(S(0.5));
        const V edge(S(0.5) - std::numeric_limits<S>::epsilon() / S(4));
        return either_nan(hz, fs, tanpi<Tier>(select(inside, x.hi, edge), select(inside, x.lo, V(S(0)))));
    }

    template<class Tier, class V>
    WRAY_INLINE pair<V> log_pair(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const auto p = log_reduce<Tier, true>(x);
        const auto t = two_sum(p.e * V(K::ln2_hi), p.hi);
        return {t.hi, t.lo + fma(p.e, V(K::ln2_lo), p.lo)};
    }

    template<class Tier, class V>
    WRAY_INLINE V omega_pair(const V &x, const V &xl) noexcept
    {
        using S = scalar_t<V>;
        using C = typename table<Tier, S>::omega;
        const V one(S(1));
        const V z = exp<Tier>(simd::min(x, V(S(-2))));
        const V series = z * fma(z, fma(z, fma(z, fma(z, V(S(125) / S(24)), V(S(-8) / S(3))), V(S(1.5))), V(S(-1))), one);
        const V xs = simd::max(x, V(S(3)));
        const V l = log<Tier>(xs);
        const V big = xs - l + (l / xs) * fma(l - V(S(2)), V(S(0.5)) / xs, one);
        V y = select(x < V(S(-2)), series, select(x < V(S(3)), numerics::rational<C>(simd::min(simd::max(x, V(S(-2))), V(S(3)))), big));
        constexpr int steps = full<Tier> ? (std::is_same_v<S, float> ? 2 : 3) : 1;
        for (int i = 0; i < steps; ++i)
            y = fma(-(y + log<Tier>(y) - x - xl), y / (one + y), y);
        if constexpr (full<Tier>)
        {
            const auto lp = log_pair<Tier>(y);
            const V r = (((x - lp.hi) - lp.lo) + xl) - y;
            y = fma(r, y / (one + y), y);
            const auto t = two_sum(x, -y);
            const V e = exp<Tier>(t.hi);
            y = select(x < V(S(0)), fma(e, t.lo + xl, e), y);
        }
        const V tiny(std::numeric_limits<S>::digits == 24 ? S(-104) : S(-746));
        const V huge(S(1) / format<S>::min_normal);
        y = select(x < V(S(-40)), exp<Tier>(x), y);
        y = select(x > huge, x - log<Tier>(simd::min(x, V(std::numeric_limits<S>::max()))), y);
        y = select(x < tiny, V(S(0)), y);
        return select(isnan(x), x, select(x == inf<V>(), x, y));
    }

    template<class Tier, class V>
    WRAY_INLINE V omega(const V &x) noexcept
    {
        return omega_pair<Tier>(x, V(scalar_t<V>(0)));
    }

    template<class V>
    WRAY_INLINE V branch_gap(const V &v) noexcept
    {
        using S = scalar_t<V>;
        constexpr int degree = std::numeric_limits<S>::digits == 24 ? 8 : 15;
        S c[static_cast<std::size_t>(degree + 1)] = {};
        S fact = S(2);
        for (int j = 0; j <= degree; ++j)
        {
            c[j] = S(j + 1) / fact;
            fact *= S(j + 3);
        }
        V acc(c[degree]);
        for (int j = degree - 1; j >= 0; --j)
            acc = fma(acc, v, V(c[j]));
        return v * v * acc;
    }

    template<class Tier, class V>
    WRAY_INLINE V product_log(const V &x) noexcept
    {
        using S = scalar_t<V>;
        using K = constants<S>;
        const V one(S(1));
        const V lp_x = simd::max(x, V(S(0.5)));
        const auto lp = log_pair<Tier>(lp_x);
        const V upper = omega_pair<Tier>(lp.hi, lp.lo);
        const auto ex = two_product(x, V(K::e_hi));
        const V gap = ((one + ex.hi) + ex.lo) + x * V(K::e_lo);
        const V g = simd::max(gap, V(S(0)));
        const V p = simd::sqrt(V(S(2)) * g);
        V v = p * fma(p, fma(p, fma(p, V(S(-43) / S(540)), V(S(11) / S(72))), V(S(-1) / S(3))), one);
        constexpr int branch_steps = full<Tier> ? (std::is_same_v<S, float> ? 2 : 3) : 1;
        for (int i = 0; i < branch_steps; ++i)
            v = select(v > V(S(0)), v - (branch_gap(v) - g) / (v * exp<Tier>(v)), v);
        const V branch = v - one;
        const V xn = simd::min(simd::max(x, V(S(-0.32))), V(S(0.5)));
        V w = log1p<Tier>(xn);
        constexpr int steps = full<Tier> ? 3 : 2;
        for (int i = 0; i < steps; ++i)
        {
            const V e = exp<Tier>(w);
            const V f = fma(w, e, -xn);
            const V w1 = w + one;
            const V d = fma(-(w + V(S(2))) / (w1 + w1), f, e * w1);
            w = select(f == V(S(0)), w, w - f / d);
        }
        if constexpr (full<Tier>)
        {
            const auto e = exp_half<Tier>(w);
            const V f = (w + w) * e.lo - residual(w + w, e.hi, xn);
            w = w - f / ((e.hi + e.hi) * (w + one));
        }
        const V lower = keep_zero(x, w);
        const V v2 = select(x < V(S(-0.32)), branch, select(x < V(S(0.5)), lower, upper));
        return select(either(isnan(x), either(gap < V(S(0)), x < V(S(-0.5)))), select(isnan(x), x, nan<V>()), select(x == inf<V>(), x, v2));
    }

    template<class V>
    WRAY_INLINE V log10_combine(const log_parts<V> &p) noexcept
    {
        using K = constants<scalar_t<V> >;
        const V inner = fma(p.hi, V(K::log10_per_ln_lo), p.lo * V(K::log10_per_ln_hi));
        const V v = fma(p.hi, V(K::log10_per_ln_hi), inner);
        const V small = fma(p.e, V(K::log10_per_log2_lo), v);
        return fma(p.e, V(K::log10_per_log2_hi), small);
    }

    template<class Tier, class V>
    V log10_edge(const V &x) noexcept
    {
        return log_specials(x, log10_combine(log_reduce<Tier, true>(x)));
    }

    template<class Tier, class V>
    WRAY_INLINE V log10(const V &x) noexcept
    {
        if (simd::any(!log_regular_mask(x)))
            return log10_edge<Tier>(x);
        return log10_combine(log_reduce<Tier, false>(x));
    }
}

WRAY_FP_END
