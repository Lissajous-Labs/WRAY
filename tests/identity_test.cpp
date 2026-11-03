#include <wray/wray.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "check.h"

namespace
{
    template <class T>
    bool same(T got, T want)
    {
        if (std::isnan(want))
            return std::isnan(got);
        return got == want && std::signbit(got) == std::signbit(want);
    }

    template <class T>
    bool annex_f_case(T x, T y)
    {
        const auto edge = [](T v) { return std::isnan(v) || std::isinf(v) || v == T(0); };
        return edge(x) || edge(y) || std::fabs(x) == T(1);
    }

    template <class T>
    void annex_f_pow()
    {
        using V = wray::simd::native<T>;
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T huge = std::numeric_limits<T>::max();
        const T big_odd = std::ldexp(T(1), std::numeric_limits<T>::digits) - T(1);
        const T big_even = std::ldexp(T(1), std::numeric_limits<T>::digits);
        const T xs[] = {T(0), -T(0), tiny, -tiny, T(0.5), T(-0.5), T(1), T(-1), T(2), T(-2), huge, -huge, inf, -inf, nan,
                        std::nextafter(T(1), T(2)), -std::nextafter(T(1), T(2)), std::nextafter(T(1), T(0)), -std::nextafter(T(1), T(0))};
        const T ys[] = {T(0), -T(0), T(0.5), T(-0.5), T(1), T(-1), T(2), T(-2), T(3), T(-3), inf, -inf, nan, huge, -huge,
                        big_odd, -big_odd, big_even, -big_even, big_odd - T(2), T(1) + std::numeric_limits<T>::epsilon()};
        int mismatches = 0;
        for (const T x: xs)
        {
            for (const T y: ys)
            {
                const T want = static_cast<T>(std::pow(static_cast<long double>(x), static_cast<long double>(y)));
                const T got = wray::pow(x, y);
                const T lane = wray::pow(V(x), V(y))[V::size - 1];
                CHECK(same(lane, got));
                bool ok;
                if (annex_f_case(x, y) || !std::isfinite(want))
                    ok = same(got, want);
                else
                    ok = std::signbit(got) == std::signbit(want) && std::isfinite(got) == std::isfinite(want) &&
                         (got == want || std::fabs(got - want) <= std::fabs(want) * T(1e-3) || (std::fabs(want) < std::numeric_limits<T>::min() * T(4) && std::fabs(got - want) <= std::numeric_limits<T>::min()));
                if (!ok && mismatches++ < 8)
                    std::printf("pow(%a, %a) = %a, C gives %a\n", static_cast<double>(x), static_cast<double>(y), static_cast<double>(got), static_cast<double>(want));
                CHECK(ok);
            }
        }
    }

    template <class T>
    using unary = T (*)(T);

    template <class V>
    using vector_unary = V (*)(V);

    template <class T, class V>
    struct named
    {
        const char *name;
        unary<T> fn;
        vector_unary<V> vfn;
    };

#define WRAY_ROW(ns, f) named<T, V>{#ns "::" #f, [](T x) { return ns::f(x); }, [](V x) { return ns::f(x); }}

    template <class T, class V>
    std::vector<named<T, V> > precise_rows()
    {
        return {WRAY_ROW(wray, exp), WRAY_ROW(wray, expm1), WRAY_ROW(wray, exp2), WRAY_ROW(wray, log), WRAY_ROW(wray, log1p), WRAY_ROW(wray, log10),
                WRAY_ROW(wray, log2), WRAY_ROW(wray, wrap_turns), WRAY_ROW(wray, sin_turns), WRAY_ROW(wray, sin), WRAY_ROW(wray, cos_turns),
                WRAY_ROW(wray, cos), WRAY_ROW(wray, tanpi), WRAY_ROW(wray, tan_turns), WRAY_ROW(wray, tan), WRAY_ROW(wray, atan), WRAY_ROW(wray, sinh),
                WRAY_ROW(wray, cosh), WRAY_ROW(wray, asinh), WRAY_ROW(wray, cbrt), WRAY_ROW(wray, erf), WRAY_ROW(wray, midi_to_hz),
                WRAY_ROW(wray, hz_to_midi), WRAY_ROW(wray, ratio_from_cents), WRAY_ROW(wray, omega), WRAY_ROW(wray, product_log),
                WRAY_ROW(wray, tanh), WRAY_ROW(wray, sigmoid), WRAY_ROW(wray, db_to_gain), WRAY_ROW(wray, gain_to_db)};
    }

    template <class T, class V>
    std::vector<named<T, V> > fast_rows()
    {
        return {WRAY_ROW(wray::fast, exp), WRAY_ROW(wray::fast, exp2), WRAY_ROW(wray::fast, log), WRAY_ROW(wray::fast, log2),
                WRAY_ROW(wray::fast, sin_turns), WRAY_ROW(wray::fast, sin), WRAY_ROW(wray::fast, cos_turns), WRAY_ROW(wray::fast, cos),
                WRAY_ROW(wray::fast, tanpi), WRAY_ROW(wray::fast, tan_turns), WRAY_ROW(wray::fast, tan), WRAY_ROW(wray::fast, atan),
                WRAY_ROW(wray::fast, omega), WRAY_ROW(wray::fast, product_log), WRAY_ROW(wray::fast, tanh), WRAY_ROW(wray::fast, sigmoid),
                WRAY_ROW(wray::fast, db_to_gain), WRAY_ROW(wray::fast, gain_to_db)};
    }

    template <class T, class V>
    std::vector<named<T, V> > balanced_rows()
    {
        return {WRAY_ROW(wray::balanced, sin), WRAY_ROW(wray::balanced, cos), WRAY_ROW(wray::balanced, tanh),
                named<T, V>{"balanced::sincos_sin", [](T x) { return wray::balanced::sincos(x).sin; }, [](V x) { return wray::balanced::sincos(x).sin; }},
                named<T, V>{"balanced::sincos_cos", [](T x) { return wray::balanced::sincos(x).cos; }, [](V x) { return wray::balanced::sincos(x).cos; }}};
    }

#undef WRAY_ROW

    template <class T, class V>
    void mixed_lanes(const std::vector<named<T, V> > &rows)
    {
        const T specials[] = {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                              T(0), -T(0), std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(), T(-1),
                              std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(), T(1e7), T(-3e9)};
        const T ordinary[] = {T(0.3), T(1.7), T(-2.5), T(10), T(-0.04), T(57), T(0.9), T(-7)};
        for (const auto &r: rows)
        {
            for (const T s: specials)
            {
                for (int lane = 0; lane < V::size; ++lane)
                {
                    T in[V::size];
                    for (int i = 0; i < V::size; ++i)
                        in[i] = i == lane ? s : ordinary[i % 8];
                    const V out = r.vfn(V::load(in));
                    bool ok = true;
                    for (int i = 0; i < V::size; ++i)
                        ok = ok && same(out[i], r.fn(in[i]));
                    if (!ok)
                        std::printf("%s: lane %d holding %a disagrees with scalar\n", r.name, lane, static_cast<double>(s));
                    CHECK(ok);
                }
            }
        }
    }

    template <class T, class V>
    void metamorphic(std::uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<T> wide(T(-1e4), T(1e4));
        std::uniform_real_distribution<T> turns(T(-64), T(64));
        int failures = 0;
        const T tan_bound = std::numeric_limits<T>::epsilon() * T(8);
        for (int n = 0; n < 20000; ++n)
        {
            const T x = n % 3 == 0 ? wide(rng) : std::ldexp(wide(rng), static_cast<int>(rng() % 40) - 20);
            const auto sc = wray::sincos(V(x));
            const auto st = wray::sincos_turns(V(x));
            bool ok = same(sc.sin[0], wray::sin(x)) && same(sc.cos[0], wray::cos(x)) && same(st.sin[0], wray::sin_turns(x)) && same(st.cos[0], wray::cos_turns(x));
            const auto scalar_sc = wray::sincos(x);
            ok = ok && same(scalar_sc.sin, wray::sin(x)) && same(scalar_sc.cos, wray::cos(x));
            const auto bsc = wray::balanced::sincos(V(x));
            const auto scalar_bsc = wray::balanced::sincos(x);
            const T bs = wray::balanced::sin(x);
            const T bc = wray::balanced::cos(x);
            ok = ok && same(bsc.sin[0], scalar_bsc.sin) && same(bsc.cos[0], scalar_bsc.cos);
            const T norm_bound = T(16) * std::numeric_limits<T>::epsilon();
            ok = ok && std::fabs(bs * bs + bc * bc - T(1)) <= norm_bound;
            ok = ok && std::fabs(scalar_bsc.sin * scalar_bsc.sin + scalar_bsc.cos * scalar_bsc.cos - T(1)) <= norm_bound;
            ok = ok && same(wray::balanced::sin(-x), -bs) && same(wray::balanced::cos(-x), bc);
            const auto negative_bsc = wray::balanced::sincos(-x);
            ok = ok && same(negative_bsc.sin, -scalar_bsc.sin) && same(negative_bsc.cos, scalar_bsc.cos);
            const T bt = wray::balanced::tanh(x);
            ok = ok && std::isfinite(bt) && bt >= T(-1) && bt <= T(1);
            ok = ok && same(wray::balanced::tanh(-x), -bt) && same(wray::balanced::tanh(V(x))[0], bt);
            if constexpr (std::is_same_v<T, float>)
            {
                const auto fsc = wray::fast::sincos(V(x));
                const auto fst = wray::fast::sincos_turns(V(x));
                ok = ok && same(fsc.sin[0], wray::fast::sin(x)) && same(fsc.cos[0], wray::fast::cos(x)) && same(fst.sin[0], wray::fast::sin_turns(x)) &&
                     same(fst.cos[0], wray::fast::cos_turns(x));
            }
            const T t = turns(rng);
            const T c = wray::cos_turns(t);
            if (std::fabs(c) > T(0.01))
            {
                const T q = wray::sin_turns(t) / c;
                const T tt = wray::tan_turns(t);
                ok = ok && std::fabs(tt - q) <= tan_bound * std::fabs(q) * (T(1) + T(1) / std::fabs(c));
            }
            const T w = wray::wrap_turns(t);
            const T k = std::nearbyint(t);
            ok = ok && w >= T(-0.5) && w <= T(0.5) && same(w, t - k);
            if (std::fabs(t) < T(1e6) && t + T(1) - T(1) == t)
                ok = ok && (same(wray::wrap_turns(t + T(1)), w) || std::fabs(w) == T(0.5));
            if (!ok && failures++ < 8)
                std::printf("metamorphic failure at x=%a t=%a (seed %llu)\n", static_cast<double>(x), static_cast<double>(t), static_cast<unsigned long long>(seed));
            CHECK(ok);
        }
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T huge = std::numeric_limits<T>::max();
        CHECK(same(wray::wrap_turns(T(0)), T(0)));
        CHECK(wray::wrap_turns(-T(0)) == T(0));
        CHECK(std::isnan(wray::wrap_turns(inf)) && std::isnan(wray::wrap_turns(-inf)) && std::isnan(wray::wrap_turns(nan)));
        CHECK(same(wray::wrap_turns(huge), T(0)) && same(wray::wrap_turns(-huge), T(0)));
        CHECK(same(wray::wrap_turns(std::numeric_limits<T>::denorm_min()), std::numeric_limits<T>::denorm_min()));
        CHECK(same(wray::wrap_turns(T(-0.75)), T(0.25)));
        CHECK(same(wray::wrap_turns(T(2.5)), T(0.5)) && same(wray::wrap_turns(T(3.5)), T(-0.5)));
    }

    template <class T, class V>
    void widths(std::uint64_t seed)
    {
        mixed_lanes<T, V>(precise_rows<T, V>());
        mixed_lanes<T, V>(balanced_rows<T, V>());
        if constexpr (std::is_same_v<T, float>)
            mixed_lanes<T, V>(fast_rows<T, V>());
        metamorphic<T, V>(seed);
    }
}

int main(int argc, char **argv)
{
    const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20261101;
    std::printf("identity: seed %llu\n", static_cast<unsigned long long>(seed));
    annex_f_pow<float>();
    annex_f_pow<double>();
    widths<float, wray::f32x4>(seed);
    widths<double, wray::f64x2>(seed);
    widths<float, wray::simd::native<float> >(seed);
    widths<double, wray::simd::native<double> >(seed);
    return wray::test::finish("identity");
}
