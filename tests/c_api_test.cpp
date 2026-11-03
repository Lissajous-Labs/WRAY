#include <wray/c/wray.h>
#include <wray/wray.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include "check.h"

namespace
{
    template <class T>
    bool same(T a, T b)
    {
        if (std::isnan(a) && std::isnan(b))
            return true;
        using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
        return std::bit_cast<U>(a) == std::bit_cast<U>(b);
    }

    template <class T>
    std::vector<T> inputs(std::mt19937_64 &g, std::size_t n)
    {
        std::uniform_real_distribution<double> wide(-200.0, 200.0);
        const T specials[] = {T(0), -T(0), T(1), T(-1), T(0.25), T(440), T(48000), std::numeric_limits<T>::infinity(),
                              -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::denorm_min()};
        std::vector<T> v(n);
        for (std::size_t i = 0; i < n; ++i)
            v[i] = i % 7 == 0 ? specials[(i / 7) % std::size(specials)] : static_cast<T>(wide(g));
        return v;
    }

    template <class T, class C, class F>
    void unary(std::mt19937_64 &g, C c, F f)
    {
        for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(7), std::size_t(64), std::size_t(1027)})
        {
            const auto x = inputs<T>(g, n);
            std::vector<T> got(n + 1, T(7)), want(n + 1, T(7));
            c(x.data(), got.data(), n);
            f(std::span<const T>(x.data(), n), std::span<T>(want.data(), n));
            bool ok = got[n] == T(7);
            for (std::size_t i = 0; i < n; ++i)
                ok = ok && same(got[i], want[i]);
            std::vector<T> io(x);
            c(io.data(), io.data(), n);
            for (std::size_t i = 0; i < n; ++i)
                ok = ok && same(io[i], want[i]);
            CHECK(ok);
        }
    }

    template <class T, class C, class F>
    void pair(std::mt19937_64 &g, C c, F f)
    {
        for (std::size_t n : {std::size_t(0), std::size_t(5), std::size_t(1027)})
        {
            const auto x = inputs<T>(g, n);
            std::vector<T> s(n), co(n), ws(n), wc(n);
            c(x.data(), s.data(), co.data(), n);
            f(std::span<const T>(x.data(), n), std::span<T>(ws.data(), n), std::span<T>(wc.data(), n));
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i)
                ok = ok && same(s[i], ws[i]) && same(co[i], wc[i]);
            CHECK(ok);
        }
    }

    template <class T, class C, class F>
    void binary(std::mt19937_64 &g, C c, F f)
    {
        for (std::size_t n : {std::size_t(0), std::size_t(3), std::size_t(1027)})
        {
            const auto a = inputs<T>(g, n);
            const auto b = inputs<T>(g, n);
            std::vector<T> got(n), want(n);
            c(a.data(), b.data(), got.data(), n);
            f(std::span<const T>(a.data(), n), std::span<const T>(b.data(), n), std::span<T>(want.data(), n));
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i)
                ok = ok && same(got[i], want[i]);
            CHECK(ok);
        }
    }
    template <class T, class Unary, class Pair>
    void balanced_abi(std::mt19937_64& g, Unary sine, Unary cosine, Pair paired, Unary hyperbolic)
    {
        constexpr std::size_t w = std::size_t(wray::simd::native<T>::size);
        for (std::size_t n = 0; n <= 4 * w + 1; ++n)
        {
            auto x = inputs<T>(g, n);
            for (std::size_t i = 0; i < n; ++i)
                if (i % 4 == 1)
                    x[i] = std::copysign(std::numeric_limits<T>::max(), T(i % 3 ? 1 : -1));
            for (const auto fn : {sine, cosine, hyperbolic})
            {
                std::vector<T> got(n + 2, T(7));
                fn(x.data(), got.data() + 1, n);
                bool ok = got.front() == T(7) && got.back() == T(7);
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(got[i + 1], fn == sine ? wray::balanced::sin(x[i]) :
                                              fn == cosine ? wray::balanced::cos(x[i]) : wray::balanced::tanh(x[i]));
                auto io = x;
                fn(io.data(), io.data(), n);
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(io[i], got[i + 1]);
                CHECK(ok);
            }
            for (int alias = 0; alias < 3; ++alias)
            {
                auto io = x;
                std::vector<T> s(n + 2, T(7)), c(n + 2, T(7));
                T* ps = alias == 1 ? io.data() : s.data() + 1;
                T* pc = alias == 2 ? io.data() : c.data() + 1;
                paired(io.data(), ps, pc, n);
                bool ok = s.front() == T(7) && s.back() == T(7) && c.front() == T(7) && c.back() == T(7);
                for (std::size_t i = 0; i < n; ++i)
                {
                    const auto expected = wray::balanced::sincos(x[i]);
                    ok = ok && same(ps[i], expected.sin) && same(pc[i], expected.cos);
                }
                CHECK(ok);
            }
        }
        const T edge[] = {std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(),
                          std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN(), -T(0)};
        T s[std::size(edge)], c[std::size(edge)];
        paired(edge, s, c, std::size(edge));
        for (std::size_t i = 0; i < std::size(edge); ++i)
        {
            const auto expected = wray::balanced::sincos(edge[i]);
            CHECK(same(s[i], expected.sin) && same(c[i], expected.cos));
        }
        sine(edge, s, std::size(edge));
        cosine(edge, c, std::size(edge));
        for (std::size_t i = 0; i < std::size(edge); ++i)
            CHECK(same(s[i], wray::balanced::sin(edge[i])) && same(c[i], wray::balanced::cos(edge[i])));
        hyperbolic(edge, s, std::size(edge));
        for (std::size_t i = 0; i < std::size(edge); ++i)
            CHECK(same(s[i], wray::balanced::tanh(edge[i])));
    }


#define WRAY_C_UNARY(ns, prefix, f, T, t) unary<T>(g, wray_##prefix##f##_##t, [](auto in, auto out) { ns::f(in, out); });
#define WRAY_C_PAIR(ns, prefix, f, T, t) pair<T>(g, wray_##prefix##f##_##t, [](auto in, auto s, auto c) { ns::f(in, s, c); });
#define WRAY_C_BINARY(ns, prefix, f, T, t) binary<T>(g, wray_##prefix##f##_##t, [](auto a, auto b, auto out) { ns::f(a, b, out); });

    void all(std::mt19937_64 &g)
    {
        WRAY_C_UNARY(wray, , exp, float, f32)
        WRAY_C_UNARY(wray, , expm1, float, f32)
        WRAY_C_UNARY(wray, , exp2, float, f32)
        WRAY_C_UNARY(wray, , log, float, f32)
        WRAY_C_UNARY(wray, , log1p, float, f32)
        WRAY_C_UNARY(wray, , log10, float, f32)
        WRAY_C_UNARY(wray, , log2, float, f32)
        WRAY_C_UNARY(wray, , wrap_turns, float, f32)
        WRAY_C_UNARY(wray, , sin_turns, float, f32)
        WRAY_C_UNARY(wray, , sin, float, f32)
        WRAY_C_UNARY(wray, , cos_turns, float, f32)
        WRAY_C_UNARY(wray, , cos, float, f32)
        WRAY_C_UNARY(wray, , tanpi, float, f32)
        WRAY_C_UNARY(wray, , tan_turns, float, f32)
        WRAY_C_UNARY(wray, , tan, float, f32)
        WRAY_C_UNARY(wray, , atan, float, f32)
        WRAY_C_UNARY(wray, , sinh, float, f32)
        WRAY_C_UNARY(wray, , cosh, float, f32)
        WRAY_C_UNARY(wray, , asinh, float, f32)
        WRAY_C_UNARY(wray, , cbrt, float, f32)
        WRAY_C_UNARY(wray, , erf, float, f32)
        WRAY_C_UNARY(wray, , midi_to_hz, float, f32)
        WRAY_C_UNARY(wray, , hz_to_midi, float, f32)
        WRAY_C_UNARY(wray, , ratio_from_cents, float, f32)
        WRAY_C_UNARY(wray, , omega, float, f32)
        WRAY_C_UNARY(wray, , product_log, float, f32)
        WRAY_C_UNARY(wray, , tanh, float, f32)
        WRAY_C_UNARY(wray, , sigmoid, float, f32)
        WRAY_C_UNARY(wray, , db_to_gain, float, f32)
        WRAY_C_UNARY(wray, , gain_to_db, float, f32)
        WRAY_C_PAIR(wray, , sincos, float, f32)
        WRAY_C_PAIR(wray, , sincos_turns, float, f32)
        WRAY_C_BINARY(wray, , smoothing_coefficient, float, f32)
        WRAY_C_BINARY(wray, , pole_from_cutoff, float, f32)
        WRAY_C_BINARY(wray, , prewarp, float, f32)
        WRAY_C_BINARY(wray, , atan2, float, f32)
        WRAY_C_BINARY(wray, , pow, float, f32)
        WRAY_C_UNARY(wray, , exp, double, f64)
        WRAY_C_UNARY(wray, , expm1, double, f64)
        WRAY_C_UNARY(wray, , exp2, double, f64)
        WRAY_C_UNARY(wray, , log, double, f64)
        WRAY_C_UNARY(wray, , log1p, double, f64)
        WRAY_C_UNARY(wray, , log10, double, f64)
        WRAY_C_UNARY(wray, , log2, double, f64)
        WRAY_C_UNARY(wray, , wrap_turns, double, f64)
        WRAY_C_UNARY(wray, , sin_turns, double, f64)
        WRAY_C_UNARY(wray, , sin, double, f64)
        WRAY_C_UNARY(wray, , cos_turns, double, f64)
        WRAY_C_UNARY(wray, , cos, double, f64)
        WRAY_C_UNARY(wray, , tanpi, double, f64)
        WRAY_C_UNARY(wray, , tan_turns, double, f64)
        WRAY_C_UNARY(wray, , tan, double, f64)
        WRAY_C_UNARY(wray, , atan, double, f64)
        WRAY_C_UNARY(wray, , sinh, double, f64)
        WRAY_C_UNARY(wray, , cosh, double, f64)
        WRAY_C_UNARY(wray, , asinh, double, f64)
        WRAY_C_UNARY(wray, , cbrt, double, f64)
        WRAY_C_UNARY(wray, , erf, double, f64)
        WRAY_C_UNARY(wray, , midi_to_hz, double, f64)
        WRAY_C_UNARY(wray, , hz_to_midi, double, f64)
        WRAY_C_UNARY(wray, , ratio_from_cents, double, f64)
        WRAY_C_UNARY(wray, , omega, double, f64)
        WRAY_C_UNARY(wray, , product_log, double, f64)
        WRAY_C_UNARY(wray, , tanh, double, f64)
        WRAY_C_UNARY(wray, , sigmoid, double, f64)
        WRAY_C_UNARY(wray, , db_to_gain, double, f64)
        WRAY_C_UNARY(wray, , gain_to_db, double, f64)
        WRAY_C_PAIR(wray, , sincos, double, f64)
        WRAY_C_PAIR(wray, , sincos_turns, double, f64)
        WRAY_C_BINARY(wray, , smoothing_coefficient, double, f64)
        WRAY_C_BINARY(wray, , pole_from_cutoff, double, f64)
        WRAY_C_BINARY(wray, , prewarp, double, f64)
        WRAY_C_BINARY(wray, , atan2, double, f64)
        WRAY_C_BINARY(wray, , pow, double, f64)
        WRAY_C_UNARY(wray::fast, fast_, exp, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, exp2, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, log, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, log2, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, sin_turns, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, sin, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, cos_turns, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, cos, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, tanpi, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, tan_turns, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, tan, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, atan, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, omega, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, product_log, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, tanh, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, sigmoid, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, db_to_gain, float, f32)
        WRAY_C_UNARY(wray::fast, fast_, gain_to_db, float, f32)
        WRAY_C_PAIR(wray::fast, fast_, sincos, float, f32)
        WRAY_C_PAIR(wray::fast, fast_, sincos_turns, float, f32)
        WRAY_C_BINARY(wray::fast, fast_, atan2, float, f32)
        WRAY_C_BINARY(wray::fast, fast_, pow, float, f32)
        balanced_abi<float>(g, wray_balanced_sin_f32, wray_balanced_cos_f32, wray_balanced_sincos_f32, wray_balanced_tanh_f32);
        balanced_abi<double>(g, wray_balanced_sin_f64, wray_balanced_cos_f64, wray_balanced_sincos_f64, wray_balanced_tanh_f64);
    }

#undef WRAY_C_UNARY
#undef WRAY_C_PAIR
#undef WRAY_C_BINARY
}

int main()
{
    std::mt19937_64 g(23);
    CHECK(wray_c_abi_version() == WRAY_C_ABI_VERSION);
    CHECK(wray_c_has_fma() == (WRAY_HAS_FMA ? 1 : 0));
    all(g);
    return wray::test::finish("c_api");
}
