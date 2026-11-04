#include <wray/wray.h>

#include <algorithm>
#include <array>
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
        return std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t> >(a) ==
               std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t> >(b);
    }

    template <class T>
    void ranges(std::mt19937_64 &g)
    {
        constexpr std::size_t w = static_cast<std::size_t>(wray::simd::native<T>::size);
        std::uniform_real_distribution<T> u(T(-4), T(4));
        for (std::size_t n = 0; n <= 8 * w + 1; ++n)
        {
            for (std::size_t offset = 0; offset < w; ++offset)
            {
                std::vector<T> a(offset + n);
                std::vector<T> b(offset + n);
                for (std::size_t i = 0; i < a.size(); ++i)
                {
                    a[i] = u(g);
                    b[i] = u(g);
                }
                const std::span<const T> in(a.data() + offset, n);
                const std::span<const T> in2(b.data() + offset, n);

                std::vector<T> out(offset + n);
                wray::exp(in, std::span<T>(out.data() + offset, n));
                bool ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(out[offset + i], wray::exp(in[i]));
                CHECK(ok);

                wray::pow(in, in2, std::span<T>(out.data() + offset, n));
                ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(out[offset + i], wray::pow(in[i], in2[i]));
                CHECK(ok);

                if (n > 0)
                {
                    std::vector<T> shorter(n - 1, T(7));
                    wray::sin(in, shorter);
                    ok = true;
                    for (std::size_t i = 0; i + 1 < n; ++i)
                        ok = ok && same(shorter[i], wray::sin(in[i]));
                    CHECK(ok);
                    std::vector<T> longer(n + 3, T(7));
                    wray::sin(in, longer);
                    for (std::size_t i = n; i < n + 3; ++i)
                        ok = ok && longer[i] == T(7);
                    CHECK(ok);
                }

                std::vector<T> io(a);
                const std::span<T> place(io.data() + offset, n);
                wray::tanh(place, place);
                ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(place[i], wray::tanh(in[i]));
                CHECK(ok);

                if constexpr (std::is_same_v<T, float>)
                {
                    wray::fast::log(in, std::span<T>(out.data() + offset, n));
                    ok = true;
                    for (std::size_t i = 0; i < n; ++i)
                        ok = ok && same(out[offset + i], wray::fast::log(in[i]));
                    CHECK(ok);
                }
            }
        }
        std::vector<T> empty;
        wray::exp(empty, empty);
        wray::pow(empty, empty, empty);
        CHECK(empty.empty());
    }

    template<class T, class Range, class Scalar>
    void paired_ranges(Range apply, Scalar scalar)
    {
        constexpr std::size_t w = static_cast<std::size_t>(wray::simd::native<T>::size);
        const std::array<T, 12> values = {
            T(0), -T(0), T(0.25), T(-0.5), T(1.75), T(-100),
            std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
            std::numeric_limits<T>::min(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()
        };
        const auto equal = [](T a, T b) { return same(a, b) || (std::isnan(a) && std::isnan(b)); };
        for (std::size_t n = 0; n <= 8 * w + 1; ++n)
        {
            for (std::size_t offset = 0; offset < w; ++offset)
            {
                std::vector<T> a(offset + n + 2);
                for (std::size_t i = 0; i < a.size(); ++i)
                    a[i] = values[i % values.size()];
                const std::span<const T> in(a.data() + offset, n);
                for (int shortened = 0; shortened < 3; ++shortened)
                {
                    std::vector<T> s(offset + n + 3, T(7));
                    std::vector<T> c(s);
                    const std::size_t ns = shortened == 1 && n ? n - 1 : n;
                    const std::size_t nc = shortened == 2 && n ? n - 1 : n;
                    const std::size_t count = std::min({n, ns, nc});
                    apply(in, std::span<T>(s.data() + offset, ns), std::span<T>(c.data() + offset, nc));
                    bool ok = true;
                    for (std::size_t i = 0; i < s.size(); ++i)
                    {
                        if (i >= offset && i < offset + count)
                        {
                            const auto expected = scalar(in[i - offset]);
                            ok = ok && equal(s[i], expected.sin) && equal(c[i], expected.cos);
                        }
                        else
                            ok = ok && s[i] == T(7) && c[i] == T(7);
                    }
                    CHECK(ok);
                }
                for (bool sine_in_place : {false, true})
                {
                    std::vector<T> io(a);
                    std::vector<T> other(offset + n + 3, T(7));
                    const std::span<T> place(io.data() + offset, n);
                    const std::span<T> out(other.data() + offset, n);
                    apply(place, sine_in_place ? place : out, sine_in_place ? out : place);
                    bool ok = true;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        const auto expected = scalar(in[i]);
                        ok = ok && equal(place[i], sine_in_place ? expected.sin : expected.cos);
                        ok = ok && equal(out[i], sine_in_place ? expected.cos : expected.sin);
                    }
                    for (std::size_t i = 0; i < io.size(); ++i)
                        if (i < offset || i >= offset + n)
                            ok = ok && equal(io[i], a[i]);
                    for (std::size_t i = 0; i < other.size(); ++i)
                        if (i < offset || i >= offset + n)
                            ok = ok && other[i] == T(7);
                    CHECK(ok);
                }
            }
        }
    }
    template<class T, class Range, class Scalar>
    void balanced_unary_ranges(Range apply, Scalar scalar)
    {
        constexpr std::size_t w = static_cast<std::size_t>(wray::simd::native<T>::size);
        const T edge = std::is_same_v<T, float> ? T(0x1.921fb6p+12) : T(0x1.921fb54442d18p+22);
        const T values[] = {T(0), -T(0), T(0.3), T(-1.7), edge, std::nextafter(edge, T(0)),
                            std::nextafter(edge, std::numeric_limits<T>::infinity()),
                            std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
                            std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(),
                            std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()};
        const auto equal = [](T a, T b) { return same(a, b) || (std::isnan(a) && std::isnan(b)); };
        for (std::size_t n = 0; n <= 8 * w + 1; ++n)
        {
            for (std::size_t offset = 0; offset < w; ++offset)
            {
                std::vector<T> a(offset + n + 2);
                for (std::size_t i = 0; i < a.size(); ++i)
                    a[i] = values[i % std::size(values)];
                const std::span<const T> in(a.data() + offset, n);
                for (int delta : {-1, 0, 3})
                {
                    const std::size_t count = delta < 0 ? (n ? n - 1 : 0) : n + std::size_t(delta);
                    std::vector<T> out(offset + count + 2, T(7));
                    apply(in, std::span<T>(out.data() + offset, count));
                    bool ok = true;
                    for (std::size_t i = 0; i < out.size(); ++i)
                        ok = ok && (i >= offset && i < offset + std::min(n, count) ? equal(out[i], scalar(in[i - offset])) : out[i] == T(7));
                    CHECK(ok);
                }
                std::vector<T> io(a);
                const std::span<T> place(io.data() + offset, n);
                apply(place, place);
                bool ok = true;
                for (std::size_t i = 0; i < io.size(); ++i)
                    ok = ok && equal(io[i], i >= offset && i < offset + n ? scalar(in[i - offset]) : a[i]);
                CHECK(ok);
            }
        }
    }
    template<class T>
    void balanced_power_ranges()
    {
        constexpr std::size_t w = static_cast<std::size_t>(wray::simd::native<T>::size);
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T bases[] = {T(1.75), T(-2), -T(0), T(0.5), inf, -inf, nan,
                           std::numeric_limits<T>::denorm_min(), std::nextafter(T(1), T(2))};
        const T exponents[] = {T(0.3), T(3), T(-3), T(0.5), T(-1), T(0), T(2), T(1), T(1e6)};
        const auto equal = [](T a, T b) { return same(a, b) || (std::isnan(a) && std::isnan(b)); };
        for (std::size_t n = 0; n <= 8 * w + 1; ++n)
            for (std::size_t offset = 0; offset < w; ++offset)
            {
                std::vector<T> a(offset + n + 3), b(offset + n + 3);
                for (std::size_t i = 0; i < a.size(); ++i)
                {
                    a[i] = bases[i % std::size(bases)];
                    b[i] = exponents[i % std::size(exponents)];
                }
                for (int shorter : {0, 1, 2, 3})
                {
                    const std::size_t na = shorter == 1 && n ? n - 1 : n;
                    const std::size_t nb = shorter == 2 && n ? n - 1 : n;
                    const std::size_t count = shorter == 3 && n ? n - 1 : n + 2;
                    const std::size_t used = std::min({na, nb, count});
                    std::vector<T> out(offset + count + 2, T(7));
                    wray::balanced::pow(std::span<const T>(a.data() + offset, na), std::span<const T>(b.data() + offset, nb),
                                        std::span<T>(out.data() + offset, count));
                    bool ok = true;
                    for (std::size_t i = 0; i < out.size(); ++i)
                        ok = ok && (i >= offset && i < offset + used ?
                                    equal(out[i], wray::balanced::pow(a[i], b[i])) : out[i] == T(7));
                    CHECK(ok);
                }
                for (bool base_in_place : {false, true})
                {
                    auto io = base_in_place ? a : b;
                    const std::span<T> place(io.data() + offset, n);
                    const std::span<const T> x(a.data() + offset, n), y(b.data() + offset, n);
                    if (base_in_place)
                        wray::balanced::pow(place, y, place);
                    else
                        wray::balanced::pow(x, place, place);
                    bool ok = true;
                    for (std::size_t i = 0; i < io.size(); ++i)
                        ok = ok && equal(io[i], i >= offset && i < offset + n ?
                                         wray::balanced::pow(a[i], b[i]) : (base_in_place ? a[i] : b[i]));
                    CHECK(ok);
                }
                auto shared = a;
                const std::span<T> place(shared.data() + offset, n);
                wray::balanced::pow(place, place, place);
                bool ok = true;
                for (std::size_t i = 0; i < shared.size(); ++i)
                    ok = ok && equal(shared[i], i >= offset && i < offset + n ? wray::balanced::pow(a[i], a[i]) : a[i]);
                CHECK(ok);
            }
    }



    template<class T>
    void paired()
    {
        paired_ranges<T>([](auto in, auto s, auto c) { wray::sincos(in, s, c); },
                         [](T x) { return wray::sincos(x); });
        paired_ranges<T>([](auto in, auto s, auto c) { wray::sincos_turns(in, s, c); },
                         [](T x) { return wray::sincos_turns(x); });
        paired_ranges<T>([](auto in, auto s, auto c) { wray::balanced::sincos(in, s, c); },
                         [](T x) { return wray::balanced::sincos(x); });
        balanced_unary_ranges<T>([](auto in, auto out) { wray::balanced::sin(in, out); }, [](T x) { return wray::balanced::sin(x); });
        balanced_unary_ranges<T>([](auto in, auto out) { wray::balanced::cos(in, out); }, [](T x) { return wray::balanced::cos(x); });
        balanced_unary_ranges<T>([](auto in, auto out) { wray::balanced::tanh(in, out); }, [](T x) { return wray::balanced::tanh(x); });
        balanced_power_ranges<T>();
        if constexpr (std::is_same_v<T, float>)
        {
            paired_ranges<T>([](auto in, auto s, auto c) { wray::fast::sincos(in, s, c); },
                             [](T x) { return wray::fast::sincos(x); });
            paired_ranges<T>([](auto in, auto s, auto c) { wray::fast::sincos_turns(in, s, c); },
                             [](T x) { return wray::fast::sincos_turns(x); });
        }
    }
}

int main()
{
    std::mt19937_64 g(17);
    ranges<float>(g);
    ranges<double>(g);
    paired<float>();
    paired<double>();
    return wray::test::finish("range");
}
