#include <wray/adaa.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <vector>

#include "check.h"

namespace
{
    template <class Shape>
    void shape_of_f(const char* name)
    {
        std::uint64_t violations = 0;
        float previous = Shape::f(0.0f);
        for (std::uint32_t bits = 0; bits < 0x7f800000u; bits += 61)
        {
            const float x = std::bit_cast<float>(bits);
            const float y = Shape::f(x);
            const float z = Shape::f(-x);
            violations += std::bit_cast<std::uint32_t>(z) != std::bit_cast<std::uint32_t>(-y);
            violations += y < previous;
            previous = y;
        }
        if (violations)
            std::printf("%s: %llu shape violations\n", name, static_cast<unsigned long long>(violations));
        CHECK(violations == 0);
    }

    template <class Shape, class T>
    void edges()
    {
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T huge = std::numeric_limits<T>::max();
        for (const T x : {T(-3), T(-0.5), T(0), T(0.25), T(1), T(7.5), huge})
            CHECK(Shape::first(x, x) == Shape::f(x));
        CHECK(Shape::first(T(0), inf) == Shape::f(inf));
        CHECK(Shape::first(-inf, T(2)) == Shape::f(-inf));
        CHECK(std::isnan(Shape::first(nan, T(1))));
        CHECK(std::isnan(Shape::first(T(1), nan)));
        CHECK(std::isfinite(Shape::first(-huge, huge)));
        CHECK(std::isfinite(Shape::first(huge, huge / T(2))));
        CHECK(std::isfinite(Shape::F1(huge / T(4))));
        wray::adaa::first_order<Shape, T> stage;
        CHECK(stage(T(0.5)) == Shape::first(T(0), T(0.5)));
        CHECK(stage(T(0.75)) == Shape::first(T(0.5), T(0.75)));
    }

    template <class T>
    double alias_power(const std::vector<T>& y, int bin)
    {
        const int n = int(y.size());
        double total = 0.0;
        for (int k = 2; k < 64; ++k)
        {
            const long long raw = static_cast<long long>(k) * bin;
            if (raw < n / 2)
                continue;
            long long folded = raw % n;
            if (folded > n / 2)
                folded = n - folded;
            double re = 0.0;
            double im = 0.0;
            for (int i = 0; i < n; ++i)
            {
                const double phase = 2.0 * std::numbers::pi * double(folded) * double(i) / double(n);
                re += double(y[std::size_t(i)]) * std::cos(phase);
                im -= double(y[std::size_t(i)]) * std::sin(phase);
            }
            total += re * re + im * im;
        }
        return total;
    }

    template <class Shape, class T>
    double aliasing_reduction()
    {
        const int n = 1 << 13;
        const int bin = 1501;
        const auto size = static_cast<std::size_t>(n);
        std::vector<T> x(size), naive(size), smooth(size);
        for (int i = 0; i < n; ++i)
            x[std::size_t(i)] = T(4.0 * std::sin(2.0 * std::numbers::pi * double(bin) * double(i) / double(n)));
        for (int i = 0; i < n; ++i)
        {
            naive[std::size_t(i)] = Shape::f(x[std::size_t(i)]);
            smooth[std::size_t(i)] = Shape::first(x[std::size_t((i + n - 1) % n)], x[std::size_t(i)]);
        }
        return 10.0 * std::log10(alias_power<T>(naive, bin) / alias_power<T>(smooth, bin));
    }

    std::uint64_t next(std::uint64_t& state)
    {
        state += 0x9e3779b97f4a7c15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    template <class T>
    T sample(std::uint64_t& state)
    {
        const std::uint64_t r = next(state);
        const double unit = double(r >> 11) * 0x1p-53;
        const int scale = int(next(state) % 8);
        const double magnitude[] = {0.25, 1.0, 1.5, 4.0, 1e3, 1e9, 1e30, 1e300};
        const double v = (2.0 * unit - 1.0) * magnitude[scale];
        const double top = double(std::numeric_limits<T>::max());
        return T(v > top ? top : v < -top ? -top : v);
    }

    template <class Shape, class T>
    void intervals()
    {
        const T huge = std::numeric_limits<T>::max();
        CHECK(Shape::first(-huge, huge) == T(0));
        CHECK(Shape::first(huge, -huge) == T(0));
        std::uint64_t state = 0x5eed;
        bool order = true;
        bool bounded = true;
        for (int i = 0; i < 200000; ++i)
        {
            T a = sample<T>(state);
            const T b = i % 4 == 0 ? -a * T(1 + 1e-4 * double(next(state) % 7)) : sample<T>(state);
            if (!std::isfinite(b))
                a = T(0);
            const T v = Shape::first(a, b);
            order = order && v == Shape::first(b, a);
            const T fa = Shape::f(a);
            const T fb = Shape::f(b);
            const T slack = T(4) * std::numeric_limits<T>::epsilon() * std::max(std::fabs(fa), std::fabs(fb));
            bounded = bounded && v >= std::min(fa, fb) - slack && v <= std::max(fa, fb) + slack;
        }
        CHECK(order);
        CHECK(bounded);
    }

    template <class T>
    void atan_limits()
    {
        const T huge = std::numeric_limits<T>::max();
        const long double half_pi = 1.570796326794896619231321691639751442L;
        const long double ulp = std::ldexp(1.0L, 1 - std::numeric_limits<T>::digits);
        CHECK(std::fabs(static_cast<long double>(wray::adaa::atan::first(T(0), huge)) - half_pi) <= 2 * ulp);
        CHECK(std::fabs(static_cast<long double>(wray::adaa::atan::first(-huge, T(0))) + half_pi) <= 2 * ulp);
    }

    void regression_pairs()
    {
        const auto within = [](float got, double exact, double ulps) {
            const double ulp = std::ldexp(1.0, std::ilogb(float(exact)) - 23);
            return std::fabs(double(got) - exact) <= ulps * ulp;
        };
        CHECK(within(wray::adaa::hard_clip::first(-0.44799888134002686f, 1.6291460990905762f), 0.4952926300004076, 1.0));
        CHECK(within(wray::adaa::soft_clip::first(1.4531803131103516f, -0.6492534875869751f), 0.24867804690919537, 3.0));
    }

    template <class Shape>
    void shape(const char* name, double floor_db)
    {
        shape_of_f<Shape>(name);
        edges<Shape, float>();
        edges<Shape, double>();
        intervals<Shape, float>();
        intervals<Shape, double>();
        const double r32 = aliasing_reduction<Shape, float>();
        const double r64 = aliasing_reduction<Shape, double>();
        std::printf("%-10s aliasing reduction f32 %.2f dB f64 %.2f dB\n", name, r32, r64);
        CHECK(r32 >= floor_db);
        CHECK(r64 >= floor_db);
    }
}

int main()
{
    shape<wray::adaa::hard_clip>("hard_clip", 3.95);
    shape<wray::adaa::soft_clip>("soft_clip", 4.1);
    shape<wray::adaa::tanh>("tanh", 3.95);
    shape<wray::adaa::atan>("atan", 4.1);
    shape<wray::adaa::algebraic>("algebraic", 4.15);
    shape<wray::adaa::radical>("radical", 4.0);
    atan_limits<float>();
    atan_limits<double>();
    regression_pairs();
    return wray::test::finish("adaa");
}
