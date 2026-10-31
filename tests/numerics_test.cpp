#include <wray/numerics/polynomial.h>
#include <wray/math/coefficients.h>

#include <array>
#include <cmath>
#include <random>

#include "check.h"

namespace
{
    struct cubic_over_quadratic
    {
        using value_type = double;
        static constexpr std::array<double, 4> p{1.0, -2.0, 0.5, 0.25};
        static constexpr std::array<double, 3> q{1.0, 0.5, 0.125};
    };

    long double exact_poly(const double* c, int n, long double x)
    {
        long double acc = 0.0L;
        for (int i = n - 1; i >= 0; --i)
            acc = acc * x + c[i];
        return acc;
    }

    template <std::size_t M>
    void compare(const std::array<double, M>& c, std::mt19937_64& g)
    {
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (int i = 0; i < 1000; ++i)
        {
            const double x = d(g);
            const long double ref = exact_poly(c.data(), int(M), x);
            double scale = 0.0;
            for (std::size_t k = 0; k < M; ++k)
                scale += std::fabs(c[k]);
            const double h = wray::numerics::horner(c, x);
            const double e = wray::numerics::estrin(c, x);
            CHECK(std::fabs(double(h - ref)) <= 4.0 * double(M) * 0x1p-53 * scale);
            CHECK(std::fabs(double(e - ref)) <= 4.0 * double(M) * 0x1p-53 * scale);
            const wray::f64x2 v = wray::numerics::estrin(c, wray::f64x2(x));
            CHECK(v[0] == e);
            CHECK(v[1] == e);
        }
    }

    template <class C>
    void cores_start_at_one()
    {
        CHECK(C::q[0] == typename C::value_type(1));
    }
}

int main()
{
    std::mt19937_64 g(5);
    compare(std::array<double, 1>{3.0}, g);
    compare(std::array<double, 2>{3.0, -1.0}, g);
    compare(std::array<double, 5>{1.0, 0.5, -0.25, 0.125, 2.0}, g);
    compare(std::array<double, 8>{1.0, -1.0, 0.5, 0.25, -0.125, 0.0625, 3.0, -2.0}, g);
    compare(std::array<double, 11>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, g);

    for (double x : {-0.75, -0.1, 0.0, 0.3, 0.9})
    {
        const double num = 1.0 - 2.0 * x + 0.5 * x * x + 0.25 * x * x * x;
        const double den = 1.0 + 0.5 * x + 0.125 * x * x;
        CHECK(std::fabs(wray::numerics::rational<cubic_over_quadratic>(x) - num / den) <= 1e-15);
        const auto parts = wray::numerics::rational_parts<cubic_over_quadratic>(x);
        CHECK(std::fabs(parts.num - num) <= 1e-15);
        CHECK(std::fabs(parts.den - den) <= 1e-15);
    }

    using namespace wray::math::coeff;
    cores_start_at_one<exp_f32>();
    cores_start_at_one<exp_f64>();
    cores_start_at_one<tan_f64>();
    cores_start_at_one<atan_f64>();
    cores_start_at_one<tanh_f64>();
    cores_start_at_one<tanh_range_fast>();
    CHECK(wray::numerics::is_polynomial<exp_f64>);
    CHECK(!wray::numerics::is_polynomial<atan_f64>);
    CHECK(std::fabs(wray::numerics::rational<exp_f64>(0.0) - 1.0) <= 1e-16);
    CHECK(std::fabs(wray::numerics::rational<atan_f64>(0.0) + 1.0 / 3.0) <= 1e-16);
    CHECK(std::fabs(wray::numerics::rational<tan_f64>(0.0) - 1.0 / 3.0) <= 1e-16);
    return wray::test::finish("numerics");
}
