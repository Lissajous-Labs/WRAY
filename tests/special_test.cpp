#include <wray/wray.h>

#include <cmath>
#include <limits>
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
    void unary_specials()
    {
        using V = wray::simd::native<T>;
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T huge = std::numeric_limits<T>::max();
        struct row
        {
            T (*fn)(T);
            V (*vfn)(V);
            T x;
            T want;
        };
        const row rows[] = {
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, T(0), T(1)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -T(0), T(1)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, inf, inf},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -inf, T(0)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, nan, nan},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, huge, inf},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -huge, T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(0), T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -T(0), -T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, tiny, tiny},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -tiny, -tiny},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(-1), -inf},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(-2), nan},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, inf, inf},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -inf, nan},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, nan, nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(1), T(0)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(10), T(1)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(100), T(2)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(0), -inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, -T(0), -inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(-1), nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, inf, inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, -inf, nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, nan, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, -T(0), -T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.25), T(1)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.25), T(-1)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.5), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.5), -T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(1), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.125), T(0.70710678118654752440)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.375), -T(0.70710678118654752440)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, inf, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, nan, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, huge, T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, -huge, -T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, -T(0), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.25), T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(-0.25), T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.5), T(-1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(1), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.125), T(0.70710678118654752440)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.375), -T(0.70710678118654752440)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, inf, nan},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, nan, nan},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, huge, T(1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0), T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.25), T(1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-0.25), T(-1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.5), inf},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-0.5), -inf},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(1), T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-1), -T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.75), T(-1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, inf, nan},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, nan, nan},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, huge, T(0)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, T(0), T(0)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -T(0), -T(0)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, tiny, tiny},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -tiny, -tiny},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, inf, inf},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -inf, T(-1)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, huge, inf},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -huge, T(-1)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, nan, nan},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, T(10), T(1024)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, T(-1), T(0.5)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, inf, inf},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, -inf, T(0)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, nan, nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(1), T(0)},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(0), -inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, -T(0), -inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(-1), nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, inf, inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, -inf, nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, nan, nan},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(8), T(3)},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(0.25), T(-2)},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, tiny, T(std::ilogb(tiny))},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(0), -inf},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, inf, inf},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, T(0), T(0)},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, -T(0), -T(0)},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, inf, nan},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, nan, nan},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, huge, nan},
            {[](T x) { return wray::cos(x); }, [](V x) { return wray::cos(x); }, T(0), T(1)},
            {[](T x) { return wray::cos(x); }, [](V x) { return wray::cos(x); }, -inf, nan},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, T(0), T(0)},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, nan, nan},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, T(0), T(0)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, -T(0), -T(0)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, inf, T(1.5707963267948966)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, -inf, T(-1.5707963267948966)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, nan, nan},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, T(0), T(0)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, inf, T(1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, -inf, T(-1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, T(50), T(1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, nan, nan},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, T(0), T(0.5)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, inf, T(1)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, -inf, T(0)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, nan, nan},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, T(0), T(1)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, T(20), T(10)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, -inf, T(0)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, inf, inf},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(1), T(0)},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(0), -inf},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(-1), nan},
        };
        for (const auto& r : rows)
        {
            CHECK(same(r.fn(r.x), r.want));
            const V v = r.vfn(V(r.x));
            for (int i = 0; i < V::size; ++i)
                CHECK(same(v[i], r.want));
        }
    }

    template <class T>
    void binary_specials()
    {
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T pi = T(3.141592653589793);
        const T half_pi = T(1.5707963267948966);
        CHECK(same(wray::atan2(T(0), T(1)), T(0)));
        CHECK(same(wray::atan2(-T(0), T(1)), -T(0)));
        CHECK(same(wray::atan2(T(0), -T(0)), pi));
        CHECK(same(wray::atan2(-T(0), -T(0)), -pi));
        CHECK(same(wray::atan2(T(0), T(0)), T(0)));
        CHECK(same(wray::atan2(T(1), T(0)), half_pi));
        CHECK(same(wray::atan2(T(-1), T(0)), -half_pi));
        CHECK(same(wray::atan2(inf, inf), wray::atan(T(1))));
        CHECK(same(wray::atan2(T(1), -inf), pi));
        CHECK(same(wray::atan2(nan, T(1)), nan));
        CHECK(same(wray::atan2(T(1), nan), nan));
        CHECK(same(wray::pow(T(2), T(10)), T(1024)));
        CHECK(same(wray::pow(T(9), T(0.5)), T(3)));
        CHECK(same(wray::pow(T(0), T(2)), T(0)));
        CHECK(same(wray::pow(T(0), T(-2)), inf));
        CHECK(same(wray::pow(nan, T(0)), T(1)));
        CHECK(same(wray::pow(T(1), nan), T(1)));
        CHECK(same(wray::pow(T(-8), T(1.0 / 3.0)), nan));
        CHECK(same(wray::pow(inf, T(2)), inf));
        CHECK(same(wray::pow(inf, T(-2)), T(0)));
        CHECK(same(wray::pow(T(0.5), inf), T(0)));
        CHECK(same(wray::pow(T(2), inf), inf));
        CHECK(same(wray::pow(T(10), T(400)), inf));
        using V = wray::simd::native<T>;
        const T huge = std::numeric_limits<T>::max();
        for (const T y : {huge, -huge, T(1e30), T(-1e30)})
        {
            CHECK(same(wray::pow(T(1), y), T(1)));
            const V v = wray::pow(V(T(1)), V(y));
            for (int i = 0; i < V::size; ++i)
                CHECK(same(v[i], T(1)));
        }
        CHECK(same(wray::pow(T(2), huge), inf));
        CHECK(same(wray::pow(T(0.5), huge), T(0)));
    }

    void ranges()
    {
        std::vector<float> in(37);
        std::vector<float> out(37);
        std::vector<double> ind(19);
        std::vector<double> outd(19);
        for (std::size_t i = 0; i < in.size(); ++i)
            in[i] = float(i) * 0.25f - 4.0f;
        for (std::size_t i = 0; i < ind.size(); ++i)
            ind[i] = double(i) * 0.5 - 3.0;
        wray::exp(in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::exp(in[i]));
        wray::fast::tanh(in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::fast::tanh(in[i]));
        wray::sin(ind, outd);
        for (std::size_t i = 0; i < ind.size(); ++i)
            CHECK(outd[i] == wray::sin(ind[i]));
        wray::atan2(in, in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::atan2(in[i], in[i]));
        const auto sc = wray::sincos(0.5f);
        CHECK(sc.sin == wray::sin(0.5f));
        CHECK(sc.cos == wray::cos(0.5f));
    }
}

int main()
{
    unary_specials<float>();
    unary_specials<double>();
    binary_specials<float>();
    binary_specials<double>();
    ranges();
    return wray::test::finish("special");
}
