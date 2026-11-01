#include <wray/wray.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "check.h"

namespace
{
    template <class T>
    std::vector<T> inputs(std::mt19937_64& g)
    {
        using U = typename wray::simd::bits_of<T>::unsigned_integer;
        std::vector<T> v;
        for (int i = 0; i < 40000; ++i)
            v.push_back(static_cast<T>(-30.0 + 60.0 * std::ldexp(static_cast<double>(g() >> 11), -53)));
        for (int i = 0; i < 40000; ++i)
            v.push_back(std::bit_cast<T>(static_cast<U>(g())));
        const T specials[] = {T(0), -T(0), T(1), T(-1), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                              std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::max(), T(1e-30)};
        for (int r = 0; r < 4; ++r)
            for (T s : specials)
                v.push_back(s);
        return v;
    }

    template <class T>
    void shuffle(std::vector<T>& v, std::mt19937_64& g)
    {
        for (std::size_t i = v.size(); i > 1; --i)
            std::swap(v[i - 1], v[static_cast<std::size_t>(g() % i)]);
    }

    template <class T>
    bool identical(T a, T b)
    {
        if (std::isnan(a) && std::isnan(b))
            return true;
        return std::bit_cast<typename wray::simd::bits_of<T>::integer>(a) == std::bit_cast<typename wray::simd::bits_of<T>::integer>(b);
    }

    template <class T, class F>
    void unary(const std::vector<T>& x, F f)
    {
        using V = wray::simd::native<T>;
        for (std::size_t i = 0; i + V::size <= x.size(); i += V::size)
        {
            const V r = f(V::load(x.data() + i));
            for (int k = 0; k < V::size; ++k)
                CHECK(identical(r[k], f(x[i + std::size_t(k)])));
        }
    }

    template <class T, class F>
    void binary(const std::vector<T>& x, const std::vector<T>& y, F f)
    {
        using V = wray::simd::native<T>;
        for (std::size_t i = 0; i + V::size <= x.size(); i += V::size)
        {
            const V r = f(V::load(x.data() + i), V::load(y.data() + i));
            for (int k = 0; k < V::size; ++k)
                CHECK(identical(r[k], f(x[i + std::size_t(k)], y[i + std::size_t(k)])));
        }
    }

    template <class T>
    void precise(std::mt19937_64& g)
    {
        const auto x = inputs<T>(g);
        auto y = inputs<T>(g);
        shuffle(y, g);
        unary(x, [](auto v) { return wray::exp(v); });
        unary(x, [](auto v) { return wray::log1p(v); });
        unary(x, [](auto v) { return wray::log10(v); });
        unary(x, [](auto v) { return wray::sin_turns(v); });
        unary(x, [](auto v) { return wray::cos_turns(v); });
        unary(x, [](auto v) { return wray::tan_turns(v); });
        unary(x, [](auto v) { return wray::expm1(v); });
        unary(x, [](auto v) { return wray::exp2(v); });
        unary(x, [](auto v) { return wray::log(v); });
        unary(x, [](auto v) { return wray::log2(v); });
        unary(x, [](auto v) { return wray::sin(v); });
        unary(x, [](auto v) { return wray::cos(v); });
        unary(x, [](auto v) { return wray::tan(v); });
        unary(x, [](auto v) { return wray::atan(v); });
        unary(x, [](auto v) { return wray::tanh(v); });
        unary(x, [](auto v) { return wray::sigmoid(v); });
        unary(x, [](auto v) { return wray::db_to_gain(v); });
        unary(x, [](auto v) { return wray::gain_to_db(v); });
        binary(x, y, [](auto a, auto b) { return wray::atan2(a, b); });
        binary(x, y, [](auto a, auto b) { return wray::pow(a, b); });
    }

    void fast(std::mt19937_64& g)
    {
        const auto x = inputs<float>(g);
        auto y = inputs<float>(g);
        shuffle(y, g);
        unary(x, [](auto v) { return wray::fast::sin_turns(v); });
        unary(x, [](auto v) { return wray::fast::cos_turns(v); });
        unary(x, [](auto v) { return wray::fast::tan_turns(v); });
        unary(x, [](auto v) { return wray::fast::exp(v); });
        unary(x, [](auto v) { return wray::fast::exp2(v); });
        unary(x, [](auto v) { return wray::fast::log(v); });
        unary(x, [](auto v) { return wray::fast::log2(v); });
        unary(x, [](auto v) { return wray::fast::sin(v); });
        unary(x, [](auto v) { return wray::fast::cos(v); });
        unary(x, [](auto v) { return wray::fast::tan(v); });
        unary(x, [](auto v) { return wray::fast::atan(v); });
        unary(x, [](auto v) { return wray::fast::tanh(v); });
        unary(x, [](auto v) { return wray::fast::sigmoid(v); });
        unary(x, [](auto v) { return wray::fast::db_to_gain(v); });
        unary(x, [](auto v) { return wray::fast::gain_to_db(v); });
        binary(x, y, [](auto a, auto b) { return wray::fast::atan2(a, b); });
        binary(x, y, [](auto a, auto b) { return wray::fast::pow(a, b); });
    }
}

int main()
{
    std::mt19937_64 g(11);
    precise<float>(g);
    precise<double>(g);
    fast(g);
    return wray::test::finish("parity");
}
