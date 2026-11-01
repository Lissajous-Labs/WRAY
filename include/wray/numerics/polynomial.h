#pragma once

#include <array>
#include <cstddef>

#include "../simd/vec.h"

namespace wray::numerics
{
    template<class V, class T, std::size_t M>
    WRAY_INLINE V horner(const std::array<T, M> &c, const V &x) noexcept
    {
        V acc(c[M - 1]);
        if constexpr (M > 1)
        {
            for (std::size_t i = M - 1; i-- > 0;)
                acc = simd::fma(acc, x, V(c[i]));
        }
        return acc;
    }

    namespace detail
    {
        template<std::size_t N, class V, std::size_t L>
        WRAY_INLINE V pair_reduce(std::array<V, L> &level, const V &power) noexcept
        {
            if constexpr (N == 1)
                return level[0];
            else
            {
                for (std::size_t i = 0; i < N / 2; ++i)
                    level[i] = simd::fma(level[2 * i + 1], power, level[2 * i]);
                if constexpr (N % 2 == 1)
                    level[N / 2] = level[N - 1];
                if constexpr (N == 2)
                    return level[0];
                else
                    return pair_reduce<(N + 1) / 2>(level, power * power);
            }
        }
    }

    template<std::size_t O, class V, class T, std::size_t M>
    WRAY_INLINE V estrin_from(const std::array<T, M> &c, const V &x) noexcept
    {
        constexpr std::size_t n = M - O;
        if constexpr (n == 1)
            return V(c[O]);
        else
        {
            std::array<V, (n + 1) / 2> level;
            for (std::size_t i = 0; i < n / 2; ++i)
                level[i] = simd::fma(V(c[O + 2 * i + 1]), x, V(c[O + 2 * i]));
            if constexpr (n % 2 == 1)
                level[n / 2] = V(c[M - 1]);
            return detail::pair_reduce<(n + 1) / 2>(level, x * x);
        }
    }

    template<class V, class T, std::size_t M>
    WRAY_INLINE V estrin(const std::array<T, M> &c, const V &x) noexcept
    {
        return estrin_from<0>(c, x);
    }

    template<class V, class T, std::size_t M>
    WRAY_INLINE V polynomial(const std::array<T, M> &c, const V &x) noexcept
    {
        if constexpr (M < 5)
            return horner(c, x);
        else
            return simd::fma(estrin_from<1>(c, x), x, V(c[0]));
    }

    template<class C>
    inline constexpr bool is_polynomial = C::q.size() == 1;

    template<class C, class V>
    WRAY_INLINE V rational(const V &x) noexcept
    {
        if constexpr (is_polynomial<C>)
            return polynomial(C::p, x);
        else
            return polynomial(C::p, x) / polynomial(C::q, x);
    }

    template<class V>
    struct fraction
    {
        V num;
        V den;
    };

    template<class C, class V>
    WRAY_INLINE fraction<V> rational_parts(const V &x) noexcept
    {
        if constexpr (is_polynomial<C>)
            return {polynomial(C::p, x), V(typename C::value_type(1))};
        else
            return {polynomial(C::p, x), polynomial(C::q, x)};
    }
}
