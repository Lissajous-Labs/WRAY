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

    template<class V, class T, std::size_t M>
    WRAY_INLINE V estrin(const std::array<T, M> &c, const V &x) noexcept
    {
        if constexpr (M == 1)
            return V(c[0]);
        else
        {
            std::array<V, (M + 1) / 2> level;
            for (std::size_t i = 0; i < M / 2; ++i)
                level[i] = simd::fma(V(c[2 * i + 1]), x, V(c[2 * i]));
            if constexpr (M % 2 == 1)
                level[M / 2] = V(c[M - 1]);
            V power = x * x;
            std::size_t count = (M + 1) / 2;
            while (count > 1)
            {
                std::size_t next = (count + 1) / 2;
                for (std::size_t i = 0; i < count / 2; ++i)
                    level[i] = simd::fma(level[2 * i + 1], power, level[2 * i]);
                if (count % 2 == 1)
                    level[count / 2] = level[count - 1];
                power = power * power;
                count = next;
            }
            return level[0];
        }
    }

    template<class C>
    inline constexpr bool is_polynomial = C::q.size() == 1;

    template<class C, class V>
    WRAY_INLINE V rational(const V &x) noexcept
    {
        if constexpr (is_polynomial<C>)
            return horner(C::p, x);
        else
            return horner(C::p, x) / horner(C::q, x);
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
            return {horner(C::p, x), V(typename C::value_type(1))};
        else
            return {horner(C::p, x), horner(C::q, x)};
    }
}
