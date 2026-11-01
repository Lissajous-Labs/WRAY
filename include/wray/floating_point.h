#pragma once

#include <limits>

#include "math/kernels.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI
{
    namespace detail
    {
        template<class T>
        bool floating_point_ok() noexcept
        {
            constexpr int digits = std::numeric_limits<T>::digits;
            volatile T one = T(1);
            volatile T tiny = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();
            volatile T big = T(1) / tiny;
            volatile T three = T(3);
            volatile T step = T(1) / T(1u << (digits / 2 + 2));
            volatile T zero = T(0);
            volatile T quiet = std::numeric_limits<T>::quiet_NaN();
            const T a = one + step;
            const T b = one - step;
            const T s = one + tiny;
            const T e = tiny - (s - one);
            bool ok = s == T(1) && e == tiny;
            ok = ok && (big + one) - big == T(0);
            ok = ok && (sizeof(T) == 4 ? three / T(7) == T(3.0 / 7.0) : three / T(10) == T(0.3));
            ok = ok && a * b - one == T(0);
            const T n = quiet;
            ok = ok && n != n;
            ok = ok && T(1) / -T(zero) < T(0);
            const auto sum = math::detail::two_sum(T(one), T(tiny));
            ok = ok && sum.hi == T(1) && sum.lo == tiny;
            const auto product = math::detail::two_product(a, b);
            ok = ok && product.hi == T(1) && product.lo == -(step * step);
            return ok;
        }
    }

    inline bool floating_point_ok() noexcept { return detail::floating_point_ok<float>() && detail::floating_point_ok<double>(); }
}

WRAY_FP_END
