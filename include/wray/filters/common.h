#pragma once

#include <complex>
#include <numbers>
#include <span>
#include <type_traits>

#include "../math.h"

namespace wray::filters::detail
{
    inline constexpr double default_sample_rate = 48000.0;
    inline constexpr double lowest_frequency = 1.0e-6;
    inline constexpr double highest_frequency = 0.4999;
    inline constexpr double lowest_q = 1.0e-3;
    inline constexpr double highest_q = 1.0e3;

    WRAY_INLINE double sample_rate_or_default(double fs) noexcept { return fs > 0.0 ? fs : default_sample_rate; }

    WRAY_INLINE double clamp_frequency(double hz, double fs) noexcept
    {
        const double lo = fs * lowest_frequency;
        const double hi = fs * highest_frequency;
        return hz > lo ? (hz < hi ? hz : hi) : lo;
    }

    WRAY_INLINE double clamp_q(double q) noexcept { return q > lowest_q ? (q < highest_q ? q : highest_q) : lowest_q; }

    WRAY_INLINE double prewarp(double hz, double fs) noexcept { return wray::tan(std::numbers::pi * (hz / fs)); }

    template<class V>
    WRAY_INLINE double realise(double c) noexcept
    {
        return static_cast<double>(static_cast<simd::scalar_t<V> >(c));
    }

    template<class V>
    WRAY_INLINE V splat(double c) noexcept
    {
        return V(static_cast<simd::scalar_t<V> >(c));
    }

    template<class To, class From>
    WRAY_INLINE To lane_cast(const From &x) noexcept
    {
        if constexpr (std::is_same_v<To, From>)
            return x;
        else if constexpr (std::is_arithmetic_v<From>)
            return static_cast<To>(x);
        else
        {
            static_assert(simd::traits<To>::width == simd::traits<From>::width);
            constexpr int n = simd::traits<From>::width;
            simd::scalar_t<From> a[n];
            simd::scalar_t<To> b[n];
            x.store(a);
            for (int i = 0; i < n; ++i)
                b[i] = static_cast<simd::scalar_t<To> >(a[i]);
            return To::load(b);
        }
    }

    WRAY_INLINE math::sincos_result<double> half_angle(double hz, double fs) noexcept
    {
        return wray::sincos(std::numbers::pi * (hz / fs));
    }
}
