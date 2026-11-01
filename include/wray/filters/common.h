#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>
#include <complex>
#include <numbers>
#include <span>
#include <type_traits>

#include "../math.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI::filters::detail
{
    inline constexpr double default_sample_rate = 48000.0;
    inline constexpr double lowest_frequency = 1.0e-6;
    inline constexpr double highest_frequency = 0.4999;
    inline constexpr double lowest_q = 1.0e-3;
    inline constexpr double highest_q = 1.0e3;

    WRAY_INLINE double sample_rate_or_default(double fs) noexcept
    {
        return std::isfinite(fs) && fs > 0.0 ? fs : default_sample_rate;
    }

    WRAY_INLINE double clamp_gain_db(double gain_db) noexcept
    {
        return std::isnan(gain_db) ? 0.0 : (gain_db < -120.0 ? -120.0 : (gain_db > 120.0 ? 120.0 : gain_db));
    }

    WRAY_INLINE double clamp_frequency(double hz, double fs) noexcept
    {
        const double lo = fs * lowest_frequency;
        const double hi = fs * highest_frequency;
        return hz > lo ? (hz < hi ? hz : hi) : lo;
    }

    WRAY_INLINE double clamp_q(double q) noexcept { return q > lowest_q ? (q < highest_q ? q : highest_q) : lowest_q; }

    WRAY_INLINE double prewarp(double hz, double fs) noexcept { return wray::tanpi(hz / fs); }

    template<class V>
    WRAY_INLINE double realise(double c) noexcept
    {
        return static_cast<double>(static_cast<simd::scalar_t<V> >(c));
    }

    template<class S>
    WRAY_INLINE S below(S x) noexcept
    {
        using U = std::conditional_t<sizeof(S) == 4, std::uint32_t, std::uint64_t>;
        return std::bit_cast<S>(std::bit_cast<U>(x) - U(1));
    }

    template<class V>
    WRAY_INLINE double open_unit(double c) noexcept
    {
        using S = simd::scalar_t<V>;
        const S lo = std::numeric_limits<S>::min();
        const S hi = below(S(1));
        const S value = static_cast<S>(c);
        return static_cast<double>(value > lo ? (value < hi ? value : hi) : lo);
    }

    template<class V>
    WRAY_INLINE auto quiet(const V &a) noexcept
    {
        using S = simd::scalar_t<V>;
        constexpr S threshold = sizeof(S) == 4 ? S(0x1p-80) : S(0x1p-200);
        return simd::abs(a) < V(threshold);
    }

    template<class V>
    WRAY_INLINE auto quiet(const V &a, const V &b) noexcept
    {
        return simd::both(quiet(a), quiet(b));
    }

    template<class V>
    WRAY_INLINE auto quiet(const V &a, const V &b, const V &c) noexcept
    {
        return simd::both(quiet(a, b), quiet(c));
    }

    template<class V, class M>
    WRAY_INLINE V awake(const M &quiet) noexcept
    {
        using S = simd::scalar_t<V>;
        return simd::select(quiet, V(S(0)), V(S(1)));
    }

    template<class V>
    WRAY_INLINE V settle(const V &hold, const V &x) noexcept
    {
        using S = simd::scalar_t<V>;
        return simd::select(hold == V(S(0)), V(S(0)), x);
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
        return wray::sincos_turns(0.5 * (hz / fs));
    }
}

WRAY_FP_END
