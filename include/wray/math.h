#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>

#include "math/kernels.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI::math
{
    template<class R>
    concept real_range = std::ranges::contiguous_range<R> && std::is_floating_point_v<std::ranges::range_value_t<R>>;

    template<class T, class F>
    void transform(std::span<const T> in, std::span<T> out, F f) noexcept
    {
        using V = simd::native<T>;
        constexpr std::size_t w = V::size;
        const std::size_t n = std::min(in.size(), out.size());
        const T *src = in.data();
        T *dst = out.data();
        std::size_t i = 0;
        for (; i + 4 * w <= n; i += 4 * w)
        {
            const V a = f(V::load(src + i));
            const V b = f(V::load(src + i + w));
            const V c = f(V::load(src + i + 2 * w));
            const V d = f(V::load(src + i + 3 * w));
            a.store(dst + i);
            b.store(dst + i + w);
            c.store(dst + i + 2 * w);
            d.store(dst + i + 3 * w);
        }
        for (; i + w <= n; i += w)
            f(V::load(src + i)).store(dst + i);
        for (; i < n; ++i)
            dst[i] = f(src[i]);
    }

    template<class T, class F>
    void transform(std::span<const T> a, std::span<const T> b, std::span<T> out, F f) noexcept
    {
        using V = simd::native<T>;
        constexpr std::size_t w = V::size;
        const std::size_t n = std::min({a.size(), b.size(), out.size()});
        std::size_t i = 0;
        for (; i + w <= n; i += w)
            f(V::load(a.data() + i), V::load(b.data() + i)).store(out.data() + i);
        for (; i < n; ++i)
            out[i] = f(a[i], b[i]);
    }

    template<class In, class Out, class F>
    void apply(const In &in, Out &&out, F f) noexcept
    {
        using T = std::ranges::range_value_t<In>;
        transform<T>(std::span<const T>(std::ranges::data(in), std::ranges::size(in)),
                     std::span<T>(std::ranges::data(out), std::ranges::size(out)), f);
    }

    template<class A, class B, class Out, class F>
    void apply(const A &a, const B &b, Out &&out, F f) noexcept
    {
        using T = std::ranges::range_value_t<A>;
        transform<T>(std::span<const T>(std::ranges::data(a), std::ranges::size(a)),
                     std::span<const T>(std::ranges::data(b), std::ranges::size(b)),
                     std::span<T>(std::ranges::data(out), std::ranges::size(out)), f);
    }

    template<class In, class SinOut, class CosOut, class F>
    void apply_pair(const In &in, SinOut &&sin_out, CosOut &&cos_out, F f) noexcept
    {
        using T = std::ranges::range_value_t<In>;
        using V = simd::native<T>;
        constexpr std::size_t w = V::size;
        const std::size_t n = std::min({std::ranges::size(in), std::ranges::size(sin_out), std::ranges::size(cos_out)});
        const T *src = std::ranges::data(in);
        T *s = std::ranges::data(sin_out);
        T *c = std::ranges::data(cos_out);
        std::size_t i = 0;
        for (; i + 2 * w <= n; i += 2 * w)
        {
            const auto a = f(V::load(src + i));
            const auto b = f(V::load(src + i + w));
            a.sin.store(s + i);
            a.cos.store(c + i);
            b.sin.store(s + i + w);
            b.cos.store(c + i + w);
        }
        for (; i + w <= n; i += w)
        {
            const auto a = f(V::load(src + i));
            a.sin.store(s + i);
            a.cos.store(c + i);
        }
        for (; i < n; ++i)
        {
            const auto a = f(src[i]);
            s[i] = a.sin;
            c[i] = a.cos;
        }
    }

    template<class T>
    using carrier = std::conditional_t<(simd::register_bytes >= 16), simd::vec<T, 16 / static_cast<int>(sizeof(T))>, T>;

    template<class V>
    inline constexpr bool lifted = std::is_floating_point_v<V> && !std::is_same_v<carrier<V>, V>;

    template<class V, class F>
    WRAY_INLINE V lift(const V &x, F f) noexcept
    {
        if constexpr (lifted<V>)
            return simd::first(f(carrier<V>(x)));
        else
            return f(x);
    }

    template<class V, class F>
    WRAY_INLINE V lift(const V &a, const V &b, F f) noexcept
    {
        if constexpr (lifted<V>)
            return simd::first(f(carrier<V>(a), carrier<V>(b)));
        else
            return f(a, b);
    }

    template<class V, class F>
    WRAY_INLINE sincos_result<V> lift_pair(const V &x, F f) noexcept
    {
        if constexpr (lifted<V>)
        {
            const auto r = f(carrier<V>(x));
            return {simd::first(r.sin), simd::first(r.cos)};
        }
        else
            return f(x);
    }
}

namespace wray::inline WRAY_ABI
{
    template<simd::real V>
    WRAY_INLINE V exp(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V expm1(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::expm1<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V exp2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp2<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log1p(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log1p<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log10(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log10<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log2<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V wrap_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::wrap_turns(v); }); }

    template<simd::real V>
    WRAY_INLINE math::sincos_result<V> sincos_turns(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos_turns<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V sin_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin_turns<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V sin(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cos_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos_turns<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cos(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE math::sincos_result<V> sincos(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tanpi(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tanpi<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tan_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan_turns<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V atan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::atan<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V smoothing_coefficient(const V &seconds, const V &fs) noexcept { return math::lift(seconds, fs, [](const auto &a, const auto &b) { return math::detail::smoothing_coefficient<math::precise>(a, b); }); }

    template<simd::real V>
    WRAY_INLINE V pole_from_cutoff(const V &hz, const V &fs) noexcept { return math::lift(hz, fs, [](const auto &a, const auto &b) { return math::detail::pole_from_cutoff<math::precise>(a, b); }); }

    template<simd::real V>
    WRAY_INLINE V prewarp(const V &hz, const V &fs) noexcept { return math::lift(hz, fs, [](const auto &a, const auto &b) { return math::detail::prewarp<math::precise>(a, b); }); }

    template<simd::real V>
    WRAY_INLINE V atan2(const V &y, const V &x) noexcept { return math::lift(y, x, [](const auto &a, const auto &b) { return math::detail::atan2<math::precise>(a, b); }); }

    template<simd::real V>
    WRAY_INLINE V sinh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sinh<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cosh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cosh<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V asinh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::asinh<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cbrt(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cbrt<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V erf(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::erf<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V midi_to_hz(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::midi_to_hz<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V hz_to_midi(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::hz_to_midi<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V ratio_from_cents(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::ratio_from_cents<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V omega(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::omega<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V product_log(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::product_log<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tanh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tanh<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V sigmoid(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sigmoid<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V pow(const V &x, const V &y) noexcept { return math::lift(x, y, [](const auto &a, const auto &b) { return math::detail::pow<math::precise>(a, b); }); }

    template<simd::real V>
    WRAY_INLINE V db_to_gain(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::db_to_gain<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V gain_to_db(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::gain_to_db<math::precise>(v); }); }

    template<math::real_range In, class Out>
    void exp(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::exp(v); }); }

    template<math::real_range In, class Out>
    void expm1(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::expm1(v); }); }

    template<math::real_range In, class Out>
    void exp2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::exp2(v); }); }

    template<math::real_range In, class Out>
    void log(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log(v); }); }

    template<math::real_range In, class Out>
    void log1p(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log1p(v); }); }

    template<math::real_range In, class Out>
    void log10(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log10(v); }); }

    template<math::real_range In, class Out>
    void log2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log2(v); }); }

    template<math::real_range In, class Out>
    void wrap_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::wrap_turns(v); }); }

    template<math::real_range In, class Out>
    void sin_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::sin_turns(v); }); }

    template<math::real_range In, class Out>
    void sin(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::sin(v); }); }

    template<math::real_range In, class Out>
    void cos_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::cos_turns(v); }); }

    template<math::real_range In, class Out>
    void cos(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::cos(v); }); }

    template<math::real_range In, class SinOut, class CosOut>
    void sincos(const In &in, SinOut &&sin_out, CosOut &&cos_out) noexcept
    {
        math::apply_pair(in, sin_out, cos_out, [](auto v) { return wray::sincos(v); });
    }

    template<math::real_range In, class SinOut, class CosOut>
    void sincos_turns(const In &in, SinOut &&sin_out, CosOut &&cos_out) noexcept
    {
        math::apply_pair(in, sin_out, cos_out, [](auto v) { return wray::sincos_turns(v); });
    }

    template<math::real_range In, class Out>
    void tanpi(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::tanpi(v); }); }

    template<math::real_range In, class Out>
    void tan_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::tan_turns(v); }); }

    template<math::real_range In, class Out>
    void tan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::tan(v); }); }

    template<math::real_range In, class Out>
    void atan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::atan(v); }); }

    template<math::real_range In, class Out>
    void sinh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::sinh(v); }); }

    template<math::real_range In, class Out>
    void cosh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::cosh(v); }); }

    template<math::real_range In, class Out>
    void asinh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::asinh(v); }); }

    template<math::real_range In, class Out>
    void cbrt(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::cbrt(v); }); }

    template<math::real_range In, class Out>
    void erf(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::erf(v); }); }

    template<math::real_range In, class Out>
    void midi_to_hz(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::midi_to_hz(v); }); }

    template<math::real_range In, class Out>
    void hz_to_midi(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::hz_to_midi(v); }); }

    template<math::real_range In, class Out>
    void ratio_from_cents(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::ratio_from_cents(v); }); }

    template<math::real_range In, class Out>
    void omega(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::omega(v); }); }

    template<math::real_range In, class Out>
    void product_log(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::product_log(v); }); }

    template<math::real_range In, class Out>
    void tanh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::tanh(v); }); }

    template<math::real_range In, class Out>
    void sigmoid(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::sigmoid(v); }); }

    template<math::real_range In, class Out>
    void db_to_gain(const In &in, Out &&out) noexcept
    {
        math::apply(in, out, [](auto v) { return wray::db_to_gain(v); });
    }

    template<math::real_range In, class Out>
    void gain_to_db(const In &in, Out &&out) noexcept
    {
        math::apply(in, out, [](auto v) { return wray::gain_to_db(v); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void smoothing_coefficient(const A &a, const B &b, Out &&out) noexcept
    {
        math::apply(a, b, out, [](auto x, auto y) { return wray::smoothing_coefficient(x, y); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void pole_from_cutoff(const A &a, const B &b, Out &&out) noexcept
    {
        math::apply(a, b, out, [](auto x, auto y) { return wray::pole_from_cutoff(x, y); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void prewarp(const A &a, const B &b, Out &&out) noexcept
    {
        math::apply(a, b, out, [](auto x, auto y) { return wray::prewarp(x, y); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void atan2(const A &y, const B &x, Out &&out) noexcept
    {
        math::apply(y, x, out, [](auto a, auto b) { return wray::atan2(a, b); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void pow(const A &x, const B &y, Out &&out) noexcept
    {
        math::apply(x, y, out, [](auto a, auto b) { return wray::pow(a, b); });
    }
}

namespace wray::inline WRAY_ABI::fast
{
    template<class V>
    concept single = std::is_same_v<simd::scalar_t<V>, float>;

    template<single V>
    WRAY_INLINE V exp(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V exp2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp2<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V log(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V log2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log2<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE math::sincos_result<V> sincos_turns(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos_turns<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V sin_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin_turns<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V sin(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V cos_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos_turns<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V cos(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE math::sincos_result<V> sincos(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V tanpi(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tanpi<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V tan_turns(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan_turns<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V tan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V atan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::atan<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V atan2(const V &y, const V &x) noexcept { return math::lift(y, x, [](const auto &a, const auto &b) { return math::detail::atan2<math::fast>(a, b); }); }

    template<single V>
    WRAY_INLINE V omega(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::omega<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V product_log(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::product_log<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V tanh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tanh<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V sigmoid(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sigmoid<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V pow(const V &x, const V &y) noexcept { return math::lift(x, y, [](const auto &a, const auto &b) { return math::detail::pow<math::fast>(a, b); }); }

    template<single V>
    WRAY_INLINE V db_to_gain(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::db_to_gain<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V gain_to_db(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::gain_to_db<math::fast>(v); }); }

    template<math::real_range In, class Out>
    void exp(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::exp(v); }); }

    template<math::real_range In, class Out>
    void exp2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::exp2(v); }); }

    template<math::real_range In, class Out>
    void log(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::log(v); }); }

    template<math::real_range In, class Out>
    void log2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::log2(v); }); }

    template<math::real_range In, class Out>
    void sin_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::sin_turns(v); }); }

    template<math::real_range In, class Out>
    void sin(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::sin(v); }); }

    template<math::real_range In, class Out>
    void cos_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::cos_turns(v); }); }

    template<math::real_range In, class Out>
    void cos(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::cos(v); }); }

    template<math::real_range In, class SinOut, class CosOut>
        requires std::is_same_v<std::ranges::range_value_t<In>, float>
    void sincos(const In &in, SinOut &&sin_out, CosOut &&cos_out) noexcept
    {
        math::apply_pair(in, sin_out, cos_out, [](auto v) { return fast::sincos(v); });
    }

    template<math::real_range In, class SinOut, class CosOut>
        requires std::is_same_v<std::ranges::range_value_t<In>, float>
    void sincos_turns(const In &in, SinOut &&sin_out, CosOut &&cos_out) noexcept
    {
        math::apply_pair(in, sin_out, cos_out, [](auto v) { return fast::sincos_turns(v); });
    }

    template<math::real_range In, class Out>
    void tanpi(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::tanpi(v); }); }

    template<math::real_range In, class Out>
    void tan_turns(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::tan_turns(v); }); }

    template<math::real_range In, class Out>
    void tan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::tan(v); }); }

    template<math::real_range In, class Out>
    void atan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::atan(v); }); }

    template<math::real_range In, class Out>
    void omega(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::omega(v); }); }

    template<math::real_range In, class Out>
    void product_log(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::product_log(v); }); }

    template<math::real_range In, class Out>
    void tanh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::tanh(v); }); }

    template<math::real_range In, class Out>
    void sigmoid(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::sigmoid(v); }); }

    template<math::real_range In, class Out>
    void db_to_gain(const In &in, Out &&out) noexcept
    {
        math::apply(in, out, [](auto v) { return fast::db_to_gain(v); });
    }

    template<math::real_range In, class Out>
    void gain_to_db(const In &in, Out &&out) noexcept
    {
        math::apply(in, out, [](auto v) { return fast::gain_to_db(v); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void atan2(const A &y, const B &x, Out &&out) noexcept
    {
        math::apply(y, x, out, [](auto a, auto b) { return fast::atan2(a, b); });
    }

    template<math::real_range A, math::real_range B, class Out>
    void pow(const A &x, const B &y, Out &&out) noexcept
    {
        math::apply(x, y, out, [](auto a, auto b) { return fast::pow(a, b); });
    }
}

namespace wray::inline WRAY_ABI::balanced
{
    template<simd::real V>
    WRAY_INLINE V sin(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin<math::balanced>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cos(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos<math::balanced>(v); }); }

    template<simd::real V>
    WRAY_INLINE math::sincos_result<V> sincos(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos<math::balanced>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tanh(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tanh<math::balanced>(v); }); }

    template<simd::real V>
    WRAY_INLINE V pow(const V &x, const V &y) noexcept { return math::lift(x, y, [](const auto &a, const auto &b) { return math::detail::pow<math::balanced>(a, b); }); }

    template<math::real_range In, class Out>
    void sin(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return balanced::sin(v); }); }

    template<math::real_range In, class Out>
    void cos(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return balanced::cos(v); }); }

    template<math::real_range In, class SinOut, class CosOut>
    void sincos(const In &in, SinOut &&sin_out, CosOut &&cos_out) noexcept
    {
        math::apply_pair(in, sin_out, cos_out, [](auto v) { return balanced::sincos(v); });
    }

    template<math::real_range In, class Out>
    void tanh(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return balanced::tanh(v); }); }

    template<math::real_range A, math::real_range B, class Out>
    void pow(const A &x, const B &y, Out &&out) noexcept { math::apply(x, y, out, [](auto a, auto b) { return balanced::pow(a, b); }); }
}

WRAY_FP_END
