#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>

#include "math/kernels.h"

namespace wray::math
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

namespace wray
{
    template<simd::real V>
    WRAY_INLINE V exp(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V exp2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::exp2<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V log2(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::log2<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V sin(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V cos(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE math::sincos_result<V> sincos(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V tan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V atan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::atan<math::precise>(v); }); }

    template<simd::real V>
    WRAY_INLINE V atan2(const V &y, const V &x) noexcept { return math::lift(y, x, [](const auto &a, const auto &b) { return math::detail::atan2<math::precise>(a, b); }); }

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
    void exp2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::exp2(v); }); }

    template<math::real_range In, class Out>
    void log(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log(v); }); }

    template<math::real_range In, class Out>
    void log2(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::log2(v); }); }

    template<math::real_range In, class Out>
    void sin(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::sin(v); }); }

    template<math::real_range In, class Out>
    void cos(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::cos(v); }); }

    template<math::real_range In, class Out>
    void tan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::tan(v); }); }

    template<math::real_range In, class Out>
    void atan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return wray::atan(v); }); }

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

namespace wray::fast
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
    WRAY_INLINE V sin(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::sin<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V cos(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::cos<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE math::sincos_result<V> sincos(const V &x) noexcept { return math::lift_pair(x, [](const auto &v) { return math::detail::sincos<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V tan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::tan<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V atan(const V &x) noexcept { return math::lift(x, [](const auto &v) { return math::detail::atan<math::fast>(v); }); }

    template<single V>
    WRAY_INLINE V atan2(const V &y, const V &x) noexcept { return math::lift(y, x, [](const auto &a, const auto &b) { return math::detail::atan2<math::fast>(a, b); }); }

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
    void sin(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::sin(v); }); }

    template<math::real_range In, class Out>
    void cos(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::cos(v); }); }

    template<math::real_range In, class Out>
    void tan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::tan(v); }); }

    template<math::real_range In, class Out>
    void atan(const In &in, Out &&out) noexcept { math::apply(in, out, [](auto v) { return fast::atan(v); }); }

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
