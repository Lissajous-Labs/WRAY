#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "math.h"

namespace wray::adaa::detail
{
    using namespace wray::math::detail;

    template<class S>
    struct gauss;

    template<>
    struct gauss<float>
    {
        static constexpr std::array<float, 4> nodes{-0.861136311594052575f, -0.339981043584856265f, 0.339981043584856265f, 0.861136311594052575f};
        static constexpr std::array<float, 4> weights{0.173927422568726929f, 0.326072577431273071f, 0.326072577431273071f, 0.173927422568726929f};
        static constexpr float span = 0.5f;
    };

    template<>
    struct gauss<double>
    {
        static constexpr std::array<double, 6> nodes{-0.932469514203152028, -0.661209386466264514, -0.238619186083196909,
                                                     0.238619186083196909,  0.661209386466264514,  0.932469514203152028};
        static constexpr std::array<double, 6> weights{0.0856622461895851725, 0.180380786524069303, 0.233956967286345523,
                                                       0.233956967286345523,  0.180380786524069303, 0.0856622461895851725};
        static constexpr double span = 0.25;
    };

    template<class V, class F>
    WRAY_INLINE V quadrature(const V &x0, const V &x1, F f) noexcept
    {
        using S = scalar_t<V>;
        using G = gauss<S>;
        const V m = x0 * V(S(0.5)) + x1 * V(S(0.5));
        const V half = (x1 - x0) * V(S(0.5));
        V sum(S(0));
        for (std::size_t i = 0; i < G::nodes.size(); ++i)
            sum = fma(V(G::weights[i]), f(fma(half, V(G::nodes[i]), m)), sum);
        return sum;
    }

    template<class V>
    WRAY_INLINE V huge_input() noexcept
    {
        using S = scalar_t<V>;
        return V(std::numeric_limits<S>::digits == 24 ? S(0x1p40) : S(0x1p300));
    }

    template<class V, class F>
    WRAY_INLINE V settle(const V &x0, const V &x1, const V &v, F f) noexcept
    {
        V r = select(abs(x1 - x0) == inf<V>(), f(select(abs(x1) == inf<V>(), x1, x0)), v);
        r = select(x0 == x1, f(x0), r);
        return select(either(isnan(x0), isnan(x1)), x0 + x1, r);
    }

    template<class V, class F, class D>
    WRAY_INLINE V mean(const V &x0, const V &x1, F f, D delta) noexcept
    {
        using S = scalar_t<V>;
        const V h = x1 - x0;
        const V p = abs(x0);
        const V q = abs(x1);
        const V lo = simd::min(p, q);
        const V hi = simd::max(p, q);
        const V d = hi - lo;
        const V m = select(d < V(gauss<S>::span), quadrature(lo, hi, f), delta(lo, hi, d) / d);
        const V v = select(q >= p, m, -m) * (d / h);
        return settle(x0, x1, select(d == V(S(0)), select(q >= p, m, -m) * (d / h), v), f);
    }

    template<class V>
    WRAY_INLINE V clip(const V &x) noexcept
    {
        using S = scalar_t<V>;
        return simd::min(simd::max(x, V(S(-1))), V(S(1)));
    }

    template<class V>
    WRAY_INLINE V soft(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V a = abs(clip(x));
        const V d = V(S(1)) - a;
        const V inner = fma(a * a * a, V(S(-1) / S(3)), a);
        const V outer = fma(d * d * (a + V(S(2))), V(S(-1) / S(3)), V(S(2) / S(3)));
        return select(isnan(x), x, copysign_from(select(a < V(S(0.5)), inner, outer), x));
    }

    template<class V>
    WRAY_INLINE V ratio(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const V a = abs(x);
        const V d = one + a;
        return select(isnan(x), x, copysign_from(select(a < one, a / d, one - one / d), x));
    }

    template<class V>
    WRAY_INLINE V root_ratio(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const V a = simd::min(abs(x), huge_input<V>());
        const auto sq = two_product(a, a);
        const auto s = two_sum(one, sq.hi);
        const V sl = s.lo + sq.lo;
        const V r = simd::sqrt(s.hi);
        const V rl = (residual(r, r, s.hi) + sl) / (r + r);
        const V v = a / r;
        const V e = residual(v, r, a) / r;
        const V small = v + fma(-v, rl / r, e);
        const V u = one / a;
        const V large = one / simd::sqrt(fma(u, u, one));
        return select(isnan(x), x, copysign_from(select(a < one, small, large), x));
    }

    template<class V, class M>
    WRAY_INLINE V folded(const V &x0, const V &x1, M mean_of_magnitudes) noexcept
    {
        const V p = abs(x0);
        const V q = abs(x1);
        const V lo = simd::min(p, q);
        const V hi = simd::max(p, q);
        const V d = hi - lo;
        const V m = mean_of_magnitudes(lo, hi, d);
        return select(q >= p, m, -m) * (d / (x1 - x0));
    }

    template<class V>
    WRAY_INLINE V hard_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        const V v = folded(x0, x1, [](const V &lo, const V &hi, const V &d) {
            const V one(S(1));
            const V half(S(0.5));
            const V mixed = fma((one - lo) * (one + lo), half, hi - one) / d;
            return select(hi <= one, (hi + lo) * half, select(lo >= one, one, mixed));
        });
        return settle(x0, x1, v, [](const V &x) { return clip(x); });
    }

    template<class V>
    WRAY_INLINE V soft_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        const V v = folded(x0, x1, [](const V &lo, const V &hi, const V &d) {
            const V one(S(1));
            const V sum = hi + lo;
            const V inner = fma(sum * fma(hi, hi, lo * lo), V(S(-1) / S(12)), sum * V(S(0.5)));
            const V edge = (one - lo) * (one + lo) * fma(fma(lo, lo, one), V(S(-1) / S(12)), V(S(0.5)));
            const V mixed = fma(hi - one, V(S(2) / S(3)), edge) / d;
            return select(hi <= one, inner, select(lo >= one, V(S(2) / S(3)), mixed));
        });
        return settle(x0, x1, v, [](const V &x) { return soft(x); });
    }

    template<class V>
    WRAY_INLINE V tanh_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        return mean(x0, x1, [](const V &x) { return wray::math::detail::tanh<math::precise>(x); }, [](const V &lo, const V &, const V &d) {
            const V e = exp<math::precise>(V(S(-2)) * lo);
            return d + log1p<math::precise>(e * expm1<math::precise>(V(S(-2)) * d) / (V(S(1)) + e));
        });
    }

    template<class V>
    WRAY_INLINE V atan_F1(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const V a = abs(x);
        const V inv = one / simd::max(a, one);
        const V wide = log<math::precise>(a) + V(S(0.5)) * log1p<math::precise>(inv * inv);
        const V narrow = V(S(0.5)) * log1p<math::precise>(x * x);
        return x * atan<math::precise>(x) - select(a > one, wide, narrow);
    }

    template<class V>
    WRAY_INLINE V atan_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        return mean(x0, x1, [](const V &x) { return atan<math::precise>(x); }, [](const V &lo, const V &hi, const V &d) {
            const V one(S(1));
            const V lead = fma(d, atan<math::precise>(hi), lo * atan<math::precise>(d / fma(lo, hi, one)));
            const V ratio_log = select(d < V(S(0.5)) * lo, log1p<math::precise>(d / lo), log<math::precise>(hi) - log<math::precise>(lo));
            const V both = fma(ratio_log, V(S(2)), log1p<math::precise>(-(d / hi) * (one + lo / hi) / fma(lo, lo, one)));
            const V ih = one / simd::max(hi, one);
            const V mixed = fma(log<math::precise>(hi), V(S(2)), log1p<math::precise>(ih * ih) - log1p<math::precise>(lo * lo));
            const V narrow = log1p<math::precise>(d * (hi + lo) / fma(lo, lo, one));
            const V spread = select(lo >= one, both, select(hi >= one, mixed, narrow));
            return fma(spread, V(S(-0.5)), lead);
        });
    }

    template<class V>
    WRAY_INLINE V ratio_F1(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V a = abs(x);
        const V u = a / (V(S(2)) + a);
        const V w = u * u;
        V p(S(1) / S(21));
        for (int k = 8; k >= 0; --k)
            p = fma(p, w, V(S(1) / S(2 * k + 3)));
        const V near = a * a / (V(S(2)) + a) - V(S(2)) * u * w * p;
        return select(a < V(S(0.25)), near, a - log1p<math::precise>(a));
    }

    template<class V>
    WRAY_INLINE V ratio_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        return mean(x0, x1, [](const V &x) { return ratio(x); }, [](const V &lo, const V &hi, const V &d) {
            return select(lo < V(S(1)), ratio_F1(hi) - ratio_F1(lo), d - log1p<math::precise>(d / (V(S(1)) + lo)));
        });
    }

    template<class V>
    WRAY_INLINE V root_norm(const V &x) noexcept
    {
        using S = scalar_t<V>;
        const V one(S(1));
        const V a = abs(x);
        const V u = one / simd::max(a, one);
        return select(a > one, a * simd::sqrt(fma(u, u, one)), simd::sqrt(fma(a, a, one)));
    }

    template<class V>
    WRAY_INLINE V root_first(const V &x0, const V &x1) noexcept
    {
        using S = scalar_t<V>;
        const V half(S(0.5));
        const V v = fma(x0, half, x1 * half) / fma(root_norm(x0), half, root_norm(x1) * half);
        return settle(x0, x1, v, [](const V &x) { return root_ratio(x); });
    }
}

namespace wray::adaa
{
    struct hard_clip
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::clip(v); }); }

        template<simd::real V>
        static V F1(const V &x) noexcept
        {
            return math::lift(x, [](const auto &v) {
                using W = std::decay_t<decltype(v)>;
                using S = simd::scalar_t<W>;
                const W a = simd::abs(v);
                return simd::select(a <= W(S(1)), v * v * W(S(0.5)), a - W(S(0.5)));
            });
        }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::hard_first(a, b); }); }
    };

    struct soft_clip
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::soft(v); }); }

        template<simd::real V>
        static V F1(const V &x) noexcept
        {
            return math::lift(x, [](const auto &v) {
                using W = std::decay_t<decltype(v)>;
                using S = simd::scalar_t<W>;
                const W a = simd::abs(v);
                const W t = v * v;
                return simd::select(a <= W(S(1)), simd::fma(t, t * W(S(-1) / S(12)), t * W(S(0.5))), simd::fma(a, W(S(2) / S(3)), W(S(-0.25))));
            });
        }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::soft_first(a, b); }); }
    };

    struct tanh
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return wray::tanh(x); }

        template<simd::real V>
        static V F1(const V &x) noexcept
        {
            return math::lift(x, [](const auto &v) {
                using W = std::decay_t<decltype(v)>;
                using S = simd::scalar_t<W>;
                using namespace math::detail;
                const W a = abs(v);
                const W s = sinh<math::precise>(a * W(S(0.5)));
                const W near = log1p<math::precise>(W(S(2)) * s * s);
                const W far = a + log1p<math::precise>(exp<math::precise>(W(S(-2)) * a)) - W(math::constants<S>::ln2);
                return select(a < W(S(1)), near, far);
            });
        }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::tanh_first(a, b); }); }
    };

    struct atan
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return wray::atan(x); }

        template<simd::real V>
        static V F1(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::atan_F1(v); }); }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::atan_first(a, b); }); }
    };

    struct algebraic
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::ratio(v); }); }

        template<simd::real V>
        static V F1(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::ratio_F1(v); }); }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::ratio_first(a, b); }); }
    };

    struct radical
    {
        template<simd::real V>
        static V f(const V &x) noexcept { return math::lift(x, [](const auto &v) { return detail::root_ratio(v); }); }

        template<simd::real V>
        static V F1(const V &x) noexcept
        {
            return math::lift(x, [](const auto &v) {
                using W = std::decay_t<decltype(v)>;
                using S = simd::scalar_t<W>;
                const W a = simd::abs(v);
                const W ac = simd::min(a, detail::huge_input<W>());
                const W t = ac * ac;
                return simd::select(a < detail::huge_input<W>(), t / (W(S(1)) + simd::sqrt(W(S(1)) + t)), a - W(S(1)));
            });
        }

        template<simd::real V>
        static V first(const V &x0, const V &x1) noexcept { return math::lift(x0, x1, [](const auto &a, const auto &b) { return detail::root_first(a, b); }); }
    };

    template<class Shape, simd::real V>
    class first_order
    {
    public:
        explicit first_order(const V &initial = V(simd::scalar_t<V>(0))) noexcept : previous_(initial) {}

        V operator()(const V &x) noexcept
        {
            const V y = Shape::first(previous_, x);
            previous_ = x;
            return y;
        }

        void reset(const V &initial = V(simd::scalar_t<V>(0))) noexcept { previous_ = initial; }

    private:
        V previous_;
    };
}
