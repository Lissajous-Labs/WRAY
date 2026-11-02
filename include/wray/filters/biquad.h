#pragma once

#include "common.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI::filters
{
    enum class biquad_type
    {
        lowpass,
        highpass,
        bandpass,
        notch,
        peak,
        low_shelf,
        high_shelf,
        allpass
    };

    [[nodiscard]] constexpr bool uses_gain(biquad_type t) noexcept
    {
        return t == biquad_type::peak || t == biquad_type::low_shelf || t == biquad_type::high_shelf;
    }

    struct biquad_coefficients
    {
        double b0 = 1.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
    };

    inline biquad_coefficients design_biquad(biquad_type type, double sample_rate, double hz, double q,
                                             double gain_db) noexcept
    {
        const double fs = detail::sample_rate_or_default(sample_rate);
        gain_db = detail::clamp_gain_db(gain_db);
        if (uses_gain(type) && gain_db == 0.0)
            return {};
        const auto half = detail::half_angle(detail::clamp_frequency(hz, fs), fs);
        const double s = half.sin;
        const double c = half.cos;
        const double ss = s * s;
        const double cc = c * c;
        const double inv_q = 1.0 / detail::clamp_q(q);
        const double alpha = s * c * inv_q;
        const double cos_w = (c - s) * (c + s);
        double b0 = 1.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double a0 = 1.0 + alpha;
        double a1 = -2.0 * cos_w;
        double a2 = 1.0 - alpha;
        switch (type)
        {
            case biquad_type::lowpass:
                b0 = ss;
                b1 = 2.0 * ss;
                b2 = ss;
                break;
            case biquad_type::highpass:
                b0 = cc;
                b1 = -2.0 * cc;
                b2 = cc;
                break;
            case biquad_type::bandpass:
                b0 = alpha;
                b1 = 0.0;
                b2 = -alpha;
                break;
            case biquad_type::notch:
                b0 = 1.0;
                b1 = a1;
                b2 = 1.0;
                break;
            case biquad_type::peak:
            {
                const double a = wray::db_to_gain(gain_db * 0.5);
                b0 = 1.0 + alpha * a;
                b1 = a1;
                b2 = 1.0 - alpha * a;
                a0 = 1.0 + alpha / a;
                a2 = 1.0 - alpha / a;
                break;
            }
            case biquad_type::low_shelf:
            {
                const double a = wray::db_to_gain(gain_db * 0.5);
                const double beta = simd::sqrt(a) * alpha;
                const double lo = cc + a * ss;
                const double hi = a * cc + ss;
                b0 = a * (lo + beta);
                b1 = 2.0 * a * (a * ss - cc);
                b2 = a * (lo - beta);
                a0 = hi + beta;
                a1 = -2.0 * (a * cc - ss);
                a2 = hi - beta;
                break;
            }
            case biquad_type::high_shelf:
            {
                const double a = wray::db_to_gain(gain_db * 0.5);
                const double beta = simd::sqrt(a) * alpha;
                const double lo = cc + a * ss;
                const double hi = a * cc + ss;
                b0 = a * (hi + beta);
                b1 = -2.0 * a * (a * cc - ss);
                b2 = a * (hi - beta);
                a0 = lo + beta;
                a1 = 2.0 * (a * ss - cc);
                a2 = lo - beta;
                break;
            }
            case biquad_type::allpass:
                b0 = a2;
                b1 = a1;
                b2 = a0;
                break;
        }
        const double inv = 1.0 / a0;
        if (type == biquad_type::allpass)
            return {a2 * inv, a1 * inv, 1.0, a1 * inv, a2 * inv};
        return {b0 * inv, b1 * inv, b2 * inv, a1 * inv, a2 * inv};
    }

    template<simd::real V, simd::real P = V>
    class biquad
    {
    public:
        using value_type = V;
        using state_type = P;

        biquad() noexcept { update(); }

        void prepare(double sample_rate) noexcept
        {
            fs_ = detail::sample_rate_or_default(sample_rate);
            update();
            reset();
        }

        void reset() noexcept
        {
            s1_ = P(simd::scalar_t<P>(0));
            s2_ = P(simd::scalar_t<P>(0));
            hold_ = P(simd::scalar_t<P>(1));
            next_ = hold_;
            last_ = hold_;
            recent_ = hold_;
        }

        void set_type(biquad_type t) noexcept
        {
            type_ = t;
            update();
        }

        void set_frequency(double hz) noexcept
        {
            hz_ = hz;
            update();
        }

        void set_q(double q) noexcept
        {
            q_ = q;
            update();
        }

        void set_gain_db(double gain_db) noexcept
        {
            gain_db_ = gain_db;
            update();
        }

        void set(biquad_type t, double hz, double q, double gain_db) noexcept
        {
            type_ = t;
            hz_ = detail::unless_nan(hz, hz_);
            q_ = detail::unless_nan(q, q_);
            gain_db_ = detail::unless_nan(gain_db, gain_db_);
            update();
        }

        void set_coefficients(const biquad_coefficients &k) noexcept { load(k); }

        [[nodiscard]] biquad_type type() const noexcept { return type_; }
        [[nodiscard]] double frequency() const noexcept { return detail::clamp_frequency(hz_, fs_); }
        [[nodiscard]] double q() const noexcept { return detail::clamp_q(q_); }
        [[nodiscard]] double gain_db() const noexcept { return detail::clamp_gain_db(gain_db_); }
        [[nodiscard]] double sample_rate() const noexcept { return fs_; }
        [[nodiscard]] const biquad_coefficients &coefficients() const noexcept { return c_; }

        [[nodiscard]] std::array<P, 2> state() const noexcept { return {s1_, s2_}; }

        WRAY_INLINE V tick(const V &x) noexcept
        {
            const P u = detail::lane_cast<P>(x);
            const P loud = detail::awake<P>(detail::quiet(u));
            const P k = simd::max(simd::max(hold_, loud), recent_);
            recent_ = simd::max(loud, last_);
            last_ = loud;
            hold_ = next_;
            next_ = detail::awake<P>(detail::quiet(s1_, s2_));
            const P y = simd::fma(b0_, u, s1_);
            s1_ = simd::fma(na1_ * k, y, simd::fma(b1_, u, s2_));
            s2_ = simd::fma(na2_ * k, y, b2_ * u);
            return detail::lane_cast<V>(y);
        }

        void process(std::span<V> io) noexcept
        {
            auto local = *this;
            for (V &x: io)
                x = local.tick(x);
            *this = local;
        }

        void process(std::span<const V> in, std::span<V> out) noexcept
        {
            const std::size_t n = in.size() < out.size() ? in.size() : out.size();
            auto local = *this;
            for (std::size_t i = 0; i < n; ++i)
                out[i] = local.tick(in[i]);
            *this = local;
        }

        [[nodiscard]] std::complex<double> response(double hz) const noexcept
        {
            const auto sc = detail::half_angle(hz, fs_);
            const double vers = 2.0 * sc.sin * sc.sin;
            const double sin_w = 2.0 * sc.sin * sc.cos;
            const std::complex<double> num(c_.b0 + c_.b1 + c_.b2 - (c_.b0 + c_.b2) * vers, (c_.b0 - c_.b2) * sin_w);
            const std::complex<double> den(1.0 + c_.a1 + c_.a2 - (1.0 + c_.a2) * vers, (1.0 - c_.a2) * sin_w);
            return num / den;
        }

    private:
        void update() noexcept { load(design_biquad(type_, fs_, hz_, q_, gain_db_)); }

        void load(const biquad_coefficients &k) noexcept
        {
            c_.b0 = detail::realise<P>(k.b0);
            c_.b1 = detail::realise<P>(k.b1);
            c_.b2 = detail::realise<P>(k.b2);
            c_.a1 = detail::realise<P>(k.a1);
            c_.a2 = detail::realise<P>(k.a2);
            using S = simd::scalar_t<P>;
            const double edge = static_cast<double>(detail::below(S(1)));
            c_.a2 = std::clamp(c_.a2, -edge, edge);
            const double limit = static_cast<double>(detail::below(static_cast<S>(1.0 + c_.a2)));
            c_.a1 = std::clamp(c_.a1, -limit, limit);
            if (k.b0 == k.a2 && k.b1 == k.a1 && k.b2 == 1.0)
            {
                c_.b0 = c_.a2;
                c_.b1 = c_.a1;
                c_.b2 = 1.0;
            }
            b0_ = detail::splat<P>(c_.b0);
            b1_ = detail::splat<P>(c_.b1);
            b2_ = detail::splat<P>(c_.b2);
            na1_ = detail::splat<P>(-c_.a1);
            na2_ = detail::splat<P>(-c_.a2);
        }

        double fs_ = detail::default_sample_rate;
        double hz_ = 1000.0;
        double q_ = std::numbers::sqrt2 * 0.5;
        double gain_db_ = 0.0;
        biquad_type type_ = biquad_type::lowpass;
        biquad_coefficients c_{};
        P b0_{};
        P b1_{};
        P b2_{};
        P na1_{};
        P na2_{};
        P s1_ = P(simd::scalar_t<P>(0));
        P s2_ = P(simd::scalar_t<P>(0));
        P hold_ = P(simd::scalar_t<P>(1));
        P next_ = P(simd::scalar_t<P>(1));
        P last_ = P(simd::scalar_t<P>(1));
        P recent_ = P(simd::scalar_t<P>(1));
    };
}

WRAY_FP_END
