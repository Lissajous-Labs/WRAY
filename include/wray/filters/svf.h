#pragma once

#include "common.h"

namespace wray::filters
{
    enum class svf_type
    {
        lowpass,
        highpass,
        bandpass,
        notch,
        peak,
        low_shelf,
        high_shelf,
        allpass,
        tilt
    };

    struct svf_coefficients
    {
        double a1 = 1.0;
        double a2 = 0.0;
        double a3 = 0.0;
        double m0 = 0.0;
        double m1 = 0.0;
        double m2 = 1.0;
    };

    inline svf_coefficients design_svf(svf_type type, double sample_rate, double hz, double q, double gain_db) noexcept
    {
        const double fs = detail::sample_rate_or_default(sample_rate);
        const double gt = detail::prewarp(detail::clamp_frequency(hz, fs), fs);
        const double kq = 1.0 / detail::clamp_q(q);
        gain_db = detail::clamp_gain_db(gain_db);
        const double a = wray::db_to_gain(gain_db * 0.5);
        const double am1 = wray::expm1(gain_db * (std::numbers::ln10 / 40.0));
        const double a2m1 = wray::expm1(gain_db * (std::numbers::ln10 / 20.0));
        double g = gt;
        double k = kq;
        svf_coefficients c;
        switch (type)
        {
            case svf_type::lowpass:
                c.m0 = 0.0;
                c.m1 = 0.0;
                c.m2 = 1.0;
                break;
            case svf_type::highpass:
                c.m0 = 1.0;
                c.m1 = -k;
                c.m2 = -1.0;
                break;
            case svf_type::bandpass:
                c.m0 = 0.0;
                c.m1 = k;
                c.m2 = 0.0;
                break;
            case svf_type::notch:
                c.m0 = 1.0;
                c.m1 = -k;
                c.m2 = 0.0;
                break;
            case svf_type::peak:
                k = kq / a;
                c.m0 = 1.0;
                c.m1 = k * a2m1;
                c.m2 = 0.0;
                break;
            case svf_type::low_shelf:
                g = gt / simd::sqrt(a);
                c.m0 = 1.0;
                c.m1 = k * am1;
                c.m2 = a2m1;
                break;
            case svf_type::high_shelf:
                g = gt * simd::sqrt(a);
                c.m0 = a * a;
                c.m1 = -k * am1 * a;
                c.m2 = -a2m1;
                break;
            case svf_type::allpass:
                c.m0 = 1.0;
                c.m1 = -2.0 * k;
                c.m2 = 0.0;
                break;
            case svf_type::tilt:
                g = gt * simd::sqrt(a);
                c.m0 = a;
                c.m1 = -k * am1;
                c.m2 = -a2m1 / a;
                break;
        }
        c.a1 = 1.0 / (1.0 + g * (g + k));
        c.a2 = g * c.a1;
        c.a3 = g * c.a2;
        return c;
    }

    template<simd::real V>
    class svf
    {
    public:
        using value_type = V;

        svf() noexcept { update(); }

        void prepare(double sample_rate) noexcept
        {
            fs_ = detail::sample_rate_or_default(sample_rate);
            update();
            reset();
        }

        void reset() noexcept
        {
            ic1_ = V(simd::scalar_t<V>(0));
            ic2_ = V(simd::scalar_t<V>(0));
        }

        void set_type(svf_type t) noexcept
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

        void set(svf_type t, double hz, double q, double gain_db) noexcept
        {
            type_ = t;
            hz_ = hz;
            q_ = q;
            gain_db_ = gain_db;
            update();
        }

        [[nodiscard]] svf_type type() const noexcept { return type_; }
        [[nodiscard]] double frequency() const noexcept { return detail::clamp_frequency(hz_, fs_); }
        [[nodiscard]] double q() const noexcept { return detail::clamp_q(q_); }
        [[nodiscard]] double gain_db() const noexcept { return detail::clamp_gain_db(gain_db_); }
        [[nodiscard]] double sample_rate() const noexcept { return fs_; }
        [[nodiscard]] const svf_coefficients &coefficients() const noexcept { return c_; }

        WRAY_INLINE V tick(const V &x) noexcept
        {
            const V v3 = x - ic2_;
            const V p1 = a1_ * ic1_;
            const V v1 = simd::fma(a2_, v3, p1);
            const V v2 = simd::fma(a3_, v3, simd::fma(a2_, ic1_, ic2_));
            ic1_ = v1 + v1 - ic1_;
            ic2_ = v2 + v2 - ic2_;
            const V direct = m0_ * x;
            return simd::fma(m2_, v2, simd::fma(m1_, v1, direct));
        }

        void process(std::span<V> io) noexcept
        {
            for (V &x: io)
                x = tick(x);
        }

        void process(std::span<const V> in, std::span<V> out) noexcept
        {
            const std::size_t n = in.size() < out.size() ? in.size() : out.size();
            for (std::size_t i = 0; i < n; ++i)
                out[i] = tick(in[i]);
        }

        [[nodiscard]] std::complex<double> response(double hz) const noexcept
        {
            const auto sc = detail::half_angle(hz, fs_);
            const std::complex<double> u(-2.0 * sc.sin * sc.sin, 2.0 * sc.sin * sc.cos);
            const double b2 = 4.0 * c_.a2 * c_.a2;
            const std::complex<double> p = u + 2.0 * (1.0 - c_.a1);
            const std::complex<double> q = u + 2.0 * c_.a3;
            const std::complex<double> det = p * q + b2;
            const double c1 = c_.m1 * c_.a1 + c_.m2 * c_.a2;
            const double c2 = c_.m2 * (1.0 - c_.a3) - c_.m1 * c_.a2;
            const double d0 = c_.m0 + c_.m1 * c_.a2 + c_.m2 * c_.a3;
            const std::complex<double> num = 2.0 * c_.a2 * c1 * u + c2 * (b2 + 2.0 * c_.a3 * p);
            return d0 + num / det;
        }

    private:
        void update() noexcept
        {
            const svf_coefficients d = design_svf(type_, fs_, hz_, q_, gain_db_);
            c_.a1 = detail::realise<V>(d.a1);
            c_.a2 = detail::realise<V>(d.a2);
            c_.a3 = detail::realise<V>(d.a3);
            c_.m0 = detail::realise<V>(d.m0);
            c_.m1 = detail::realise<V>(d.m1);
            c_.m2 = detail::realise<V>(d.m2);
            a1_ = detail::splat<V>(c_.a1);
            a2_ = detail::splat<V>(c_.a2);
            a3_ = detail::splat<V>(c_.a3);
            m0_ = detail::splat<V>(c_.m0);
            m1_ = detail::splat<V>(c_.m1);
            m2_ = detail::splat<V>(c_.m2);
        }

        double fs_ = detail::default_sample_rate;
        double hz_ = 1000.0;
        double q_ = std::numbers::sqrt2 * 0.5;
        double gain_db_ = 0.0;
        svf_type type_ = svf_type::lowpass;
        svf_coefficients c_{};
        V a1_{};
        V a2_{};
        V a3_{};
        V m0_{};
        V m1_{};
        V m2_{};
        V ic1_ = V(simd::scalar_t<V>(0));
        V ic2_ = V(simd::scalar_t<V>(0));
    };
}
