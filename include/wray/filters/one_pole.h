#pragma once

#include "common.h"

namespace wray::filters
{
    enum class one_pole_type
    {
        lowpass,
        highpass,
        allpass
    };

    template<simd::real V>
    class tpt_one_pole
    {
    public:
        using value_type = V;

        tpt_one_pole() noexcept { update(); }

        void prepare(double sample_rate) noexcept
        {
            fs_ = detail::sample_rate_or_default(sample_rate);
            update();
            reset();
        }

        void reset() noexcept { s_ = V(simd::scalar_t<V>(0)); }

        void set_type(one_pole_type t) noexcept
        {
            type_ = t;
            update();
        }

        void set_frequency(double hz) noexcept
        {
            hz_ = hz;
            update();
        }

        void set(one_pole_type t, double hz) noexcept
        {
            type_ = t;
            hz_ = hz;
            update();
        }

        [[nodiscard]] one_pole_type type() const noexcept { return type_; }
        [[nodiscard]] double frequency() const noexcept { return detail::clamp_frequency(hz_, fs_); }
        [[nodiscard]] double sample_rate() const noexcept { return fs_; }
        [[nodiscard]] double gain() const noexcept { return big_g_; }

        WRAY_INLINE V tick(const V &x) noexcept
        {
            const V v = (x - s_) * g_;
            const V lp = v + s_;
            s_ = lp + v;
            const V direct = cx_ * x;
            return simd::fma(clp_, lp, direct);
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
            const double gc = big_g_ * sc.cos;
            const std::complex<double> den(gc, (1.0 - big_g_) * sc.sin);
            const std::complex<double> num = mix_x_ * den + mix_lp_ * gc;
            return num / den;
        }

    private:
        void update() noexcept
        {
            const double g = detail::prewarp(detail::clamp_frequency(hz_, fs_), fs_);
            big_g_ = detail::open_unit<V>(g / (1.0 + g));
            switch (type_)
            {
                case one_pole_type::lowpass:
                    mix_x_ = 0.0;
                    mix_lp_ = 1.0;
                    break;
                case one_pole_type::highpass:
                    mix_x_ = 1.0;
                    mix_lp_ = -1.0;
                    break;
                case one_pole_type::allpass:
                    mix_x_ = -1.0;
                    mix_lp_ = 2.0;
                    break;
            }
            g_ = detail::splat<V>(big_g_);
            cx_ = detail::splat<V>(mix_x_);
            clp_ = detail::splat<V>(mix_lp_);
        }

        double fs_ = detail::default_sample_rate;
        double hz_ = 1000.0;
        one_pole_type type_ = one_pole_type::lowpass;
        double big_g_ = 0.0;
        double mix_x_ = 0.0;
        double mix_lp_ = 1.0;
        V g_ = V(simd::scalar_t<V>(0));
        V cx_ = V(simd::scalar_t<V>(0));
        V clp_ = V(simd::scalar_t<V>(1));
        V s_ = V(simd::scalar_t<V>(0));
    };
}
