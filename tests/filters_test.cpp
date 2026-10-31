#include <wray/wray.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.h"

namespace
{
    using cd = std::complex<double>;
    using ld = long double;
    using cld = std::complex<long double>;
    namespace wf = wray::filters;

    constexpr ld pi_ld = 3.141592653589793238462643383279502884L;
    constexpr std::size_t fft_size = 65536;
    constexpr std::size_t noise_length = 200000;
    constexpr int grid_points = 128;
    constexpr double floor_gain = 1.0e-6;
    constexpr double tail_floor = 1.0e-20;

    struct config
    {
        int type;
        double fs;
        double f0;
        double q;
        double gain;
    };

    namespace oracle
    {
        ld warp(ld hz, ld fs) { return std::tan(pi_ld * hz / fs); }

        cld zavalishin(wf::one_pole_type t, const config &c, double f)
        {
            const cld s(0.0L, warp(f, c.fs) / warp(c.f0, c.fs));
            const cld one(1.0L, 0.0L);
            switch (t)
            {
                case wf::one_pole_type::lowpass:
                    return one / (one + s);
                case wf::one_pole_type::highpass:
                    return s / (one + s);
                case wf::one_pole_type::allpass:
                    return (one - s) / (one + s);
            }
            return {};
        }

        struct simper_row
        {
            ld g;
            ld k;
            ld m0;
            ld m1;
            ld m2;
        };

        simper_row simper_table(wf::svf_type t, const config &c)
        {
            const ld a = std::pow(10.0L, static_cast<ld>(c.gain) / 40.0L);
            const ld g = warp(c.f0, c.fs);
            const ld k = 1.0L / static_cast<ld>(c.q);
            switch (t)
            {
                case wf::svf_type::lowpass:
                    return {g, k, 0.0L, 0.0L, 1.0L};
                case wf::svf_type::highpass:
                    return {g, k, 1.0L, -k, -1.0L};
                case wf::svf_type::bandpass:
                    return {g, k, 0.0L, 1.0L, 0.0L};
                case wf::svf_type::notch:
                    return {g, k, 1.0L, -k, 0.0L};
                case wf::svf_type::peak:
                {
                    const ld kb = 1.0L / (static_cast<ld>(c.q) * a);
                    return {g, kb, 1.0L, kb * (a * a - 1.0L), 0.0L};
                }
                case wf::svf_type::low_shelf:
                    return {g / std::sqrt(a), k, 1.0L, k * (a - 1.0L), a * a - 1.0L};
                case wf::svf_type::high_shelf:
                case wf::svf_type::tilt:
                    return {g * std::sqrt(a), k, a * a, k * (1.0L - a) * a, 1.0L - a * a};
                case wf::svf_type::allpass:
                    return {g, k, 1.0L, -2.0L * k, 0.0L};
            }
            return {g, k, 0.0L, 0.0L, 1.0L};
        }

        cld simper(wf::svf_type t, const config &c, double f)
        {
            const simper_row r = simper_table(t, c);
            const cld s(0.0L, warp(f, c.fs) / r.g);
            const cld h = (r.m0 * s * s + (r.m0 * r.k + r.m1) * s + (r.m0 + r.m2)) / (s * s + r.k * s + 1.0L);
            if (t == wf::svf_type::bandpass)
                return r.k * h;
            if (t == wf::svf_type::tilt)
                return h / std::pow(10.0L, static_cast<ld>(c.gain) / 40.0L);
            return h;
        }

        cld cookbook(wf::biquad_type t, const config &c, double f)
        {
            const ld a = std::pow(10.0L, static_cast<ld>(c.gain) / 40.0L);
            const ld q = static_cast<ld>(c.q);
            const ld ra = std::sqrt(a);
            const cld s(0.0L, warp(f, c.fs) / warp(c.f0, c.fs));
            const cld s2 = s * s;
            const cld den = s2 + s / q + 1.0L;
            switch (t)
            {
                case wf::biquad_type::lowpass:
                    return 1.0L / den;
                case wf::biquad_type::highpass:
                    return s2 / den;
                case wf::biquad_type::bandpass:
                    return (s / q) / den;
                case wf::biquad_type::notch:
                    return (s2 + 1.0L) / den;
                case wf::biquad_type::allpass:
                    return (s2 - s / q + 1.0L) / den;
                case wf::biquad_type::peak:
                    return (s2 + s * (a / q) + 1.0L) / (s2 + s / (a * q) + 1.0L);
                case wf::biquad_type::low_shelf:
                    return a * (s2 + (ra / q) * s + a) / (a * s2 + (ra / q) * s + 1.0L);
                case wf::biquad_type::high_shelf:
                    return a * (a * s2 + (ra / q) * s + 1.0L) / (s2 + (ra / q) * s + a);
            }
            return {};
        }
    }

    struct one_pole_kind
    {
        static constexpr const char *name = "tpt_one_pole";
        static constexpr int types = 3;
        static constexpr bool has_q = false;

        static const char *type_name(int t)
        {
            constexpr const char *n[] = {"lowpass", "highpass", "allpass"};
            return n[t];
        }

        static bool gained(int) { return false; }

        template<class V, class P>
        using filter = wf::tpt_one_pole<V>;

        template<class F>
        static void set(F &f, const config &c)
        {
            f.prepare(c.fs);
            f.set(static_cast<wf::one_pole_type>(c.type), c.f0);
        }

        static cd reference(const config &c, double f)
        {
            const cld h = oracle::zavalishin(static_cast<wf::one_pole_type>(c.type), c, f);
            return {static_cast<double>(h.real()), static_cast<double>(h.imag())};
        }
    };

    struct svf_kind
    {
        static constexpr const char *name = "svf";
        static constexpr int types = 9;
        static constexpr bool has_q = true;

        static const char *type_name(int t)
        {
            constexpr const char *n[] = {"lowpass", "highpass", "bandpass", "notch", "peak", "low_shelf", "high_shelf", "allpass", "tilt"};
            return n[t];
        }

        static bool gained(int t)
        {
            const auto s = static_cast<wf::svf_type>(t);
            return s == wf::svf_type::peak || s == wf::svf_type::low_shelf || s == wf::svf_type::high_shelf || s == wf::svf_type::tilt;
        }

        template<class V, class P>
        using filter = wf::svf<V>;

        template<class F>
        static void set(F &f, const config &c)
        {
            f.prepare(c.fs);
            f.set(static_cast<wf::svf_type>(c.type), c.f0, c.q, c.gain);
        }

        static cd reference(const config &c, double f)
        {
            const cld h = oracle::simper(static_cast<wf::svf_type>(c.type), c, f);
            return {static_cast<double>(h.real()), static_cast<double>(h.imag())};
        }
    };

    struct biquad_kind
    {
        static constexpr const char *name = "biquad";
        static constexpr int types = 8;
        static constexpr bool has_q = true;

        static const char *type_name(int t)
        {
            constexpr const char *n[] = {"lowpass", "highpass", "bandpass", "notch", "peak", "low_shelf", "high_shelf", "allpass"};
            return n[t];
        }

        static bool gained(int t) { return wf::uses_gain(static_cast<wf::biquad_type>(t)); }

        template<class V, class P>
        using filter = wf::biquad<V, P>;

        template<class F>
        static void set(F &f, const config &c)
        {
            f.prepare(c.fs);
            f.set(static_cast<wf::biquad_type>(c.type), c.f0, c.q, c.gain);
        }

        static cd reference(const config &c, double f)
        {
            const cld h = oracle::cookbook(static_cast<wf::biquad_type>(c.type), c, f);
            return {static_cast<double>(h.real()), static_cast<double>(h.imag())};
        }
    };

    std::vector<double> log_space(double lo, double hi, int n)
    {
        std::vector<double> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = lo * std::pow(hi / lo, static_cast<double>(i) / static_cast<double>(n - 1));
        return v;
    }

    const std::vector<double> &nu_grid()
    {
        static const std::vector<double> g = log_space(2.0e-4, 0.49, grid_points);
        return g;
    }

    template<class K>
    std::vector<config> make_box(const std::vector<double> &f0_hz, bool include_top, const std::vector<double> &qs,
                                 const std::vector<double> &gains)
    {
        std::vector<config> box;
        const std::vector<double> one_q{0.7071067811865476};
        const std::vector<double> no_gain{0.0};
        for (const double fs: {44100.0, 48000.0, 96000.0})
        {
            std::vector<double> f0s = f0_hz;
            if (include_top)
                f0s.push_back(0.45 * fs);
            for (int t = 0; t < K::types; ++t)
                for (const double f0: f0s)
                    for (const double q: K::has_q ? qs : one_q)
                        for (const double g: K::gained(t) ? gains : no_gain)
                            box.push_back({t, fs, f0, q, g});
        }
        return box;
    }

    template<class K>
    std::vector<config> oracle_box()
    {
        std::vector<double> f0 = log_space(20.0, 19845.0, 15);
        return make_box<K>(f0, true, {0.1, 0.25, 0.5, 0.7071067811865476, 1.0, 2.0, 5.0, 10.0, 20.0},
                           {-24.0, -12.0, -6.0, -1.0, 0.0, 1.0, 6.0, 12.0, 24.0});
    }

    template<class K>
    std::vector<config> impulse_box()
    {
        return make_box<K>({20.0, 200.0, 2000.0, 12000.0}, true, {0.1, 0.7071067811865476, 20.0}, {-24.0, 0.0, 24.0});
    }

    double relative_error(cd got, cd ref)
    {
        const double d = std::abs(got - ref);
        return d / std::max(std::abs(ref), floor_gain);
    }

    template<class F>
    void parallel_for(std::size_t n, F &&f)
    {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < hw; ++t)
            pool.emplace_back([&] {
                for (std::size_t i = next++; i < n; i = next++)
                    f(i);
            });
        for (auto &th: pool)
            th.join();
    }

    struct worst
    {
        double value = 0.0;
        config at{};
        double f = 0.0;

        void add(double v, const config &c, double hz)
        {
            if (v > value || std::isnan(v))
            {
                value = std::isnan(v) ? std::numeric_limits<double>::infinity() : v;
                at = c;
                f = hz;
            }
        }

        void merge(const worst &o)
        {
            if (o.value > value)
                *this = o;
        }
    };

    struct fft
    {
        std::vector<cd> twiddle;
        std::vector<std::uint32_t> reverse;

        fft() : twiddle(fft_size / 2), reverse(fft_size)
        {
            for (std::size_t k = 0; k < fft_size / 2; ++k)
            {
                const ld a = -2.0L * pi_ld * static_cast<ld>(k) / static_cast<ld>(fft_size);
                twiddle[k] = cd(static_cast<double>(std::cos(a)), static_cast<double>(std::sin(a)));
            }
            const int bits = std::countr_zero(fft_size);
            for (std::size_t i = 0; i < fft_size; ++i)
            {
                std::uint32_t r = 0;
                for (int b = 0; b < bits; ++b)
                    r |= static_cast<std::uint32_t>((i >> b) & 1u) << (bits - 1 - b);
                reverse[i] = r;
            }
        }

        void run(std::vector<cd> &a) const
        {
            for (std::size_t i = 0; i < fft_size; ++i)
                if (i < reverse[i])
                    std::swap(a[i], a[reverse[i]]);
            for (std::size_t len = 2; len <= fft_size; len <<= 1)
            {
                const std::size_t half = len / 2;
                const std::size_t step = fft_size / len;
                for (std::size_t i = 0; i < fft_size; i += len)
                    for (std::size_t j = 0; j < half; ++j)
                    {
                        const cd w = twiddle[j * step];
                        const cd x = a[i + j + half];
                        const cd v(x.real() * w.real() - x.imag() * w.imag(), x.real() * w.imag() + x.imag() * w.real());
                        const cd u = a[i + j];
                        a[i + j] = u + v;
                        a[i + j + half] = u - v;
                    }
            }
        }
    };

    const fft &transform()
    {
        static const fft f;
        return f;
    }

    template<class T>
    std::vector<T> noise(std::size_t n, std::uint64_t seed)
    {
        std::mt19937_64 g(seed);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<T> v(n);
        for (auto &x: v)
            x = static_cast<T>(d(g));
        return v;
    }

    template<class T>
    bool same_bits(T a, T b) { return std::bit_cast<typename wray::simd::bits_of<T>::integer>(a) == std::bit_cast<typename wray::simd::bits_of<T>::integer>(b); }

    struct comparison
    {
        worst rel;
        worst peak;
        std::vector<double> envelope;

        void merge(const comparison &o)
        {
            rel.merge(o.rel);
            peak.merge(o.peak);
            if (envelope.size() < o.envelope.size())
                envelope.resize(o.envelope.size(), 0.0);
            for (std::size_t j = 0; j < o.envelope.size(); ++j)
                envelope[j] = std::max(envelope[j], o.envelope[j]);
        }
    };

    comparison compare(const config &c, const std::vector<double> &hz, const std::vector<cd> &got, const std::vector<cd> &ref)
    {
        comparison r;
        double top = 0.0;
        for (const cd h: ref)
            top = std::max(top, std::abs(h));
        r.envelope.assign(hz.size(), 0.0);
        for (std::size_t j = 0; j < hz.size(); ++j)
        {
            const double e = relative_error(got[j], ref[j]);
            const double p = std::abs(got[j] - ref[j]) / top;
            r.rel.add(e, c, hz[j]);
            r.peak.add(p, c, hz[j]);
            r.envelope[j] = std::isnan(e) ? std::numeric_limits<double>::infinity() : e;
        }
        return r;
    }

    struct limits
    {
        double oracle_rel;
        double oracle_peak;
        double impulse_rel;
        double impulse_peak;
        double bound;
    };

    struct summary
    {
        std::string filter;
        std::string precision;
        limits limit{};
        std::size_t oracle_configs = 0;
        comparison oracle;
        std::size_t impulse_configs = 0;
        comparison impulse;
        std::size_t max_impulse_samples = 0;
        std::size_t max_tail_samples = 0;
        double max_bound_ratio = 0.0;
        bool decayed = true;
        bool all_finite = true;
        bool recovered = true;
    };

    template<class K, class V, class P>
    void oracle_check(summary &s)
    {
        const auto box = oracle_box<K>();
        const auto &nu = nu_grid();
        std::vector<comparison> w(box.size());
        parallel_for(box.size(), [&](std::size_t i) {
            typename K::template filter<V, P> f;
            K::set(f, box[i]);
            std::vector<double> hz(nu.size());
            std::vector<cd> got(nu.size());
            std::vector<cd> ref(nu.size());
            for (std::size_t j = 0; j < nu.size(); ++j)
            {
                hz[j] = nu[j] * box[i].fs;
                got[j] = f.response(hz[j]);
                ref[j] = K::reference(box[i], hz[j]);
            }
            w[i] = compare(box[i], hz, got, ref);
        });
        s.oracle_configs = box.size();
        for (const auto &c: w)
            s.oracle.merge(c);
    }

    template<class V>
    double as_double(V v) { return static_cast<double>(v); }

    struct impulse_result
    {
        comparison error;
        std::size_t samples = 0;
        std::size_t tail_samples = 0;
        double l1 = 0.0;
        double bound_ratio = 0.0;
        bool decayed = false;
        bool finite = true;
        bool recovered = true;
    };

    template<class K, class V, class P>
    impulse_result impulse_and_stability(const config &c, std::uint64_t seed)
    {
        using T = V;
        constexpr double decay = std::is_same_v<wray::simd::scalar_t<P>, float> ? 1.0e-30 : 1.0e-40;
        constexpr std::size_t max_blocks = 512;
        impulse_result r;
        typename K::template filter<V, P> f;
        K::set(f, c);

        std::vector<cd> folded(fft_size, cd(0.0, 0.0));
        double peak = 0.0;
        std::size_t blocks = 0;
        T x = T(1);
        while (blocks < max_blocks)
        {
            double block_max = 0.0;
            for (std::size_t n = 0; n < fft_size; ++n)
            {
                const double y = as_double(f.tick(x));
                x = T(0);
                folded[n] += cd(y, 0.0);
                r.l1 += std::fabs(y);
                block_max = std::max(block_max, std::fabs(y));
                if (!std::isfinite(y))
                    r.finite = false;
            }
            peak = std::max(peak, block_max);
            ++blocks;
            if (block_max <= decay * peak)
                break;
        }
        r.samples = blocks * fft_size;
        transform().run(folded);

        const auto &nu = nu_grid();
        std::vector<double> hz(nu.size());
        std::vector<cd> got(nu.size());
        std::vector<cd> ref(nu.size());
        for (std::size_t j = 0; j < nu.size(); ++j)
        {
            const auto k = static_cast<std::size_t>(std::llround(nu[j] * static_cast<double>(fft_size)));
            hz[j] = static_cast<double>(k) * c.fs / static_cast<double>(fft_size);
            got[j] = folded[k];
            ref[j] = f.response(hz[j]);
        }
        r.error = compare(c, hz, got, ref);

        const auto in = noise<T>(noise_length, seed);
        f.reset();
        double out_max = 0.0;
        for (const T v: in)
        {
            const double y = as_double(f.tick(v));
            if (!std::isfinite(y))
                r.finite = false;
            out_max = std::max(out_max, std::fabs(y));
        }
        r.bound_ratio = out_max / r.l1;
        for (std::size_t b = 0; b < max_blocks && !r.decayed; ++b)
        {
            double block_max = 0.0;
            for (std::size_t n = 0; n < fft_size; ++n)
                block_max = std::max(block_max, std::fabs(as_double(f.tick(T(0)))));
            r.tail_samples += fft_size;
            r.decayed = block_max <= tail_floor * r.l1;
        }

        f.tick(T(1.0e30));
        for (std::size_t n = 0; n < fft_size; ++n)
        {
            const double y = as_double(f.tick(T(0)));
            if (!std::isfinite(y))
                r.finite = false;
        }

        typename K::template filter<V, P> fresh;
        K::set(fresh, c);
        for (const T poison: {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()})
        {
            f.tick(poison);
            for (std::size_t n = 0; n < 64; ++n)
                f.tick(in[n]);
            f.reset();
            fresh.reset();
            for (std::size_t n = 0; n < 4096; ++n)
            {
                const T a = f.tick(in[n]);
                const T b = fresh.tick(in[n]);
                if (!std::isfinite(a) || !same_bits(a, b))
                    r.recovered = false;
            }
        }
        return r;
    }

    template<class K, class V, class P>
    void impulse_check(summary &s)
    {
        const auto box = impulse_box<K>();
        std::vector<impulse_result> res(box.size());
        parallel_for(box.size(), [&](std::size_t i) { res[i] = impulse_and_stability<K, V, P>(box[i], 0x5eed0000u + i); });
        s.impulse_configs = box.size();
        for (const auto &r: res)
        {
            s.impulse.merge(r.error);
            s.max_impulse_samples = std::max(s.max_impulse_samples, r.samples);
            s.max_tail_samples = std::max(s.max_tail_samples, r.tail_samples);
            s.max_bound_ratio = std::max(s.max_bound_ratio, r.bound_ratio);
            s.decayed = s.decayed && r.decayed;
            s.all_finite = s.all_finite && r.finite;
            s.recovered = s.recovered && r.recovered;
        }
    }

    template<class K, class S, class V, class SP, class VP>
    std::size_t parity_check(bool &identical)
    {
        constexpr int w = wray::simd::traits<V>::width;
        const auto box = impulse_box<K>();
        std::vector<char> ok(box.size(), 1);
        parallel_for(box.size(), [&](std::size_t i) {
            typename K::template filter<V, VP> lanes;
            typename K::template filter<S, SP> scalar[w];
            K::set(lanes, box[i]);
            for (auto &f: scalar)
                K::set(f, box[i]);
            std::vector<std::vector<S> > in(static_cast<std::size_t>(w));
            for (int l = 0; l < w; ++l)
                in[static_cast<std::size_t>(l)] = noise<S>(8192, 0xab00u + i * 8u + static_cast<std::size_t>(l));
            in[0][100] = S(1.0e30);
            in[1 % w][0] = S(1);
            for (std::size_t n = 1; n < 8192; ++n)
                in[1 % w][n] = S(0);
            for (std::size_t n = 0; n < 8192; ++n)
            {
                S buf[w];
                for (int l = 0; l < w; ++l)
                    buf[l] = in[static_cast<std::size_t>(l)][n];
                const V y = lanes.tick(V::load(buf));
                for (int l = 0; l < w; ++l)
                    if (!same_bits(y[l], scalar[l].tick(buf[l])))
                        ok[i] = 0;
            }
            for (const double hz: {20.0, 1000.0, 0.3 * box[i].fs})
            {
                const cd a = lanes.response(hz);
                const cd b = scalar[0].response(hz);
                if (!same_bits(a.real(), b.real()) || !same_bits(a.imag(), b.imag()))
                    ok[i] = 0;
            }
        });
        identical = std::all_of(ok.begin(), ok.end(), [](char c) { return c != 0; });
        return box.size();
    }

    struct parity_entry
    {
        std::string filter;
        std::string lanes;
        std::size_t configs;
        bool identical;
    };

    struct chart_entry
    {
        std::string filter;
        std::string type;
        config c;
        std::vector<double> hz;
        std::vector<cd> response_f64;
        std::vector<cd> response_f32;
        std::vector<cd> reference;
        std::vector<double> measured_hz;
        std::vector<cd> measured_f64;
        std::vector<cd> analytic_f64;
    };

    template<class K>
    void chart_data(std::vector<chart_entry> &out)
    {
        for (int t = 0; t < K::types; ++t)
        {
            const config c{t, 48000.0, 1000.0, K::has_q ? 2.0 : 0.7071067811865476, K::gained(t) ? 12.0 : 0.0};
            chart_entry e{K::name, K::type_name(t), c, {}, {}, {}, {}, {}, {}, {}};
            typename K::template filter<double, double> fd;
            typename K::template filter<float, float> ff;
            K::set(fd, c);
            K::set(ff, c);
            for (const double nu: log_space(2.0e-4, 0.49, 400))
            {
                const double hz = nu * c.fs;
                e.hz.push_back(hz);
                e.response_f64.push_back(fd.response(hz));
                e.response_f32.push_back(ff.response(hz));
                e.reference.push_back(K::reference(c, hz));
            }
            std::vector<cd> folded(fft_size, cd(0.0, 0.0));
            double x = 1.0;
            for (int b = 0; b < 64; ++b)
                for (std::size_t n = 0; n < fft_size; ++n)
                {
                    folded[n] += cd(fd.tick(x), 0.0);
                    x = 0.0;
                }
            transform().run(folded);
            std::size_t last = 0;
            for (const double nu: log_space(2.0e-4, 0.49, 400))
            {
                const auto k = static_cast<std::size_t>(std::llround(nu * static_cast<double>(fft_size)));
                if (k == last)
                    continue;
                last = k;
                const double hz = static_cast<double>(k) * c.fs / static_cast<double>(fft_size);
                e.measured_hz.push_back(hz);
                e.measured_f64.push_back(folded[k]);
                e.analytic_f64.push_back(fd.response(hz));
            }
            out.push_back(std::move(e));
        }
    }

    std::string num(double v)
    {
        if (std::isinf(v))
            return v > 0 ? "1e308" : "-1e308";
        if (std::isnan(v))
            return "null";
        char b[40];
        std::snprintf(b, sizeof b, "%.9g", v);
        return b;
    }

    std::string list(const std::vector<double> &v)
    {
        std::string s = "[";
        for (std::size_t i = 0; i < v.size(); ++i)
            s += (i ? "," : "") + num(v[i]);
        return s + "]";
    }

    std::string complex_list(const std::vector<cd> &v)
    {
        std::string s = "[";
        for (std::size_t i = 0; i < v.size(); ++i)
            s += std::string(i ? "," : "") + "[" + num(v[i].real()) + "," + num(v[i].imag()) + "]";
        return s + "]";
    }

    std::string config_json(const config &c, const char *type)
    {
        return std::string("{\"type\":\"") + type + "\",\"fs\":" + num(c.fs) + ",\"f0\":" + num(c.f0) + ",\"q\":" + num(c.q) +
               ",\"gain_db\":" + num(c.gain) + "}";
    }

    template<class K>
    void print_worst(const char *what, const worst &w)
    {
        std::printf("    %-13s %.3e at %s f0=%g q=%g gain=%g fs=%g f=%g\n", what, w.value, K::type_name(w.at.type), w.at.f0, w.at.q, w.at.gain,
                    w.at.fs, w.f);
    }

    template<class K, class V, class P>
    summary run_family(const char *precision, const limits &limit)
    {
        summary s;
        s.filter = K::name;
        s.precision = precision;
        s.limit = limit;
        oracle_check<K, V, P>(s);
        impulse_check<K, V, P>(s);
        std::printf("%s %s: oracle %zu configs, impulse/stability %zu configs, impulse up to %zu samples, tail up to %zu samples, "
                    "max |y|/L1 %.9f, decayed %d, finite %d, recovered %d\n",
                    s.filter.c_str(), precision, s.oracle_configs, s.impulse_configs, s.max_impulse_samples, s.max_tail_samples,
                    s.max_bound_ratio, s.decayed ? 1 : 0, s.all_finite ? 1 : 0, s.recovered ? 1 : 0);
        print_worst<K>("oracle rel", s.oracle.rel);
        print_worst<K>("oracle peak", s.oracle.peak);
        print_worst<K>("impulse rel", s.impulse.rel);
        print_worst<K>("impulse peak", s.impulse.peak);
        CHECK(s.oracle.rel.value <= limit.oracle_rel);
        CHECK(s.oracle.peak.value <= limit.oracle_peak);
        CHECK(s.impulse.rel.value <= limit.impulse_rel);
        CHECK(s.impulse.peak.value <= limit.impulse_peak);
        CHECK(s.max_bound_ratio <= limit.bound);
        CHECK(s.decayed);
        CHECK(s.all_finite);
        CHECK(s.recovered);
        return s;
    }

    template<class K>
    std::string worst_json(const worst &w)
    {
        return "{\"value\":" + num(w.value) + ",\"hz\":" + num(w.f) + ",\"config\":" + config_json(w.at, K::type_name(w.at.type)) + "}";
    }

    template<class K>
    std::string summary_json(const summary &s)
    {
        return "{\"filter\":\"" + s.filter + "\",\"precision\":\"" + s.precision + "\",\"oracle_configs\":" + std::to_string(s.oracle_configs) +
               ",\"oracle_rel\":" + worst_json<K>(s.oracle.rel) + ",\"oracle_rel_limit\":" + num(s.limit.oracle_rel) + ",\"oracle_peak\":" +
               worst_json<K>(s.oracle.peak) + ",\"oracle_peak_limit\":" + num(s.limit.oracle_peak) + ",\"oracle_envelope\":" +
               list(s.oracle.envelope) + ",\"impulse_configs\":" + std::to_string(s.impulse_configs) + ",\"impulse_rel\":" +
               worst_json<K>(s.impulse.rel) + ",\"impulse_rel_limit\":" + num(s.limit.impulse_rel) + ",\"impulse_peak\":" +
               worst_json<K>(s.impulse.peak) + ",\"impulse_peak_limit\":" + num(s.limit.impulse_peak) + ",\"impulse_envelope\":" +
               list(s.impulse.envelope) + ",\"max_impulse_samples\":" + std::to_string(s.max_impulse_samples) + ",\"noise_samples\":" +
               std::to_string(noise_length) + ",\"max_output_over_l1\":" + num(s.max_bound_ratio) + ",\"bound_limit\":" + num(s.limit.bound) +
               ",\"max_tail_samples\":" + std::to_string(s.max_tail_samples) + ",\"tail_floor_over_l1\":" + num(tail_floor) +
               ",\"decayed\":" + (s.decayed ? "true" : "false") + ",\"all_finite\":" + (s.all_finite ? "true" : "false") +
               ",\"recovered_after_reset\":" + (s.recovered ? "true" : "false") + "}";
    }
}

int main(int argc, char **argv)
{
    const char *json = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc)
            json = argv[++i];

    std::vector<std::string> summaries;
    {
        const auto a = run_family<one_pole_kind, double, double>("double", {1, 1, 1, 1, 1});
        const auto b = run_family<one_pole_kind, float, float>("float", {1, 1, 1, 1, 2});
        summaries.push_back(summary_json<one_pole_kind>(a));
        summaries.push_back(summary_json<one_pole_kind>(b));
    }
    {
        const auto a = run_family<svf_kind, double, double>("double", {1, 1, 1, 1, 1});
        const auto b = run_family<svf_kind, float, float>("float", {1, 1, 1, 1, 2});
        summaries.push_back(summary_json<svf_kind>(a));
        summaries.push_back(summary_json<svf_kind>(b));
    }
    {
        const auto a = run_family<biquad_kind, double, double>("double", {1, 1, 1, 1, 1});
        const auto b = run_family<biquad_kind, float, double>("float_double", {1, 1, 1, 1, 2});
        const auto c = run_family<biquad_kind, float, float>("float", {1e9, 1e9, 1e9, 1e9, 2});
        summaries.push_back(summary_json<biquad_kind>(a));
        summaries.push_back(summary_json<biquad_kind>(b));
        summaries.push_back(summary_json<biquad_kind>(c));
    }

    using wray::f32x4;
    using wray::f64x2;
    using f64x4 = wray::simd::vec<double, 4>;
    std::vector<parity_entry> parity;
    auto record = [&](const char *filter, const char *lanes, std::size_t n, bool same) {
        std::printf("parity %-13s %-24s %zu configs, bit-identical %d\n", filter, lanes, n, same ? 1 : 0);
        CHECK(same);
        parity.push_back({filter, lanes, n, same});
    };
    bool same = false;
    std::size_t n = parity_check<one_pole_kind, float, f32x4, float, f32x4>(same);
    record("tpt_one_pole", "f32x4 vs float", n, same);
    n = parity_check<one_pole_kind, double, f64x2, double, f64x2>(same);
    record("tpt_one_pole", "f64x2 vs double", n, same);
    n = parity_check<svf_kind, float, f32x4, float, f32x4>(same);
    record("svf", "f32x4 vs float", n, same);
    n = parity_check<svf_kind, double, f64x2, double, f64x2>(same);
    record("svf", "f64x2 vs double", n, same);
    n = parity_check<biquad_kind, float, f32x4, float, f32x4>(same);
    record("biquad", "f32x4 vs float", n, same);
    n = parity_check<biquad_kind, double, f64x2, double, f64x2>(same);
    record("biquad", "f64x2 vs double", n, same);
    n = parity_check<biquad_kind, float, f32x4, double, f64x4>(same);
    record("biquad", "f32x4/f64x4 vs float/double", n, same);

    if (json)
    {
        std::vector<chart_entry> charts;
        chart_data<one_pole_kind>(charts);
        chart_data<svf_kind>(charts);
        chart_data<biquad_kind>(charts);
        std::FILE *fp = std::fopen(json, "w");
        if (!fp)
        {
            std::printf("cannot write %s\n", json);
            return 1;
        }
        std::string s = "{\n\"fft_size\":" + std::to_string(fft_size) + ",\n\"floor_gain\":" + num(floor_gain) + ",\n\"nu\":" + list(nu_grid()) +
                        ",\n\"summaries\":[\n";
        for (std::size_t i = 0; i < summaries.size(); ++i)
            s += (i ? ",\n" : "") + summaries[i];
        s += "\n],\n\"parity\":[\n";
        for (std::size_t i = 0; i < parity.size(); ++i)
            s += std::string(i ? ",\n" : "") + "{\"filter\":\"" + parity[i].filter + "\",\"lanes\":\"" + parity[i].lanes + "\",\"configs\":" +
                 std::to_string(parity[i].configs) + ",\"bit_identical\":" + (parity[i].identical ? "true" : "false") + "}";
        s += "\n],\n\"charts\":[\n";
        for (std::size_t i = 0; i < charts.size(); ++i)
        {
            const auto &e = charts[i];
            s += std::string(i ? ",\n" : "") + "{\"filter\":\"" + e.filter + "\",\"config\":" + config_json(e.c, e.type.c_str()) + ",\"hz\":" +
                 list(e.hz) + ",\"response_f64\":" + complex_list(e.response_f64) + ",\"response_f32\":" + complex_list(e.response_f32) +
                 ",\"reference\":" + complex_list(e.reference) + ",\"measured_hz\":" + list(e.measured_hz) + ",\"measured_f64\":" +
                 complex_list(e.measured_f64) + ",\"analytic_f64\":" + complex_list(e.analytic_f64) + "}";
        }
        s += "\n]\n}\n";
        std::fputs(s.c_str(), fp);
        std::fclose(fp);
    }
    return wray::test::finish("filters");
}
