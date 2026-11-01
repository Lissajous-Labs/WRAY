#include <wray/wray.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.h"

#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace
{
    constexpr double margin = WRAY_HAS_FMA ? 0.0 : 0.5;

    using batch_f32 = std::function<void(const std::vector<float>&, std::vector<float>&)>;
    using batch_f64 = std::function<void(const std::vector<double>&, std::vector<double>&)>;

    double ulp_of(double ref, int precision, int emin)
    {
        if (ref == 0.0 || std::isinf(ref))
            return std::ldexp(1.0, emin - precision + 1);
        const int e = std::max(std::ilogb(ref), emin);
        return std::ldexp(1.0, e - precision + 1);
    }

    double limit_for(int precision)
    {
        return precision == 24 ? 0x1p128 * (1.0 - 0x1p-25) : std::numeric_limits<double>::infinity();
    }

    double error_ulp(double got, double ref, int precision, int emin)
    {
        if (std::isnan(ref))
            return std::isnan(got) ? 0.0 : std::numeric_limits<double>::infinity();
        if (std::isnan(got))
            return std::numeric_limits<double>::infinity();
        if (std::isinf(got))
        {
            const bool ok = std::fabs(ref) >= limit_for(precision) && std::signbit(got) == std::signbit(ref);
            return ok || got == ref ? 0.0 : std::numeric_limits<double>::infinity();
        }
        return std::fabs(got - ref) / ulp_of(ref, precision, emin);
    }

    double error_rel(double got, double ref)
    {
        if (std::isnan(ref))
            return std::isnan(got) ? 0.0 : std::numeric_limits<double>::infinity();
        if (ref == 0.0)
            return got == 0.0 ? 0.0 : std::fabs(got);
        if (std::isinf(ref) || std::isinf(got))
            return got == ref ? 0.0 : std::numeric_limits<double>::infinity();
        return std::fabs(got - ref) / std::fabs(ref);
    }

    struct unary_case
    {
        const char* name;
        std::function<double(double)> reference;
        std::function<bool(float)> domain;
        batch_f32 precise;
        batch_f32 fast;
        std::function<bool(float)> fast_domain;
        batch_f64 precise64;
        std::function<double(std::mt19937_64&)> sample64;
        double contract_f32;
        double contract_fast;
        double contract_f64;
    };

    template <class F>
    batch_f32 batch32(F f)
    {
        return [f](const std::vector<float>& in, std::vector<float>& out) { wray::math::apply(in, out, f); };
    }

    template <class F>
    batch_f64 batch64(F f)
    {
        return [f](const std::vector<double>& in, std::vector<double>& out) { wray::math::apply(in, out, f); };
    }

    bool finite(float x) { return std::isfinite(x); }
    bool all(float) { return true; }
    bool positive_normal(float x) { return x >= 0x1p-126f && std::isfinite(x); }

    double decibels(double x)
    {
        constexpr double c = 0.16609640474436813;
        constexpr double c_lo = -8.345257784509387e-18;
        const double hi = x * c;
        const double lo = std::fma(x, c, -hi) + x * c_lo;
        if (std::fabs(hi) > 4096.0)
            return std::exp2(hi);
        return std::exp2(hi) * (1.0 + lo * 0.6931471805599453);
    }

    double sin_turns_ref(double x)
    {
        const double w = x - std::nearbyint(x);
        const double a = std::fabs(w) > 0.25 ? 0.5 - std::fabs(w) : std::fabs(w);
        return std::copysign(std::sin(6.283185307179586 * a), w);
    }

    double cos_turns_ref(double x)
    {
        return std::sin(6.283185307179586 * (0.25 - std::fabs(x - std::nearbyint(x))));
    }

    double tan_turns_ref(double x)
    {
        const double w = x - std::nearbyint(x);
        if (std::fabs(w) == 0.5)
            return std::copysign(std::numeric_limits<double>::infinity(), x);
        if (std::fabs(w) > 0.25)
            return -1.0 / std::tan(3.141592653589793 * (w - std::copysign(0.5, w)));
        return std::tan(3.141592653589793 * w);
    }

    double uniform(std::mt19937_64& g, double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }

    double bit_uniform(std::mt19937_64& g, double lo, double hi)
    {
        const auto a = std::bit_cast<std::uint64_t>(lo);
        const auto b = std::bit_cast<std::uint64_t>(hi);
        return std::bit_cast<double>(std::uniform_int_distribution<std::uint64_t>(a, b)(g));
    }

    double signed_bits(std::mt19937_64& g, double lo, double hi)
    {
        const double v = bit_uniform(g, lo, hi);
        return (g() & 1) ? -v : v;
    }

    std::vector<unary_case> cases()
    {
        constexpr double trig32 = 0x1.921fb6p+12;
        constexpr double trig64 = 0x1.921fb54442d18p+22;
        return {
            {"cosh", [](double x) { return std::cosh(x); }, all, batch32([](auto v) { return wray::cosh(v); }), {}, all,
             batch64([](auto v) { return wray::cosh(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -3.0, 3.0) : signed_bits(g, 0x1p-1074, 711.0); }, 0.9, 0.0, 0.9},
            {"sinh", [](double x) { return std::sinh(x); }, all, batch32([](auto v) { return wray::sinh(v); }), {}, all,
             batch64([](auto v) { return wray::sinh(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -3.0, 3.0) : signed_bits(g, 0x1p-1074, 711.0); }, 0.85, 0.0, 0.85},
            {"tan_turns", tan_turns_ref, all, batch32([](auto v) { return wray::tan_turns(v); }), batch32([](auto v) { return wray::fast::tan_turns(v); }), all,
             batch64([](auto v) { return wray::tan_turns(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -1.0, 1.0) : signed_bits(g, 1e-300, 1e300); }, 1.2, 0x1p-16, 2.0},
            {"cos_turns", cos_turns_ref, all, batch32([](auto v) { return wray::cos_turns(v); }), batch32([](auto v) { return wray::fast::cos_turns(v); }), all,
             batch64([](auto v) { return wray::cos_turns(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -1.0, 1.0) : signed_bits(g, 1e-300, 1e300); }, 0.8, 0x1p-16, 0.8},
            {"sin_turns", sin_turns_ref, all, batch32([](auto v) { return wray::sin_turns(v); }), batch32([](auto v) { return wray::fast::sin_turns(v); }), all,
             batch64([](auto v) { return wray::sin_turns(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -1.0, 1.0) : signed_bits(g, 1e-300, 1e300); }, 1.0, 0x1p-16, 0.85},
            {"log10", [](double x) { return std::log10(x); }, all, batch32([](auto v) { return wray::log10(v); }), {}, all,
             batch64([](auto v) { return wray::log10(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, 0.5, 20.0) : bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.4, 0.0, 1.0},
            {"log1p", [](double x) { return std::log1p(x); }, all, batch32([](auto v) { return wray::log1p(v); }), {}, all,
             batch64([](auto v) { return wray::log1p(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -1.0, 4.0) : bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.0, 0.0, 0.95},
            {"expm1", [](double x) { return std::expm1(x); }, all, batch32([](auto v) { return wray::expm1(v); }), {}, all,
             batch64([](auto v) { return wray::expm1(v); }), [](std::mt19937_64& g) { return g() & 1 ? uniform(g, -745.0, 709.78) : signed_bits(g, 0x1p-1074, 2.0); },
             2.0, 0.0, 1.75},
            {"exp", [](double x) { return std::exp(x); }, all, batch32([](auto v) { return wray::exp(v); }), batch32([](auto v) { return wray::fast::exp(v); }),
             [](float x) { return x >= -87.0f && x <= 88.0f; }, batch64([](auto v) { return wray::exp(v); }),
             [](std::mt19937_64& g) { return uniform(g, -745.0, 709.78); }, 1.0, 0x1p-16, 1.0},
            {"exp2", [](double x) { return std::exp2(x); }, all, batch32([](auto v) { return wray::exp2(v); }), batch32([](auto v) { return wray::fast::exp2(v); }),
             [](float x) { return x >= -126.0f && x <= 127.0f; }, batch64([](auto v) { return wray::exp2(v); }),
             [](std::mt19937_64& g) { return uniform(g, -1075.0, 1023.99); }, 1.0, 0x1p-16, 1.0},
            {"log", [](double x) { return std::log(x); }, all, batch32([](auto v) { return wray::log(v); }), batch32([](auto v) { return wray::fast::log(v); }),
             positive_normal, batch64([](auto v) { return wray::log(v); }),
             [](std::mt19937_64& g) { return bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.0, 0x1p-16, 1.0},
            {"log2", [](double x) { return std::log2(x); }, all, batch32([](auto v) { return wray::log2(v); }), batch32([](auto v) { return wray::fast::log2(v); }),
             positive_normal, batch64([](auto v) { return wray::log2(v); }),
             [](std::mt19937_64& g) { return bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.5, 0x1p-16, 1.0},
            {"sin", [](double x) { return std::sin(x); }, [](float x) { return std::fabs(x) <= float(trig32); }, batch32([](auto v) { return wray::sin(v); }),
             batch32([](auto v) { return wray::fast::sin(v); }), [](float x) { return std::fabs(x) <= float(trig32); }, batch64([](auto v) { return wray::sin(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 1e-300, trig64); }, 1.0, 0x1p-16, 1.0},
            {"cos", [](double x) { return std::cos(x); }, [](float x) { return std::fabs(x) <= float(trig32); }, batch32([](auto v) { return wray::cos(v); }),
             batch32([](auto v) { return wray::fast::cos(v); }), [](float x) { return std::fabs(x) <= float(trig32); }, batch64([](auto v) { return wray::cos(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 1e-300, trig64); }, 1.0, 0x1p-16, 1.0},
            {"tan", [](double x) { return std::tan(x); }, [](float x) { return std::fabs(x) <= float(trig32); }, batch32([](auto v) { return wray::tan(v); }),
             batch32([](auto v) { return wray::fast::tan(v); }), [](float x) { return std::fabs(x) <= float(trig32); }, batch64([](auto v) { return wray::tan(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 1e-300, trig64); }, 1.75, 0x1p-16, 2.5},
            {"atan", [](double x) { return std::atan(x); }, all, batch32([](auto v) { return wray::atan(v); }), batch32([](auto v) { return wray::fast::atan(v); }),
             finite, batch64([](auto v) { return wray::atan(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.0, 0x1p-16, 1.1},
            {"tanh", [](double x) { return std::tanh(x); }, all, batch32([](auto v) { return wray::tanh(v); }), batch32([](auto v) { return wray::fast::tanh(v); }),
             finite, batch64([](auto v) { return wray::tanh(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 0x1p-1074, 40.0); }, 1.35, 0x1p-16, 1.3},
            {"sigmoid", [](double x) { return x < 0.0 ? std::exp(x) / (1.0 + std::exp(x)) : 1.0 / (1.0 + std::exp(-x)); }, all, batch32([](auto v) { return wray::sigmoid(v); }),
             batch32([](auto v) { return wray::fast::sigmoid(v); }), [](float x) { return std::fabs(x) <= 87.0f; }, batch64([](auto v) { return wray::sigmoid(v); }),
             [](std::mt19937_64& g) { return uniform(g, -745.0, 745.0); }, 2.0, 0x1p-16, 1.75},
            {"db_to_gain", decibels, all, batch32([](auto v) { return wray::db_to_gain(v); }),
             batch32([](auto v) { return wray::fast::db_to_gain(v); }), [](float x) { return std::fabs(x) <= 750.0f; }, batch64([](auto v) { return wray::db_to_gain(v); }),
             [](std::mt19937_64& g) { return uniform(g, -6400.0, 6100.0); }, 1.5, 0x1p-16, 1.5},
            {"gain_to_db", [](double x) { return 20.0 * std::log10(x); }, all, batch32([](auto v) { return wray::gain_to_db(v); }),
             batch32([](auto v) { return wray::fast::gain_to_db(v); }), positive_normal, batch64([](auto v) { return wray::gain_to_db(v); }),
             [](std::mt19937_64& g) { return bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.5, 0x1p-16, 1.5},
        };
    }

    enum class host
    {
        nearest,
        upward,
        downward,
        toward_zero,
        flush
    };

    constexpr host hosts[] = {host::nearest, host::upward, host::downward, host::toward_zero, host::flush};
    constexpr const char* host_names[] = {"nearest", "upward", "downward", "toward_zero", "flush"};

    void set_flush(bool on)
    {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        std::uint64_t fpcr = 0;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr = on ? (fpcr | (std::uint64_t(1) << 24)) : (fpcr & ~(std::uint64_t(1) << 24));
        asm volatile("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__SSE2__) || defined(_M_X64)
        _mm_setcsr(on ? (_mm_getcsr() | 0x8040u) : (_mm_getcsr() & ~0x8040u));
#else
        (void)on;
#endif
    }

    void enter(host h)
    {
        const int modes[] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO, FE_TONEAREST};
        std::fesetround(modes[static_cast<int>(h)]);
        set_flush(h == host::flush);
    }

    template <class T>
    double host_error(T x, T got, double ref, bool relative, host h, int precision, int emin)
    {
        const double top = double(std::numeric_limits<T>::max());
        const double tiny = double(std::numeric_limits<T>::min());
        if (h == host::flush && (std::fabs(double(x)) < tiny || std::fabs(ref) < tiny) && x != T(0))
            return std::isnan(got) && !std::isnan(ref) ? std::numeric_limits<double>::infinity() : 0.0;
        if (h != host::nearest && h != host::flush && std::fabs(ref) > top && std::fabs(double(got)) == top && std::signbit(got) == std::signbit(ref))
            return 0.0;
        return relative ? error_rel(double(got), ref) : error_ulp(double(got), ref, precision, emin);
    }

    struct result
    {
        double worst = 0.0;
        float worst_x = 0.0f;
    };

    result sweep_f32(const unary_case& c, const batch_f32& fn, bool relative, std::uint64_t stride, const std::function<bool(float)>& domain,
                     host h = host::nearest)
    {
        if (!fn)
            return {};
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<result> partial(threads);
        std::vector<std::thread> pool;
        const std::uint64_t total = (1ull << 32) / stride;
        for (unsigned t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t] {
                constexpr std::size_t block = 4096;
                std::vector<float> in;
                std::vector<float> out(block);
                in.reserve(block);
                result& r = partial[t];
                for (std::uint64_t i = t * block; i < total; i += std::uint64_t(threads) * block)
                {
                    in.clear();
                    for (std::uint64_t j = i; j < std::min(total, i + block); ++j)
                    {
                        const float x = std::bit_cast<float>(static_cast<std::uint32_t>(j * stride));
                        const bool subnormal = x != 0.0f && std::fabs(x) < 0x1p-126f;
                        if (domain(x) && !(relative && subnormal))
                            in.push_back(x);
                    }
                    out.resize(in.size());
                    enter(h);
                    fn(in, out);
                    enter(host::nearest);
                    for (std::size_t k = 0; k < in.size(); ++k)
                    {
                        const double e = host_error(in[k], out[k], c.reference(double(in[k])), relative, h, 24, -126);
                        if (e > r.worst)
                        {
                            r.worst = e;
                            r.worst_x = in[k];
                        }
                    }
                }
            });
        }
        for (auto& th : pool)
            th.join();
        result best;
        for (const auto& r : partial)
            if (r.worst > best.worst)
                best = r;
        return best;
    }

    double sweep_f64(const unary_case& c, std::size_t samples, double& worst_x, host h = host::nearest)
    {
        std::mt19937_64 g(42);
        std::vector<double> in(samples);
        std::vector<double> out(samples);
        for (auto& x : in)
            x = c.sample64(g);
        enter(h);
        c.precise64(in, out);
        enter(host::nearest);
        double worst = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double e = host_error(in[i], out[i], c.reference(in[i]), false, h, 53, -1022);
            if (e > worst)
            {
                worst = e;
                worst_x = in[i];
            }
        }
        return worst;
    }

    template <class T>
    T smaller(T a, T b)
    {
        return a != T(0) && (b == T(0) || std::fabs(a) < std::fabs(b)) ? a : b;
    }

    void binary_checks(std::size_t samples, bool report, bool with_atan2, bool with_pow, host h = host::nearest)
    {
        const double slack = h == host::nearest || h == host::flush ? 0.0 : 1.0 + margin;
        const char* label = host_names[static_cast<int>(h)];
        std::mt19937_64 g(9);
        std::vector<float> y(samples), x(samples), out(samples), fout(samples);
        std::vector<double> yd(samples), xd(samples), outd(samples);
        for (std::size_t i = 0; i < samples; ++i)
        {
            y[i] = float(signed_bits(g, 0x1p-149, 0x1.fffffep+127));
            x[i] = float(signed_bits(g, 0x1p-149, 0x1.fffffep+127));
            yd[i] = signed_bits(g, 0x1p-1074, 0x1.fffffffffffffp+1023);
            xd[i] = signed_bits(g, 0x1p-1074, 0x1.fffffffffffffp+1023);
        }
        enter(h);
        wray::math::apply(y, x, out, [](auto a, auto b) { return wray::atan2(a, b); });
        wray::math::apply(y, x, fout, [](auto a, auto b) { return wray::fast::atan2(a, b); });
        wray::math::apply(yd, xd, outd, [](auto a, auto b) { return wray::atan2(a, b); });
        enter(host::nearest);
        double w32 = 0.0, wf = 0.0, w64 = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double r = std::atan2(double(y[i]), double(x[i]));
            w32 = std::max(w32, host_error(smaller(y[i], x[i]), out[i], r, false, h, 24, -126));
            const double af = std::fabs(double(fout[i]) - r);
            wf = std::max(wf, std::min(host_error(smaller(y[i], x[i]), fout[i], r, true, h, 24, -126), af));
            w64 = std::max(w64, host_error(smaller(yd[i], xd[i]), outd[i], std::atan2(yd[i], xd[i]), false, h, 53, -1022));
        }
        if (report && with_atan2)
            std::printf("%-11s f32 %8.3f ulp   fast %.3e   f64 %8.3f ulp (vs libm) %s\n", "atan2", w32, wf, w64, label);
        CHECK(!with_atan2 || w32 <= 1.0 + margin + slack);
        CHECK(!with_atan2 || wf <= 0x1p-16 + slack * 0x1p-23);
        CHECK(!with_atan2 || w64 <= 2.5 + margin + slack);

        for (std::size_t i = 0; i < samples; ++i)
        {
            x[i] = float(bit_uniform(g, 1e-6, 1e6));
            y[i] = float(uniform(g, -6.0, 6.0));
            xd[i] = bit_uniform(g, 1e-60, 1e60);
            yd[i] = uniform(g, -5.0, 5.0);
        }
        enter(h);
        wray::math::apply(x, y, out, [](auto a, auto b) { return wray::pow(a, b); });
        wray::math::apply(x, y, fout, [](auto a, auto b) { return wray::fast::pow(a, b); });
        wray::math::apply(xd, yd, outd, [](auto a, auto b) { return wray::pow(a, b); });
        enter(host::nearest);
        w32 = wf = w64 = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double r = std::pow(double(x[i]), double(y[i]));
            w32 = std::max(w32, host_error(x[i], out[i], r, false, h, 24, -126));
            const double z = std::fabs(double(y[i]) * std::log2(double(x[i])));
            if (z < 120.0)
                wf = std::max(wf, host_error(x[i], fout[i], r, true, h, 24, -126) / (1.0 + z));
            w64 = std::max(w64, host_error(xd[i], outd[i], std::pow(xd[i], yd[i]), false, h, 53, -1022));
        }
        if (report && with_pow)
            std::printf("%-11s f32 %8.3f ulp   fast %.3e   f64 %8.3f ulp (vs libm) %s\n", "pow", w32, wf, w64, label);
        CHECK(!with_pow || w32 <= 2.0 + margin + slack);
        CHECK(!with_pow || wf <= 0x1p-16 + slack * 0x1p-23);
        CHECK(!with_pow || w64 <= 2.5 + margin + slack);
    }

    void host_checks(const std::function<bool(const char*)>& selected, std::uint64_t stride, std::size_t samples, bool report)
    {
        for (host h : hosts)
        {
            if (h == host::nearest)
                continue;
            const double slack = h == host::flush ? 0.0 : 1.0 + margin;
            const char* label = host_names[static_cast<int>(h)];
            for (const auto& c : cases())
            {
                if (!selected(c.name))
                    continue;
                const auto p = sweep_f32(c, c.precise, false, stride, c.domain, h);
                const auto f = sweep_f32(c, c.fast, true, stride, c.fast_domain, h);
                double wx = 0.0;
                const double d = sweep_f64(c, samples, wx, h);
                if (report)
                    std::printf("%-11s f32 %8.3f ulp @ %-14.9g fast %.3e @ %-14.9g f64 %8.3f ulp @ %.17g %s\n", c.name, p.worst, double(p.worst_x), f.worst,
                                double(f.worst_x), d, wx, label);
                CHECK(p.worst <= c.contract_f32 + margin + slack);
                CHECK(f.worst <= c.contract_fast + slack * 0x1p-23);
                CHECK(d <= c.contract_f64 + 2.0 + margin + slack);
            }
            binary_checks(samples, report, selected("atan2"), selected("pow"), h);
        }
    }

    struct monotone
    {
        double lo;
        double hi;
        int dir;
    };

    enum class parity
    {
        none,
        odd,
        even
    };

    enum expectation
    {
        any,
        defined,
        bounded
    };

    struct shape
    {
        const char* name;
        parity symmetry;
        std::vector<monotone> monotonic;
        double lo;
        double hi;
        std::function<expectation(double, bool)> expect;
        double core;
        std::vector<double> edges;
    };

    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double half_pi = 1.5707963267948966;
    constexpr double pi = 3.141592653589793;

    expectation everywhere(double x, bool) { return std::isnan(x) ? any : bounded; }

    expectation below(double x, double limit) { return std::isnan(x) ? any : x < limit ? bounded : defined; }

    expectation finite_only(double x, bool) { return std::isfinite(x) ? bounded : any; }

    expectation logarithmic(double x, bool) { return x > 0.0 && x < inf ? bounded : x >= 0.0 ? defined : any; }

    expectation periodic(double x, bool wide)
    {
        const double limit = wide ? wray::math::constants<double>::trig_arg_max : double(wray::math::constants<float>::trig_arg_max);
        return std::fabs(x) <= limit ? bounded : any;
    }

    std::vector<double> multiples(double step, double offset, int count)
    {
        std::vector<double> v;
        for (int k = -count; k <= count; ++k)
            v.push_back((k + offset) * step);
        return v;
    }

    std::vector<double> powers(double base)
    {
        std::vector<double> v{0x1p-1022, 0x1p-1074};
        for (int k = -1020; k <= 1020; k += 17)
            v.push_back(std::ldexp(base, k));
        return v;
    }

    std::vector<double> joined(std::vector<double> a, const std::vector<double>& b)
    {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    }

    std::vector<shape> shapes()
    {
        constexpr double ln2 = 0.6931471805599453;
        constexpr double db2 = 6.020599913279624;
        const std::vector<double> trig = joined(multiples(pi / 4.0, 0.0, 40), {1e5 * half_pi, 0x1.921fb54442d18p+22, 6433.98, 102943.0, 1e6});
        const std::vector<double> logs = joined(powers(1.0), powers(1.4142135623730951));
        return {
            {"exp", parity::none, {{-inf, inf, 1}}, 0.0, inf, [](double x, bool w) { return below(x, w ? 709.78 : 88.72); }, 20.0,
             joined(multiples(ln2, 0.5, 60), {86.0, 707.0, 88.72, 709.78, 745.1, 708.39, 103.97, 87.0})},
            {"log1p", parity::none, {{-1.0, inf, 1}}, -inf, inf, [](double x, bool) { return x > -1.0 && x < inf ? bounded : x >= -1.0 ? defined : any; }, 2.0, {-0.5, 0x1p-25, 0x1p-54, 1.0, 0.4142135623730951, -0.2928932188134524, 1e300}},
            {"log10", parity::none, {{0.0, inf, 1}}, -inf, inf, logarithmic, 4.0, logs},
            {"sin_turns", parity::odd, {{-0.25, 0.25, 1}}, -1.0, 1.0, finite_only, 1.0, joined(multiples(0.125, 0.0, 16), {1e6 + 0.25, 4194303.75, 0x1p-30})},
            {"cos_turns", parity::even, {{-0.5, 0.0, 1}, {0.0, 0.5, -1}}, -1.0, 1.0, finite_only, 1.0, joined(multiples(0.125, 0.0, 16), {1e6 + 0.25, 4194303.75, 0x1p-30})},
            {"tan_turns", parity::odd, {{-0.49999999999999994, 0.49999999999999994, 1}}, -inf, inf, [](double x, bool) { return std::isnan(x) || std::isinf(x) ? any : std::fabs(x - std::nearbyint(x)) == 0.5 ? defined : bounded; }, 1.0, joined(multiples(0.125, 0.0, 16), {0.25, 0.5, 1e6 + 0.25, 4194303.75, 0x1p-30})},
            {"sinh", parity::odd, {{-inf, inf, 1}}, -inf, inf, [](double x, bool w) { return std::fabs(x) < (w ? 710.47 : 89.41) ? bounded : std::isnan(x) ? any : defined; }, 3.0, {1.0, 0x1p-30, 86.0, 707.0, 88.72, 89.41, 709.78, 710.47, 0.3466, 1.0397}},
            {"cosh", parity::even, {{-inf, 0.0, -1}, {0.0, inf, 1}}, 1.0, inf, [](double x, bool w) { return std::fabs(x) < (w ? 710.47 : 89.41) ? bounded : std::isnan(x) ? any : defined; }, 3.0, {1.0, 0x1p-30, 86.0, 707.0, 88.72, 89.41, 709.78, 710.47, 0.3466, 1.0397}},
            {"expm1", parity::none, {{-inf, inf, 1}}, -1.0, inf, [](double x, bool w) { return below(x, w ? 709.78 : 88.72); }, 2.0,
             joined(multiples(ln2, 0.5, 60), {86.0, 707.0, 88.72, 709.78, 0x1p-25, 0x1p-54, 17.3, 37.5})},
            {"exp2", parity::none, {{-inf, inf, 1}}, 0.0, inf, [](double x, bool w) { return below(x, w ? 1024.0 : 128.0); }, 20.0,
             joined(multiples(1.0, 0.5, 60), {124.0, 1020.0, 1022.0, 1074.0, 126.0, 149.0})},
            {"log", parity::none, {{0.0, inf, 1}}, -inf, inf, logarithmic, 4.0, logs},
            {"log2", parity::none, {{0.0, inf, 1}}, -inf, inf, logarithmic, 4.0, logs},
            {"sin", parity::odd, {{-half_pi, half_pi, 1}}, -1.0, 1.0, periodic, 10.0, trig},
            {"cos", parity::even, {{-pi, 0.0, 1}, {0.0, pi, -1}}, -1.0, 1.0, periodic, 10.0, trig},
            {"tan", parity::odd, {{-half_pi, half_pi, 1}}, -inf, inf, periodic, 10.0, trig},
            {"atan", parity::odd, {{-inf, inf, 1}}, -half_pi, half_pi, everywhere, 4.0, {1.0, 0x1p-30, 0x1p30, 0x1p60, 1e300}},
            {"tanh", parity::odd, {{-inf, inf, 1}}, -1.0, 1.0, everywhere, 4.0,
             {double(wray::math::constants<double>::tanh_switch), double(wray::math::constants<float>::tanh_switch), 19.25, 9.01, 5.9, 0x1p-30}},
            {"sigmoid", parity::none, {{-inf, inf, 1}}, 0.0, 1.0, everywhere, 20.0, joined(multiples(ln2, 0.5, 30), {86.0, 707.0, 745.1, 103.97})},
            {"db_to_gain", parity::none, {{-inf, inf, 1}}, 0.0, inf, [](double x, bool w) { return below(x, w ? 6153.6 : 770.6); }, 120.0,
             joined(multiples(db2, 0.5, 60), {124.0 * db2, 1020.0 * db2, 1074.0 * db2, 149.0 * db2})},
            {"gain_to_db", parity::none, {{0.0, inf, 1}}, -inf, inf, logarithmic, 4.0, logs},
        };
    }

    struct tally
    {
        static constexpr const char* kinds[4] = {"monotonic", "parity", "bounds", "finite"};
        std::array<std::uint64_t, 4> count{};
        std::array<double, 4> where{};

        void add(std::size_t k, double x)
        {
            if (count[k]++ == 0)
                where[k] = x;
        }

        void merge(const tally& o)
        {
            for (std::size_t k = 0; k < 4; ++k)
            {
                if (o.count[k] && !count[k])
                    where[k] = o.where[k];
                count[k] += o.count[k];
            }
        }

        std::uint64_t total() const { return count[0] + count[1] + count[2] + count[3]; }
    };

    template <class T>
    using bits_of = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;

    template <class T>
    struct shape_runner
    {
        const shape& s;
        std::function<void(const std::vector<T>&, std::vector<T>&)> fn;
        std::function<bool(T)> domain;
        std::vector<T> xs, neg, ys, yn;

        bool same(T a, T b) const { return (std::isnan(a) && std::isnan(b)) || std::bit_cast<bits_of<T>>(a) == std::bit_cast<bits_of<T>>(b); }

        void point(T x, T y, tally& t) const
        {
            if (!domain(x))
                return;
            const expectation e = s.expect(double(x), sizeof(T) == 8);
            if ((e != any && std::isnan(y)) || (e == bounded && !std::isfinite(y)))
                t.add(3, double(x));
            if (!std::isnan(y) && (y < T(s.lo) || y > T(s.hi)))
                t.add(2, double(x));
        }

        void step(T a, T b, T fa, T fb, tally& t) const
        {
            if (!domain(a) || !domain(b))
                return;
            for (const auto& m : s.monotonic)
                if (double(a) >= m.lo && double(b) <= m.hi && !(m.dir > 0 ? fb >= fa : fb <= fa))
                    t.add(0, double(a));
        }

        void run(T start, std::size_t length, tally& t)
        {
            const bits_of<T> first = std::bit_cast<bits_of<T>>(start);
            const bits_of<T> last = std::bit_cast<bits_of<T>>(std::numeric_limits<T>::infinity());
            xs.clear();
            for (bits_of<T> o = first; xs.size() < length && o <= last; ++o)
                xs.push_back(std::bit_cast<T>(o));
            const std::size_t n = xs.size();
            neg.resize(n);
            ys.resize(n);
            yn.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                neg[i] = -xs[i];
            fn(xs, ys);
            fn(neg, yn);
            for (std::size_t i = 0; i < n; ++i)
            {
                point(xs[i], ys[i], t);
                point(neg[i], yn[i], t);
                if (domain(xs[i]) && ((s.symmetry == parity::odd && !same(yn[i], -ys[i])) || (s.symmetry == parity::even && !same(yn[i], ys[i]))))
                    t.add(1, double(xs[i]));
                if (i + 1 < n)
                {
                    step(xs[i], xs[i + 1], ys[i], ys[i + 1], t);
                    step(neg[i + 1], neg[i], yn[i + 1], yn[i], t);
                }
            }
            if (first == 0 && n > 0)
                step(neg[0], xs[0], yn[0], ys[0], t);
        }
    };

    template <class T, class Start>
    tally shape_parallel(const shape& s, const std::function<void(const std::vector<T>&, std::vector<T>&)>& fn, const std::function<bool(T)>& domain,
                         std::uint64_t runs, std::size_t length, Start start)
    {
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<tally> partial(threads);
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
                shape_runner<T> r{s, fn, domain, {}, {}, {}, {}};
                for (std::uint64_t i = t; i < runs; i += threads)
                    r.run(start(i), length, partial[t]);
            });
        for (auto& th : pool)
            th.join();
        tally all;
        for (const auto& p : partial)
            all.merge(p);
        return all;
    }

    tally shape_f32(const shape& s, const batch_f32& fn, const std::function<bool(float)>& domain, std::uint64_t stride)
    {
        if (!fn)
            return {};
        constexpr std::size_t run = 4096;
        const std::uint64_t runs = (0x7f800000ull / run + 1 + stride - 1) / stride;
        return shape_parallel<float>(s, fn, domain, runs, run + 1,
                                     [stride](std::uint64_t i) { return std::bit_cast<float>(static_cast<std::uint32_t>(i * stride * run)); });
    }

    double before(double x, int k)
    {
        const auto b = std::bit_cast<std::uint64_t>(std::fabs(x));
        return std::bit_cast<double>(b > std::uint64_t(k) ? b - std::uint64_t(k) : 0);
    }

    tally shape_f64(const shape& s, const unary_case& c, std::size_t samples)
    {
        std::mt19937_64 g(17);
        std::vector<double> starts;
        for (std::size_t i = 0; i < samples; ++i)
            starts.push_back(before(i % 2 ? c.sample64(g) : uniform(g, -s.core, s.core), 8));
        const std::size_t sampled = starts.size();
        for (double e : s.edges)
            starts.push_back(before(e, 64));
        const std::function<bool(double)> all64 = [](double) { return true; };
        tally t = shape_parallel<double>(s, c.precise64, all64, sampled, 17, [&](std::uint64_t i) { return starts[i]; });
        t.merge(shape_parallel<double>(s, c.precise64, all64, starts.size() - sampled, 129, [&](std::uint64_t i) { return starts[sampled + i]; }));
        return t;
    }

    struct exemption
    {
        const char* name;
        const char* tier;
        const char* kind;
        const char* reason;
        bool without_fma = false;
    };

    constexpr exemption exemptions[] = {
        {"atan", "f64", "monotonic", "near |x| = 1 one input step moves atan by half an ulp, the same size as the rounding of the x^5 tail"},
        {"tanh", "fast", "monotonic", "a monotone saturation needs the exp form of wray::tanh, which costs as much as the precise tier"},
        {"atan", "f32", "monotonic", "without FMA the core's fused steps round twice", true},
        {"atan", "fast", "monotonic", "without FMA the core's fused steps round twice", true},
        {"atan2", "f32", "monotonic", "without FMA the quotient residual is not exact", true},
        {"atan2", "f64", "monotonic", "without FMA the quotient residual is not exact", true},
    };

    bool shape_report(const char* name, const char* tier, const tally& t, bool report)
    {
        bool ok = true;
        if (report || t.total())
            std::printf("%-11s %-4s shape%s\n", name, tier, t.total() ? "" : "  ok");
        for (std::size_t k = 0; k < 4; ++k)
        {
            if (!t.count[k])
                continue;
            const auto e = std::ranges::find_if(exemptions, [&](const exemption& x) {
                return std::strcmp(x.name, name) == 0 && std::strcmp(x.tier, tier) == 0 && std::strcmp(x.kind, tally::kinds[k]) == 0;
            });
            const bool exempt = e != std::end(exemptions) && (!e->without_fma || !WRAY_HAS_FMA);
            ok = ok && exempt;
            std::printf("    %s %llu @ %.17g%s%s\n", tally::kinds[k], static_cast<unsigned long long>(t.count[k]), t.where[k], exempt ? "  exempt: " : "",
                        exempt ? e->reason : "");
        }
        return ok;
    }

    void shape_checks(const std::function<bool(const char*)>& selected, std::uint64_t stride, std::size_t samples, bool report)
    {
        const auto all_cases = cases();
        for (const auto& s : shapes())
        {
            if (!selected(s.name))
                continue;
            const auto c = std::ranges::find_if(all_cases, [&](const unary_case& u) { return std::strcmp(u.name, s.name) == 0; });
            CHECK(shape_report(s.name, "f32", shape_f32(s, c->precise, all, stride), report));
            CHECK(shape_report(s.name, "fast", shape_f32(s, c->fast, c->fast_domain, stride), report));
            CHECK(shape_report(s.name, "f64", shape_f64(s, *c, samples), report));
        }
    }

    template <class T, class F>
    tally binary_shape(F f, std::size_t samples, bool fast_tier)
    {
        std::mt19937_64 g(23);
        const T top = std::numeric_limits<T>::max();
        const T pi_t = T(pi);
        tally t;
        std::vector<T> y(16), x(16), ny(16), r(16), rn(16);
        for (std::size_t i = 0; i < samples; ++i)
        {
            const T x0 = T(signed_bits(g, double(std::numeric_limits<T>::denorm_min()), double(top)));
            T y0 = T(bit_uniform(g, double(std::numeric_limits<T>::denorm_min()), double(top)));
            if (fast_tier)
                y0 = T(bit_uniform(g, 0x1p-100, 0x1p100));
            for (std::size_t k = 0; k < 16; ++k)
            {
                y[k] = std::bit_cast<T>(bits_of<T>(std::bit_cast<bits_of<T>>(y0) + bits_of<T>(k)));
                ny[k] = -y[k];
                x[k] = x0;
            }
            wray::math::apply(y, x, r, f);
            wray::math::apply(ny, x, rn, f);
            for (std::size_t k = 0; k < 16; ++k)
            {
                if (!std::isfinite(r[k]))
                    t.add(3, double(y[k]));
                if (std::fabs(r[k]) > pi_t)
                    t.add(2, double(y[k]));
                if (std::bit_cast<bits_of<T>>(rn[k]) != std::bit_cast<bits_of<T>>(T(-r[k])))
                    t.add(1, double(y[k]));
                if (k > 0 && (x0 > T(0) ? r[k] < r[k - 1] : r[k] > r[k - 1]))
                    t.add(0, double(y[k]));
            }
        }
        return t;
    }

    template <class T, class F>
    tally power_shape(F f, std::size_t samples, bool fast_tier)
    {
        std::mt19937_64 g(29);
        constexpr double top = sizeof(T) == 8 ? 1023.0 : 127.0;
        tally t;
        std::vector<T> x(16), y(16), r(16);
        for (std::size_t i = 0; i < samples; ++i)
        {
            const T x0 = T(bit_uniform(g, fast_tier ? 0x1p-60 : double(std::numeric_limits<T>::denorm_min()), fast_tier ? 0x1p60 : double(std::numeric_limits<T>::max())));
            const T y0 = T(uniform(g, -2.0 * top, 2.0 * top) / std::max(1.0, std::fabs(std::log2(double(x0)))));
            for (std::size_t k = 0; k < 16; ++k)
            {
                x[k] = std::bit_cast<T>(bits_of<T>(std::bit_cast<bits_of<T>>(x0) + bits_of<T>(k)));
                y[k] = y0;
            }
            wray::math::apply(x, y, r, f);
            for (std::size_t k = 0; k < 16; ++k)
            {
                const double z = std::fabs(double(y0) * std::log2(double(x[k])));
                if (std::isnan(r[k]) || (z < top - 1.0 && !std::isfinite(r[k])))
                    t.add(3, double(x[k]));
                if (r[k] < T(0))
                    t.add(2, double(x[k]));
                if (k > 0 && (y0 > T(0) ? r[k] < r[k - 1] : r[k] > r[k - 1]))
                    t.add(0, double(x[k]));
            }
        }
        return t;
    }

    void binary_shape_checks(std::size_t samples, bool report, bool with_atan2, bool with_pow)
    {
        if (with_atan2)
        {
            CHECK(shape_report("atan2", "f32", binary_shape<float>([](auto a, auto b) { return wray::atan2(a, b); }, samples, false), report));
            CHECK(shape_report("atan2", "fast", binary_shape<float>([](auto a, auto b) { return wray::fast::atan2(a, b); }, samples, true), report));
            CHECK(shape_report("atan2", "f64", binary_shape<double>([](auto a, auto b) { return wray::atan2(a, b); }, samples, false), report));
        }
        if (with_pow)
        {
            CHECK(shape_report("pow", "f32", power_shape<float>([](auto a, auto b) { return wray::pow(a, b); }, samples, false), report));
            CHECK(shape_report("pow", "fast", power_shape<float>([](auto a, auto b) { return wray::fast::pow(a, b); }, samples, true), report));
            CHECK(shape_report("pow", "f64", power_shape<double>([](auto a, auto b) { return wray::pow(a, b); }, samples, false), report));
        }
    }
}

int main(int argc, char** argv)
{
    std::uint64_t stride = 61;
    std::size_t samples64 = 1u << 22;
    bool report = false;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--only") == 0)
            while (i + 1 < argc && argv[i + 1][0] != '-')
                only.emplace_back(argv[++i]);
        if (std::strcmp(argv[i], "--exhaustive") == 0)
        {
            stride = 1;
            samples64 = 1u << 26;
        }
        if (std::strcmp(argv[i], "--report") == 0)
            report = true;
    }
    const auto selected = [&only](const char* name) { return only.empty() || std::ranges::find(only, name) != only.end(); };
    for (const auto& c : cases())
    {
        if (!selected(c.name))
            continue;
        const auto p = sweep_f32(c, c.precise, false, stride, c.domain);
        const auto f = sweep_f32(c, c.fast, true, stride, c.fast_domain);
        double wx = 0.0;
        const double d = sweep_f64(c, samples64, wx);
        if (report)
            std::printf("%-11s f32 %8.3f ulp @ %-14.9g fast %.3e @ %-14.9g f64 %8.3f ulp @ %.17g (vs libm)\n", c.name, p.worst, double(p.worst_x), f.worst,
                        double(f.worst_x), d, wx);
        CHECK(p.worst <= c.contract_f32 + margin);
        CHECK(f.worst <= c.contract_fast);
        CHECK(d <= c.contract_f64 + 2.0 + margin);
    }
    binary_checks(1u << 22, report, selected("atan2"), selected("pow"));
    shape_checks(selected, stride, samples64 / 16, report);
    binary_shape_checks(1u << 16, report, selected("atan2"), selected("pow"));
    host_checks(selected, stride == 1 ? 4 : stride * 16, stride == 1 ? samples64 / 8 : samples64 / 32, report);
    return wray::test::finish("accuracy");
}
