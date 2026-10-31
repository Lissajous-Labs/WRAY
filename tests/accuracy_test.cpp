#include <wray/wray.h>

#include <algorithm>
#include <bit>
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
             [](std::mt19937_64& g) { return signed_bits(g, 1e-300, trig64); }, 2.5, 0x1p-16, 2.5},
            {"atan", [](double x) { return std::atan(x); }, all, batch32([](auto v) { return wray::atan(v); }), batch32([](auto v) { return wray::fast::atan(v); }),
             finite, batch64([](auto v) { return wray::atan(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 2.0, 0x1p-16, 1.5},
            {"tanh", [](double x) { return std::tanh(x); }, all, batch32([](auto v) { return wray::tanh(v); }), batch32([](auto v) { return wray::fast::tanh(v); }),
             finite, batch64([](auto v) { return wray::tanh(v); }),
             [](std::mt19937_64& g) { return signed_bits(g, 0x1p-1074, 40.0); }, 1.5, 0x1p-16, 2.5},
            {"sigmoid", [](double x) { return x < 0.0 ? std::exp(x) / (1.0 + std::exp(x)) : 1.0 / (1.0 + std::exp(-x)); }, all, batch32([](auto v) { return wray::sigmoid(v); }),
             batch32([](auto v) { return wray::fast::sigmoid(v); }), [](float x) { return std::fabs(x) <= 87.0f; }, batch64([](auto v) { return wray::sigmoid(v); }),
             [](std::mt19937_64& g) { return uniform(g, -745.0, 745.0); }, 2.5, 0x1p-16, 2.0},
            {"db_to_gain", decibels, all, batch32([](auto v) { return wray::db_to_gain(v); }),
             batch32([](auto v) { return wray::fast::db_to_gain(v); }), [](float x) { return std::fabs(x) <= 750.0f; }, batch64([](auto v) { return wray::db_to_gain(v); }),
             [](std::mt19937_64& g) { return uniform(g, -6400.0, 6100.0); }, 1.5, 0x1p-16, 1.5},
            {"gain_to_db", [](double x) { return 20.0 * std::log10(x); }, all, batch32([](auto v) { return wray::gain_to_db(v); }),
             batch32([](auto v) { return wray::fast::gain_to_db(v); }), positive_normal, batch64([](auto v) { return wray::gain_to_db(v); }),
             [](std::mt19937_64& g) { return bit_uniform(g, 0x1p-1074, 0x1.fffffffffffffp+1023); }, 1.5, 0x1p-16, 1.5},
        };
    }

    struct result
    {
        double worst = 0.0;
        float worst_x = 0.0f;
    };

    result sweep_f32(const unary_case& c, const batch_f32& fn, bool relative, std::uint64_t stride, const std::function<bool(float)>& domain)
    {
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
                    fn(in, out);
                    for (std::size_t k = 0; k < in.size(); ++k)
                    {
                        const double ref = c.reference(double(in[k]));
                        const double e = relative ? error_rel(double(out[k]), ref) : error_ulp(double(out[k]), ref, 24, -126);
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

    double sweep_f64(const unary_case& c, std::size_t samples, double& worst_x)
    {
        std::mt19937_64 g(42);
        std::vector<double> in(samples);
        std::vector<double> out(samples);
        for (auto& x : in)
            x = c.sample64(g);
        c.precise64(in, out);
        double worst = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double e = error_ulp(out[i], c.reference(in[i]), 53, -1022);
            if (e > worst)
            {
                worst = e;
                worst_x = in[i];
            }
        }
        return worst;
    }

    void binary_checks(std::size_t samples, bool report, bool with_atan2, bool with_pow)
    {
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
        wray::math::apply(y, x, out, [](auto a, auto b) { return wray::atan2(a, b); });
        wray::math::apply(y, x, fout, [](auto a, auto b) { return wray::fast::atan2(a, b); });
        wray::math::apply(yd, xd, outd, [](auto a, auto b) { return wray::atan2(a, b); });
        double w32 = 0.0, wf = 0.0, w64 = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double r = std::atan2(double(y[i]), double(x[i]));
            w32 = std::max(w32, error_ulp(double(out[i]), r, 24, -126));
            const double af = std::fabs(double(fout[i]) - r);
            wf = std::max(wf, std::min(error_rel(double(fout[i]), r), af));
            w64 = std::max(w64, error_ulp(outd[i], std::atan2(yd[i], xd[i]), 53, -1022));
        }
        if (report && with_atan2)
            std::printf("%-11s f32 %8.3f ulp   fast %.3e   f64 %8.3f ulp (vs libm)\n", "atan2", w32, wf, w64);
        CHECK(!with_atan2 || w32 <= 2.0 + margin);
        CHECK(!with_atan2 || wf <= 0x1p-16);
        CHECK(!with_atan2 || w64 <= 2.5 + margin);

        for (std::size_t i = 0; i < samples; ++i)
        {
            x[i] = float(bit_uniform(g, 1e-6, 1e6));
            y[i] = float(uniform(g, -6.0, 6.0));
            xd[i] = bit_uniform(g, 1e-60, 1e60);
            yd[i] = uniform(g, -5.0, 5.0);
        }
        wray::math::apply(x, y, out, [](auto a, auto b) { return wray::pow(a, b); });
        wray::math::apply(x, y, fout, [](auto a, auto b) { return wray::fast::pow(a, b); });
        wray::math::apply(xd, yd, outd, [](auto a, auto b) { return wray::pow(a, b); });
        w32 = wf = w64 = 0.0;
        for (std::size_t i = 0; i < samples; ++i)
        {
            const double r = std::pow(double(x[i]), double(y[i]));
            w32 = std::max(w32, error_ulp(double(out[i]), r, 24, -126));
            const double z = std::fabs(double(y[i]) * std::log2(double(x[i])));
            if (z < 120.0)
                wf = std::max(wf, error_rel(double(fout[i]), r) / (1.0 + z));
            w64 = std::max(w64, error_ulp(outd[i], std::pow(xd[i], yd[i]), 53, -1022));
        }
        if (report && with_pow)
            std::printf("%-11s f32 %8.3f ulp   fast %.3e   f64 %8.3f ulp (vs libm)\n", "pow", w32, wf, w64);
        CHECK(!with_pow || w32 <= 2.0 + margin);
        CHECK(!with_pow || wf <= 0x1p-16);
        CHECK(!with_pow || w64 <= 2.5 + margin);
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
    return wray::test::finish("accuracy");
}
