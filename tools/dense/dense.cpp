#include <wray/wray.h>

#include <quadmath.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{
    using quad = __float128;

    const quad pi = strtoflt128("3.14159265358979323846264338327950288419716939937510", nullptr);

    struct piece
    {
        enum kind_t
        {
            values,
            bits,
            near_turns,
            near_quadrants
        } kind;
        double lo;
        double hi;
        bool signed_;
        double weight;
    };

    struct unary_entry
    {
        const char *name;
        std::vector<piece> domain;
        std::function<void(const std::vector<double> &, std::vector<double> &)> run;
        std::function<quad(double)> reference;
    };

    struct binary_entry
    {
        const char *name;
        std::function<void(std::mt19937_64 &, std::size_t, std::vector<double> &, std::vector<double> &)> sample;
        std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> run;
        std::function<quad(double, double)> reference;
    };

    constexpr double tiny = std::numeric_limits<double>::denorm_min();
    constexpr double huge = std::numeric_limits<double>::max();
    constexpr double trig = 0x1.921fb54442d18p+22;

    double uniform(std::mt19937_64 &g, double lo, double hi)
    {
        return lo + (hi - lo) * (static_cast<double>(g() >> 11) * 0x1p-53);
    }

    double bit_uniform(std::mt19937_64 &g, double lo, double hi, bool signed_)
    {
        const auto a = std::bit_cast<std::uint64_t>(lo);
        const auto b = std::bit_cast<std::uint64_t>(hi);
        const double v = std::bit_cast<double>(a + g() % (b - a + 1));
        return signed_ && (g() & 1) ? -v : v;
    }

    double draw(std::mt19937_64 &g, const std::vector<piece> &domain)
    {
        double total = 0;
        for (const auto &p: domain)
            total += p.weight;
        double pick = uniform(g, 0, total);
        for (const auto &p: domain)
        {
            if (pick > p.weight)
            {
                pick -= p.weight;
                continue;
            }
            switch (p.kind)
            {
                case piece::values:
                    return uniform(g, p.lo, p.hi);
                case piece::bits:
                    return bit_uniform(g, p.lo, p.hi, p.signed_);
                case piece::near_turns:
                {
                    const double k = static_cast<double>(static_cast<std::int64_t>(g() % 1024) - 512) / 8.0;
                    return std::nextafter(k, (g() & 1) ? huge : -huge);
                }
                case piece::near_quadrants:
                {
                    const double k = static_cast<double>(static_cast<std::int64_t>(g() % 8388608) - 4194304);
                    return std::nextafter(k * (M_PI / 2), (g() & 1) ? huge : -huge);
                }
            }
        }
        return 0;
    }

    struct turn
    {
        quad sin;
        quad cos;
    };

    turn turns(double r)
    {
        const double k = std::nearbyint(4 * r);
        const quad t = 2 * pi * static_cast<quad>(r - k / 4);
        const quad s = t == 0 ? 0 : sinq(t);
        const quad c = cosq(t);
        switch (static_cast<int>(k) & 3)
        {
            case 0:
                return {s, c};
            case 1:
                return {c, -s};
            case 2:
                return {-s, -c};
            default:
                return {-c, s};
        }
    }

    double frac(double x)
    {
        return x - std::nearbyint(x);
    }

    quad tan_pi(double x)
    {
        const double r = frac(x);
        if (std::fabs(r) <= 0.25)
            return tanq(pi * static_cast<quad>(r));
        const double c = r - std::copysign(0.5, r);
        return c == 0 ? (x > 0 ? HUGE_VALQ : -HUGE_VALQ) : -1 / tanq(pi * static_cast<quad>(c));
    }

    quad tan_ratio(const turn &t, double x)
    {
        return t.cos == 0 ? (x > 0 ? HUGE_VALQ : -HUGE_VALQ) : t.sin / t.cos;
    }

    quad product_log_reference(double x)
    {
        if (x == 0)
            return 0;
        quad w = static_cast<quad>(wray::product_log(x));
        const quad xq = x;
        for (int i = 0; i < 8; ++i)
        {
            const quad e = expq(w);
            const quad f = w * e - xq;
            const quad d = e * (w + 1) - (w + 2) * f / (2 * w + 2);
            if (d == 0 || isnanq(d))
                break;
            w -= f / d;
        }
        return w;
    }

    quad omega_reference(double x)
    {
        quad w = static_cast<quad>(wray::omega(x));
        if (w <= 0)
            w = expq(static_cast<quad>(x));
        const quad xq = x;
        for (int i = 0; i < 8 && w > 0; ++i)
        {
            const quad f = w + logq(w) - xq;
            w -= f * w / (w + 1);
        }
        return w;
    }

    double ulp_of(quad ref)
    {
        if (ref == 0 || isinfq(ref))
            return std::ldexp(1.0, -1074);
        int e = 0;
        frexpq(ref, &e);
        return std::ldexp(1.0, std::max(e - 1, -1022) - 52);
    }

    double error(double got, quad ref)
    {
        if (std::isnan(got))
            return isnanq(ref) ? 0.0 : std::numeric_limits<double>::infinity();
        if (std::isinf(got))
        {
            const quad top = ldexpq(1, 1024) * (1 - ldexpq(1, -54));
            return fabsq(ref) >= top && (got > 0) == (ref > 0) ? 0.0 : std::numeric_limits<double>::infinity();
        }
        return static_cast<double>(fabsq(static_cast<quad>(got) - ref) / static_cast<quad>(ulp_of(ref)));
    }

    template <class F>
    std::function<void(const std::vector<double> &, std::vector<double> &)> vector_run(F f)
    {
        return [f](const std::vector<double> &in, std::vector<double> &out) { wray::math::apply(in, out, f); };
    }

    template <class F>
    std::function<void(const std::vector<double> &, const std::vector<double> &, std::vector<double> &)> vector_run2(F f)
    {
        return [f](const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &out) { wray::math::apply(a, b, out, f); };
    }

    std::vector<unary_entry> unary_table()
    {
        using p = piece;
        const std::vector<piece> turn_domain{{p::values, -1, 1, false, 2}, {p::bits, tiny, huge, true, 1}, {p::near_turns, 0, 0, false, 1}};
        const std::vector<piece> trig_domain{{p::values, -2 * M_PI, 2 * M_PI, false, 4}, {p::values, -trig, trig, false, 4}, {p::bits, 1e-6, trig, true, 3}, {p::near_quadrants, 0, 0, false, 4}, {p::bits, trig, huge, true, 1}};
        const std::vector<piece> logs{{p::bits, tiny, huge, false, 1}, {p::values, 0.5, 2.0, false, 1}};
        auto balanced_domain = trig_domain;
        balanced_domain.push_back({p::bits, tiny, huge, true, 2});
        constexpr auto root = std::bit_cast<std::uint64_t>(0x1.a9adcc7f96cf0p+20);
        balanced_domain.push_back({p::bits, std::bit_cast<double>(root - 8), std::bit_cast<double>(root + 8), true, 1});
        return {
            {"exp", {{p::values, -745.0, 709.78, false, 1}, {p::bits, 1e-6, 709.78, true, 1}}, vector_run([](auto v) { return wray::exp(v); }), [](double x) { return expq(x); }},
            {"expm1", {{p::values, -2.0, 2.0, false, 1}, {p::bits, tiny, 709.78, true, 1}}, vector_run([](auto v) { return wray::expm1(v); }), [](double x) { return expm1q(x); }},
            {"exp2", {{p::values, -1074.9, 1023.99, false, 1}, {p::bits, 1e-6, 1023.99, true, 1}}, vector_run([](auto v) { return wray::exp2(v); }), [](double x) { return exp2q(x); }},
            {"log", logs, vector_run([](auto v) { return wray::log(v); }), [](double x) { return logq(x); }},
            {"log2", logs, vector_run([](auto v) { return wray::log2(v); }), [](double x) { return log2q(x); }},
            {"log10", {{p::bits, tiny, huge, false, 1}, {p::values, 0.5, 20.0, false, 1}}, vector_run([](auto v) { return wray::log10(v); }), [](double x) { return log10q(x); }},
            {"log1p", {{p::values, -0.75, 2.0, false, 2}, {p::bits, tiny, huge, false, 1}, {p::bits, tiny, 0.999, false, 1}}, vector_run([](auto v) { return wray::log1p(v); }), [](double x) { return log1pq(x); }},
            {"gain_to_db", logs, vector_run([](auto v) { return wray::gain_to_db(v); }), [](double x) { return 20 * log10q(x); }},
            {"sin_turns", turn_domain, vector_run([](auto v) { return wray::sin_turns(v); }), [](double x) { return turns(frac(x)).sin; }},
            {"cos_turns", turn_domain, vector_run([](auto v) { return wray::cos_turns(v); }), [](double x) { return turns(frac(x)).cos; }},
            {"tanpi", turn_domain, vector_run([](auto v) { return wray::tanpi(v); }), tan_pi},
            {"tan_turns", turn_domain, vector_run([](auto v) { return wray::tan_turns(v); }), [](double x) { return tan_ratio(turns(frac(x)), x); }},
            {"sin", trig_domain, vector_run([](auto v) { return wray::sin(v); }), [](double x) { return sinq(x); }},
            {"cos", trig_domain, vector_run([](auto v) { return wray::cos(v); }), [](double x) { return cosq(x); }},
            {"tan", trig_domain, vector_run([](auto v) { return wray::tan(v); }), [](double x) { return tanq(x); }},
            {"atan", {{p::bits, tiny, huge, true, 1}, {p::values, -4, 4, false, 1}}, vector_run([](auto v) { return wray::atan(v); }), [](double x) { return atanq(x); }},
            {"sinh", {{p::values, -3, 3, false, 2}, {p::values, -710.5, 710.5, false, 1}, {p::bits, tiny, 710.5, true, 1}}, vector_run([](auto v) { return wray::sinh(v); }), [](double x) { return sinhq(x); }},
            {"cosh", {{p::values, -3, 3, false, 2}, {p::values, -710.5, 710.5, false, 1}, {p::bits, tiny, 710.5, true, 1}}, vector_run([](auto v) { return wray::cosh(v); }), [](double x) { return coshq(x); }},
            {"asinh", {{p::values, -4, 4, false, 1}, {p::bits, tiny, huge, true, 1}}, vector_run([](auto v) { return wray::asinh(v); }), [](double x) { return asinhq(x); }},
            {"cbrt", {{p::values, -10, 10, false, 1}, {p::bits, tiny, huge, true, 1}}, vector_run([](auto v) { return wray::cbrt(v); }), [](double x) { return cbrtq(x); }},
            {"erf", {{p::values, -6.5, 6.5, false, 1}, {p::bits, tiny, huge, true, 1}}, vector_run([](auto v) { return wray::erf(v); }), [](double x) { return erfq(x); }},
            {"midi_to_hz", {{p::values, -100, 250, false, 1}, {p::values, -12900, 12300, false, 1}}, vector_run([](auto v) { return wray::midi_to_hz(v); }), [](double x) { return 440 * exp2q((static_cast<quad>(x) - 69) / 12); }},
            {"hz_to_midi", {{p::values, 1e-3, 1e6, false, 1}, {p::bits, tiny, huge, false, 1}}, vector_run([](auto v) { return wray::hz_to_midi(v); }), [](double x) { return 69 + 12 * log2q(static_cast<quad>(x) / 440); }},
            {"ratio_from_cents", {{p::values, -2400, 2400, false, 1}, {p::values, -1290000, 1220000, false, 1}}, vector_run([](auto v) { return wray::ratio_from_cents(v); }), [](double x) { return exp2q(static_cast<quad>(x) / 1200); }},
            {"omega", {{p::values, -10, 30, false, 1}, {p::bits, tiny, huge, true, 1}}, vector_run([](auto v) { return wray::omega(v); }), omega_reference},
            {"product_log", {{p::values, -0.36787944, 10, false, 2}, {p::bits, tiny, huge, false, 1}, {p::values, -0.36787944, -0.36, false, 1}}, vector_run([](auto v) { return wray::product_log(v); }), product_log_reference},
            {"tanh", {{p::values, -20, 20, false, 1}, {p::bits, tiny, 30, true, 1}}, vector_run([](auto v) { return wray::tanh(v); }), [](double x) { return tanhq(x); }},
            {"sigmoid", {{p::values, -744, 744, false, 1}}, vector_run([](auto v) { return wray::sigmoid(v); }), [](double x) { return 1 / (1 + expq(-static_cast<quad>(x))); }},
            {"db_to_gain", {{p::values, -6000, 6000, false, 1}}, vector_run([](auto v) { return wray::db_to_gain(v); }), [](double x) { return powq(10, static_cast<quad>(x) / 20); }},
            {"balanced_sin", balanced_domain, vector_run([](auto v) { return wray::balanced::sin(v); }), [](double x) { return sinq(x); }},
            {"balanced_cos", balanced_domain, vector_run([](auto v) { return wray::balanced::cos(v); }), [](double x) { return cosq(x); }},
            {"balanced_sincos_sin", balanced_domain, vector_run([](auto v) { return wray::balanced::sincos(v).sin; }), [](double x) { return sinq(x); }},
            {"balanced_sincos_cos", balanced_domain, vector_run([](auto v) { return wray::balanced::sincos(v).cos; }), [](double x) { return cosq(x); }},
            {"balanced_tanh", {{p::values, -20, 20, false, 4}, {p::values, -1, 1, false, 2}, {p::bits, tiny, huge, true, 3}},
             vector_run([](auto v) { return wray::balanced::tanh(v); }), [](double x) { return tanhq(x); }},
        };
    }

    std::vector<binary_entry> binary_table()
    {
        const auto maps = [](bool time) {
            return [time](std::mt19937_64 &g, std::size_t n, std::vector<double> &a, std::vector<double> &b) {
                const double rates[] = {44100.0, 48000.0, 96000.0, 192000.0};
                for (std::size_t i = 0; i < n; ++i)
                {
                    const double r = std::exp(uniform(g, std::log(1e-9), std::log(0.5)));
                    b[i] = (g() & 1) ? rates[g() % 4] : uniform(g, 8000.0, 768000.0);
                    a[i] = time ? 1 / (r * b[i]) : r * b[i];
                }
            };
        };
        return {
            {"atan2", [](std::mt19937_64 &g, std::size_t n, std::vector<double> &a, std::vector<double> &b) {
                 for (std::size_t i = 0; i < n; ++i)
                 {
                     a[i] = (g() & 1) ? bit_uniform(g, tiny, huge, true) : uniform(g, -4, 4);
                     b[i] = (g() & 1) ? bit_uniform(g, tiny, huge, true) : uniform(g, -4, 4);
                 }
             }, vector_run2([](auto y, auto x) { return wray::atan2(y, x); }), [](double y, double x) { return atan2q(y, x); }},
            {"pow", [](std::mt19937_64 &g, std::size_t n, std::vector<double> &a, std::vector<double> &b) {
                 for (std::size_t i = 0; i < n; ++i)
                 {
                     const auto k = g() % 4;
                     if (k == 0)
                     {
                         a[i] = -bit_uniform(g, 1e-3, 1e3, false);
                         b[i] = static_cast<double>(static_cast<std::int64_t>(g() % 25) - 12);
                     } else if (k == 1)
                     {
                         a[i] = std::exp2(uniform(g, -1, 1));
                         b[i] = uniform(g, -1000, 1000) / std::log2(a[i]);
                     } else
                     {
                         a[i] = bit_uniform(g, 1e-10, 1e10, false);
                         b[i] = uniform(g, -3, 3);
                     }
                 }
             }, vector_run2([](auto x, auto y) { return wray::pow(x, y); }), [](double x, double y) { return powq(x, y); }},
            {"smoothing_coefficient", maps(true), vector_run2([](auto s, auto fs) { return wray::smoothing_coefficient(s, fs); }),
             [](double s, double fs) { return -expm1q(-1 / (static_cast<quad>(s) * fs)); }},
            {"pole_from_cutoff", maps(false), vector_run2([](auto hz, auto fs) { return wray::pole_from_cutoff(hz, fs); }),
             [](double hz, double fs) { return expq(-2 * pi * hz / fs); }},
            {"prewarp", maps(false), vector_run2([](auto hz, auto fs) { return wray::prewarp(hz, fs); }), [](double hz, double fs) { return tanq(pi * hz / fs); }},
        };
    }

    struct worst
    {
        double ulp = 0;
        double x = 0;
        double y = 0;
    };

    template <class Work>
    worst parallel(std::uint64_t samples, std::uint64_t seed, Work work)
    {
        constexpr std::size_t block = 1 << 15;
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<std::uint64_t> next{0};
        std::mutex lock;
        worst all;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t)
            pool.emplace_back([&] {
                worst mine;
                for (;;)
                {
                    const std::uint64_t index = next.fetch_add(1);
                    if (index * block >= samples)
                        break;
                    std::mt19937_64 g(seed * 0x9e3779b97f4a7c15ull + index);
                    const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(block, samples - index * block));
                    work(g, n, mine);
                }
                const std::lock_guard<std::mutex> guard(lock);
                if (mine.ulp > all.ulp || (std::isnan(mine.ulp) && !std::isnan(all.ulp)))
                    all = mine;
            });
        for (auto &t: pool)
            t.join();
        return all;
    }

    void keep(worst &w, double e, double x, double y)
    {
        if (e > w.ulp || std::isnan(e))
            w = {e, x, y};
    }
}

int main(int argc, char **argv)
{
    std::uint64_t samples = 1000000;
    std::uint64_t seed = 1;
    double pow_scale = 4.0;
    std::vector<std::string> only;
    bool references = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc)
            samples = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--pow-scale") == 0 && i + 1 < argc)
            pow_scale = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--references") == 0)
            references = true;
        else
            only.emplace_back(argv[i]);
    }
    const auto wanted = [&](const char *name) { return only.empty() || std::find(only.begin(), only.end(), name) != only.end(); };
    std::printf("seed %llu samples %llu fma %d\n", static_cast<unsigned long long>(seed), static_cast<unsigned long long>(samples), WRAY_HAS_FMA);
    if (references)
    {
        std::mt19937_64 g(seed);
        char text[64];
        for (const auto &u: unary_table())
            if (wanted(u.name))
                for (std::uint64_t i = 0; i < samples; ++i)
                {
                    const double x = draw(g, u.domain);
                    quadmath_snprintf(text, sizeof text, "%.36Qe", u.reference(x));
                    std::printf("%s %a 0 %s\n", u.name, x, text);
                }
        for (const auto &b: binary_table())
            if (wanted(b.name))
            {
                std::vector<double> x(samples), y(samples);
                b.sample(g, static_cast<std::size_t>(samples), x, y);
                for (std::uint64_t i = 0; i < samples; ++i)
                {
                    quadmath_snprintf(text, sizeof text, "%.36Qe", b.reference(x[i], y[i]));
                    std::printf("%s %a %a %s\n", b.name, x[i], y[i], text);
                }
            }
        return 0;
    }
    for (const auto &u: unary_table())
    {
        if (!wanted(u.name))
            continue;
        const worst w = parallel(samples, seed, [&](std::mt19937_64 &g, std::size_t n, worst &mine) {
            std::vector<double> in(n), out(n);
            for (auto &x: in)
                x = draw(g, u.domain);
            u.run(in, out);
            for (std::size_t i = 0; i < n; ++i)
                keep(mine, error(out[i], u.reference(in[i])), in[i], 0);
        });
        std::printf("%s %.6g %a %a\n", u.name, w.ulp, w.x, w.y);
        std::fflush(stdout);
    }
    for (const auto &b: binary_table())
    {
        if (!wanted(b.name))
            continue;
        const bool is_pow = std::strcmp(b.name, "pow") == 0;
        const worst w = parallel(samples, seed, [&](std::mt19937_64 &g, std::size_t n, worst &mine) {
            std::vector<double> x(n), y(n), out(n);
            b.sample(g, n, x, y);
            b.run(x, y, out);
            for (std::size_t i = 0; i < n; ++i)
            {
                double e = error(out[i], b.reference(x[i], y[i]));
                if (is_pow)
                    e /= std::max(1.0, std::fabs(y[i] * std::log2(std::fabs(x[i]))) / pow_scale);
                keep(mine, e, x[i], y[i]);
            }
        });
        std::printf("%s %.6g %a %a\n", b.name, w.ulp, w.x, w.y);
        std::fflush(stdout);
    }
    return 0;
}
