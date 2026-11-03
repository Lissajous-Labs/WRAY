#include <wray/wray.h>
#include "baselines.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace
{
    constexpr std::size_t kSize = 4096;
    constexpr int kRounds = 25;
    constexpr int kRepeats = 40;
    constexpr int kChainRepeats = 8;

    template <class T>
    struct workload
    {
        std::vector<T> a;
        std::vector<T> b;
    };

    template <class T>
    struct throughput
    {
        std::string name;
        std::function<void(const workload<T>&, std::span<T>, std::span<T>)> run;
    };

    template <class T>
    struct latency
    {
        std::string name;
        std::function<T(const workload<T>&, T)> run;
    };

    template <class T>
    struct suite
    {
        std::vector<throughput<T>> throughput_rows;
        std::vector<latency<T>> latency_rows;
    };

    struct none
    {
    };

    template <class T>
    std::vector<T> uniform(std::mt19937_64& rng, double lo, double hi)
    {
        std::uniform_real_distribution<double> d(lo, hi);
        std::vector<T> v(kSize);
        for (auto& x : v)
            x = static_cast<T>(d(rng));
        return v;
    }

    template <class T>
    std::vector<T> log_uniform(std::mt19937_64& rng, double lo, double hi)
    {
        std::uniform_real_distribution<double> d(std::log(lo), std::log(hi));
        std::vector<T> v(kSize);
        for (auto& x : v)
            x = static_cast<T>(std::exp(d(rng)));
        return v;
    }

    template <class T>
    workload<T> make(const std::string& name, std::mt19937_64& rng)
    {
        workload<T> w;
        if (name == "product_log")
            w.a = uniform<T>(rng, -0.3, 10);
        else if (name == "omega")
            w.a = uniform<T>(rng, -10, 30);
        else if (name == "smoothing_coefficient")
        {
            w.a = log_uniform<T>(rng, 1e-4, 10);
            w.b = uniform<T>(rng, 44100, 192000);
        }
        else if (name == "pole_from_cutoff" || name == "prewarp")
        {
            w.a = log_uniform<T>(rng, 20, 20000);
            w.b = uniform<T>(rng, 44100, 192000);
        }
        else if (name == "ratio_from_cents")
            w.a = uniform<T>(rng, -2400, 2400);
        else if (name == "hz_to_midi")
            w.a = log_uniform<T>(rng, 20, 20000);
        else if (name == "midi_to_hz")
            w.a = uniform<T>(rng, 0, 127);
        else if (name == "erf")
            w.a = uniform<T>(rng, -4, 4);
        else if (name == "cbrt")
            w.a = uniform<T>(rng, -100, 100);
        else if (name == "asinh")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "cosh")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "sinh")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "tanpi")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "tan_turns")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "cos_turns")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "sin_turns")
            w.a = uniform<T>(rng, -10, 10);
        else if (name == "log10")
            w.a = log_uniform<T>(rng, 1e-30, 1e30);
        else if (name == "log1p")
            w.a = uniform<T>(rng, -0.9, 10);
        else if (name == "exp" || name == "sigmoid")
            w.a = uniform<T>(rng, -80, 80);
        else if (name == "expm1")
            w.a = uniform<T>(rng, -4, 4);
        else if (name == "exp2")
            w.a = uniform<T>(rng, -120, 120);
        else if (name == "log" || name == "log2" || name == "gain_to_db")
            w.a = log_uniform<T>(rng, 1e-30, 1e30);
        else if (name == "sin" || name == "cos" || name == "tan" || name == "sincos")
            w.a = uniform<T>(rng, -100, 100);
        else if (name == "atan")
            w.a = uniform<T>(rng, -50, 50);
        else if (name == "tanh")
            w.a = uniform<T>(rng, -6, 6);
        else if (name == "db_to_gain")
            w.a = uniform<T>(rng, -120, 24);
        else if (name == "atan2")
        {
            w.a = uniform<T>(rng, -10, 10);
            w.b = uniform<T>(rng, -10, 10);
        }
        else if (name == "pow")
        {
            w.a = log_uniform<T>(rng, 1e-3, 1e3);
            w.b = uniform<T>(rng, -4, 4);
        }
        return w;
    }

    template <class T>
    std::vector<T> mirrored(std::mt19937_64& rng, double lo, double hi)
    {
        std::vector<T> v = uniform<T>(rng, lo, hi);
        for (auto& x : v)
            x = (rng() & 1) ? -x : x;
        return v;
    }

    template <class T>
    workload<T> edge(const std::string& name, std::mt19937_64& rng)
    {
        constexpr bool wide = std::is_same_v<T, double>;
        const double tiny = double(std::numeric_limits<T>::denorm_min());
        const double normal = double(std::numeric_limits<T>::min());
        workload<T> w;
        if (name == "product_log")
            w.a = mirrored<T>(rng, 1e20, 1e30);
        else if (name == "omega")
            w.a = mirrored<T>(rng, 1e20, 1e30);
        else if (name == "ratio_from_cents")
            w.a = mirrored<T>(rng, 1e6, 1e7);
        else if (name == "hz_to_midi")
            w.a = log_uniform<T>(rng, double(std::numeric_limits<T>::denorm_min()), double(std::numeric_limits<T>::min()));
        else if (name == "midi_to_hz")
            w.a = mirrored<T>(rng, 1e4, 1e5);
        else if (name == "erf")
            w.a = mirrored<T>(rng, 10, 100);
        else if (name == "cbrt")
            w.a = log_uniform<T>(rng, double(std::numeric_limits<T>::denorm_min()), double(std::numeric_limits<T>::min()));
        else if (name == "asinh")
            w.a = mirrored<T>(rng, 1e20, 1e30);
        else if (name == "cosh")
            w.a = mirrored<T>(rng, std::is_same_v<T, double> ? 708.0 : 87.0, std::is_same_v<T, double> ? 760.0 : 110.0);
        else if (name == "sinh")
            w.a = mirrored<T>(rng, std::is_same_v<T, double> ? 708.0 : 87.0, std::is_same_v<T, double> ? 760.0 : 110.0);
        else if (name == "tanpi")
            w.a = mirrored<T>(rng, 1e6, 1e7);
        else if (name == "tan_turns")
            w.a = mirrored<T>(rng, 1e6, 1e7);
        else if (name == "cos_turns")
            w.a = mirrored<T>(rng, 1e6, 1e7);
        else if (name == "sin_turns")
            w.a = mirrored<T>(rng, 1e6, 1e7);
        else if (name == "log10")
            w.a = log_uniform<T>(rng, double(std::numeric_limits<T>::denorm_min()), double(std::numeric_limits<T>::min()));
        else if (name == "log1p")
            w.a = uniform<T>(rng, -3, -1.5);
        else if (name == "exp" || name == "expm1")
            w.a = mirrored<T>(rng, wide ? 708.0 : 87.0, wide ? 760.0 : 110.0);
        else if (name == "sigmoid")
            w.a = mirrored<T>(rng, wide ? 708.0 : 87.0, wide ? 760.0 : 110.0);
        else if (name == "exp2")
            w.a = mirrored<T>(rng, wide ? 1021.0 : 125.0, wide ? 1100.0 : 160.0);
        else if (name == "log" || name == "log2" || name == "gain_to_db")
            w.a = log_uniform<T>(rng, tiny, normal);
        else if (name == "sin" || name == "cos" || name == "tan" || name == "sincos")
            w.a = mirrored<T>(rng, wide ? 1e6 : 4e3, wide ? 6.5e6 : 6.4e3);
        else if (name == "atan")
            w.a = mirrored<T>(rng, 1e20, 1e30);
        else if (name == "tanh")
            w.a = mirrored<T>(rng, 20.0, 40.0);
        else if (name == "db_to_gain")
            w.a = mirrored<T>(rng, wide ? 6200.0 : 760.0, wide ? 6500.0 : 900.0);
        else if (name == "atan2")
        {
            w.a = log_uniform<T>(rng, normal, normal * 1e3);
            w.b = log_uniform<T>(rng, 1.0, 1e3);
        }
        else if (name == "pow")
        {
            w.a = log_uniform<T>(rng, tiny, normal);
            w.b = uniform<T>(rng, 0.01, 0.5);
        }
        return w;
    }

    template <class T, class F>
    throughput<T> simd_loop(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::span<T> out, std::span<T>) { wray::math::apply(w.a, out, f); }};
    }

    template <class T, class F>
    throughput<T> simd_loop2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::span<T> out, std::span<T>) { wray::math::apply(w.a, w.b, out, f); }};
    }

    template <class T, class F>
    latency<T> chain(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, T tiny) {
                    T x = w.a[0];
                    for (std::size_t i = 1; i < kSize; ++i)
                        x = w.a[i] + f(x) * tiny;
                    return x;
                }};
    }

    template <class T, class F>
    latency<T> chain2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, T tiny) {
                    T x = w.a[0];
                    for (std::size_t i = 1; i < kSize; ++i)
                        x = w.a[i] + f(x, w.b[i - 1]) * tiny;
                    return x;
                }};
    }

    template <class T, class R, class P, class Q>
    void unary(suite<T>& s, R reference, P precise, Q fast)
    {
        if constexpr (!std::is_same_v<R, none>)
        {
            s.latency_rows.push_back(chain<T>("libm", reference));
        }
        s.throughput_rows.push_back(simd_loop<T>("wray", precise));
        s.latency_rows.push_back(chain<T>("wray", [precise](T x) { return precise(x); }));
        if constexpr (std::is_same_v<T, float> && !std::is_same_v<Q, none>)
        {
            s.throughput_rows.push_back(simd_loop<T>("wray fast", fast));
            s.latency_rows.push_back(chain<T>("wray fast", [fast](T x) { return fast(x); }));
        }
    }

    template <class T, class R, class P, class Q>
    void binary(suite<T>& s, R reference, P precise, Q fast)
    {
        s.throughput_rows.push_back(simd_loop2<T>("wray", precise));
        s.latency_rows.push_back(chain2<T>("libm", reference));
        s.latency_rows.push_back(chain2<T>("wray", [precise](T a, T b) { return precise(a, b); }));
        if constexpr (std::is_same_v<T, float> && !std::is_same_v<Q, none>)
        {
            s.throughput_rows.push_back(simd_loop2<T>("wray fast", fast));
            s.latency_rows.push_back(chain2<T>("wray fast", [fast](T a, T b) { return fast(a, b); }));
        }
    }

    template <class T>
    suite<T> variants(const std::string& n)
    {
        suite<T> s;
        if (n == "exp")
            unary(s, [](T x) { return std::exp(x); }, [](auto x) { return wray::exp(x); }, [](auto x) { return wray::fast::exp(x); });
        if (n == "log1p")
            unary(s, [](T x) { return std::log1p(x); }, [](auto x) { return wray::log1p(x); }, none{});
        if (n == "log10")
            unary(s, [](T x) { return std::log10(x); }, [](auto x) { return wray::log10(x); }, none{});
        if (n == "sin_turns")
            unary(s, [](T x) { return std::sin(T(6.283185307179586) * x); }, [](auto x) { return wray::sin_turns(x); }, [](auto x) { return wray::fast::sin_turns(x); });
        if (n == "cos_turns")
            unary(s, [](T x) { return std::cos(T(6.283185307179586) * x); }, [](auto x) { return wray::cos_turns(x); }, [](auto x) { return wray::fast::cos_turns(x); });
        if (n == "tan_turns")
            unary(s, [](T x) { return std::tan(T(6.283185307179586) * x); }, [](auto x) { return wray::tan_turns(x); }, [](auto x) { return wray::fast::tan_turns(x); });
        if (n == "tanpi")
            unary(s, [](T x) { return std::tan(T(3.141592653589793) * x); }, [](auto x) { return wray::tanpi(x); }, [](auto x) { return wray::fast::tanpi(x); });
        if (n == "sinh")
            unary(s, [](T x) { return std::sinh(x); }, [](auto x) { return wray::sinh(x); }, none{});
        if (n == "cosh")
            unary(s, [](T x) { return std::cosh(x); }, [](auto x) { return wray::cosh(x); }, none{});
        if (n == "asinh")
            unary(s, [](T x) { return std::asinh(x); }, [](auto x) { return wray::asinh(x); }, none{});
        if (n == "cbrt")
            unary(s, [](T x) { return std::cbrt(x); }, [](auto x) { return wray::cbrt(x); }, none{});
        if (n == "erf")
            unary(s, [](T x) { return std::erf(x); }, [](auto x) { return wray::erf(x); }, none{});
        if (n == "midi_to_hz")
            unary(s, [](T x) { return T(440) * std::exp2((x - T(69)) / T(12)); }, [](auto x) { return wray::midi_to_hz(x); }, none{});
        if (n == "hz_to_midi")
            unary(s, [](T x) { return T(69) + T(12) * std::log2(x / T(440)); }, [](auto x) { return wray::hz_to_midi(x); }, none{});
        if (n == "ratio_from_cents")
            unary(s, [](T x) { return std::exp2(x / T(1200)); }, [](auto x) { return wray::ratio_from_cents(x); }, none{});
        if (n == "smoothing_coefficient")
            binary(s, [](T t, T fs) { return -std::expm1(T(-1) / (t * fs)); }, [](auto t, auto fs) { return wray::smoothing_coefficient(t, fs); }, none{});
        if (n == "pole_from_cutoff")
            binary(s, [](T hz, T fs) { return std::exp(T(-2) * std::numbers::pi_v<T> * hz / fs); }, [](auto hz, auto fs) { return wray::pole_from_cutoff(hz, fs); }, none{});
        if (n == "prewarp")
            binary(s, [](T hz, T fs) { return std::tan(std::numbers::pi_v<T> * hz / fs); }, [](auto hz, auto fs) { return wray::prewarp(hz, fs); }, none{});
        if (n == "omega")
            unary(s, none{}, [](auto x) { return wray::omega(x); }, [](auto x) { return wray::fast::omega(x); });
        if (n == "product_log")
            unary(s, none{}, [](auto x) { return wray::product_log(x); }, [](auto x) { return wray::fast::product_log(x); });
        if (n == "expm1")
            unary(s, [](T x) { return std::expm1(x); }, [](auto x) { return wray::expm1(x); }, none{});
        if (n == "exp2")
            unary(s, [](T x) { return std::exp2(x); }, [](auto x) { return wray::exp2(x); }, [](auto x) { return wray::fast::exp2(x); });
        if (n == "log")
            unary(s, [](T x) { return std::log(x); }, [](auto x) { return wray::log(x); }, [](auto x) { return wray::fast::log(x); });
        if (n == "log2")
            unary(s, [](T x) { return std::log2(x); }, [](auto x) { return wray::log2(x); }, [](auto x) { return wray::fast::log2(x); });
        if (n == "sin")
        {
            unary(s, [](T x) { return std::sin(x); }, [](auto x) { return wray::sin(x); }, [](auto x) { return wray::fast::sin(x); });
            s.throughput_rows.push_back(simd_loop<T>("wray balanced", [](auto x) { return wray::balanced::sin(x); }));
            s.latency_rows.push_back(chain<T>("wray balanced", [](T x) { return wray::balanced::sin(x); }));
        }
        if (n == "cos")
        {
            unary(s, [](T x) { return std::cos(x); }, [](auto x) { return wray::cos(x); }, [](auto x) { return wray::fast::cos(x); });
            s.throughput_rows.push_back(simd_loop<T>("wray balanced", [](auto x) { return wray::balanced::cos(x); }));
            s.latency_rows.push_back(chain<T>("wray balanced", [](T x) { return wray::balanced::cos(x); }));
        }
        if (n == "sincos")
        {
            s.throughput_rows.push_back({"wray", [](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                wray::sincos(w.a, out, cos_out);
            }});
            s.latency_rows.push_back(chain<T>("libm", [](T x) { return std::sin(x) + std::cos(x); }));
            s.latency_rows.push_back(chain<T>("wray", [](T x) {
                const auto r = wray::sincos(x);
                return r.sin + r.cos;
            }));
            s.throughput_rows.push_back({"wray balanced", [](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                wray::balanced::sincos(w.a, out, cos_out);
            }});
            s.latency_rows.push_back(chain<T>("wray balanced", [](T x) {
                const auto r = wray::balanced::sincos(x);
                return r.sin + r.cos;
            }));
            if constexpr (std::is_same_v<T, float>)
            {
                s.throughput_rows.push_back({"wray fast", [](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                    wray::fast::sincos(w.a, out, cos_out);
                }});
                s.latency_rows.push_back(chain<T>("wray fast", [](T x) {
                    const auto r = wray::fast::sincos(x);
                    return r.sin + r.cos;
                }));
            }
        }
        if (n == "tan")
            unary(s, [](T x) { return std::tan(x); }, [](auto x) { return wray::tan(x); }, [](auto x) { return wray::fast::tan(x); });
        if (n == "atan")
            unary(s, [](T x) { return std::atan(x); }, [](auto x) { return wray::atan(x); }, [](auto x) { return wray::fast::atan(x); });
        if (n == "atan2")
            binary(s, [](T y, T x) { return std::atan2(y, x); }, [](auto y, auto x) { return wray::atan2(y, x); },
                   [](auto y, auto x) { return wray::fast::atan2(y, x); });
        if (n == "tanh")
        {
            unary(s, [](T x) { return std::tanh(x); }, [](auto x) { return wray::tanh(x); }, [](auto x) { return wray::fast::tanh(x); });
            s.throughput_rows.push_back(simd_loop<T>("wray balanced", [](auto x) { return wray::balanced::tanh(x); }));
            s.latency_rows.push_back(chain<T>("wray balanced", [](T x) { return wray::balanced::tanh(x); }));
        }
        if (n == "sigmoid")
            unary(s, [](T x) { return T(1) / (T(1) + std::exp(-x)); }, [](auto x) { return wray::sigmoid(x); },
                  [](auto x) { return wray::fast::sigmoid(x); });
        if (n == "pow")
            binary(s, [](T x, T y) { return std::pow(x, y); }, [](auto x, auto y) { return wray::pow(x, y); },
                   [](auto x, auto y) { return wray::fast::pow(x, y); });
        if (n == "db_to_gain")
            unary(s, [](T x) { return std::exp2(x * (std::numbers::ln10_v<T> / (T(20) * std::numbers::ln2_v<T>))); }, [](auto x) { return wray::db_to_gain(x); },
                  [](auto x) { return wray::fast::db_to_gain(x); });
        if (n == "gain_to_db")
            unary(s, [](T x) { return T(20) * std::log10(x); }, [](auto x) { return wray::gain_to_db(x); },
                  [](auto x) { return wray::fast::gain_to_db(x); });
        if (const auto f = wray_bench::libm<T>(n))
        {
            s.throughput_rows.insert(s.throughput_rows.begin(), {"libm", [f](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                f(w.a.data(), w.b.data(), out.data(), cos_out.data(), w.a.size());
            }});
        }
#if defined(__APPLE__)
        if (const auto f = wray_bench::vforce<T>(n))
        {
            const auto position = s.throughput_rows.begin() + (!s.throughput_rows.empty() && s.throughput_rows.front().name == "libm" ? 1 : 0);
            s.throughput_rows.insert(position, {"vforce", [f](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                f(w.a.data(), w.b.data(), out.data(), cos_out.data(), w.a.size());
            }});
        }
#endif
        const auto add_reference = [&](const char *name, wray_bench::function<T> f) {
            if (!f)
                return;
            const auto position = std::find_if(s.throughput_rows.begin(), s.throughput_rows.end(), [](const auto &row) { return row.name.starts_with("wray"); });
            s.throughput_rows.insert(position, {name, [f](const workload<T>& w, std::span<T> out, std::span<T> cos_out) {
                f(w.a.data(), w.b.data(), out.data(), cos_out.data(), w.a.size());
            }});
        };
#if defined(WRAY_BENCH_HAS_SLEEF)
        add_reference("sleef_u10", wray_bench::sleef<T>(n, wray_bench::sleef_accuracy::u10));
        add_reference("sleef_u35", wray_bench::sleef<T>(n, wray_bench::sleef_accuracy::u35));
#endif
#if defined(WRAY_BENCH_HAS_LIBMVEC)
        add_reference("libmvec", wray_bench::libmvec<T>(n));
#endif
        return s;
    }

    template <class T>
    bool agrees(T got, T want)
    {
        if (std::isnan(want))
            return std::isnan(got);
        if (std::isinf(want))
            return got == want;
        const T scale = std::max(std::abs(want), T(1e-3));
        return std::abs(got - want) <= scale * T(std::is_same_v<T, float> ? 1e-4 : 1e-10);
    }

    template <class T>
    void validate(const char *type, const char *fn, suite<T> &s, const workload<T> &w)
    {
        const auto reference = std::find_if(s.throughput_rows.begin(), s.throughput_rows.end(), [](const auto &row) { return row.name == "libm"; });
        if (reference == s.throughput_rows.end())
            return;
        std::vector<T> want(kSize), want_cos(kSize), got(kSize), got_cos(kSize);
        reference->run(w, want, want_cos);
        const bool pair = std::strcmp(fn, "sincos") == 0;
        for (auto &row : s.throughput_rows)
        {
            if (&row == &*reference || row.name.starts_with("wray"))
                continue;
            row.run(w, got, got_cos);
            for (std::size_t i = 0; i < kSize; ++i)
                if (!agrees(got[i], want[i]) || (pair && !agrees(got_cos[i], want_cos[i])))
                {
                    std::fprintf(stderr, "%s %s: %s disagrees with libm at input %.9g: %.9g vs %.9g\n", type, fn, row.name.c_str(),
                                 double(w.a[i]), double(got[i]), double(want[i]));
                    std::exit(3);
                }
        }
    }

    struct timing
    {
        std::vector<double> samples;

        void add(double ns) { samples.push_back(ns); }

        double quantile(double q) const
        {
            std::vector<double> v = samples;
            std::sort(v.begin(), v.end());
            return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5)];
        }
    };

    struct comparison
    {
        std::string_view baseline;
        timing log_ratios;

        std::array<double, 3> interval() const
        {
            constexpr std::size_t count = 5000;
            std::vector<double> bootstrap(count);
            std::vector<double> sample(log_ratios.samples.size());
            std::mt19937_64 rng(17);
            std::uniform_int_distribution<std::size_t> pick(0, sample.size() - 1);
            for (auto &value : bootstrap)
            {
                for (auto &x : sample)
                    x = log_ratios.samples[pick(rng)];
                std::sort(sample.begin(), sample.end());
                value = sample[sample.size() / 2];
            }
            std::sort(bootstrap.begin(), bootstrap.end());
            return {std::exp(log_ratios.quantile(0.5)),
                    std::exp(bootstrap[static_cast<std::size_t>(0.025 * (count - 1) + 0.5)]),
                    std::exp(bootstrap[static_cast<std::size_t>(0.975 * (count - 1) + 0.5)])};
        }
    };

    void emit(std::FILE* json, bool& first, const char* type, const char* fn, const char* variant, const char* mode,
              const timing& t, const char* baseline = "none", const std::vector<comparison>& comparisons = {})
    {
        const double best = t.quantile(0.0);
        std::printf("%-4s %-10s %-22s %-11s %8.3f ns  median %8.3f  p90 %8.3f  baseline %s\n",
                    type, mode, fn, variant, best, t.quantile(0.5), t.quantile(0.9), baseline);
        std::fprintf(json, "%s\n  {\"type\": \"%s\", \"function\": \"%s\", \"variant\": \"%s\", \"mode\": \"%s\", \"baseline\": \"%s\", \"ns\": %.5f, \"median\": %.5f, \"p90\": %.5f, \"rounds\": %d, \"sample_count\": %zu, \"samples_ns\": [",
                     first ? "" : ",", type, fn, variant, mode, baseline, best, t.quantile(0.5), t.quantile(0.9), kRounds, t.samples.size());
        for (std::size_t i = 0; i < t.samples.size(); ++i)
            std::fprintf(json, "%s%.7f", i ? ", " : "", t.samples[i]);
        std::fprintf(json, "], \"comparisons\": [");
        for (std::size_t i = 0; i < comparisons.size(); ++i)
        {
            const auto &c = comparisons[i];
            const auto bounds = c.interval();
            std::printf("     vs %-6s %.3fx [%.3f, %.3f] paired 95%% CI\n",
                        c.baseline.data(), bounds[0], bounds[1], bounds[2]);
            std::fprintf(json, "%s{\"baseline\": \"%s\", \"speedup\": %.7f, \"ci_low\": %.7f, \"ci_high\": %.7f, \"log_ratios\": [",
                         i ? ", " : "", c.baseline.data(), bounds[0], bounds[1], bounds[2]);
            for (std::size_t j = 0; j < c.log_ratios.samples.size(); ++j)
                std::fprintf(json, "%s%.9f", j ? ", " : "", c.log_ratios.samples[j]);
            std::fprintf(json, "]}");
        }
        std::fprintf(json, "]}");
        first = false;
    }

    template <class T>
    void run(const char* type, std::FILE* json, bool& first, const std::vector<std::string>& only, const char* selected_baseline)
    {
        const char* names[] = {"product_log", "omega", "smoothing_coefficient", "pole_from_cutoff", "prewarp", "ratio_from_cents", "hz_to_midi", "midi_to_hz", "erf", "cbrt", "asinh", "cosh", "sinh", "tanpi", "tan_turns", "cos_turns", "sin_turns", "log10", "log1p", "exp", "expm1", "exp2", "log", "log2", "sin", "cos", "sincos", "tan", "atan", "atan2", "tanh", "sigmoid", "pow", "db_to_gain",
                               "gain_to_db"};
        std::mt19937_64 rng(7);
        std::mt19937_64 edge_rng(11);
        volatile T sink = 0;
        volatile T zero = 0;
        for (const char* n : names)
        {
            auto w = make<T>(n, rng);
            auto e = edge<T>(n, edge_rng);
            if (e.a.empty())
                e = w;
            if (!only.empty() && std::find(only.begin(), only.end(), n) == only.end())
                continue;
            auto s = variants<T>(n);
            const auto count = s.throughput_rows.size();
            std::vector<timing> best(count);
            std::vector<timing> best_chain(s.latency_rows.size());
            std::vector<timing> best_edge(count);
            std::vector<std::vector<comparison>> comparisons(count);
            std::vector<std::size_t> references;
            const char *baseline = "none";
            for (std::size_t k = 0; k < count; ++k)
            {
                if (!s.throughput_rows[k].name.starts_with("wray"))
                    references.push_back(k);
                if (s.throughput_rows[k].name == selected_baseline)
                    baseline = selected_baseline;
            }
            validate(type, n, s, w);
            std::vector<T> out(kSize), cos_out(kSize);
            for (auto &row : s.throughput_rows)
                row.run(w, out, cos_out);
            for (std::size_t k = 0; k < count; ++k)
            {
                if (!s.throughput_rows[k].name.starts_with("wray"))
                    continue;
                for (const auto reference : references)
                    comparisons[k].push_back({s.throughput_rows[reference].name, {}});
                comparisons[k].push_back({"self", {}});
            }
            for (int round = 0; round < kRounds; ++round)
            {
                const auto measure = [&](std::size_t k, const workload<T> &input) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kRepeats; ++r)
                        s.throughput_rows[k].run(input, out, cos_out);
                    const auto t1 = std::chrono::steady_clock::now();
                    const auto index = static_cast<std::size_t>(round) % kSize;
                    sink = sink + out[index] + (std::strcmp(n, "sincos") == 0 ? cos_out[index] : T(0));
                    return std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kRepeats) * double(kSize));
                };
                for (std::size_t k = 0; k < count; ++k)
                {
                    if (!s.throughput_rows[k].name.starts_with("wray"))
                        continue;
                    for (std::size_t p = 0; p <= references.size(); ++p)
                    {
                        const auto reference = p == references.size() ? k : references[p];
                        const std::array<std::size_t, 4> order = round % 2
                            ? std::array<std::size_t, 4>{k, reference, reference, k}
                            : std::array<std::size_t, 4>{reference, k, k, reference};
                        std::array<double, 4> ns;
                        for (std::size_t j = 0; j < order.size(); ++j)
                            ns[j] = measure(order[j], w);
                        const double outside = (ns[0] + ns[3]) * 0.5;
                        const double inside = (ns[1] + ns[2]) * 0.5;
                        const double a = round % 2 ? inside : outside;
                        const double b = round % 2 ? outside : inside;
                        comparisons[k][p].log_ratios.add(std::log(a / b));
                        if (p < references.size())
                        {
                            best[reference].add(a);
                            best[k].add(b);
                        }
                        else if (references.empty())
                            best[k].add((a + b) * 0.5);
                    }
                    best_edge[k].add(measure(k, e));
                }
                for (std::size_t k = 0; k < s.latency_rows.size(); ++k)
                {
                    const T tiny = zero;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kChainRepeats; ++r)
                        sink = sink + s.latency_rows[k].run(w, tiny);
                    const auto t1 = std::chrono::steady_clock::now();
                    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kChainRepeats) * double(kSize - 1));
                    best_chain[k].add(ns);
                }
            }
            for (std::size_t k = 0; k < s.throughput_rows.size(); ++k)
                emit(json, first, type, n, s.throughput_rows[k].name.c_str(), "throughput", best[k], baseline, comparisons[k]);
            for (std::size_t k = 0; k < s.throughput_rows.size(); ++k)
                if (!best_edge[k].samples.empty())
                    emit(json, first, type, n, s.throughput_rows[k].name.c_str(), "edge", best_edge[k]);
            const char *latency_baseline = references.empty() ? "none" : "libm";
            for (std::size_t k = 0; k < s.latency_rows.size(); ++k)
                emit(json, first, type, n, s.latency_rows[k].name.c_str(), "latency", best_chain[k], latency_baseline);
        }
    }

    template <class F>
    timing time_block(F f, std::size_t items)
    {
        timing t;
        for (int round = 0; round < kRounds; ++round)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < kRepeats; ++r)
                f();
            const auto t1 = std::chrono::steady_clock::now();
            t.add(std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kRepeats) * double(items)));
        }
        return t;
    }

    template <class Shape, class T>
    void adaa_rows(const char* type, const char* name, std::FILE* json, bool& first, const std::vector<T>& drive)
    {
        std::vector<T> out(drive.size());
        volatile T sink = 0;
        const auto pointwise = time_block([&] {
            for (std::size_t i = 0; i < drive.size(); ++i)
                out[i] = Shape::f(drive[i]);
            sink = sink + out[drive.size() / 2];
        }, drive.size());
        wray::adaa::first_order<Shape, T> stage;
        const auto first_order = time_block([&] {
            for (std::size_t i = 0; i < drive.size(); ++i)
                out[i] = stage(drive[i]);
            sink = sink + out[drive.size() / 2];
        }, drive.size());
        const std::string fn = std::string("adaa_") + name;
        emit(json, first, type, fn.c_str(), "pointwise", "block", pointwise);
        emit(json, first, type, fn.c_str(), "first_order", "block", first_order);
    }

    template <class T>
    void dsp(const char* type, std::FILE* json, bool& first)
    {
        constexpr std::size_t block = 512;
        std::vector<T> drive(block);
        for (std::size_t i = 0; i < block; ++i)
            drive[i] = T(4) * std::sin(T(2) * std::numbers::pi_v<T> * T(1000) * T(i) / T(48000));
        adaa_rows<wray::adaa::hard_clip, T>(type, "hard_clip", json, first, drive);
        adaa_rows<wray::adaa::soft_clip, T>(type, "soft_clip", json, first, drive);
        adaa_rows<wray::adaa::tanh, T>(type, "tanh", json, first, drive);
        adaa_rows<wray::adaa::atan, T>(type, "atan", json, first, drive);
        adaa_rows<wray::adaa::algebraic, T>(type, "algebraic", json, first, drive);
        adaa_rows<wray::adaa::radical, T>(type, "radical", json, first, drive);

        std::vector<T> io = drive;
        wray::filters::biquad<T> bq;
        bq.prepare(48000.0);
        bq.set(wray::filters::biquad_type::peak, 1000.0, 0.7, 6.0);
        emit(json, first, type, "biquad", "wray", "block", time_block([&] { bq.process(std::span<T>(io)); }, block));
        wray::filters::svf<T> sv;
        sv.prepare(48000.0);
        sv.set(wray::filters::svf_type::peak, 1000.0, 0.7, 6.0);
        emit(json, first, type, "svf", "wray", "block", time_block([&] { sv.process(std::span<T>(io)); }, block));
        wray::filters::tpt_one_pole<T> op;
        op.prepare(48000.0);
        op.set(wray::filters::one_pole_type::lowpass, 1000.0);
        emit(json, first, type, "one_pole", "wray", "block", time_block([&] { op.process(std::span<T>(io)); }, block));

        std::vector<double> hz(block), q(block), gain(block);
        std::mt19937_64 rng(3);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (std::size_t i = 0; i < block; ++i)
        {
            hz[i] = 20.0 * std::pow(1000.0, u(rng));
            q[i] = 0.3 + 5.0 * u(rng);
            gain[i] = -24.0 + 48.0 * u(rng);
        }
        volatile double sink = 0;
        emit(json, first, type, "design_biquad", "wray", "update", time_block([&] {
            double acc = 0;
            for (std::size_t i = 0; i < block; ++i)
                acc += wray::filters::design_biquad(wray::filters::biquad_type::peak, 48000.0, hz[i], q[i], gain[i]).b0;
            sink = sink + acc;
        }, block));
        emit(json, first, type, "design_svf", "wray", "update", time_block([&] {
            double acc = 0;
            for (std::size_t i = 0; i < block; ++i)
                acc += wray::filters::design_svf(wray::filters::svf_type::peak, 48000.0, hz[i], q[i], gain[i]).a2;
            sink = sink + acc;
        }, block));
        emit(json, first, type, "biquad_set", "wray", "update", time_block([&] {
            for (std::size_t i = 0; i < block; ++i)
                bq.set(wray::filters::biquad_type::peak, hz[i], q[i], gain[i]);
            sink = sink + bq.coefficients().b0;
        }, block));
    }

    std::string cpu_name()
    {
        if (const char *name = std::getenv("WRAY_BENCH_CPU"))
            return name;
#if defined(__APPLE__)
        char buf[256] = {};
        std::size_t len = sizeof buf;
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
            return buf;
#elif defined(__linux__)
        if (std::FILE* f = std::fopen("/proc/cpuinfo", "r"))
        {
            char line[512];
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, "model name", 10) == 0 || std::strncmp(line, "CPU part", 8) == 0)
                {
                    std::fclose(f);
                    std::string s = std::strchr(line, ':') ? std::strchr(line, ':') + 2 : line;
                    s.erase(s.find_last_not_of(" \n") + 1);
                    return s;
                }
            std::fclose(f);
        }
#endif
        return "unknown";
    }

    std::string compiler()
    {
#if defined(__clang__)
        return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
        return "msvc " + std::to_string(_MSC_VER);
#else
        return "unknown";
#endif
    }

    const char* backend()
    {
#if defined(WRAY_SIMD_AVX512)
        return "avx512";
#elif defined(WRAY_SIMD_AVX2)
        return "avx2";
#elif defined(WRAY_SIMD_NEON)
        return "neon";
#elif defined(WRAY_SIMD_SSE2)
        return "sse2";
#else
        return "scalar";
#endif
    }

    std::string platform()
    {
#if defined(__APPLE__)
        char version[128] = {};
        std::size_t length = sizeof version;
        if (sysctlbyname("kern.osproductversion", version, &length, nullptr, 0) == 0)
            return std::string("macOS ") + version;
        return "macOS";
#elif defined(_WIN32)
        return "Windows";
#elif defined(__linux__)
        return "Linux";
#else
        return "unknown";
#endif
    }

    std::string escaped(std::string s)
    {
        std::string out;
        for (char c : s)
        {
            if (c == '"' || c == '\\')
                out += '\\';
            out += c;
        }
        return out;
    }
}

int main(int argc, char** argv)
{
    const char* path = "bench.json";
#if defined(__APPLE__)
    const char* baseline = "vforce";
#else
    const char* baseline = "libm";
#endif
    bool has_path = false;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        if (arg == "--help")
        {
            std::puts("Usage: wray_bench [output.json] [--baseline libm|vforce|sleef_u10|sleef_u35|libmvec] [function ...]");
            std::puts("Every available baseline is recorded; selection sets the primary denominator.");
            std::puts("Vector baselines in this build: " WRAY_BENCH_VECTOR_BASELINES);
            std::puts("Latency uses scalar libm. Throughput uses paired ABBA rounds, a self-control, and bootstrap intervals.");
            return 0;
        }
        if (arg == "--baseline")
        {
            const std::string_view known[] = {"libm", "vforce", "sleef_u10", "sleef_u35", "libmvec"};
            if (++i == argc || std::find(std::begin(known), std::end(known), std::string_view(argv[i])) == std::end(known))
            {
                std::fputs("--baseline requires libm, vforce, sleef_u10, sleef_u35 or libmvec\n", stderr);
                return 2;
            }
            baseline = argv[i];
        }
        else if (arg.starts_with("--"))
        {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 2;
        }
        else if (!has_path)
        {
            path = argv[i];
            has_path = true;
        }
        else
            only.emplace_back(arg);
    }
#if !defined(__APPLE__)
    if (std::strcmp(baseline, "vforce") == 0)
    {
        std::fputs("vForce is available only on macOS\n", stderr);
        return 2;
    }
#endif
#if !defined(WRAY_BENCH_HAS_SLEEF)
    if (std::string_view(baseline).starts_with("sleef"))
    {
        std::fputs("This build has no SLEEF; install SLEEF 3.9 or newer where CMake can find it\n", stderr);
        return 2;
    }
#endif
#if !defined(WRAY_BENCH_HAS_LIBMVEC)
    if (std::strcmp(baseline, "libmvec") == 0)
    {
        std::fputs("libmvec is available only with glibc 2.35 or newer on x86-64\n", stderr);
        return 2;
    }
#endif
    std::FILE* json = std::fopen(path, "w");
    if (!json)
    {
        std::perror(path);
        return 1;
    }
    const char *revision = std::getenv("WRAY_BENCH_REVISION");
    if (!revision)
        revision = std::getenv("GITHUB_SHA");
    std::fprintf(json, "{\n\"meta\": {\"cpu\": \"%s\", \"platform\": \"%s\", \"compiler\": \"%s\", \"flags\": \"%s\", \"backend\": \"%s\", \"revision\": \"%s\", \"baseline\": \"%s\", \"libm_vectorization\": \"disabled\", \"libm_flags\": \"%s\", \"vector_baselines\": \"%s\", \"fma\": %s, \"rounds\": %d, \"repeats\": %d, \"sampling\": \"alternating ABBA; self-control; median paired ratios; 5000 bootstrap resamples; 95%% intervals\", \"statistic\": \"ns is minimum sample; median and p90 are sample quantiles\"},\n\"rows\": [",
                 escaped(cpu_name()).c_str(), escaped(platform()).c_str(), escaped(compiler()).c_str(), escaped(WRAY_BENCH_FLAGS).c_str(),
                 backend(), escaped(revision ? revision : "unrecorded").c_str(), baseline, escaped(WRAY_BENCH_LIBM_FLAGS).c_str(),
                 escaped(WRAY_BENCH_VECTOR_BASELINES).c_str(), WRAY_HAS_FMA ? "true" : "false", kRounds, kRepeats);
    bool first = true;
    run<float>("f32", json, first, only, baseline);
    run<double>("f64", json, first, only, baseline);
    if (only.empty())
    {
        dsp<float>("f32", json, first);
        dsp<double>("f64", json, first);
    }
    std::fprintf(json, "\n]\n}\n");
    std::fclose(json);
    return 0;
}
