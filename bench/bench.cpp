#include <wray/wray.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

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
        std::function<void(const workload<T>&, std::vector<T>&)> run;
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
    throughput<T> scalar_loop(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) {
                    for (std::size_t i = 0; i < kSize; ++i)
                        out[i] = f(w.a[i]);
                }};
    }

    template <class T, class F>
    throughput<T> scalar_loop2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) {
                    for (std::size_t i = 0; i < kSize; ++i)
                        out[i] = f(w.a[i], w.b[i]);
                }};
    }

    template <class T, class F>
    throughput<T> simd_loop(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) { wray::math::apply(w.a, out, f); }};
    }

    template <class T, class F>
    throughput<T> simd_loop2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) { wray::math::apply(w.a, w.b, out, f); }};
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
            s.throughput_rows.push_back(scalar_loop<T>("std", reference));
            s.latency_rows.push_back(chain<T>("std", reference));
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
        s.throughput_rows.push_back(scalar_loop2<T>("std", reference));
        s.throughput_rows.push_back(simd_loop2<T>("wray", precise));
        s.latency_rows.push_back(chain2<T>("std", reference));
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
            binary(s, [](T t, T fs) { return T(1) - std::exp(T(-1) / (t * fs)); }, [](auto t, auto fs) { return wray::smoothing_coefficient(t, fs); }, none{});
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
            unary(s, [](T x) { return std::sin(x); }, [](auto x) { return wray::sin(x); }, [](auto x) { return wray::fast::sin(x); });
        if (n == "cos")
            unary(s, [](T x) { return std::cos(x); }, [](auto x) { return wray::cos(x); }, [](auto x) { return wray::fast::cos(x); });
        if (n == "sincos")
            unary(s, [](T x) { return std::sin(x) + std::cos(x); },
                  [](auto x) {
                      const auto r = wray::sincos(x);
                      return r.sin + r.cos;
                  },
                  [](auto x) {
                      const auto r = wray::fast::sincos(x);
                      return r.sin + r.cos;
                  });
        if (n == "tan")
            unary(s, [](T x) { return std::tan(x); }, [](auto x) { return wray::tan(x); }, [](auto x) { return wray::fast::tan(x); });
        if (n == "atan")
            unary(s, [](T x) { return std::atan(x); }, [](auto x) { return wray::atan(x); }, [](auto x) { return wray::fast::atan(x); });
        if (n == "atan2")
            binary(s, [](T y, T x) { return std::atan2(y, x); }, [](auto y, auto x) { return wray::atan2(y, x); },
                   [](auto y, auto x) { return wray::fast::atan2(y, x); });
        if (n == "tanh")
            unary(s, [](T x) { return std::tanh(x); }, [](auto x) { return wray::tanh(x); }, [](auto x) { return wray::fast::tanh(x); });
        if (n == "sigmoid")
            unary(s, [](T x) { return T(1) / (T(1) + std::exp(-x)); }, [](auto x) { return wray::sigmoid(x); },
                  [](auto x) { return wray::fast::sigmoid(x); });
        if (n == "pow")
            binary(s, [](T x, T y) { return std::pow(x, y); }, [](auto x, auto y) { return wray::pow(x, y); },
                   [](auto x, auto y) { return wray::fast::pow(x, y); });
        if (n == "db_to_gain")
            unary(s, [](T x) { return std::pow(T(10), x / T(20)); }, [](auto x) { return wray::db_to_gain(x); },
                  [](auto x) { return wray::fast::db_to_gain(x); });
        if (n == "gain_to_db")
            unary(s, [](T x) { return T(20) * std::log10(x); }, [](auto x) { return wray::gain_to_db(x); },
                  [](auto x) { return wray::fast::gain_to_db(x); });
        return s;
    }

    void emit(std::FILE* json, bool& first, const char* type, const char* fn, const char* variant, const char* mode, double ns)
    {
        std::printf("%-4s %-10s %-12s %-11s %8.3f ns\n", type, mode, fn, variant, ns);
        std::fprintf(json, "%s\n  {\"type\": \"%s\", \"function\": \"%s\", \"variant\": \"%s\", \"mode\": \"%s\", \"ns\": %.5f}", first ? "" : ",", type, fn,
                     variant, mode, ns);
        first = false;
    }

    template <class T>
    void run(const char* type, std::FILE* json, bool& first, const std::vector<std::string>& only)
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
            std::vector<double> best(s.throughput_rows.size(), 1e30);
            std::vector<double> best_chain(s.latency_rows.size(), 1e30);
            std::vector<double> best_edge(s.throughput_rows.size(), 1e30);
            std::vector<T> out(kSize);
            for (int round = 0; round < kRounds; ++round)
            {
                for (std::size_t k = 0; k < s.throughput_rows.size(); ++k)
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kRepeats; ++r)
                        s.throughput_rows[k].run(w, out);
                    const auto t1 = std::chrono::steady_clock::now();
                    sink = sink + out[static_cast<std::size_t>(round) % kSize];
                    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kRepeats) * double(kSize));
                    best[k] = std::min(best[k], ns);
                    if (s.throughput_rows[k].name.starts_with("wray"))
                    {
                        const auto e0 = std::chrono::steady_clock::now();
                        for (int r = 0; r < kRepeats; ++r)
                            s.throughput_rows[k].run(e, out);
                        const auto e1 = std::chrono::steady_clock::now();
                        sink = sink + out[static_cast<std::size_t>(round) % kSize];
                        best_edge[k] = std::min(best_edge[k], std::chrono::duration<double, std::nano>(e1 - e0).count() / (double(kRepeats) * double(kSize)));
                    }
                }
                for (std::size_t k = 0; k < s.latency_rows.size(); ++k)
                {
                    const T tiny = zero;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kChainRepeats; ++r)
                        sink = sink + s.latency_rows[k].run(w, tiny);
                    const auto t1 = std::chrono::steady_clock::now();
                    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kChainRepeats) * double(kSize - 1));
                    best_chain[k] = std::min(best_chain[k], ns);
                }
            }
            for (std::size_t k = 0; k < s.throughput_rows.size(); ++k)
                emit(json, first, type, n, s.throughput_rows[k].name.c_str(), "throughput", best[k]);
            for (std::size_t k = 0; k < s.throughput_rows.size(); ++k)
                if (best_edge[k] < 1e30)
                    emit(json, first, type, n, s.throughput_rows[k].name.c_str(), "edge", best_edge[k]);
            for (std::size_t k = 0; k < s.latency_rows.size(); ++k)
                emit(json, first, type, n, s.latency_rows[k].name.c_str(), "latency", best_chain[k]);
        }
    }
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "bench.json";
    std::vector<std::string> only;
    for (int i = 2; i < argc; ++i)
        only.emplace_back(argv[i]);
    std::FILE* json = std::fopen(path, "w");
    std::fprintf(json, "[");
    bool first = true;
    run<float>("f32", json, first, only);
    run<double>("f64", json, first, only);
    std::fprintf(json, "\n]\n");
    std::fclose(json);
    return 0;
}
