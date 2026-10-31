#include <wray/wray.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace
{
    constexpr std::size_t kSize = 4096;
    constexpr int kRounds = 25;
    constexpr int kRepeats = 40;

    template <class T>
    struct workload
    {
        std::string name;
        std::vector<T> a;
        std::vector<T> b;
    };

    template <class T>
    struct variant
    {
        std::string name;
        std::function<void(const workload<T>&, std::vector<T>&)> run;
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
        workload<T> w{name, {}, {}};
        if (name == "exp" || name == "sigmoid")
            w.a = uniform<T>(rng, -80, 80);
        else if (name == "exp2")
            w.a = uniform<T>(rng, -120, 120);
        else if (name == "log" || name == "log2" || name == "gain_to_db")
            w.a = log_uniform<T>(rng, 1e-30, 1e30);
        else if (name == "sin" || name == "cos" || name == "tan")
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

    template <class T, class F>
    variant<T> scalar_loop(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) {
                    for (std::size_t i = 0; i < kSize; ++i)
                        out[i] = f(w.a[i]);
                }};
    }

    template <class T, class F>
    variant<T> scalar_loop2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) {
                    for (std::size_t i = 0; i < kSize; ++i)
                        out[i] = f(w.a[i], w.b[i]);
                }};
    }

    template <class T, class F>
    variant<T> simd_loop(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) { wray::math::apply(w.a, out, f); }};
    }

    template <class T, class F>
    variant<T> simd_loop2(const char* name, F f)
    {
        return {name, [f](const workload<T>& w, std::vector<T>& out) { wray::math::apply(w.a, w.b, out, f); }};
    }

    template <class T>
    T std_fn(const std::string& n, T x)
    {
        if (n == "exp") return std::exp(x);
        if (n == "exp2") return std::exp2(x);
        if (n == "log") return std::log(x);
        if (n == "log2") return std::log2(x);
        if (n == "sin") return std::sin(x);
        if (n == "cos") return std::cos(x);
        if (n == "tan") return std::tan(x);
        if (n == "atan") return std::atan(x);
        if (n == "tanh") return std::tanh(x);
        if (n == "sigmoid") return T(1) / (T(1) + std::exp(-x));
        if (n == "db_to_gain") return std::pow(T(10), x / T(20));
        if (n == "gain_to_db") return T(20) * std::log10(x);
        return x;
    }

    template <class T>
    std::vector<variant<T>> variants(const std::string& n)
    {
        std::vector<variant<T>> v;
        if (n == "atan2")
        {
            v.push_back(scalar_loop2<T>("std", [](T y, T x) { return std::atan2(y, x); }));
            v.push_back(simd_loop2<T>("wray", [](auto y, auto x) { return wray::atan2(y, x); }));
            if constexpr (std::is_same_v<T, float>)
                v.push_back(simd_loop2<T>("wray fast", [](auto y, auto x) { return wray::fast::atan2(y, x); }));
#if defined(__APPLE__)
            v.push_back({"accelerate", [](const workload<T>& w, std::vector<T>& out) {
                             const int n = static_cast<int>(kSize);
                             if constexpr (std::is_same_v<T, float>) vvatan2f(out.data(), w.a.data(), w.b.data(), &n);
                             else vvatan2(out.data(), w.a.data(), w.b.data(), &n);
                         }});
#endif
            return v;
        }
        if (n == "pow")
        {
            v.push_back(scalar_loop2<T>("std", [](T x, T y) { return std::pow(x, y); }));
            v.push_back(simd_loop2<T>("wray", [](auto x, auto y) { return wray::pow(x, y); }));
            if constexpr (std::is_same_v<T, float>)
                v.push_back(simd_loop2<T>("wray fast", [](auto x, auto y) { return wray::fast::pow(x, y); }));
#if defined(__APPLE__)
            v.push_back({"accelerate", [](const workload<T>& w, std::vector<T>& out) {
                             const int n = static_cast<int>(kSize);
                             if constexpr (std::is_same_v<T, float>) vvpowf(out.data(), w.b.data(), w.a.data(), &n);
                             else vvpow(out.data(), w.b.data(), w.a.data(), &n);
                         }});
#endif
            return v;
        }
        v.push_back(scalar_loop<T>("std", [n](T x) { return std_fn<T>(n, x); }));
        using fn = std::function<void(const std::vector<T>&, std::vector<T>&)>;
        fn simd;
        fn fast;
        fn accel;
        auto bind = [](auto f) { return fn([f](const std::vector<T>& in, std::vector<T>& out) { wray::math::apply(in, out, f); }); };
        if (n == "exp") { simd = bind([](auto x) { return wray::exp(x); }); }
        if (n == "exp2") { simd = bind([](auto x) { return wray::exp2(x); }); }
        if (n == "log") { simd = bind([](auto x) { return wray::log(x); }); }
        if (n == "log2") { simd = bind([](auto x) { return wray::log2(x); }); }
        if (n == "sin") { simd = bind([](auto x) { return wray::sin(x); }); }
        if (n == "cos") { simd = bind([](auto x) { return wray::cos(x); }); }
        if (n == "tan") { simd = bind([](auto x) { return wray::tan(x); }); }
        if (n == "atan") { simd = bind([](auto x) { return wray::atan(x); }); }
        if (n == "tanh") { simd = bind([](auto x) { return wray::tanh(x); }); }
        if (n == "sigmoid") { simd = bind([](auto x) { return wray::sigmoid(x); }); }
        if (n == "db_to_gain") { simd = bind([](auto x) { return wray::db_to_gain(x); }); }
        if (n == "gain_to_db") { simd = bind([](auto x) { return wray::gain_to_db(x); }); }
        if constexpr (std::is_same_v<T, float>)
        {
            if (n == "exp") { fast = bind([](auto x) { return wray::fast::exp(x); }); }
            if (n == "exp2") { fast = bind([](auto x) { return wray::fast::exp2(x); }); }
            if (n == "log") { fast = bind([](auto x) { return wray::fast::log(x); }); }
            if (n == "log2") { fast = bind([](auto x) { return wray::fast::log2(x); }); }
            if (n == "sin") { fast = bind([](auto x) { return wray::fast::sin(x); }); }
            if (n == "cos") { fast = bind([](auto x) { return wray::fast::cos(x); }); }
            if (n == "tan") { fast = bind([](auto x) { return wray::fast::tan(x); }); }
            if (n == "atan") { fast = bind([](auto x) { return wray::fast::atan(x); }); }
            if (n == "tanh") { fast = bind([](auto x) { return wray::fast::tanh(x); }); }
            if (n == "sigmoid") { fast = bind([](auto x) { return wray::fast::sigmoid(x); }); }
            if (n == "db_to_gain") { fast = bind([](auto x) { return wray::fast::db_to_gain(x); }); }
            if (n == "gain_to_db") { fast = bind([](auto x) { return wray::fast::gain_to_db(x); }); }
        }
#if defined(__APPLE__)
        auto vf = [](auto f32, auto f64) {
            return fn([f32, f64](const std::vector<T>& in, std::vector<T>& out) {
                const int c = static_cast<int>(in.size());
                if constexpr (std::is_same_v<T, float>) f32(out.data(), in.data(), &c);
                else f64(out.data(), in.data(), &c);
            });
        };
        if (n == "exp") accel = vf(vvexpf, vvexp);
        if (n == "exp2") accel = vf(vvexp2f, vvexp2);
        if (n == "log") accel = vf(vvlogf, vvlog);
        if (n == "log2") accel = vf(vvlog2f, vvlog2);
        if (n == "sin") accel = vf(vvsinf, vvsin);
        if (n == "cos") accel = vf(vvcosf, vvcos);
        if (n == "tan") accel = vf(vvtanf, vvtan);
        if (n == "atan") accel = vf(vvatanf, vvatan);
        if (n == "tanh") accel = vf(vvtanhf, vvtanh);
#endif
        v.push_back({"wray", [simd](const workload<T>& w, std::vector<T>& out) { simd(w.a, out); }});
        if (fast)
            v.push_back({"wray fast", [fast](const workload<T>& w, std::vector<T>& out) { fast(w.a, out); }});
        if (accel)
            v.push_back({"accelerate", [accel](const workload<T>& w, std::vector<T>& out) { accel(w.a, out); }});
        return v;
    }

    template <class T>
    void run(const char* type, std::FILE* json, bool& first)
    {
        const char* names[] = {"exp", "exp2", "log", "log2", "sin", "cos", "tan", "atan", "atan2", "tanh", "sigmoid", "pow", "db_to_gain", "gain_to_db"};
        std::mt19937_64 rng(7);
        volatile T sink = 0;
        for (const char* n : names)
        {
            auto w = make<T>(n, rng);
            auto vs = variants<T>(n);
            std::vector<double> best(vs.size(), 1e30);
            std::vector<T> out(kSize);
            for (int round = 0; round < kRounds; ++round)
            {
                for (std::size_t k = 0; k < vs.size(); ++k)
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kRepeats; ++r)
                        vs[k].run(w, out);
                    const auto t1 = std::chrono::steady_clock::now();
                    sink = sink + out[static_cast<std::size_t>(round) % kSize];
                    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(kRepeats) * double(kSize));
                    best[k] = std::min(best[k], ns);
                }
            }
            for (std::size_t k = 0; k < vs.size(); ++k)
            {
                std::printf("%-4s %-11s %-11s %8.3f ns\n", type, n, vs[k].name.c_str(), best[k]);
                std::fprintf(json, "%s\n  {\"type\": \"%s\", \"function\": \"%s\", \"variant\": \"%s\", \"ns\": %.5f}", first ? "" : ",", type, n, vs[k].name.c_str(), best[k]);
                first = false;
            }
        }
    }
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "bench.json";
    std::FILE* json = std::fopen(path, "w");
    std::fprintf(json, "[");
    bool first = true;
    run<float>("f32", json, first);
    run<double>("f64", json, first);
    std::fprintf(json, "\n]\n");
    std::fclose(json);
    return 0;
}
