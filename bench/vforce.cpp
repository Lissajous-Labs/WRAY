#include "baselines.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <type_traits>

namespace
{
    int count(std::size_t n)
    {
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            std::abort();
        return static_cast<int>(n);
    }

    template<class T, auto Float, auto Double>
    void call(const T *in, T *out, std::size_t size)
    {
        const int n = count(size);
        if constexpr (std::is_same_v<T, float>) Float(out, in, &n);
        else Double(out, in, &n);
    }

    template<class T, auto Float, auto Double>
    wray_bench::function<T> native()
    {
        return [](const T *a, const T *, T *out, T *, std::size_t n) { call<T, Float, Double>(a, out, n); };
    }

    template<class T, auto Float, auto Double, class F>
    wray_bench::function<T> scaled(F)
    {
        return [](const T *a, const T *, T *out, T *, std::size_t n) {
            std::transform(a, a + n, out, F{});
            call<T, Float, Double>(out, out, n);
        };
    }
}

namespace wray_bench
{
    template<class T>
    function<T> vforce(std::string_view name)
    {
#define WRAY_BENCH_VFORCE(operation) if (name == #operation) return native<T, vv##operation##f, vv##operation>()
        WRAY_BENCH_VFORCE(exp);
        WRAY_BENCH_VFORCE(exp2);
        WRAY_BENCH_VFORCE(expm1);
        WRAY_BENCH_VFORCE(log);
        WRAY_BENCH_VFORCE(log2);
        WRAY_BENCH_VFORCE(log10);
        WRAY_BENCH_VFORCE(log1p);
        WRAY_BENCH_VFORCE(sin);
        WRAY_BENCH_VFORCE(cos);
        WRAY_BENCH_VFORCE(tan);
        WRAY_BENCH_VFORCE(atan);
        WRAY_BENCH_VFORCE(sinh);
        WRAY_BENCH_VFORCE(cosh);
        WRAY_BENCH_VFORCE(tanh);
        WRAY_BENCH_VFORCE(asinh);
        WRAY_BENCH_VFORCE(cbrt);
#undef WRAY_BENCH_VFORCE
        if (name == "sincos")
            return [](const T *a, const T *, T *out, T *cos_out, std::size_t size) {
                const int n = count(size);
                if constexpr (std::is_same_v<T, float>) vvsincosf(out, cos_out, a, &n);
                else vvsincos(out, cos_out, a, &n);
            };
        if (name == "pow")
            return [](const T *a, const T *b, T *out, T *, std::size_t size) {
                const int n = count(size);
                if constexpr (std::is_same_v<T, float>) vvpowf(out, b, a, &n);
                else vvpow(out, b, a, &n);
            };
        if (name == "atan2")
            return [](const T *a, const T *b, T *out, T *, std::size_t size) {
                const int n = count(size);
                if constexpr (std::is_same_v<T, float>) vvatan2f(out, a, b, &n);
                else vvatan2(out, a, b, &n);
            };
        if (name == "sin_turns")
            return scaled<T, vvsinf, vvsin>([](T x) { return T(2) * std::numbers::pi_v<T> * x; });
        if (name == "cos_turns")
            return scaled<T, vvcosf, vvcos>([](T x) { return T(2) * std::numbers::pi_v<T> * x; });
        if (name == "tan_turns")
            return scaled<T, vvtanf, vvtan>([](T x) { return T(2) * std::numbers::pi_v<T> * x; });
        if (name == "tanpi")
            return scaled<T, vvtanf, vvtan>([](T x) { return std::numbers::pi_v<T> * x; });
        if (name == "ratio_from_cents")
            return scaled<T, vvexp2f, vvexp2>([](T x) { return x / T(1200); });
        if (name == "db_to_gain")
            return scaled<T, vvexp2f, vvexp2>([](T x) { return x * (std::numbers::ln10_v<T> / (T(20) * std::numbers::ln2_v<T>)); });
        if (name == "sigmoid")
            return [](const T *a, const T *, T *out, T *, std::size_t n) {
                std::transform(a, a + n, out, [](T x) { return -x; });
                call<T, vvexpf, vvexp>(out, out, n);
                std::transform(out, out + n, out, [](T x) { return T(1) / (T(1) + x); });
            };
        if (name == "midi_to_hz")
            return [](const T *a, const T *, T *out, T *, std::size_t n) {
                std::transform(a, a + n, out, [](T x) { return (x - T(69)) / T(12); });
                call<T, vvexp2f, vvexp2>(out, out, n);
                std::transform(out, out + n, out, [](T x) { return T(440) * x; });
            };
        if (name == "hz_to_midi")
            return [](const T *a, const T *, T *out, T *, std::size_t n) {
                std::transform(a, a + n, out, [](T x) { return x / T(440); });
                call<T, vvlog2f, vvlog2>(out, out, n);
                std::transform(out, out + n, out, [](T x) { return T(69) + T(12) * x; });
            };
        if (name == "gain_to_db")
            return [](const T *a, const T *, T *out, T *, std::size_t n) {
                call<T, vvlog10f, vvlog10>(a, out, n);
                std::transform(out, out + n, out, [](T x) { return T(20) * x; });
            };
        if (name == "smoothing_coefficient")
            return [](const T *a, const T *b, T *out, T *, std::size_t n) {
                std::transform(a, a + n, b, out, [](T t, T fs) { return T(-1) / (t * fs); });
                call<T, vvexpm1f, vvexpm1>(out, out, n);
                std::transform(out, out + n, out, [](T x) { return -x; });
            };
        if (name == "pole_from_cutoff")
            return [](const T *a, const T *b, T *out, T *, std::size_t n) {
                std::transform(a, a + n, b, out, [](T hz, T fs) { return T(-2) * std::numbers::pi_v<T> * hz / fs; });
                call<T, vvexpf, vvexp>(out, out, n);
            };
        if (name == "prewarp")
            return [](const T *a, const T *b, T *out, T *, std::size_t n) {
                std::transform(a, a + n, b, out, [](T hz, T fs) { return std::numbers::pi_v<T> * hz / fs; });
                call<T, vvtanf, vvtan>(out, out, n);
            };
        return nullptr;
    }

    template function<float> vforce<float>(std::string_view);
    template function<double> vforce<double>(std::string_view);
}
