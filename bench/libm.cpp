#include "baselines.h"

#include <cmath>
#include <numbers>

namespace
{
    template<class T, class F>
    wray_bench::function<T> unary(F)
    {
        return [](const T *a, const T *, T *out, T *, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = F{}(a[i]);
        };
    }

    template<class T, class F>
    wray_bench::function<T> binary(F)
    {
        return [](const T *a, const T *b, T *out, T *, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = F{}(a[i], b[i]);
        };
    }
}

namespace wray_bench
{
    template<class T>
    function<T> libm(std::string_view name)
    {
#define WRAY_BENCH_LIBM(name) if (name_key == #name) return unary<T>([](T x) { return std::name(x); })
        const auto name_key = name;
        WRAY_BENCH_LIBM(exp);
        WRAY_BENCH_LIBM(exp2);
        WRAY_BENCH_LIBM(expm1);
        WRAY_BENCH_LIBM(log);
        WRAY_BENCH_LIBM(log2);
        WRAY_BENCH_LIBM(log10);
        WRAY_BENCH_LIBM(log1p);
        WRAY_BENCH_LIBM(sin);
        WRAY_BENCH_LIBM(cos);
        WRAY_BENCH_LIBM(tan);
        WRAY_BENCH_LIBM(atan);
        WRAY_BENCH_LIBM(sinh);
        WRAY_BENCH_LIBM(cosh);
        WRAY_BENCH_LIBM(tanh);
        WRAY_BENCH_LIBM(asinh);
        WRAY_BENCH_LIBM(cbrt);
        WRAY_BENCH_LIBM(erf);
#undef WRAY_BENCH_LIBM
        if (name == "sincos")
            return [](const T *a, const T *, T *out, T *cos_out, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                {
                    const T x = a[i];
                    out[i] = std::sin(x);
                    cos_out[i] = std::cos(x);
                }
            };
        if (name == "pow")
            return binary<T>([](T x, T y) { return std::pow(x, y); });
        if (name == "atan2")
            return binary<T>([](T y, T x) { return std::atan2(y, x); });
        if (name == "sin_turns")
            return unary<T>([](T x) { return std::sin(T(2) * std::numbers::pi_v<T> * x); });
        if (name == "cos_turns")
            return unary<T>([](T x) { return std::cos(T(2) * std::numbers::pi_v<T> * x); });
        if (name == "tan_turns")
            return unary<T>([](T x) { return std::tan(T(2) * std::numbers::pi_v<T> * x); });
        if (name == "tanpi")
            return unary<T>([](T x) { return std::tan(std::numbers::pi_v<T> * x); });
        if (name == "sigmoid")
            return unary<T>([](T x) { return T(1) / (T(1) + std::exp(-x)); });
        if (name == "midi_to_hz")
            return unary<T>([](T x) { return T(440) * std::exp2((x - T(69)) / T(12)); });
        if (name == "hz_to_midi")
            return unary<T>([](T x) { return T(69) + T(12) * std::log2(x / T(440)); });
        if (name == "ratio_from_cents")
            return unary<T>([](T x) { return std::exp2(x / T(1200)); });
        if (name == "db_to_gain")
            return unary<T>([](T x) { return std::exp2(x * (std::numbers::ln10_v<T> / (T(20) * std::numbers::ln2_v<T>))); });
        if (name == "gain_to_db")
            return unary<T>([](T x) { return T(20) * std::log10(x); });
        if (name == "smoothing_coefficient")
            return binary<T>([](T t, T fs) { return -std::expm1(T(-1) / (t * fs)); });
        if (name == "pole_from_cutoff")
            return binary<T>([](T hz, T fs) { return std::exp(T(-2) * std::numbers::pi_v<T> * hz / fs); });
        if (name == "prewarp")
            return binary<T>([](T hz, T fs) { return std::tan(std::numbers::pi_v<T> * hz / fs); });
        return nullptr;
    }

    template function<float> libm<float>(std::string_view);
    template function<double> libm<double>(std::string_view);
}
