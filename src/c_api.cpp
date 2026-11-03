#include <wray/c/wray.h>
#include <wray/wray.h>

#include <span>

namespace
{
    template<class T>
    std::span<const T> in_span(const T *p, size_t n) noexcept { return {p, n}; }

    template<class T>
    std::span<T> out_span(T *p, size_t n) noexcept { return {p, n}; }
}

extern "C"
{
    int wray_c_abi_version(void) { return WRAY_C_ABI_VERSION; }

    const char *wray_c_backend(void)
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

    int wray_c_has_fma(void) { return WRAY_HAS_FMA ? 1 : 0; }

#define WRAY_C_UNARY(ns, prefix, f, T, t) \
    void wray_##prefix##f##_##t(const T *in, T *out, size_t n) { ns::f(in_span(in, n), out_span(out, n)); }
#define WRAY_C_PAIR(ns, prefix, f, T, t) \
    void wray_##prefix##f##_##t(const T *in, T *sin_out, T *cos_out, size_t n) { ns::f(in_span(in, n), out_span(sin_out, n), out_span(cos_out, n)); }
#define WRAY_C_BINARY(ns, prefix, f, T, t) \
    void wray_##prefix##f##_##t(const T *a, const T *b, T *out, size_t n) { ns::f(in_span(a, n), in_span(b, n), out_span(out, n)); }

    WRAY_C_UNARY(wray, , exp, float, f32)
    WRAY_C_UNARY(wray, , expm1, float, f32)
    WRAY_C_UNARY(wray, , exp2, float, f32)
    WRAY_C_UNARY(wray, , log, float, f32)
    WRAY_C_UNARY(wray, , log1p, float, f32)
    WRAY_C_UNARY(wray, , log10, float, f32)
    WRAY_C_UNARY(wray, , log2, float, f32)
    WRAY_C_UNARY(wray, , wrap_turns, float, f32)
    WRAY_C_UNARY(wray, , sin_turns, float, f32)
    WRAY_C_UNARY(wray, , sin, float, f32)
    WRAY_C_UNARY(wray, , cos_turns, float, f32)
    WRAY_C_UNARY(wray, , cos, float, f32)
    WRAY_C_UNARY(wray, , tanpi, float, f32)
    WRAY_C_UNARY(wray, , tan_turns, float, f32)
    WRAY_C_UNARY(wray, , tan, float, f32)
    WRAY_C_UNARY(wray, , atan, float, f32)
    WRAY_C_UNARY(wray, , sinh, float, f32)
    WRAY_C_UNARY(wray, , cosh, float, f32)
    WRAY_C_UNARY(wray, , asinh, float, f32)
    WRAY_C_UNARY(wray, , cbrt, float, f32)
    WRAY_C_UNARY(wray, , erf, float, f32)
    WRAY_C_UNARY(wray, , midi_to_hz, float, f32)
    WRAY_C_UNARY(wray, , hz_to_midi, float, f32)
    WRAY_C_UNARY(wray, , ratio_from_cents, float, f32)
    WRAY_C_UNARY(wray, , omega, float, f32)
    WRAY_C_UNARY(wray, , product_log, float, f32)
    WRAY_C_UNARY(wray, , tanh, float, f32)
    WRAY_C_UNARY(wray, , sigmoid, float, f32)
    WRAY_C_UNARY(wray, , db_to_gain, float, f32)
    WRAY_C_UNARY(wray, , gain_to_db, float, f32)
    WRAY_C_PAIR(wray, , sincos, float, f32)
    WRAY_C_PAIR(wray, , sincos_turns, float, f32)
    WRAY_C_BINARY(wray, , smoothing_coefficient, float, f32)
    WRAY_C_BINARY(wray, , pole_from_cutoff, float, f32)
    WRAY_C_BINARY(wray, , prewarp, float, f32)
    WRAY_C_BINARY(wray, , atan2, float, f32)
    WRAY_C_BINARY(wray, , pow, float, f32)
    WRAY_C_UNARY(wray, , exp, double, f64)
    WRAY_C_UNARY(wray, , expm1, double, f64)
    WRAY_C_UNARY(wray, , exp2, double, f64)
    WRAY_C_UNARY(wray, , log, double, f64)
    WRAY_C_UNARY(wray, , log1p, double, f64)
    WRAY_C_UNARY(wray, , log10, double, f64)
    WRAY_C_UNARY(wray, , log2, double, f64)
    WRAY_C_UNARY(wray, , wrap_turns, double, f64)
    WRAY_C_UNARY(wray, , sin_turns, double, f64)
    WRAY_C_UNARY(wray, , sin, double, f64)
    WRAY_C_UNARY(wray, , cos_turns, double, f64)
    WRAY_C_UNARY(wray, , cos, double, f64)
    WRAY_C_UNARY(wray, , tanpi, double, f64)
    WRAY_C_UNARY(wray, , tan_turns, double, f64)
    WRAY_C_UNARY(wray, , tan, double, f64)
    WRAY_C_UNARY(wray, , atan, double, f64)
    WRAY_C_UNARY(wray, , sinh, double, f64)
    WRAY_C_UNARY(wray, , cosh, double, f64)
    WRAY_C_UNARY(wray, , asinh, double, f64)
    WRAY_C_UNARY(wray, , cbrt, double, f64)
    WRAY_C_UNARY(wray, , erf, double, f64)
    WRAY_C_UNARY(wray, , midi_to_hz, double, f64)
    WRAY_C_UNARY(wray, , hz_to_midi, double, f64)
    WRAY_C_UNARY(wray, , ratio_from_cents, double, f64)
    WRAY_C_UNARY(wray, , omega, double, f64)
    WRAY_C_UNARY(wray, , product_log, double, f64)
    WRAY_C_UNARY(wray, , tanh, double, f64)
    WRAY_C_UNARY(wray, , sigmoid, double, f64)
    WRAY_C_UNARY(wray, , db_to_gain, double, f64)
    WRAY_C_UNARY(wray, , gain_to_db, double, f64)
    WRAY_C_PAIR(wray, , sincos, double, f64)
    WRAY_C_PAIR(wray, , sincos_turns, double, f64)
    WRAY_C_BINARY(wray, , smoothing_coefficient, double, f64)
    WRAY_C_BINARY(wray, , pole_from_cutoff, double, f64)
    WRAY_C_BINARY(wray, , prewarp, double, f64)
    WRAY_C_BINARY(wray, , atan2, double, f64)
    WRAY_C_BINARY(wray, , pow, double, f64)
    WRAY_C_UNARY(wray::fast, fast_, exp, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, exp2, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, log, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, log2, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, sin_turns, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, sin, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, cos_turns, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, cos, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, tanpi, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, tan_turns, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, tan, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, atan, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, omega, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, product_log, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, tanh, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, sigmoid, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, db_to_gain, float, f32)
    WRAY_C_UNARY(wray::fast, fast_, gain_to_db, float, f32)
    WRAY_C_PAIR(wray::fast, fast_, sincos, float, f32)
    WRAY_C_PAIR(wray::fast, fast_, sincos_turns, float, f32)
    WRAY_C_BINARY(wray::fast, fast_, atan2, float, f32)
    WRAY_C_BINARY(wray::fast, fast_, pow, float, f32)
    WRAY_C_UNARY(wray::balanced, balanced_, sin, float, f32)
    WRAY_C_UNARY(wray::balanced, balanced_, cos, float, f32)
    WRAY_C_PAIR(wray::balanced, balanced_, sincos, float, f32)
    WRAY_C_UNARY(wray::balanced, balanced_, sin, double, f64)
    WRAY_C_UNARY(wray::balanced, balanced_, cos, double, f64)
    WRAY_C_PAIR(wray::balanced, balanced_, sincos, double, f64)

#undef WRAY_C_UNARY
#undef WRAY_C_PAIR
#undef WRAY_C_BINARY
}
