#include "baselines.h"

#include <wray/simd/config.h>

#include <sleef.h>

#include <cstddef>
#include <type_traits>

#define WRAY_BENCH_JOIN_(a, b, c, d, e, f) a##b##c##d##e##f
#define WRAY_BENCH_JOIN(a, b, c, d, e, f) WRAY_BENCH_JOIN_(a, b, c, d, e, f)

#if defined(WRAY_SIMD_AVX512)
#define WRAY_BENCH_ISA avx512f
#define WRAY_BENCH_FW 16
#define WRAY_BENCH_DW 8
#elif defined(WRAY_SIMD_AVX2)
#define WRAY_BENCH_ISA avx2
#define WRAY_BENCH_FW 8
#define WRAY_BENCH_DW 4
#elif defined(WRAY_SIMD_SSE2)
#define WRAY_BENCH_ISA sse2
#define WRAY_BENCH_FW 4
#define WRAY_BENCH_DW 2
#elif defined(WRAY_SIMD_NEON)
#define WRAY_BENCH_ISA advsimd
#define WRAY_BENCH_FW 4
#define WRAY_BENCH_DW 2
#else
#error "The SLEEF baseline needs a SIMD backend"
#endif

#define WRAY_BENCH_F(fn, acc) WRAY_BENCH_JOIN(Sleef_, fn, f, WRAY_BENCH_FW, _##acc, WRAY_BENCH_ISA)
#define WRAY_BENCH_D(fn, acc) WRAY_BENCH_JOIN(Sleef_, fn, d, WRAY_BENCH_DW, _##acc, WRAY_BENCH_ISA)

namespace
{
#if defined(WRAY_SIMD_AVX512)
    using vf = __m512;
    using vd = __m512d;
    vf load(const float *p) { return _mm512_loadu_ps(p); }
    vd load(const double *p) { return _mm512_loadu_pd(p); }
    void store(float *p, vf v) { _mm512_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm512_storeu_pd(p, v); }
#elif defined(WRAY_SIMD_AVX2)
    using vf = __m256;
    using vd = __m256d;
    vf load(const float *p) { return _mm256_loadu_ps(p); }
    vd load(const double *p) { return _mm256_loadu_pd(p); }
    void store(float *p, vf v) { _mm256_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm256_storeu_pd(p, v); }
#elif defined(WRAY_SIMD_SSE2)
    using vf = __m128;
    using vd = __m128d;
    vf load(const float *p) { return _mm_loadu_ps(p); }
    vd load(const double *p) { return _mm_loadu_pd(p); }
    void store(float *p, vf v) { _mm_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm_storeu_pd(p, v); }
#else
    using vf = float32x4_t;
    using vd = float64x2_t;
    vf load(const float *p) { return vld1q_f32(p); }
    vd load(const double *p) { return vld1q_f64(p); }
    void store(float *p, vf v) { vst1q_f32(p, v); }
    void store(double *p, vd v) { vst1q_f64(p, v); }
#endif

    template<class T>
    constexpr std::size_t width = std::is_same_v<T, float> ? WRAY_BENCH_FW : WRAY_BENCH_DW;

    template<class T, class G>
    void blocks(std::size_t n, G g)
    {
        constexpr std::size_t w = width<T>;
        std::size_t i = 0;
        for (; i + w <= n; i += w)
            g(i);
        if (i < n && n >= w)
            g(n - w);
    }

    template<class T, auto Float, auto Double>
    void unary(const T *a, const T *, T *out, T *, std::size_t n)
    {
        blocks<T>(n, [&](std::size_t i) {
            if constexpr (std::is_same_v<T, float>) store(out + i, Float(load(a + i)));
            else store(out + i, Double(load(a + i)));
        });
    }

    template<class T, auto Float, auto Double>
    void binary(const T *a, const T *b, T *out, T *, std::size_t n)
    {
        blocks<T>(n, [&](std::size_t i) {
            if constexpr (std::is_same_v<T, float>) store(out + i, Float(load(a + i), load(b + i)));
            else store(out + i, Double(load(a + i), load(b + i)));
        });
    }

    template<class T, auto Float, auto Double>
    void pair(const T *a, const T *, T *out, T *cos_out, std::size_t n)
    {
        blocks<T>(n, [&](std::size_t i) {
            if constexpr (std::is_same_v<T, float>)
            {
                const auto r = Float(load(a + i));
                store(out + i, r.x);
                store(cos_out + i, r.y);
            } else
            {
                const auto r = Double(load(a + i));
                store(out + i, r.x);
                store(cos_out + i, r.y);
            }
        });
    }
}

#define WRAY_BENCH_UNARY(fn, acc) if (name == #fn) return unary<T, WRAY_BENCH_F(fn, acc), WRAY_BENCH_D(fn, acc)>
#define WRAY_BENCH_BINARY(fn, acc) if (name == #fn) return binary<T, WRAY_BENCH_F(fn, acc), WRAY_BENCH_D(fn, acc)>
#define WRAY_BENCH_PAIR(fn, acc) if (name == #fn) return pair<T, WRAY_BENCH_F(fn, acc), WRAY_BENCH_D(fn, acc)>

namespace wray_bench
{
    template<class T>
    function<T> sleef(std::string_view name, sleef_accuracy accuracy)
    {
        if (accuracy == sleef_accuracy::u10)
        {
            WRAY_BENCH_UNARY(sin, u10);
            WRAY_BENCH_UNARY(cos, u10);
            WRAY_BENCH_UNARY(tan, u10);
            WRAY_BENCH_PAIR(sincos, u10);
            WRAY_BENCH_UNARY(exp, u10);
            WRAY_BENCH_UNARY(exp2, u10);
            WRAY_BENCH_UNARY(expm1, u10);
            WRAY_BENCH_UNARY(log, u10);
            WRAY_BENCH_UNARY(log2, u10);
            WRAY_BENCH_UNARY(log10, u10);
            WRAY_BENCH_UNARY(log1p, u10);
            WRAY_BENCH_BINARY(pow, u10);
            WRAY_BENCH_UNARY(sinh, u10);
            WRAY_BENCH_UNARY(cosh, u10);
            WRAY_BENCH_UNARY(tanh, u10);
            WRAY_BENCH_UNARY(asinh, u10);
            WRAY_BENCH_UNARY(atan, u10);
            WRAY_BENCH_BINARY(atan2, u10);
            WRAY_BENCH_UNARY(cbrt, u10);
            WRAY_BENCH_UNARY(erf, u10);
            return nullptr;
        }
        WRAY_BENCH_UNARY(sin, u35);
        WRAY_BENCH_UNARY(cos, u35);
        WRAY_BENCH_UNARY(tan, u35);
        WRAY_BENCH_PAIR(sincos, u35);
        WRAY_BENCH_UNARY(exp2, u35);
        WRAY_BENCH_UNARY(log, u35);
        WRAY_BENCH_UNARY(log2, u35);
        WRAY_BENCH_UNARY(sinh, u35);
        WRAY_BENCH_UNARY(cosh, u35);
        WRAY_BENCH_UNARY(tanh, u35);
        WRAY_BENCH_UNARY(atan, u35);
        WRAY_BENCH_BINARY(atan2, u35);
        WRAY_BENCH_UNARY(cbrt, u35);
        return nullptr;
    }

    template function<float> sleef<float>(std::string_view, sleef_accuracy);
    template function<double> sleef<double>(std::string_view, sleef_accuracy);
}
