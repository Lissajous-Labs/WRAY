#include "baselines.h"

#include <wray/simd/config.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#define WRAY_BENCH_JOIN_(a, b, c, d, e, f) a##b##c##d##e##f
#define WRAY_BENCH_JOIN(a, b, c, d, e, f) WRAY_BENCH_JOIN_(a, b, c, d, e, f)

#if defined(WRAY_SIMD_AVX512)
#define WRAY_BENCH_ISA e
#define WRAY_BENCH_FW 16
#define WRAY_BENCH_DW 8
using vf = __m512;
using vd = __m512d;
using vi = __m512i;
namespace
{
    vi load_pointers(const void *p) { return _mm512_loadu_si512(p); }
    vf load(const float *p) { return _mm512_loadu_ps(p); }
    vd load(const double *p) { return _mm512_loadu_pd(p); }
    void store(float *p, vf v) { _mm512_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm512_storeu_pd(p, v); }
}
#elif defined(WRAY_SIMD_AVX2)
#define WRAY_BENCH_ISA d
#define WRAY_BENCH_FW 8
#define WRAY_BENCH_DW 4
using vf = __m256;
using vd = __m256d;
using vi = __m256i;
namespace
{
    vi load_pointers(const void *p) { return _mm256_loadu_si256(static_cast<const vi *>(p)); }
    vf load(const float *p) { return _mm256_loadu_ps(p); }
    vd load(const double *p) { return _mm256_loadu_pd(p); }
    void store(float *p, vf v) { _mm256_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm256_storeu_pd(p, v); }
}
#elif defined(WRAY_SIMD_SSE2)
#define WRAY_BENCH_ISA b
#define WRAY_BENCH_FW 4
#define WRAY_BENCH_DW 2
using vf = __m128;
using vd = __m128d;
using vi = __m128i;
namespace
{
    vi load_pointers(const void *p) { return _mm_loadu_si128(static_cast<const vi *>(p)); }
    vf load(const float *p) { return _mm_loadu_ps(p); }
    vd load(const double *p) { return _mm_loadu_pd(p); }
    void store(float *p, vf v) { _mm_storeu_ps(p, v); }
    void store(double *p, vd v) { _mm_storeu_pd(p, v); }
}
#else
#error "The libmvec baseline needs an x86 SIMD backend"
#endif

#define WRAY_BENCH_F(fn, args) WRAY_BENCH_JOIN(_ZGV, WRAY_BENCH_ISA, N, WRAY_BENCH_FW, args##_, fn##f)
#define WRAY_BENCH_D(fn, args) WRAY_BENCH_JOIN(_ZGV, WRAY_BENCH_ISA, N, WRAY_BENCH_DW, args##_, fn)
#define WRAY_BENCH_DECLARE_UNARY(fn) vf WRAY_BENCH_F(fn, v)(vf); vd WRAY_BENCH_D(fn, v)(vd);
#define WRAY_BENCH_DECLARE_BINARY(fn) vf WRAY_BENCH_F(fn, vv)(vf, vf); vd WRAY_BENCH_D(fn, vv)(vd, vd);

extern "C"
{
    WRAY_BENCH_DECLARE_UNARY(sin)
    WRAY_BENCH_DECLARE_UNARY(cos)
    WRAY_BENCH_DECLARE_UNARY(tan)
    WRAY_BENCH_DECLARE_UNARY(exp)
    WRAY_BENCH_DECLARE_UNARY(exp2)
    WRAY_BENCH_DECLARE_UNARY(expm1)
    WRAY_BENCH_DECLARE_UNARY(log)
    WRAY_BENCH_DECLARE_UNARY(log2)
    WRAY_BENCH_DECLARE_UNARY(log10)
    WRAY_BENCH_DECLARE_UNARY(log1p)
    WRAY_BENCH_DECLARE_UNARY(sinh)
    WRAY_BENCH_DECLARE_UNARY(cosh)
    WRAY_BENCH_DECLARE_UNARY(tanh)
    WRAY_BENCH_DECLARE_UNARY(asinh)
    WRAY_BENCH_DECLARE_UNARY(atan)
    WRAY_BENCH_DECLARE_UNARY(cbrt)
    WRAY_BENCH_DECLARE_UNARY(erf)
    WRAY_BENCH_DECLARE_BINARY(pow)
    WRAY_BENCH_DECLARE_BINARY(atan2)
    void WRAY_BENCH_F(sincos, vvv)(vf, vi, vi, vi, vi);
    void WRAY_BENCH_D(sincos, vvv)(vd, vi, vi);
}

namespace
{
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

    template<class T>
    void pair(const T *a, const T *, T *out, T *cos_out, std::size_t n)
    {
        constexpr std::size_t w = width<T>;
        constexpr std::size_t half = sizeof(vi) / sizeof(std::uintptr_t);
        blocks<T>(n, [&](std::size_t i) {
            std::uintptr_t s[w], c[w];
            for (std::size_t k = 0; k < w; ++k)
            {
                s[k] = reinterpret_cast<std::uintptr_t>(out + i + k);
                c[k] = reinterpret_cast<std::uintptr_t>(cos_out + i + k);
            }
            if constexpr (std::is_same_v<T, float>)
                WRAY_BENCH_F(sincos, vvv)(load(a + i), load_pointers(s), load_pointers(s + half), load_pointers(c), load_pointers(c + half));
            else
                WRAY_BENCH_D(sincos, vvv)(load(a + i), load_pointers(s), load_pointers(c));
        });
    }
}

#define WRAY_BENCH_UNARY(fn) if (name == #fn) return unary<T, WRAY_BENCH_F(fn, v), WRAY_BENCH_D(fn, v)>
#define WRAY_BENCH_BINARY(fn) if (name == #fn) return binary<T, WRAY_BENCH_F(fn, vv), WRAY_BENCH_D(fn, vv)>

namespace wray_bench
{
    template<class T>
    function<T> libmvec(std::string_view name)
    {
        WRAY_BENCH_UNARY(sin);
        WRAY_BENCH_UNARY(cos);
        WRAY_BENCH_UNARY(tan);
        WRAY_BENCH_UNARY(exp);
        WRAY_BENCH_UNARY(exp2);
        WRAY_BENCH_UNARY(expm1);
        WRAY_BENCH_UNARY(log);
        WRAY_BENCH_UNARY(log2);
        WRAY_BENCH_UNARY(log10);
        WRAY_BENCH_UNARY(log1p);
        WRAY_BENCH_UNARY(sinh);
        WRAY_BENCH_UNARY(cosh);
        WRAY_BENCH_UNARY(tanh);
        WRAY_BENCH_UNARY(asinh);
        WRAY_BENCH_UNARY(atan);
        WRAY_BENCH_UNARY(cbrt);
        WRAY_BENCH_UNARY(erf);
        WRAY_BENCH_BINARY(pow);
        WRAY_BENCH_BINARY(atan2);
        if (name == "sincos")
            return pair<T>;
        return nullptr;
    }

    template function<float> libmvec<float>(std::string_view);
    template function<double> libmvec<double>(std::string_view);
}
