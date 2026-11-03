#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "config.h"

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI::simd
{
    template<class T>
    struct bits_of;

    template<>
    struct bits_of<float>
    {
        using integer = std::int32_t;
        using unsigned_integer = std::uint32_t;
        static constexpr int width = 32;
    };

    template<>
    struct bits_of<double>
    {
        using integer = std::int64_t;
        using unsigned_integer = std::uint64_t;
        static constexpr int width = 64;
    };

    template<>
    struct bits_of<std::int32_t>
    {
        using integer = std::int32_t;
        using unsigned_integer = std::uint32_t;
        static constexpr int width = 32;
    };

    template<>
    struct bits_of<std::int64_t>
    {
        using integer = std::int64_t;
        using unsigned_integer = std::uint64_t;
        static constexpr int width = 64;
    };

    template<class T, class U>
    WRAY_INLINE T round_even(T x) noexcept
    {
        constexpr T limit = std::numeric_limits<T>::digits == 24 ? T(0x1p23) : T(0x1p52);
        if (!(std::fabs(x) < limit))
            return x;
        const T m = std::copysign(limit, x);
        const T c = x + m;
        const T r = c - m;
        const T d = x - r;
        const bool odd = (std::bit_cast<U>(c) & U(1)) != 0;
        T step = T(0);
        if (d > T(0.5) || (d == T(0.5) && odd))
            step = T(1);
        else if (d < T(-0.5) || (d == T(-0.5) && odd))
            step = T(-1);
        return std::copysign(r + step, x);
    }

    WRAY_INLINE float nearest(float x) noexcept
    {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        return __builtin_roundevenf(x);
#elif defined(__SSE4_1__) && !defined(WRAY_SIMD_SCALAR)
        return _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
        return round_even<float, std::uint32_t>(x);
#endif
    }

    WRAY_INLINE double nearest(double x) noexcept
    {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        return __builtin_roundeven(x);
#elif defined(__SSE4_1__) && !defined(WRAY_SIMD_SCALAR)
        return _mm_cvtsd_f64(_mm_round_sd(_mm_setzero_pd(), _mm_set_sd(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
        return round_even<double, std::uint64_t>(x);
#endif
    }

    WRAY_INLINE float first(float x) noexcept { return x; }
    WRAY_INLINE double first(double x) noexcept { return x; }

    template<int Bits, int N>
    struct mask
    {
        bool v[static_cast<std::size_t>(N)];
    };

    template<class T, int N>
    struct vec
    {
        using value_type = T;
        static constexpr int size = N;
        T v[static_cast<std::size_t>(N)];

        vec() = default;

        WRAY_INLINE vec(T s) noexcept : vec(s, std::make_integer_sequence<int, N>{}) {}

        WRAY_INLINE static vec load(const T *p) noexcept
        {
            vec r;
            for (int i = 0; i < N; ++i)
                r.v[i] = p[i];
            return r;
        }

        WRAY_INLINE void store(T *p) const noexcept
        {
            for (int i = 0; i < N; ++i)
                p[i] = v[i];
        }

        WRAY_INLINE T operator[](int i) const noexcept { return v[i]; }

    private:
        template<int... I>
        WRAY_INLINE vec(T s, std::integer_sequence<int, I...>) noexcept : v{((void)I, s)...} {}
    };

    template<class T, int N>
    using mask_for = mask<bits_of<T>::width, N>;

    template<class T, int N>
    using int_for = vec<typename bits_of<T>::integer, N>;

    using f32x4 = vec<float, 4>;
    using f64x2 = vec<double, 2>;
    using i32x4 = vec<std::int32_t, 4>;
    using i64x2 = vec<std::int64_t, 2>;
    using m32x4 = mask<32, 4>;
    using m64x2 = mask<64, 2>;
    using f32x8 = vec<float, 8>;
    using f64x4 = vec<double, 4>;
    using i32x8 = vec<std::int32_t, 8>;
    using i64x4 = vec<std::int64_t, 4>;
    using m32x8 = mask<32, 8>;
    using m64x4 = mask<64, 4>;
    using f32x16 = vec<float, 16>;
    using f64x8 = vec<double, 8>;
    using i32x16 = vec<std::int32_t, 16>;
    using i64x8 = vec<std::int64_t, 8>;
    using m32x16 = mask<32, 16>;
    using m64x8 = mask<64, 8>;

#if defined(WRAY_SIMD_AVX512)
    inline constexpr int register_bytes = 64;
#elif defined(WRAY_SIMD_AVX2)
    inline constexpr int register_bytes = 32;
#elif defined(WRAY_SIMD_NEON) || defined(WRAY_SIMD_SSE2)
    inline constexpr int register_bytes = 16;
#else
    inline constexpr int register_bytes = 0;
#endif

#if defined(WRAY_SIMD_SCALAR)
    inline constexpr int native_bytes = 16;
#else
    inline constexpr int native_bytes = register_bytes;
#endif

    template<int Bytes, int N>
    concept wide = register_bytes > 0 && Bytes * N > register_bytes && (N & (N - 1)) == 0;

    template<class T, int N>
    concept split = wide<static_cast<int>(sizeof(T)), N>;

    template<class T, int N>
        requires split<T, N>
    struct vec<T, N>
    {
        using value_type = T;
        using half = vec<T, N / 2>;
        static constexpr int size = N;
        half lo;
        half hi;

        vec() = default;

        WRAY_INLINE vec(const half &l, const half &h) noexcept : lo(l), hi(h)
        {
        }

        WRAY_INLINE vec(T s) noexcept : lo(s), hi(s)
        {
        }

        WRAY_INLINE static vec load(const T *p) noexcept { return {half::load(p), half::load(p + N / 2)}; }

        WRAY_INLINE void store(T *p) const noexcept
        {
            lo.store(p);
            hi.store(p + N / 2);
        }

        WRAY_INLINE T operator[](int i) const noexcept { return i < N / 2 ? lo[i] : hi[i - N / 2]; }
    };

    template<int B, int N>
        requires wide<B / 8, N>
    struct mask<B, N>
    {
        mask<B, N / 2> lo;
        mask<B, N / 2> hi;
    };

    template<class T, int N, class F>
    WRAY_INLINE vec<T, N> map(const vec<T, N> &a, F f) noexcept
    {
        vec<T, N> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = f(a.v[i]);
        return r;
    }

    template<class T, int N, class F>
    WRAY_INLINE vec<T, N> zip(const vec<T, N> &a, const vec<T, N> &b, F f) noexcept
    {
        vec<T, N> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = f(a.v[i], b.v[i]);
        return r;
    }

    template<class T, int N, class F>
    WRAY_INLINE mask_for<T, N> test(const vec<T, N> &a, const vec<T, N> &b, F f) noexcept
    {
        mask_for<T, N> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = f(a.v[i], b.v[i]);
        return r;
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator+(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            using U = std::make_unsigned_t<T>;
            return zip(a, b, [](T x, T y) { return static_cast<T>(static_cast<U>(x) + static_cast<U>(y)); });
        } else
            return zip(a, b, [](T x, T y) { return T(x + y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator-(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            using U = std::make_unsigned_t<T>;
            return zip(a, b, [](T x, T y) { return static_cast<T>(static_cast<U>(x) - static_cast<U>(y)); });
        } else
            return zip(a, b, [](T x, T y) { return T(x - y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator*(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return T(x * y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator/(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return T(x / y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator-(const vec<T, N> &a) noexcept { return map(a, [](T x) { return T(-x); }); }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator&(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return T(x & y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator|(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return T(x | y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator^(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return T(x ^ y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator<<(const vec<T, N> &a, int s) noexcept
    {
        using U = typename bits_of<T>::unsigned_integer;
        return map(a, [s](T x) { return static_cast<T>(static_cast<U>(x) << s); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> shift_right_logical(const vec<T, N> &a, int s) noexcept
    {
        using U = typename bits_of<T>::unsigned_integer;
        return map(a, [s](T x) { return static_cast<T>(static_cast<U>(x) >> s); });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator==(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x == y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator!=(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x != y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator<(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x < y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator<=(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x <= y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator>(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x > y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> operator>=(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return test(a, b, [](T x, T y) { return x >= y; });
    }

    template<class T, int N>
    WRAY_INLINE mask_for<T, N> nonzero(const vec<T, N> &a) noexcept
    {
        return test(a, a, [](T x, T) { return x != 0; });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> fma(const vec<T, N> &a, const vec<T, N> &b, const vec<T, N> &c) noexcept
    {
        vec<T, N> r;
        for (int i = 0; i < N; ++i)
        {
#if WRAY_HAS_FMA
            r.v[i] = WRAY_FMA(a.v[i], b.v[i], c.v[i]);
#else
            T p = a.v[i] * b.v[i];
            r.v[i] = p + c.v[i];
#endif
        }
        return r;
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> min(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return y < x ? y : x; });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> max(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return zip(a, b, [](T x, T y) { return x < y ? y : x; });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> abs(const vec<T, N> &a) noexcept { return map(a, [](T x) { return std::fabs(x); }); }

    template<class T, int N>
    WRAY_INLINE vec<T, N> sqrt(const vec<T, N> &a) noexcept { return map(a, [](T x) { return WRAY_SQRT(x); }); }

    template<class T, int N>
    WRAY_INLINE vec<T, N> nearest(const vec<T, N> &a) noexcept { return map(a, [](T x) { return nearest(x); }); }

    template<class T, int N>
    WRAY_INLINE T first(const vec<T, N> &a) noexcept { return a.v[0]; }

    template<class T, int N>
    WRAY_INLINE vec<T, N> select(const mask_for<T, N> &m, const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        vec<T, N> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = m.v[i] ? a.v[i] : b.v[i];
        return r;
    }

    template<class T, int N>
    WRAY_INLINE int_for<T, N> as_bits(const vec<T, N> &a) noexcept
    {
        int_for<T, N> r;
        for (int i = 0; i < N; ++i)
            r.v[i] = std::bit_cast<typename bits_of<T>::integer>(a.v[i]);
        return r;
    }

    template<class V, class I>
    WRAY_INLINE V from_bits(const I &a) noexcept
    {
        V r;
        for (int i = 0; i < V::size; ++i)
            r.v[i] = std::bit_cast<typename V::value_type>(a.v[i]);
        return r;
    }

    template<int B, int N>
    WRAY_INLINE mask<B, N> operator&(const mask<B, N> &a, const mask<B, N> &b) noexcept
    {
        mask<B, N> r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] && b.v[i];
        return r;
    }

    template<int B, int N>
    WRAY_INLINE mask<B, N> operator|(const mask<B, N> &a, const mask<B, N> &b) noexcept
    {
        mask<B, N> r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] || b.v[i];
        return r;
    }

    template<int B, int N>
    WRAY_INLINE mask<B, N> operator!(const mask<B, N> &a) noexcept
    {
        mask<B, N> r;
        for (int i = 0; i < N; ++i) r.v[i] = !a.v[i];
        return r;
    }

    template<int B, int N>
    WRAY_INLINE bool any(const mask<B, N> &a) noexcept
    {
        for (int i = 0; i < N; ++i) if (a.v[i]) return true;
        return false;
    }

    template<int B, int N>
    WRAY_INLINE bool all(const mask<B, N> &a) noexcept
    {
        for (int i = 0; i < N; ++i) if (!a.v[i]) return false;
        return true;
    }

#if defined(WRAY_SIMD_NEON)

    template<>
    struct mask<32, 4>
    {
        uint32x4_t v;
    };

    template<>
    struct mask<64, 2>
    {
        uint64x2_t v;
    };

    template<>
    struct vec<float, 4>
    {
        using value_type = float;
        static constexpr int size = 4;
        float32x4_t v;

        vec() = default;

        WRAY_INLINE vec(float32x4_t x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(float s) noexcept : v(vdupq_n_f32(s))
        {
        }

        WRAY_INLINE static vec load(const float *p) noexcept { return vld1q_f32(p); }
        WRAY_INLINE void store(float *p) const noexcept { vst1q_f32(p, v); }
        WRAY_INLINE float operator[](int i) const noexcept
        {
            float t[4];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<double, 2>
    {
        using value_type = double;
        static constexpr int size = 2;
        float64x2_t v;

        vec() = default;

        WRAY_INLINE vec(float64x2_t x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(double s) noexcept : v(vdupq_n_f64(s))
        {
        }

        WRAY_INLINE static vec load(const double *p) noexcept { return vld1q_f64(p); }
        WRAY_INLINE void store(double *p) const noexcept { vst1q_f64(p, v); }
        WRAY_INLINE double operator[](int i) const noexcept
        {
            double t[2];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int32_t, 4>
    {
        using value_type = std::int32_t;
        static constexpr int size = 4;
        int32x4_t v;

        vec() = default;

        WRAY_INLINE vec(int32x4_t x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int32_t s) noexcept : v(vdupq_n_s32(s))
        {
        }

        WRAY_INLINE static vec load(const std::int32_t *p) noexcept { return vld1q_s32(p); }
        WRAY_INLINE void store(std::int32_t *p) const noexcept { vst1q_s32(p, v); }
        WRAY_INLINE std::int32_t operator[](int i) const noexcept
        {
            std::int32_t t[4];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int64_t, 2>
    {
        using value_type = std::int64_t;
        static constexpr int size = 2;
        int64x2_t v;

        vec() = default;

        WRAY_INLINE vec(int64x2_t x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int64_t s) noexcept : v(vdupq_n_s64(s))
        {
        }

        WRAY_INLINE static vec load(const std::int64_t *p) noexcept { return vld1q_s64(p); }
        WRAY_INLINE void store(std::int64_t *p) const noexcept { vst1q_s64(p, v); }
        WRAY_INLINE std::int64_t operator[](int i) const noexcept
        {
            std::int64_t t[2];
            store(t);
            return t[i];
        }
    };

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return vaddq_f32(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return vsubq_f32(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return vmulq_f32(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return vdivq_f32(a.v, b.v); }
#endif
    WRAY_INLINE f32x4 operator-(const f32x4 &a) noexcept { return vnegq_f32(a.v); }
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept { return vfmaq_f32(c.v, a.v, b.v); }
    WRAY_INLINE f32x4 min(const f32x4 &a, const f32x4 &b) noexcept { return vminq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 max(const f32x4 &a, const f32x4 &b) noexcept { return vmaxq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 abs(const f32x4 &a) noexcept { return vabsq_f32(a.v); }
    WRAY_INLINE f32x4 sqrt(const f32x4 &a) noexcept { return vsqrtq_f32(a.v); }
    WRAY_INLINE f32x4 nearest(const f32x4 &a) noexcept { return vrndnq_f32(a.v); }
    WRAY_INLINE float first(const f32x4 &a) noexcept { return vgetq_lane_f32(a.v, 0); }
    WRAY_INLINE m32x4 operator==(const f32x4 &a, const f32x4 &b) noexcept { return {vceqq_f32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator!=(const f32x4 &a, const f32x4 &b) noexcept { return {vmvnq_u32(vceqq_f32(a.v, b.v))}; }
    WRAY_INLINE m32x4 operator<(const f32x4 &a, const f32x4 &b) noexcept { return {vcltq_f32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator<=(const f32x4 &a, const f32x4 &b) noexcept { return {vcleq_f32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator>(const f32x4 &a, const f32x4 &b) noexcept { return {vcgtq_f32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator>=(const f32x4 &a, const f32x4 &b) noexcept { return {vcgeq_f32(a.v, b.v)}; }
    WRAY_INLINE f32x4 select(const m32x4 &m, const f32x4 &a, const f32x4 &b) noexcept
    {
        return vbslq_f32(m.v, a.v, b.v);
    }
    WRAY_INLINE i32x4 as_bits(const f32x4 &a) noexcept { return vreinterpretq_s32_f32(a.v); }
    template<>
    WRAY_INLINE f32x4 from_bits<f32x4, i32x4>(const i32x4 &a) noexcept { return vreinterpretq_f32_s32(a.v); }

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return vaddq_f64(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return vsubq_f64(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return vmulq_f64(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return vdivq_f64(a.v, b.v); }
#endif
    WRAY_INLINE f64x2 operator-(const f64x2 &a) noexcept { return vnegq_f64(a.v); }
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept { return vfmaq_f64(c.v, a.v, b.v); }
    WRAY_INLINE f64x2 min(const f64x2 &a, const f64x2 &b) noexcept { return vminq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 max(const f64x2 &a, const f64x2 &b) noexcept { return vmaxq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 abs(const f64x2 &a) noexcept { return vabsq_f64(a.v); }
    WRAY_INLINE f64x2 sqrt(const f64x2 &a) noexcept { return vsqrtq_f64(a.v); }
    WRAY_INLINE f64x2 nearest(const f64x2 &a) noexcept { return vrndnq_f64(a.v); }
    WRAY_INLINE double first(const f64x2 &a) noexcept { return vgetq_lane_f64(a.v, 0); }
    WRAY_INLINE m64x2 operator==(const f64x2 &a, const f64x2 &b) noexcept { return {vceqq_f64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator!=(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {veorq_u64(vceqq_f64(a.v, b.v), vdupq_n_u64(~0ull))};
    }
    WRAY_INLINE m64x2 operator<(const f64x2 &a, const f64x2 &b) noexcept { return {vcltq_f64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator<=(const f64x2 &a, const f64x2 &b) noexcept { return {vcleq_f64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator>(const f64x2 &a, const f64x2 &b) noexcept { return {vcgtq_f64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator>=(const f64x2 &a, const f64x2 &b) noexcept { return {vcgeq_f64(a.v, b.v)}; }
    WRAY_INLINE f64x2 select(const m64x2 &m, const f64x2 &a, const f64x2 &b) noexcept
    {
        return vbslq_f64(m.v, a.v, b.v);
    }
    WRAY_INLINE i64x2 as_bits(const f64x2 &a) noexcept { return vreinterpretq_s64_f64(a.v); }
    template<>
    WRAY_INLINE f64x2 from_bits<f64x2, i64x2>(const i64x2 &a) noexcept { return vreinterpretq_f64_s64(a.v); }

    WRAY_INLINE i32x4 operator+(const i32x4 &a, const i32x4 &b) noexcept { return vaddq_s32(a.v, b.v); }
    WRAY_INLINE i32x4 operator-(const i32x4 &a, const i32x4 &b) noexcept { return vsubq_s32(a.v, b.v); }
    WRAY_INLINE i32x4 operator&(const i32x4 &a, const i32x4 &b) noexcept { return vandq_s32(a.v, b.v); }
    WRAY_INLINE i32x4 operator|(const i32x4 &a, const i32x4 &b) noexcept { return vorrq_s32(a.v, b.v); }
    WRAY_INLINE i32x4 operator^(const i32x4 &a, const i32x4 &b) noexcept { return veorq_s32(a.v, b.v); }
    WRAY_INLINE i32x4 operator<<(const i32x4 &a, int s) noexcept { return vshlq_s32(a.v, vdupq_n_s32(s)); }
    WRAY_INLINE i32x4 shift_right_logical(const i32x4 &a, int s) noexcept
    {
        return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a.v), vdupq_n_s32(-s)));
    }
    WRAY_INLINE m32x4 nonzero(const i32x4 &a) noexcept { return {vtstq_s32(a.v, a.v)}; }
    WRAY_INLINE m32x4 operator==(const i32x4 &a, const i32x4 &b) noexcept { return {vceqq_s32(a.v, b.v)}; }

    WRAY_INLINE i64x2 operator+(const i64x2 &a, const i64x2 &b) noexcept { return vaddq_s64(a.v, b.v); }
    WRAY_INLINE i64x2 operator-(const i64x2 &a, const i64x2 &b) noexcept { return vsubq_s64(a.v, b.v); }
    WRAY_INLINE i64x2 operator&(const i64x2 &a, const i64x2 &b) noexcept { return vandq_s64(a.v, b.v); }
    WRAY_INLINE i64x2 operator|(const i64x2 &a, const i64x2 &b) noexcept { return vorrq_s64(a.v, b.v); }
    WRAY_INLINE i64x2 operator^(const i64x2 &a, const i64x2 &b) noexcept { return veorq_s64(a.v, b.v); }
    WRAY_INLINE i64x2 operator<<(const i64x2 &a, int s) noexcept { return vshlq_s64(a.v, vdupq_n_s64(s)); }
    WRAY_INLINE i64x2 shift_right_logical(const i64x2 &a, int s) noexcept
    {
        return vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(a.v), vdupq_n_s64(-s)));
    }
    WRAY_INLINE m64x2 nonzero(const i64x2 &a) noexcept { return {vtstq_s64(a.v, a.v)}; }
    WRAY_INLINE m64x2 operator==(const i64x2 &a, const i64x2 &b) noexcept { return {vceqq_s64(a.v, b.v)}; }

    WRAY_INLINE m32x4 operator&(const m32x4 &a, const m32x4 &b) noexcept { return {vandq_u32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator|(const m32x4 &a, const m32x4 &b) noexcept { return {vorrq_u32(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator!(const m32x4 &a) noexcept { return {vmvnq_u32(a.v)}; }
    WRAY_INLINE bool any(const m32x4 &a) noexcept { return vmaxvq_u32(a.v) != 0; }
    WRAY_INLINE bool all(const m32x4 &a) noexcept { return vminvq_u32(a.v) != 0; }
    WRAY_INLINE m64x2 operator&(const m64x2 &a, const m64x2 &b) noexcept { return {vandq_u64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator|(const m64x2 &a, const m64x2 &b) noexcept { return {vorrq_u64(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator!(const m64x2 &a) noexcept { return {veorq_u64(a.v, vdupq_n_u64(~0ull))}; }
    WRAY_INLINE bool any(const m64x2 &a) noexcept { return vmaxvq_u32(vreinterpretq_u32_u64(a.v)) != 0; }
    WRAY_INLINE bool all(const m64x2 &a) noexcept { return vminvq_u32(vreinterpretq_u32_u64(a.v)) != 0; }

#elif defined(WRAY_SIMD_SSE2)

    template<>
    struct mask<32, 4>
    {
        __m128i v;
    };

    template<>
    struct mask<64, 2>
    {
        __m128i v;
    };

    template<>
    struct vec<float, 4>
    {
        using value_type = float;
        static constexpr int size = 4;
        __m128 v;

        vec() = default;

        WRAY_INLINE vec(__m128 x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(float s) noexcept : v(_mm_set1_ps(s))
        {
        }

        WRAY_INLINE static vec load(const float *p) noexcept { return _mm_loadu_ps(p); }
        WRAY_INLINE void store(float *p) const noexcept { _mm_storeu_ps(p, v); }
        WRAY_INLINE float operator[](int i) const noexcept
        {
            float t[4];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<double, 2>
    {
        using value_type = double;
        static constexpr int size = 2;
        __m128d v;

        vec() = default;

        WRAY_INLINE vec(__m128d x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(double s) noexcept : v(_mm_set1_pd(s))
        {
        }

        WRAY_INLINE static vec load(const double *p) noexcept { return _mm_loadu_pd(p); }
        WRAY_INLINE void store(double *p) const noexcept { _mm_storeu_pd(p, v); }
        WRAY_INLINE double operator[](int i) const noexcept
        {
            double t[2];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int32_t, 4>
    {
        using value_type = std::int32_t;
        static constexpr int size = 4;
        __m128i v;

        vec() = default;

        WRAY_INLINE vec(__m128i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int32_t s) noexcept : v(_mm_set1_epi32(s))
        {
        }

        WRAY_INLINE static vec load(const std::int32_t *p) noexcept
        {
            return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        }

        WRAY_INLINE void store(std::int32_t *p) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v); }
        WRAY_INLINE std::int32_t operator[](int i) const noexcept
        {
            std::int32_t t[4];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int64_t, 2>
    {
        using value_type = std::int64_t;
        static constexpr int size = 2;
        __m128i v;

        vec() = default;

        WRAY_INLINE vec(__m128i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int64_t s) noexcept : v(_mm_set1_epi64x(s))
        {
        }

        WRAY_INLINE static vec load(const std::int64_t *p) noexcept
        {
            return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        }

        WRAY_INLINE void store(std::int64_t *p) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v); }
        WRAY_INLINE std::int64_t operator[](int i) const noexcept
        {
            std::int64_t t[2];
            store(t);
            return t[i];
        }
    };

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return _mm_add_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return _mm_sub_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return _mm_mul_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return _mm_div_ps(a.v, b.v); }
#endif
    WRAY_INLINE f32x4 operator-(const f32x4 &a) noexcept { return _mm_xor_ps(a.v, _mm_set1_ps(-0.0f)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept
    {
        return _mm_fmadd_ps(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept
    {
        return WRAY_VOP(_mm_add_ps, +, WRAY_VOP(_mm_mul_ps, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f32x4 min(const f32x4 &a, const f32x4 &b) noexcept { return _mm_min_ps(a.v, b.v); }
    WRAY_INLINE f32x4 max(const f32x4 &a, const f32x4 &b) noexcept { return _mm_max_ps(a.v, b.v); }
    WRAY_INLINE f32x4 abs(const f32x4 &a) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a.v); }
    WRAY_INLINE f32x4 sqrt(const f32x4 &a) noexcept { return _mm_sqrt_ps(a.v); }
    WRAY_INLINE f32x4 nearest(const f32x4 &a) noexcept
    {
#if defined(__SSE4_1__)
        return _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        const __m128 sign = _mm_set1_ps(-0.0f);
        const __m128 m = _mm_or_ps(_mm_set1_ps(0x1p23f), _mm_and_ps(a.v, sign));
        const __m128 c = WRAY_VOP(_mm_add_ps, +, a.v, m);
        const __m128 r = WRAY_VOP(_mm_sub_ps, -, c, m);
        const __m128 d = WRAY_VOP(_mm_sub_ps, -, a.v, r);
        const __m128 odd = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(c), _mm_set1_epi32(1)), _mm_set1_epi32(1)));
        const __m128 half = _mm_set1_ps(0.5f);
        const __m128 low = _mm_set1_ps(-0.5f);
        const __m128 up = _mm_or_ps(_mm_cmpgt_ps(d, half), _mm_and_ps(_mm_cmpeq_ps(d, half), odd));
        const __m128 down = _mm_or_ps(_mm_cmplt_ps(d, low), _mm_and_ps(_mm_cmpeq_ps(d, low), odd));
        const __m128 step = _mm_or_ps(_mm_and_ps(up, _mm_set1_ps(1.0f)), _mm_and_ps(down, _mm_set1_ps(-1.0f)));
        const __m128 v = WRAY_VOP(_mm_add_ps, +, r, step);
        const __m128 rounded = _mm_or_ps(_mm_andnot_ps(sign, v), _mm_and_ps(a.v, sign));
        const __m128 keep = _mm_cmpnlt_ps(_mm_andnot_ps(sign, a.v), _mm_set1_ps(0x1p23f));
        return _mm_or_ps(_mm_and_ps(keep, a.v), _mm_andnot_ps(keep, rounded));
#endif
    }
    WRAY_INLINE float first(const f32x4 &a) noexcept { return _mm_cvtss_f32(a.v); }
    WRAY_INLINE m32x4 operator==(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmpeq_ps(a.v, b.v))};
    }
    WRAY_INLINE m32x4 operator!=(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmpneq_ps(a.v, b.v))};
    }
    WRAY_INLINE m32x4 operator<(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmplt_ps(a.v, b.v))};
    }
    WRAY_INLINE m32x4 operator<=(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmple_ps(a.v, b.v))};
    }
    WRAY_INLINE m32x4 operator>(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmpgt_ps(a.v, b.v))};
    }
    WRAY_INLINE m32x4 operator>=(const f32x4 &a, const f32x4 &b) noexcept
    {
        return {_mm_castps_si128(_mm_cmpge_ps(a.v, b.v))};
    }
    WRAY_INLINE f32x4 select(const m32x4 &m, const f32x4 &a, const f32x4 &b) noexcept
    {
        const __m128 k = _mm_castsi128_ps(m.v);
        return _mm_or_ps(_mm_and_ps(k, a.v), _mm_andnot_ps(k, b.v));
    }
    WRAY_INLINE i32x4 as_bits(const f32x4 &a) noexcept { return _mm_castps_si128(a.v); }
    template<>
    WRAY_INLINE f32x4 from_bits<f32x4, i32x4>(const i32x4 &a) noexcept { return _mm_castsi128_ps(a.v); }

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return _mm_add_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return _mm_sub_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return _mm_mul_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return _mm_div_pd(a.v, b.v); }
#endif
    WRAY_INLINE f64x2 operator-(const f64x2 &a) noexcept { return _mm_xor_pd(a.v, _mm_set1_pd(-0.0)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept
    {
        return _mm_fmadd_pd(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept
    {
        return WRAY_VOP(_mm_add_pd, +, WRAY_VOP(_mm_mul_pd, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f64x2 min(const f64x2 &a, const f64x2 &b) noexcept { return _mm_min_pd(a.v, b.v); }
    WRAY_INLINE f64x2 max(const f64x2 &a, const f64x2 &b) noexcept { return _mm_max_pd(a.v, b.v); }
    WRAY_INLINE f64x2 abs(const f64x2 &a) noexcept { return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v); }
    WRAY_INLINE f64x2 sqrt(const f64x2 &a) noexcept { return _mm_sqrt_pd(a.v); }
    WRAY_INLINE f64x2 nearest(const f64x2 &a) noexcept
    {
#if defined(__SSE4_1__)
        return _mm_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        const __m128d sign = _mm_set1_pd(-0.0);
        const __m128d m = _mm_or_pd(_mm_set1_pd(0x1p52), _mm_and_pd(a.v, sign));
        const __m128d c = WRAY_VOP(_mm_add_pd, +, a.v, m);
        const __m128d r = WRAY_VOP(_mm_sub_pd, -, c, m);
        const __m128d d = WRAY_VOP(_mm_sub_pd, -, a.v, r);
        const __m128i one = _mm_set_epi32(0, 1, 0, 1);
        const __m128i low_bit = _mm_cmpeq_epi32(_mm_and_si128(_mm_castpd_si128(c), one), one);
        const __m128d odd = _mm_castsi128_pd(_mm_shuffle_epi32(low_bit, _MM_SHUFFLE(2, 2, 0, 0)));
        const __m128d half = _mm_set1_pd(0.5);
        const __m128d low = _mm_set1_pd(-0.5);
        const __m128d up = _mm_or_pd(_mm_cmpgt_pd(d, half), _mm_and_pd(_mm_cmpeq_pd(d, half), odd));
        const __m128d down = _mm_or_pd(_mm_cmplt_pd(d, low), _mm_and_pd(_mm_cmpeq_pd(d, low), odd));
        const __m128d step = _mm_or_pd(_mm_and_pd(up, _mm_set1_pd(1.0)), _mm_and_pd(down, _mm_set1_pd(-1.0)));
        const __m128d v = WRAY_VOP(_mm_add_pd, +, r, step);
        const __m128d rounded = _mm_or_pd(_mm_andnot_pd(sign, v), _mm_and_pd(a.v, sign));
        const __m128d keep = _mm_cmpnlt_pd(_mm_andnot_pd(sign, a.v), _mm_set1_pd(0x1p52));
        return _mm_or_pd(_mm_and_pd(keep, a.v), _mm_andnot_pd(keep, rounded));
#endif
    }
    WRAY_INLINE double first(const f64x2 &a) noexcept { return _mm_cvtsd_f64(a.v); }
    WRAY_INLINE m64x2 operator==(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmpeq_pd(a.v, b.v))};
    }
    WRAY_INLINE m64x2 operator!=(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmpneq_pd(a.v, b.v))};
    }
    WRAY_INLINE m64x2 operator<(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmplt_pd(a.v, b.v))};
    }
    WRAY_INLINE m64x2 operator<=(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmple_pd(a.v, b.v))};
    }
    WRAY_INLINE m64x2 operator>(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmpgt_pd(a.v, b.v))};
    }
    WRAY_INLINE m64x2 operator>=(const f64x2 &a, const f64x2 &b) noexcept
    {
        return {_mm_castpd_si128(_mm_cmpge_pd(a.v, b.v))};
    }
    WRAY_INLINE f64x2 select(const m64x2 &m, const f64x2 &a, const f64x2 &b) noexcept
    {
        const __m128d k = _mm_castsi128_pd(m.v);
        return _mm_or_pd(_mm_and_pd(k, a.v), _mm_andnot_pd(k, b.v));
    }
    WRAY_INLINE i64x2 as_bits(const f64x2 &a) noexcept { return _mm_castpd_si128(a.v); }
    template<>
    WRAY_INLINE f64x2 from_bits<f64x2, i64x2>(const i64x2 &a) noexcept { return _mm_castsi128_pd(a.v); }

    WRAY_INLINE i32x4 operator+(const i32x4 &a, const i32x4 &b) noexcept { return _mm_add_epi32(a.v, b.v); }
    WRAY_INLINE i32x4 operator-(const i32x4 &a, const i32x4 &b) noexcept { return _mm_sub_epi32(a.v, b.v); }
    WRAY_INLINE i32x4 operator&(const i32x4 &a, const i32x4 &b) noexcept { return _mm_and_si128(a.v, b.v); }
    WRAY_INLINE i32x4 operator|(const i32x4 &a, const i32x4 &b) noexcept { return _mm_or_si128(a.v, b.v); }
    WRAY_INLINE i32x4 operator^(const i32x4 &a, const i32x4 &b) noexcept { return _mm_xor_si128(a.v, b.v); }
    WRAY_INLINE i32x4 operator<<(const i32x4 &a, int s) noexcept { return _mm_sll_epi32(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i32x4 shift_right_logical(const i32x4 &a, int s) noexcept
    {
        return _mm_srl_epi32(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m32x4 nonzero(const i32x4 &a) noexcept
    {
        return {_mm_xor_si128(_mm_cmpeq_epi32(a.v, _mm_setzero_si128()), _mm_set1_epi32(-1))};
    }
    WRAY_INLINE m32x4 operator==(const i32x4 &a, const i32x4 &b) noexcept { return {_mm_cmpeq_epi32(a.v, b.v)}; }

    WRAY_INLINE i64x2 operator+(const i64x2 &a, const i64x2 &b) noexcept { return _mm_add_epi64(a.v, b.v); }
    WRAY_INLINE i64x2 operator-(const i64x2 &a, const i64x2 &b) noexcept { return _mm_sub_epi64(a.v, b.v); }
    WRAY_INLINE i64x2 operator&(const i64x2 &a, const i64x2 &b) noexcept { return _mm_and_si128(a.v, b.v); }
    WRAY_INLINE i64x2 operator|(const i64x2 &a, const i64x2 &b) noexcept { return _mm_or_si128(a.v, b.v); }
    WRAY_INLINE i64x2 operator^(const i64x2 &a, const i64x2 &b) noexcept { return _mm_xor_si128(a.v, b.v); }
    WRAY_INLINE i64x2 operator<<(const i64x2 &a, int s) noexcept { return _mm_sll_epi64(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i64x2 shift_right_logical(const i64x2 &a, int s) noexcept
    {
        return _mm_srl_epi64(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m64x2 operator==(const i64x2 &a, const i64x2 &b) noexcept
    {
        const __m128i e = _mm_cmpeq_epi32(a.v, b.v);
        return {_mm_and_si128(e, _mm_shuffle_epi32(e, _MM_SHUFFLE(2, 3, 0, 1)))};
    }
    WRAY_INLINE m64x2 nonzero(const i64x2 &a) noexcept
    {
        return {_mm_xor_si128((a == i64x2(std::int64_t(0))).v, _mm_set1_epi32(-1))};
    }

    WRAY_INLINE m32x4 operator&(const m32x4 &a, const m32x4 &b) noexcept { return {_mm_and_si128(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator|(const m32x4 &a, const m32x4 &b) noexcept { return {_mm_or_si128(a.v, b.v)}; }
    WRAY_INLINE m32x4 operator!(const m32x4 &a) noexcept { return {_mm_xor_si128(a.v, _mm_set1_epi32(-1))}; }
    WRAY_INLINE bool any(const m32x4 &a) noexcept { return _mm_movemask_epi8(a.v) != 0; }
    WRAY_INLINE bool all(const m32x4 &a) noexcept { return _mm_movemask_epi8(a.v) == 0xFFFF; }
    WRAY_INLINE m64x2 operator&(const m64x2 &a, const m64x2 &b) noexcept { return {_mm_and_si128(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator|(const m64x2 &a, const m64x2 &b) noexcept { return {_mm_or_si128(a.v, b.v)}; }
    WRAY_INLINE m64x2 operator!(const m64x2 &a) noexcept { return {_mm_xor_si128(a.v, _mm_set1_epi32(-1))}; }
    WRAY_INLINE bool any(const m64x2 &a) noexcept { return _mm_movemask_epi8(a.v) != 0; }
    WRAY_INLINE bool all(const m64x2 &a) noexcept { return _mm_movemask_epi8(a.v) == 0xFFFF; }

#if defined(WRAY_SIMD_AVX2)

    template<>
    struct mask<32, 8>
    {
        __m256i v;
    };

    template<>
    struct mask<64, 4>
    {
        __m256i v;
    };

    template<>
    struct vec<float, 8>
    {
        using value_type = float;
        static constexpr int size = 8;
        __m256 v;

        vec() = default;

        WRAY_INLINE vec(__m256 x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(float s) noexcept : v(_mm256_set1_ps(s))
        {
        }

        WRAY_INLINE static vec load(const float *p) noexcept { return _mm256_loadu_ps(p); }
        WRAY_INLINE void store(float *p) const noexcept { _mm256_storeu_ps(p, v); }
        WRAY_INLINE float operator[](int i) const noexcept
        {
            float t[8];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<double, 4>
    {
        using value_type = double;
        static constexpr int size = 4;
        __m256d v;

        vec() = default;

        WRAY_INLINE vec(__m256d x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(double s) noexcept : v(_mm256_set1_pd(s))
        {
        }

        WRAY_INLINE static vec load(const double *p) noexcept { return _mm256_loadu_pd(p); }
        WRAY_INLINE void store(double *p) const noexcept { _mm256_storeu_pd(p, v); }
        WRAY_INLINE double operator[](int i) const noexcept
        {
            double t[4];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int32_t, 8>
    {
        using value_type = std::int32_t;
        static constexpr int size = 8;
        __m256i v;

        vec() = default;

        WRAY_INLINE vec(__m256i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int32_t s) noexcept : v(_mm256_set1_epi32(s))
        {
        }

        WRAY_INLINE static vec load(const std::int32_t *p) noexcept
        {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        }

        WRAY_INLINE void store(std::int32_t *p) const noexcept { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
        WRAY_INLINE std::int32_t operator[](int i) const noexcept
        {
            std::int32_t t[8];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int64_t, 4>
    {
        using value_type = std::int64_t;
        static constexpr int size = 4;
        __m256i v;

        vec() = default;

        WRAY_INLINE vec(__m256i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int64_t s) noexcept : v(_mm256_set1_epi64x(s))
        {
        }

        WRAY_INLINE static vec load(const std::int64_t *p) noexcept
        {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        }

        WRAY_INLINE void store(std::int64_t *p) const noexcept { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
        WRAY_INLINE std::int64_t operator[](int i) const noexcept
        {
            std::int64_t t[4];
            store(t);
            return t[i];
        }
    };

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x8 operator+(const f32x8 &a, const f32x8 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f32x8 operator+(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_add_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x8 operator-(const f32x8 &a, const f32x8 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f32x8 operator-(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_sub_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x8 operator*(const f32x8 &a, const f32x8 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f32x8 operator*(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_mul_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x8 operator/(const f32x8 &a, const f32x8 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f32x8 operator/(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_div_ps(a.v, b.v); }
#endif
    WRAY_INLINE f32x8 operator-(const f32x8 &a) noexcept { return _mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f32x8 fma(const f32x8 &a, const f32x8 &b, const f32x8 &c) noexcept
    {
        return _mm256_fmadd_ps(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f32x8 fma(const f32x8 &a, const f32x8 &b, const f32x8 &c) noexcept
    {
        return WRAY_VOP(_mm256_add_ps, +, WRAY_VOP(_mm256_mul_ps, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f32x8 min(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_min_ps(a.v, b.v); }
    WRAY_INLINE f32x8 max(const f32x8 &a, const f32x8 &b) noexcept { return _mm256_max_ps(a.v, b.v); }
    WRAY_INLINE f32x8 abs(const f32x8 &a) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v); }
    WRAY_INLINE f32x8 sqrt(const f32x8 &a) noexcept { return _mm256_sqrt_ps(a.v); }
    WRAY_INLINE f32x8 nearest(const f32x8 &a) noexcept { return _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
    WRAY_INLINE float first(const f32x8 &a) noexcept { return _mm256_cvtss_f32(a.v); }
    WRAY_INLINE m32x8 operator==(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ))};
    }
    WRAY_INLINE m32x8 operator!=(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_UQ))};
    }
    WRAY_INLINE m32x8 operator<(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ))};
    }
    WRAY_INLINE m32x8 operator<=(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ))};
    }
    WRAY_INLINE m32x8 operator>(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ))};
    }
    WRAY_INLINE m32x8 operator>=(const f32x8 &a, const f32x8 &b) noexcept
    {
        return {_mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ))};
    }
    WRAY_INLINE f32x8 select(const m32x8 &m, const f32x8 &a, const f32x8 &b) noexcept
    {
        return _mm256_blendv_ps(b.v, a.v, _mm256_castsi256_ps(m.v));
    }
    WRAY_INLINE i32x8 as_bits(const f32x8 &a) noexcept { return _mm256_castps_si256(a.v); }
    template<>
    WRAY_INLINE f32x8 from_bits<f32x8, i32x8>(const i32x8 &a) noexcept { return _mm256_castsi256_ps(a.v); }

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x4 operator+(const f64x4 &a, const f64x4 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f64x4 operator+(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_add_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x4 operator-(const f64x4 &a, const f64x4 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f64x4 operator-(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_sub_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x4 operator*(const f64x4 &a, const f64x4 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f64x4 operator*(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_mul_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x4 operator/(const f64x4 &a, const f64x4 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f64x4 operator/(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_div_pd(a.v, b.v); }
#endif
    WRAY_INLINE f64x4 operator-(const f64x4 &a) noexcept { return _mm256_xor_pd(a.v, _mm256_set1_pd(-0.0)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f64x4 fma(const f64x4 &a, const f64x4 &b, const f64x4 &c) noexcept
    {
        return _mm256_fmadd_pd(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f64x4 fma(const f64x4 &a, const f64x4 &b, const f64x4 &c) noexcept
    {
        return WRAY_VOP(_mm256_add_pd, +, WRAY_VOP(_mm256_mul_pd, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f64x4 min(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_min_pd(a.v, b.v); }
    WRAY_INLINE f64x4 max(const f64x4 &a, const f64x4 &b) noexcept { return _mm256_max_pd(a.v, b.v); }
    WRAY_INLINE f64x4 abs(const f64x4 &a) noexcept { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a.v); }
    WRAY_INLINE f64x4 sqrt(const f64x4 &a) noexcept { return _mm256_sqrt_pd(a.v); }
    WRAY_INLINE f64x4 nearest(const f64x4 &a) noexcept { return _mm256_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
    WRAY_INLINE double first(const f64x4 &a) noexcept { return _mm256_cvtsd_f64(a.v); }
    WRAY_INLINE m64x4 operator==(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ))};
    }
    WRAY_INLINE m64x4 operator!=(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_UQ))};
    }
    WRAY_INLINE m64x4 operator<(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ))};
    }
    WRAY_INLINE m64x4 operator<=(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ))};
    }
    WRAY_INLINE m64x4 operator>(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ))};
    }
    WRAY_INLINE m64x4 operator>=(const f64x4 &a, const f64x4 &b) noexcept
    {
        return {_mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ))};
    }
    WRAY_INLINE f64x4 select(const m64x4 &m, const f64x4 &a, const f64x4 &b) noexcept
    {
        return _mm256_blendv_pd(b.v, a.v, _mm256_castsi256_pd(m.v));
    }
    WRAY_INLINE i64x4 as_bits(const f64x4 &a) noexcept { return _mm256_castpd_si256(a.v); }
    template<>
    WRAY_INLINE f64x4 from_bits<f64x4, i64x4>(const i64x4 &a) noexcept { return _mm256_castsi256_pd(a.v); }

    WRAY_INLINE i32x8 operator+(const i32x8 &a, const i32x8 &b) noexcept { return _mm256_add_epi32(a.v, b.v); }
    WRAY_INLINE i32x8 operator-(const i32x8 &a, const i32x8 &b) noexcept { return _mm256_sub_epi32(a.v, b.v); }
    WRAY_INLINE i32x8 operator&(const i32x8 &a, const i32x8 &b) noexcept { return _mm256_and_si256(a.v, b.v); }
    WRAY_INLINE i32x8 operator|(const i32x8 &a, const i32x8 &b) noexcept { return _mm256_or_si256(a.v, b.v); }
    WRAY_INLINE i32x8 operator^(const i32x8 &a, const i32x8 &b) noexcept { return _mm256_xor_si256(a.v, b.v); }
    WRAY_INLINE i32x8 operator<<(const i32x8 &a, int s) noexcept { return _mm256_sll_epi32(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i32x8 shift_right_logical(const i32x8 &a, int s) noexcept
    {
        return _mm256_srl_epi32(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m32x8 operator==(const i32x8 &a, const i32x8 &b) noexcept { return {_mm256_cmpeq_epi32(a.v, b.v)}; }
    WRAY_INLINE m32x8 nonzero(const i32x8 &a) noexcept
    {
        return {_mm256_xor_si256(_mm256_cmpeq_epi32(a.v, _mm256_setzero_si256()), _mm256_set1_epi32(-1))};
    }

    WRAY_INLINE i64x4 operator+(const i64x4 &a, const i64x4 &b) noexcept { return _mm256_add_epi64(a.v, b.v); }
    WRAY_INLINE i64x4 operator-(const i64x4 &a, const i64x4 &b) noexcept { return _mm256_sub_epi64(a.v, b.v); }
    WRAY_INLINE i64x4 operator&(const i64x4 &a, const i64x4 &b) noexcept { return _mm256_and_si256(a.v, b.v); }
    WRAY_INLINE i64x4 operator|(const i64x4 &a, const i64x4 &b) noexcept { return _mm256_or_si256(a.v, b.v); }
    WRAY_INLINE i64x4 operator^(const i64x4 &a, const i64x4 &b) noexcept { return _mm256_xor_si256(a.v, b.v); }
    WRAY_INLINE i64x4 operator<<(const i64x4 &a, int s) noexcept { return _mm256_sll_epi64(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i64x4 shift_right_logical(const i64x4 &a, int s) noexcept
    {
        return _mm256_srl_epi64(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m64x4 operator==(const i64x4 &a, const i64x4 &b) noexcept { return {_mm256_cmpeq_epi64(a.v, b.v)}; }
    WRAY_INLINE m64x4 nonzero(const i64x4 &a) noexcept
    {
        return {_mm256_xor_si256(_mm256_cmpeq_epi64(a.v, _mm256_setzero_si256()), _mm256_set1_epi32(-1))};
    }

    WRAY_INLINE m32x8 operator&(const m32x8 &a, const m32x8 &b) noexcept { return {_mm256_and_si256(a.v, b.v)}; }
    WRAY_INLINE m32x8 operator|(const m32x8 &a, const m32x8 &b) noexcept { return {_mm256_or_si256(a.v, b.v)}; }
    WRAY_INLINE m32x8 operator!(const m32x8 &a) noexcept { return {_mm256_xor_si256(a.v, _mm256_set1_epi32(-1))}; }
    WRAY_INLINE bool any(const m32x8 &a) noexcept { return _mm256_testz_si256(a.v, a.v) == 0; }
    WRAY_INLINE bool all(const m32x8 &a) noexcept { return _mm256_movemask_epi8(a.v) == -1; }
    WRAY_INLINE m64x4 operator&(const m64x4 &a, const m64x4 &b) noexcept { return {_mm256_and_si256(a.v, b.v)}; }
    WRAY_INLINE m64x4 operator|(const m64x4 &a, const m64x4 &b) noexcept { return {_mm256_or_si256(a.v, b.v)}; }
    WRAY_INLINE m64x4 operator!(const m64x4 &a) noexcept { return {_mm256_xor_si256(a.v, _mm256_set1_epi32(-1))}; }
    WRAY_INLINE bool any(const m64x4 &a) noexcept { return _mm256_testz_si256(a.v, a.v) == 0; }
    WRAY_INLINE bool all(const m64x4 &a) noexcept { return _mm256_movemask_epi8(a.v) == -1; }

#endif

#if defined(WRAY_SIMD_AVX512)

    template<>
    struct mask<32, 16>
    {
        __mmask16 v;
    };

    template<>
    struct mask<64, 8>
    {
        __mmask8 v;
    };

    template<>
    struct vec<float, 16>
    {
        using value_type = float;
        static constexpr int size = 16;
        __m512 v;

        vec() = default;

        WRAY_INLINE vec(__m512 x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(float s) noexcept : v(_mm512_set1_ps(s))
        {
        }

        WRAY_INLINE static vec load(const float *p) noexcept { return _mm512_loadu_ps(p); }
        WRAY_INLINE void store(float *p) const noexcept { _mm512_storeu_ps(p, v); }
        WRAY_INLINE float operator[](int i) const noexcept
        {
            float t[16];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<double, 8>
    {
        using value_type = double;
        static constexpr int size = 8;
        __m512d v;

        vec() = default;

        WRAY_INLINE vec(__m512d x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(double s) noexcept : v(_mm512_set1_pd(s))
        {
        }

        WRAY_INLINE static vec load(const double *p) noexcept { return _mm512_loadu_pd(p); }
        WRAY_INLINE void store(double *p) const noexcept { _mm512_storeu_pd(p, v); }
        WRAY_INLINE double operator[](int i) const noexcept
        {
            double t[8];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int32_t, 16>
    {
        using value_type = std::int32_t;
        static constexpr int size = 16;
        __m512i v;

        vec() = default;

        WRAY_INLINE vec(__m512i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int32_t s) noexcept : v(_mm512_set1_epi32(s))
        {
        }

        WRAY_INLINE static vec load(const std::int32_t *p) noexcept { return _mm512_loadu_si512(p); }
        WRAY_INLINE void store(std::int32_t *p) const noexcept { _mm512_storeu_si512(p, v); }
        WRAY_INLINE std::int32_t operator[](int i) const noexcept
        {
            std::int32_t t[16];
            store(t);
            return t[i];
        }
    };

    template<>
    struct vec<std::int64_t, 8>
    {
        using value_type = std::int64_t;
        static constexpr int size = 8;
        __m512i v;

        vec() = default;

        WRAY_INLINE vec(__m512i x) noexcept : v(x)
        {
        }

        WRAY_INLINE vec(std::int64_t s) noexcept : v(_mm512_set1_epi64(s))
        {
        }

        WRAY_INLINE static vec load(const std::int64_t *p) noexcept { return _mm512_loadu_si512(p); }
        WRAY_INLINE void store(std::int64_t *p) const noexcept { _mm512_storeu_si512(p, v); }
        WRAY_INLINE std::int64_t operator[](int i) const noexcept
        {
            std::int64_t t[8];
            store(t);
            return t[i];
        }
    };

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x16 operator+(const f32x16 &a, const f32x16 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f32x16 operator+(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_add_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x16 operator-(const f32x16 &a, const f32x16 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f32x16 operator-(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_sub_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x16 operator*(const f32x16 &a, const f32x16 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f32x16 operator*(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_mul_ps(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f32x16 operator/(const f32x16 &a, const f32x16 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f32x16 operator/(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_div_ps(a.v, b.v); }
#endif
    WRAY_INLINE f32x16 operator-(const f32x16 &a) noexcept
    {
        return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(std::numeric_limits<std::int32_t>::min())));
    }
#if WRAY_HAS_FMA
    WRAY_INLINE f32x16 fma(const f32x16 &a, const f32x16 &b, const f32x16 &c) noexcept
    {
        return _mm512_fmadd_ps(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f32x16 fma(const f32x16 &a, const f32x16 &b, const f32x16 &c) noexcept
    {
        return WRAY_VOP(_mm512_add_ps, +, WRAY_VOP(_mm512_mul_ps, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f32x16 min(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_min_ps(a.v, b.v); }
    WRAY_INLINE f32x16 max(const f32x16 &a, const f32x16 &b) noexcept { return _mm512_max_ps(a.v, b.v); }
    WRAY_INLINE f32x16 abs(const f32x16 &a) noexcept { return _mm512_abs_ps(a.v); }
    WRAY_INLINE f32x16 sqrt(const f32x16 &a) noexcept { return _mm512_sqrt_ps(a.v); }
    WRAY_INLINE f32x16 nearest(const f32x16 &a) noexcept { return _mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
    WRAY_INLINE float first(const f32x16 &a) noexcept { return _mm_cvtss_f32(_mm512_castps512_ps128(a.v)); }
    WRAY_INLINE m32x16 operator==(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ)}; }
    WRAY_INLINE m32x16 operator!=(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_UQ)}; }
    WRAY_INLINE m32x16 operator<(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ)}; }
    WRAY_INLINE m32x16 operator<=(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ)}; }
    WRAY_INLINE m32x16 operator>(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ)}; }
    WRAY_INLINE m32x16 operator>=(const f32x16 &a, const f32x16 &b) noexcept { return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ)}; }
    WRAY_INLINE f32x16 select(const m32x16 &m, const f32x16 &a, const f32x16 &b) noexcept
    {
        return _mm512_mask_blend_ps(m.v, b.v, a.v);
    }
    WRAY_INLINE i32x16 as_bits(const f32x16 &a) noexcept { return _mm512_castps_si512(a.v); }
    template<>
    WRAY_INLINE f32x16 from_bits<f32x16, i32x16>(const i32x16 &a) noexcept { return _mm512_castsi512_ps(a.v); }

#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x8 operator+(const f64x8 &a, const f64x8 &b) noexcept { return a.v + b.v; }
#else
    WRAY_INLINE f64x8 operator+(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_add_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x8 operator-(const f64x8 &a, const f64x8 &b) noexcept { return a.v - b.v; }
#else
    WRAY_INLINE f64x8 operator-(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_sub_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x8 operator*(const f64x8 &a, const f64x8 &b) noexcept { return a.v * b.v; }
#else
    WRAY_INLINE f64x8 operator*(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_mul_pd(a.v, b.v); }
#endif
#if defined(__GNUC__) || defined(__clang__)
    WRAY_INLINE f64x8 operator/(const f64x8 &a, const f64x8 &b) noexcept { return a.v / b.v; }
#else
    WRAY_INLINE f64x8 operator/(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_div_pd(a.v, b.v); }
#endif
    WRAY_INLINE f64x8 operator-(const f64x8 &a) noexcept
    {
        return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v), _mm512_set1_epi64(std::numeric_limits<std::int64_t>::min())));
    }
#if WRAY_HAS_FMA
    WRAY_INLINE f64x8 fma(const f64x8 &a, const f64x8 &b, const f64x8 &c) noexcept
    {
        return _mm512_fmadd_pd(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f64x8 fma(const f64x8 &a, const f64x8 &b, const f64x8 &c) noexcept
    {
        return WRAY_VOP(_mm512_add_pd, +, WRAY_VOP(_mm512_mul_pd, *, a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f64x8 min(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_min_pd(a.v, b.v); }
    WRAY_INLINE f64x8 max(const f64x8 &a, const f64x8 &b) noexcept { return _mm512_max_pd(a.v, b.v); }
    WRAY_INLINE f64x8 abs(const f64x8 &a) noexcept { return _mm512_abs_pd(a.v); }
    WRAY_INLINE f64x8 sqrt(const f64x8 &a) noexcept { return _mm512_sqrt_pd(a.v); }
    WRAY_INLINE f64x8 nearest(const f64x8 &a) noexcept { return _mm512_roundscale_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
    WRAY_INLINE double first(const f64x8 &a) noexcept { return _mm_cvtsd_f64(_mm512_castpd512_pd128(a.v)); }
    WRAY_INLINE m64x8 operator==(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ)}; }
    WRAY_INLINE m64x8 operator!=(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_UQ)}; }
    WRAY_INLINE m64x8 operator<(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ)}; }
    WRAY_INLINE m64x8 operator<=(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ)}; }
    WRAY_INLINE m64x8 operator>(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ)}; }
    WRAY_INLINE m64x8 operator>=(const f64x8 &a, const f64x8 &b) noexcept { return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ)}; }
    WRAY_INLINE f64x8 select(const m64x8 &m, const f64x8 &a, const f64x8 &b) noexcept
    {
        return _mm512_mask_blend_pd(m.v, b.v, a.v);
    }
    WRAY_INLINE i64x8 as_bits(const f64x8 &a) noexcept { return _mm512_castpd_si512(a.v); }
    template<>
    WRAY_INLINE f64x8 from_bits<f64x8, i64x8>(const i64x8 &a) noexcept { return _mm512_castsi512_pd(a.v); }

    WRAY_INLINE i32x16 operator+(const i32x16 &a, const i32x16 &b) noexcept { return _mm512_add_epi32(a.v, b.v); }
    WRAY_INLINE i32x16 operator-(const i32x16 &a, const i32x16 &b) noexcept { return _mm512_sub_epi32(a.v, b.v); }
    WRAY_INLINE i32x16 operator&(const i32x16 &a, const i32x16 &b) noexcept { return _mm512_and_si512(a.v, b.v); }
    WRAY_INLINE i32x16 operator|(const i32x16 &a, const i32x16 &b) noexcept { return _mm512_or_si512(a.v, b.v); }
    WRAY_INLINE i32x16 operator^(const i32x16 &a, const i32x16 &b) noexcept { return _mm512_xor_si512(a.v, b.v); }
    WRAY_INLINE i32x16 operator<<(const i32x16 &a, int s) noexcept { return _mm512_sll_epi32(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i32x16 shift_right_logical(const i32x16 &a, int s) noexcept
    {
        return _mm512_srl_epi32(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m32x16 operator==(const i32x16 &a, const i32x16 &b) noexcept { return {_mm512_cmpeq_epi32_mask(a.v, b.v)}; }
    WRAY_INLINE m32x16 nonzero(const i32x16 &a) noexcept { return {_mm512_test_epi32_mask(a.v, a.v)}; }

    WRAY_INLINE i64x8 operator+(const i64x8 &a, const i64x8 &b) noexcept { return _mm512_add_epi64(a.v, b.v); }
    WRAY_INLINE i64x8 operator-(const i64x8 &a, const i64x8 &b) noexcept { return _mm512_sub_epi64(a.v, b.v); }
    WRAY_INLINE i64x8 operator&(const i64x8 &a, const i64x8 &b) noexcept { return _mm512_and_si512(a.v, b.v); }
    WRAY_INLINE i64x8 operator|(const i64x8 &a, const i64x8 &b) noexcept { return _mm512_or_si512(a.v, b.v); }
    WRAY_INLINE i64x8 operator^(const i64x8 &a, const i64x8 &b) noexcept { return _mm512_xor_si512(a.v, b.v); }
    WRAY_INLINE i64x8 operator<<(const i64x8 &a, int s) noexcept { return _mm512_sll_epi64(a.v, _mm_cvtsi32_si128(s)); }
    WRAY_INLINE i64x8 shift_right_logical(const i64x8 &a, int s) noexcept
    {
        return _mm512_srl_epi64(a.v, _mm_cvtsi32_si128(s));
    }
    WRAY_INLINE m64x8 operator==(const i64x8 &a, const i64x8 &b) noexcept { return {_mm512_cmpeq_epi64_mask(a.v, b.v)}; }
    WRAY_INLINE m64x8 nonzero(const i64x8 &a) noexcept { return {_mm512_test_epi64_mask(a.v, a.v)}; }

    WRAY_INLINE m32x16 operator&(const m32x16 &a, const m32x16 &b) noexcept { return {static_cast<__mmask16>(a.v & b.v)}; }
    WRAY_INLINE m32x16 operator|(const m32x16 &a, const m32x16 &b) noexcept { return {static_cast<__mmask16>(a.v | b.v)}; }
    WRAY_INLINE m32x16 operator!(const m32x16 &a) noexcept { return {static_cast<__mmask16>(~a.v)}; }
    WRAY_INLINE bool any(const m32x16 &a) noexcept { return a.v != 0; }
    WRAY_INLINE bool all(const m32x16 &a) noexcept { return a.v == 0xFFFF; }
    WRAY_INLINE m64x8 operator&(const m64x8 &a, const m64x8 &b) noexcept { return {static_cast<__mmask8>(a.v & b.v)}; }
    WRAY_INLINE m64x8 operator|(const m64x8 &a, const m64x8 &b) noexcept { return {static_cast<__mmask8>(a.v | b.v)}; }
    WRAY_INLINE m64x8 operator!(const m64x8 &a) noexcept { return {static_cast<__mmask8>(~a.v)}; }
    WRAY_INLINE bool any(const m64x8 &a) noexcept { return a.v != 0; }
    WRAY_INLINE bool all(const m64x8 &a) noexcept { return a.v == 0xFF; }

#endif

#endif

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator+(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo + b.lo, a.hi + b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator-(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo - b.lo, a.hi - b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator*(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo * b.lo, a.hi * b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator/(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo / b.lo, a.hi / b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator-(const vec<T, N> &a) noexcept { return {-a.lo, -a.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator&(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo & b.lo, a.hi & b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator|(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo | b.lo, a.hi | b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator^(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo ^ b.lo, a.hi ^ b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> operator<<(const vec<T, N> &a, int s) noexcept { return {a.lo << s, a.hi << s}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> shift_right_logical(const vec<T, N> &a, int s) noexcept
    {
        return {shift_right_logical(a.lo, s), shift_right_logical(a.hi, s)};
    }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator==(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo == b.lo, a.hi == b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator!=(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo != b.lo, a.hi != b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator<(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo < b.lo, a.hi < b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator<=(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo <= b.lo, a.hi <= b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator>(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo > b.lo, a.hi > b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> operator>=(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {a.lo >= b.lo, a.hi >= b.hi}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE mask_for<T, N> nonzero(const vec<T, N> &a) noexcept { return {nonzero(a.lo), nonzero(a.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> fma(const vec<T, N> &a, const vec<T, N> &b, const vec<T, N> &c) noexcept
    {
        return {fma(a.lo, b.lo, c.lo), fma(a.hi, b.hi, c.hi)};
    }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> min(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {min(a.lo, b.lo), min(a.hi, b.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> max(const vec<T, N> &a, const vec<T, N> &b) noexcept { return {max(a.lo, b.lo), max(a.hi, b.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> abs(const vec<T, N> &a) noexcept { return {abs(a.lo), abs(a.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> sqrt(const vec<T, N> &a) noexcept { return {sqrt(a.lo), sqrt(a.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> nearest(const vec<T, N> &a) noexcept { return {nearest(a.lo), nearest(a.hi)}; }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE T first(const vec<T, N> &a) noexcept { return first(a.lo); }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE vec<T, N> select(const mask_for<T, N> &m, const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
        return {select(m.lo, a.lo, b.lo), select(m.hi, a.hi, b.hi)};
    }

    template<class T, int N>
        requires split<T, N>
    WRAY_INLINE int_for<T, N> as_bits(const vec<T, N> &a) noexcept { return {as_bits(a.lo), as_bits(a.hi)}; }

    template<class V, class I>
        requires split<typename V::value_type, V::size>
    WRAY_INLINE V from_bits(const I &a) noexcept
    {
        return {from_bits<typename V::half>(a.lo), from_bits<typename V::half>(a.hi)};
    }

    template<int B, int N>
        requires wide<B / 8, N>
    WRAY_INLINE mask<B, N> operator&(const mask<B, N> &a, const mask<B, N> &b) noexcept { return {a.lo & b.lo, a.hi & b.hi}; }

    template<int B, int N>
        requires wide<B / 8, N>
    WRAY_INLINE mask<B, N> operator|(const mask<B, N> &a, const mask<B, N> &b) noexcept { return {a.lo | b.lo, a.hi | b.hi}; }

    template<int B, int N>
        requires wide<B / 8, N>
    WRAY_INLINE mask<B, N> operator!(const mask<B, N> &a) noexcept { return {!a.lo, !a.hi}; }

    template<int B, int N>
        requires wide<B / 8, N>
    WRAY_INLINE bool any(const mask<B, N> &a) noexcept { return any(a.lo | a.hi); }

    template<int B, int N>
        requires wide<B / 8, N>
    WRAY_INLINE bool all(const mask<B, N> &a) noexcept { return all(a.lo & a.hi); }

    WRAY_INLINE float fma(float a, float b, float c) noexcept
    {
#if WRAY_HAS_FMA
        return WRAY_FMA(a, b, c);
#else
        float p = a * b;
        return p + c;
#endif
    }

    WRAY_INLINE double fma(double a, double b, double c) noexcept
    {
#if WRAY_HAS_FMA
        return WRAY_FMA(a, b, c);
#else
        double p = a * b;
        return p + c;
#endif
    }

    WRAY_INLINE float abs(float a) noexcept { return std::fabs(a); }
    WRAY_INLINE double abs(double a) noexcept { return std::fabs(a); }
    WRAY_INLINE float sqrt(float a) noexcept { return WRAY_SQRT(a); }
    WRAY_INLINE double sqrt(double a) noexcept { return WRAY_SQRT(a); }
    WRAY_INLINE float min(float a, float b) noexcept { return b < a ? b : a; }
    WRAY_INLINE double min(double a, double b) noexcept { return b < a ? b : a; }
    WRAY_INLINE float max(float a, float b) noexcept { return a < b ? b : a; }
    WRAY_INLINE double max(double a, double b) noexcept { return a < b ? b : a; }
    WRAY_INLINE float select(bool m, float a, float b) noexcept { return m ? a : b; }
    WRAY_INLINE double select(bool m, double a, double b) noexcept { return m ? a : b; }
    WRAY_INLINE std::int32_t as_bits(float a) noexcept { return std::bit_cast<std::int32_t>(a); }
    WRAY_INLINE std::int64_t as_bits(double a) noexcept { return std::bit_cast<std::int64_t>(a); }

    template<>
    WRAY_INLINE float from_bits<float, std::int32_t>(const std::int32_t &a) noexcept { return std::bit_cast<float>(a); }

    template<>
    WRAY_INLINE double from_bits<double, std::int64_t>(const std::int64_t &a) noexcept
    {
        return std::bit_cast<double>(a);
    }

    WRAY_INLINE std::int32_t shift_right_logical(std::int32_t a, int s) noexcept
    {
        return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) >> s);
    }

    WRAY_INLINE std::int64_t shift_right_logical(std::int64_t a, int s) noexcept
    {
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) >> s);
    }

    WRAY_INLINE std::int32_t shift_left(std::int32_t a, int s) noexcept
    {
        return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) << s);
    }

    WRAY_INLINE std::int64_t shift_left(std::int64_t a, int s) noexcept
    {
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) << s);
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> shift_left(const vec<T, N> &a, int s) noexcept { return a << s; }

    WRAY_INLINE bool nonzero(std::int32_t a) noexcept { return a != 0; }
    WRAY_INLINE bool nonzero(std::int64_t a) noexcept { return a != 0; }
    WRAY_INLINE bool both(bool a, bool b) noexcept { return a && b; }
    WRAY_INLINE bool either(bool a, bool b) noexcept { return a || b; }

    template<int B, int N>
    WRAY_INLINE mask<B, N> both(const mask<B, N> &a, const mask<B, N> &b) noexcept { return a & b; }

    template<int B, int N>
    WRAY_INLINE mask<B, N> either(const mask<B, N> &a, const mask<B, N> &b) noexcept { return a | b; }

    WRAY_INLINE bool any(bool a) noexcept { return a; }
    WRAY_INLINE bool all(bool a) noexcept { return a; }

    template<class V>
    struct traits;

    template<>
    struct traits<float>
    {
        using scalar = float;
        using integer = std::int32_t;
        using mask = bool;
        static constexpr int width = 1;
    };

    template<>
    struct traits<double>
    {
        using scalar = double;
        using integer = std::int64_t;
        using mask = bool;
        static constexpr int width = 1;
    };

    template<class T, int N>
    struct traits<vec<T, N> >
    {
        using scalar = T;
        using integer = int_for<T, N>;
        using mask = mask_for<T, N>;
        static constexpr int width = N;
    };

    template<class V>
    using scalar_t = typename traits<V>::scalar;

    template<class V>
    using integer_t = typename traits<V>::integer;

    template<class V>
    using mask_t = typename traits<V>::mask;

    template<class V>
    concept real = std::is_floating_point_v<scalar_t<V> >;

    template<class T>
    inline constexpr int native_width = native_bytes / static_cast<int>(sizeof(T));

    template<class T>
    using native = vec<T, native_width<T> >;
}

namespace wray::inline WRAY_ABI
{
    using simd::vec;
    using simd::f32x4;
    using simd::f64x2;
}

WRAY_FP_END
