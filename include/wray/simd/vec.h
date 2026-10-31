#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "config.h"

namespace wray::simd
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

    template<int Bits, int N>
    struct mask
    {
        bool v[N];
    };

    template<class T, int N>
    struct vec
    {
        using value_type = T;
        static constexpr int size = N;
        T v[N];

        vec() = default;

        WRAY_INLINE vec(T s) noexcept
        {
            for (int i = 0; i < N; ++i)
                v[i] = s;
        }

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
    };

    template<class T, int N>
    using mask_for = mask<bits_of<T>::width, N>;

    template<class T, int N>
    using int_for = vec<typename bits_of<T>::integer, N>;

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
        return zip(a, b, [](T x, T y) { return T(x + y); });
    }

    template<class T, int N>
    WRAY_INLINE vec<T, N> operator-(const vec<T, N> &a, const vec<T, N> &b) noexcept
    {
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
            r.v[i] = std::fma(a.v[i], b.v[i], c.v[i]);
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
    WRAY_INLINE vec<T, N> sqrt(const vec<T, N> &a) noexcept { return map(a, [](T x) { return std::sqrt(x); }); }

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

    using f32x4 = vec<float, 4>;
    using f64x2 = vec<double, 2>;
    using i32x4 = vec<std::int32_t, 4>;
    using i64x2 = vec<std::int64_t, 2>;
    using m32x4 = mask<32, 4>;
    using m64x2 = mask<64, 2>;

    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return vaddq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return vsubq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return vmulq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return vdivq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 operator-(const f32x4 &a) noexcept { return vnegq_f32(a.v); }
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept { return vfmaq_f32(c.v, a.v, b.v); }
    WRAY_INLINE f32x4 min(const f32x4 &a, const f32x4 &b) noexcept { return vminq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 max(const f32x4 &a, const f32x4 &b) noexcept { return vmaxq_f32(a.v, b.v); }
    WRAY_INLINE f32x4 abs(const f32x4 &a) noexcept { return vabsq_f32(a.v); }
    WRAY_INLINE f32x4 sqrt(const f32x4 &a) noexcept { return vsqrtq_f32(a.v); }
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

    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return vaddq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return vsubq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return vmulq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return vdivq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 operator-(const f64x2 &a) noexcept { return vnegq_f64(a.v); }
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept { return vfmaq_f64(c.v, a.v, b.v); }
    WRAY_INLINE f64x2 min(const f64x2 &a, const f64x2 &b) noexcept { return vminq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 max(const f64x2 &a, const f64x2 &b) noexcept { return vmaxq_f64(a.v, b.v); }
    WRAY_INLINE f64x2 abs(const f64x2 &a) noexcept { return vabsq_f64(a.v); }
    WRAY_INLINE f64x2 sqrt(const f64x2 &a) noexcept { return vsqrtq_f64(a.v); }
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
    WRAY_INLINE bool any(const m64x2 &a) noexcept { return (vgetq_lane_u64(a.v, 0) | vgetq_lane_u64(a.v, 1)) != 0; }
    WRAY_INLINE bool all(const m64x2 &a) noexcept { return (vgetq_lane_u64(a.v, 0) & vgetq_lane_u64(a.v, 1)) != 0; }

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

    using f32x4 = vec<float, 4>;
    using f64x2 = vec<double, 2>;
    using i32x4 = vec<std::int32_t, 4>;
    using i64x2 = vec<std::int64_t, 2>;
    using m32x4 = mask<32, 4>;
    using m64x2 = mask<64, 2>;

    WRAY_INLINE f32x4 operator+(const f32x4 &a, const f32x4 &b) noexcept { return _mm_add_ps(a.v, b.v); }
    WRAY_INLINE f32x4 operator-(const f32x4 &a, const f32x4 &b) noexcept { return _mm_sub_ps(a.v, b.v); }
    WRAY_INLINE f32x4 operator*(const f32x4 &a, const f32x4 &b) noexcept { return _mm_mul_ps(a.v, b.v); }
    WRAY_INLINE f32x4 operator/(const f32x4 &a, const f32x4 &b) noexcept { return _mm_div_ps(a.v, b.v); }
    WRAY_INLINE f32x4 operator-(const f32x4 &a) noexcept { return _mm_xor_ps(a.v, _mm_set1_ps(-0.0f)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept
    {
        return _mm_fmadd_ps(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f32x4 fma(const f32x4 &a, const f32x4 &b, const f32x4 &c) noexcept
    {
        return _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f32x4 min(const f32x4 &a, const f32x4 &b) noexcept { return _mm_min_ps(a.v, b.v); }
    WRAY_INLINE f32x4 max(const f32x4 &a, const f32x4 &b) noexcept { return _mm_max_ps(a.v, b.v); }
    WRAY_INLINE f32x4 abs(const f32x4 &a) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a.v); }
    WRAY_INLINE f32x4 sqrt(const f32x4 &a) noexcept { return _mm_sqrt_ps(a.v); }
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

    WRAY_INLINE f64x2 operator+(const f64x2 &a, const f64x2 &b) noexcept { return _mm_add_pd(a.v, b.v); }
    WRAY_INLINE f64x2 operator-(const f64x2 &a, const f64x2 &b) noexcept { return _mm_sub_pd(a.v, b.v); }
    WRAY_INLINE f64x2 operator*(const f64x2 &a, const f64x2 &b) noexcept { return _mm_mul_pd(a.v, b.v); }
    WRAY_INLINE f64x2 operator/(const f64x2 &a, const f64x2 &b) noexcept { return _mm_div_pd(a.v, b.v); }
    WRAY_INLINE f64x2 operator-(const f64x2 &a) noexcept { return _mm_xor_pd(a.v, _mm_set1_pd(-0.0)); }
#if WRAY_HAS_FMA
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept
    {
        return _mm_fmadd_pd(a.v, b.v, c.v);
    }
#else
    WRAY_INLINE f64x2 fma(const f64x2 &a, const f64x2 &b, const f64x2 &c) noexcept
    {
        return _mm_add_pd(_mm_mul_pd(a.v, b.v), c.v);
    }
#endif
    WRAY_INLINE f64x2 min(const f64x2 &a, const f64x2 &b) noexcept { return _mm_min_pd(a.v, b.v); }
    WRAY_INLINE f64x2 max(const f64x2 &a, const f64x2 &b) noexcept { return _mm_max_pd(a.v, b.v); }
    WRAY_INLINE f64x2 abs(const f64x2 &a) noexcept { return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v); }
    WRAY_INLINE f64x2 sqrt(const f64x2 &a) noexcept { return _mm_sqrt_pd(a.v); }
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

#endif

#if !defined(WRAY_SIMD_NEON) && !defined(WRAY_SIMD_SSE2)
    using f32x4 = vec<float, 4>;
    using f64x2 = vec<double, 2>;
    using i32x4 = vec<std::int32_t, 4>;
    using i64x2 = vec<std::int64_t, 2>;
    using m32x4 = mask<32, 4>;
    using m64x2 = mask<64, 2>;
#endif

    WRAY_INLINE float fma(float a, float b, float c) noexcept
    {
#if WRAY_HAS_FMA
        return std::fma(a, b, c);
#else
        float p = a * b;
        return p + c;
#endif
    }

    WRAY_INLINE double fma(double a, double b, double c) noexcept
    {
#if WRAY_HAS_FMA
        return std::fma(a, b, c);
#else
        double p = a * b;
        return p + c;
#endif
    }

    WRAY_INLINE float abs(float a) noexcept { return std::fabs(a); }
    WRAY_INLINE double abs(double a) noexcept { return std::fabs(a); }
    WRAY_INLINE float sqrt(float a) noexcept { return std::sqrt(a); }
    WRAY_INLINE double sqrt(double a) noexcept { return std::sqrt(a); }
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
    inline constexpr int native_width = 16 / static_cast<int>(sizeof(T));

    template<class T>
    using native = vec<T, native_width<T> >;
}

namespace wray
{
    using simd::vec;
    using simd::f32x4;
    using simd::f64x2;
}
