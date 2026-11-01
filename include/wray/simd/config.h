#pragma once

#if defined(__FAST_MATH__) && !defined(WRAY_ALLOW_FAST_MATH)
#error "wray relies on IEEE rounding for compensated arithmetic; -ffast-math breaks it. Define WRAY_ALLOW_FAST_MATH to build anyway."
#endif

#if defined(WRAY_SIMD_SCALAR)
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define WRAY_SIMD_NEON 1
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#define WRAY_SIMD_SSE2 1
#include <emmintrin.h>
#if defined(__AVX2__)
#define WRAY_SIMD_AVX2 1
#endif
#if defined(__AVX512F__)
#define WRAY_SIMD_AVX512 1
#endif
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#if defined(__FMA__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#else
#define WRAY_SIMD_SCALAR 1
#endif

#if defined(__aarch64__) || defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define WRAY_HAS_FMA 1
#else
#define WRAY_HAS_FMA 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define WRAY_INLINE inline __attribute__((always_inline))
#else
#define WRAY_INLINE inline __forceinline
#endif
