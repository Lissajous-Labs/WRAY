#pragma once

#if defined(WRAY_SIMD_SCALAR)
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define WRAY_SIMD_NEON 1
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#define WRAY_SIMD_SSE2 1
#include <emmintrin.h>
#if defined(__FMA__)
#include <immintrin.h>
#endif
#else
#define WRAY_SIMD_SCALAR 1
#endif

#if defined(__aarch64__) || defined(__FMA__)
#define WRAY_HAS_FMA 1
#else
#define WRAY_HAS_FMA 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define WRAY_INLINE inline __attribute__((always_inline))
#else
#define WRAY_INLINE inline __forceinline
#endif
