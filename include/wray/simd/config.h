#pragma once

#include <cfloat>

#if !defined(WRAY_ALLOW_FAST_MATH)
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "wray relies on IEEE rounding for compensated arithmetic; fast floating-point models break it. Define WRAY_ALLOW_FAST_MATH to build anyway."
#elif defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "wray handles infinities and NaN explicitly; -ffinite-math-only removes those checks. Define WRAY_ALLOW_FAST_MATH to build anyway."
#elif defined(__ASSOCIATIVE_MATH__) || defined(__RECIPROCAL_MATH__) || defined(__NO_SIGNED_ZEROS__)
#error "wray relies on unreassociated IEEE arithmetic and signed zeros; -fassociative-math, -freciprocal-math and -fno-signed-zeros break it. Define WRAY_ALLOW_FAST_MATH to build anyway."
#endif
#endif

#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0 && !defined(WRAY_ALLOW_FAST_MATH)
#error "wray needs float and double arithmetic evaluated in their own precision (FLT_EVAL_METHOD 0); on 32-bit x86 build with -msse2 -mfpmath=sse."
#endif

#if defined(WRAY_SIMD_SCALAR)
#elif (defined(__ARM_NEON) && defined(__aarch64__)) || defined(_M_ARM64)
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

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define WRAY_HAS_FMA 1
#else
#define WRAY_HAS_FMA 0
#endif

#if defined(WRAY_SIMD_AVX512)
#define WRAY_ABI_BACKEND avx512
#elif defined(WRAY_SIMD_AVX2)
#define WRAY_ABI_BACKEND avx2
#elif defined(WRAY_SIMD_SSE2)
#define WRAY_ABI_BACKEND sse2
#elif defined(WRAY_SIMD_NEON)
#define WRAY_ABI_BACKEND neon
#else
#define WRAY_ABI_BACKEND scalar
#endif
#define WRAY_ABI_JOIN2(a, b) abi_##a##_##b
#define WRAY_ABI_JOIN(a, b) WRAY_ABI_JOIN2(a, b)
#if WRAY_HAS_FMA
#define WRAY_ABI WRAY_ABI_JOIN(WRAY_ABI_BACKEND, fma)
#else
#define WRAY_ABI WRAY_ABI_JOIN(WRAY_ABI_BACKEND, nofma)
#endif

#if defined(__clang__)
#define WRAY_FP_BEGIN _Pragma("float_control(push)") _Pragma("float_control(precise, on)") _Pragma("clang fp contract(off) reassociate(off)")
#define WRAY_FP_END _Pragma("float_control(pop)")
#elif defined(_MSC_VER) && defined(_M_FP_CONTRACT)
#define WRAY_FP_BEGIN __pragma(fp_contract(off))
#define WRAY_FP_END __pragma(fp_contract(on))
#else
#define WRAY_FP_BEGIN
#define WRAY_FP_END
#endif

#if defined(__clang__)
#if __has_builtin(__builtin_elementwise_fma)
#define WRAY_FMA(a, b, c) __builtin_elementwise_fma(a, b, c)
#endif
#if __has_builtin(__builtin_elementwise_sqrt)
#define WRAY_SQRT(a) __builtin_elementwise_sqrt(a)
#endif
#endif
#ifndef WRAY_FMA
#define WRAY_FMA(a, b, c) std::fma(a, b, c)
#endif
#ifndef WRAY_SQRT
#define WRAY_SQRT(a) std::sqrt(a)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define WRAY_INLINE inline __attribute__((always_inline))
#define WRAY_VOP(intrinsic, op, a, b) ((a) op (b))
#else
#define WRAY_INLINE inline __forceinline
#define WRAY_VOP(intrinsic, op, a, b) intrinsic(a, b)
#endif
