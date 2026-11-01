#pragma once

#include <cstdint>

#include "simd/config.h"

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define WRAY_FLUSH_AARCH64 1
#elif defined(__x86_64__) || defined(_M_X64) || defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define WRAY_FLUSH_X86 1
#include <xmmintrin.h>
#endif

WRAY_FP_BEGIN

namespace wray::inline WRAY_ABI
{
    class scoped_flush_denormals
    {
    public:
#if defined(WRAY_FLUSH_AARCH64)
        using state = std::uint64_t;
        static constexpr state flags = state(1) << 24;

        static state read() noexcept
        {
            state v = 0;
            asm volatile("mrs %0, fpcr" : "=r"(v));
            return v;
        }

        static void write(state v) noexcept { asm volatile("msr fpcr, %0" : : "r"(v)); }
#elif defined(WRAY_FLUSH_X86)
        using state = unsigned int;
        static constexpr state flags = 0x8040u;

        static state read() noexcept { return _mm_getcsr(); }

        static void write(state v) noexcept { _mm_setcsr(v); }
#else
        using state = unsigned int;
        static constexpr state flags = 0;

        static state read() noexcept { return 0; }

        static void write(state) noexcept {}
#endif

        scoped_flush_denormals() noexcept : saved_(read()) { write(saved_ | flags); }

        ~scoped_flush_denormals() { write(saved_); }

        scoped_flush_denormals(const scoped_flush_denormals &) = delete;
        scoped_flush_denormals &operator=(const scoped_flush_denormals &) = delete;

    private:
        state saved_;
    };
}

WRAY_FP_END
