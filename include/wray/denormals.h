#pragma once

#include <cstdint>

#include "simd/config.h"

#if defined(WRAY_SIMD_SSE2)
#include <xmmintrin.h>
#endif

namespace wray
{
    class scoped_flush_denormals
    {
    public:
#if defined(WRAY_SIMD_NEON)
        using state = std::uint64_t;
        static constexpr state flags = state(1) << 24;

        static state read() noexcept
        {
            state v = 0;
            asm volatile("mrs %0, fpcr" : "=r"(v));
            return v;
        }

        static void write(state v) noexcept { asm volatile("msr fpcr, %0" : : "r"(v)); }
#elif defined(WRAY_SIMD_SSE2)
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
