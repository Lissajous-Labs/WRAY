#include <wray/wray.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <vector>

#include "check.h"

namespace
{
    template <class T>
    bool same(T got, T want)
    {
        if (std::isnan(want))
            return std::isnan(got);
        return got == want && std::signbit(got) == std::signbit(want);
    }

    template <class T>
    void unary_specials()
    {
        using V = wray::simd::native<T>;
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T huge = std::numeric_limits<T>::max();
        struct row
        {
            T (*fn)(T);
            V (*vfn)(V);
            T x;
            T want;
        };
        const row rows[] = {
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, T(0), T(1)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -T(0), T(1)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, inf, inf},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -inf, T(0)},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, nan, nan},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, huge, inf},
            {[](T x) { return wray::exp(x); }, [](V x) { return wray::exp(x); }, -huge, T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(0), T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -T(0), -T(0)},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, tiny, tiny},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -tiny, -tiny},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(-1), -inf},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, T(-2), nan},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, inf, inf},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, -inf, nan},
            {[](T x) { return wray::log1p(x); }, [](V x) { return wray::log1p(x); }, nan, nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(1), T(0)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(10), T(1)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(100), T(2)},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(0), -inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, -T(0), -inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, T(-1), nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, inf, inf},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, -inf, nan},
            {[](T x) { return wray::log10(x); }, [](V x) { return wray::log10(x); }, nan, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, -T(0), -T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.25), T(1)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.25), T(-1)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.5), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.5), -T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(1), T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(0.125), T(0.70710678118654752440)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, T(-0.375), -T(0.70710678118654752440)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, inf, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, nan, nan},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, huge, T(0)},
            {[](T x) { return wray::sin_turns(x); }, [](V x) { return wray::sin_turns(x); }, -huge, -T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, -T(0), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.25), T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(-0.25), T(0)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.5), T(-1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(1), T(1)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.125), T(0.70710678118654752440)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, T(0.375), -T(0.70710678118654752440)},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, inf, nan},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, nan, nan},
            {[](T x) { return wray::cos_turns(x); }, [](V x) { return wray::cos_turns(x); }, huge, T(1)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(0), T(0)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(0.25), T(1)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(-0.25), T(-1)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(0.5), inf},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(-0.5), -inf},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(1), T(0)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(-1), -T(0)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, T(0.75), T(-1)},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, inf, nan},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, nan, nan},
            {[](T x) { return wray::tanpi(x); }, [](V x) { return wray::tanpi(x); }, huge, T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0), T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.125), T(1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-0.125), T(-1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.25), inf},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-0.25), -inf},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.5), T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(-0.5), -T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, T(0.375), T(-1)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, inf, nan},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, nan, nan},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, huge, T(0)},
            {[](T x) { return wray::tan_turns(x); }, [](V x) { return wray::tan_turns(x); }, -huge, -T(0)},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, T(0), T(0)},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, -T(0), -T(0)},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, tiny, tiny},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, -tiny, -tiny},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, inf, inf},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, -inf, -inf},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, huge, inf},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, -huge, -inf},
            {[](T x) { return wray::sinh(x); }, [](V x) { return wray::sinh(x); }, nan, nan},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, T(0), T(1)},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, -T(0), T(1)},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, tiny, T(1)},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, inf, inf},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, -inf, inf},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, huge, inf},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, -huge, inf},
            {[](T x) { return wray::cosh(x); }, [](V x) { return wray::cosh(x); }, nan, nan},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, T(0), T(0)},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, -T(0), -T(0)},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, tiny, tiny},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, -tiny, -tiny},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, inf, inf},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, -inf, -inf},
            {[](T x) { return wray::asinh(x); }, [](V x) { return wray::asinh(x); }, nan, nan},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(0), T(0)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, -T(0), -T(0)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(8), T(2)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(-27), T(-3)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(0.125), T(0.5)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(1000), T(10)},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, inf, inf},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, -inf, -inf},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, nan, nan},
            {[](T x) { return wray::cbrt(x); }, [](V x) { return wray::cbrt(x); }, T(1), T(1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, T(0), T(0)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, -T(0), -T(0)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, inf, T(1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, -inf, T(-1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, T(10), T(1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, T(-10), T(-1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, huge, T(1)},
            {[](T x) { return wray::erf(x); }, [](V x) { return wray::erf(x); }, nan, nan},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, T(69), T(440)},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, T(81), T(880)},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, T(57), T(220)},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, T(21), T(27.5)},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, inf, inf},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, -inf, T(0)},
            {[](T x) { return wray::midi_to_hz(x); }, [](V x) { return wray::midi_to_hz(x); }, nan, nan},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(440), T(69)},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(880), T(81)},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(220), T(57)},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(27.5), T(21)},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(0), -inf},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, T(-1), nan},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, inf, inf},
            {[](T x) { return wray::hz_to_midi(x); }, [](V x) { return wray::hz_to_midi(x); }, nan, nan},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, T(0), T(1)},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, T(1200), T(2)},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, T(-2400), T(0.25)},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, T(3600), T(8)},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, inf, inf},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, -inf, T(0)},
            {[](T x) { return wray::ratio_from_cents(x); }, [](V x) { return wray::ratio_from_cents(x); }, nan, nan},
            {[](T x) { return wray::omega(x); }, [](V x) { return wray::omega(x); }, T(1), T(1)},
            {[](T x) { return wray::omega(x); }, [](V x) { return wray::omega(x); }, -inf, T(0)},
            {[](T x) { return wray::omega(x); }, [](V x) { return wray::omega(x); }, inf, inf},
            {[](T x) { return wray::omega(x); }, [](V x) { return wray::omega(x); }, nan, nan},
            {[](T x) { return wray::product_log(x); }, [](V x) { return wray::product_log(x); }, T(0), T(0)},
            {[](T x) { return wray::product_log(x); }, [](V x) { return wray::product_log(x); }, -T(0), -T(0)},
            {[](T x) { return wray::product_log(x); }, [](V x) { return wray::product_log(x); }, inf, inf},
            {[](T x) { return wray::product_log(x); }, [](V x) { return wray::product_log(x); }, T(-1), nan},
            {[](T x) { return wray::product_log(x); }, [](V x) { return wray::product_log(x); }, nan, nan},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, T(0), T(0)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -T(0), -T(0)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, tiny, tiny},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -tiny, -tiny},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, inf, inf},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -inf, T(-1)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, huge, inf},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, -huge, T(-1)},
            {[](T x) { return wray::expm1(x); }, [](V x) { return wray::expm1(x); }, nan, nan},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, T(10), T(1024)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, T(-1), T(0.5)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, inf, inf},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, -inf, T(0)},
            {[](T x) { return wray::exp2(x); }, [](V x) { return wray::exp2(x); }, nan, nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(1), T(0)},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(0), -inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, -T(0), -inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, T(-1), nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, inf, inf},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, -inf, nan},
            {[](T x) { return wray::log(x); }, [](V x) { return wray::log(x); }, nan, nan},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(8), T(3)},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(0.25), T(-2)},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, tiny, T(std::ilogb(tiny))},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, T(0), -inf},
            {[](T x) { return wray::log2(x); }, [](V x) { return wray::log2(x); }, inf, inf},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, T(0), T(0)},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, -T(0), -T(0)},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, inf, nan},
            {[](T x) { return wray::sin(x); }, [](V x) { return wray::sin(x); }, nan, nan},
            {[](T x) { return wray::cos(x); }, [](V x) { return wray::cos(x); }, T(0), T(1)},
            {[](T x) { return wray::cos(x); }, [](V x) { return wray::cos(x); }, -inf, nan},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, T(0), T(0)},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tan(x); }, [](V x) { return wray::tan(x); }, nan, nan},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, T(0), T(0)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, -T(0), -T(0)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, inf, T(1.5707963267948966)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, -inf, T(-1.5707963267948966)},
            {[](T x) { return wray::atan(x); }, [](V x) { return wray::atan(x); }, nan, nan},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, T(0), T(0)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, -T(0), -T(0)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, inf, T(1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, -inf, T(-1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, T(50), T(1)},
            {[](T x) { return wray::tanh(x); }, [](V x) { return wray::tanh(x); }, nan, nan},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, T(0), T(0.5)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, inf, T(1)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, -inf, T(0)},
            {[](T x) { return wray::sigmoid(x); }, [](V x) { return wray::sigmoid(x); }, nan, nan},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, T(0), T(1)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, T(20), T(10)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, -inf, T(0)},
            {[](T x) { return wray::db_to_gain(x); }, [](V x) { return wray::db_to_gain(x); }, inf, inf},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(1), T(0)},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(0), -inf},
            {[](T x) { return wray::gain_to_db(x); }, [](V x) { return wray::gain_to_db(x); }, T(-1), nan},
        };
        for (const auto& r : rows)
        {
            CHECK(same(r.fn(r.x), r.want));
            const V v = r.vfn(V(r.x));
            for (int i = 0; i < V::size; ++i)
                CHECK(same(v[i], r.want));
        }
    }
    template <class T>
    void balanced_specials()
    {
        using V = wray::simd::native<T>;
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        for (const T x : {T(0), -T(0), std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(),
                          std::numeric_limits<T>::min(), -std::numeric_limits<T>::min(), inf, -inf, nan,
                          std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()})
        {
            const T expected = std::isnan(x) ? nan : std::fabs(x) > T(20) ? std::copysign(T(1), x) : x;
            CHECK(same(wray::balanced::tanh(x), expected));
            const V y = wray::balanced::tanh(V(x));
            for (int i = 0; i < V::size; ++i)
                CHECK(same(y[i], expected));
        }
        const T transition(0x1.1542e8p+3);
        const T before = std::nextafter(transition, T(0));
        CHECK(wray::balanced::tanh(before) <= wray::balanced::tanh(transition));
        CHECK(wray::balanced::tanh(-transition) <= wray::balanced::tanh(-before));
        for (const T x : {T(0), -T(0)})
        {
            const auto sc = wray::balanced::sincos(x);
            const auto vc = wray::balanced::sincos(V(x));
            CHECK(same(wray::balanced::sin(x), x));
            CHECK(same(wray::balanced::cos(x), T(1)));
            CHECK(same(sc.sin, x) && same(sc.cos, T(1)));
            for (int i = 0; i < V::size; ++i)
            {
                CHECK(same(wray::balanced::sin(V(x))[i], x));
                CHECK(same(wray::balanced::cos(V(x))[i], T(1)));
                CHECK(same(vc.sin[i], x) && same(vc.cos[i], T(1)));
            }
        }
        for (const T x : {inf, -inf, nan})
        {
            const auto sc = wray::balanced::sincos(x);
            const auto vc = wray::balanced::sincos(V(x));
            CHECK(std::isnan(wray::balanced::sin(x)) && std::isnan(wray::balanced::cos(x)));
            CHECK(std::isnan(sc.sin) && std::isnan(sc.cos));
            for (int i = 0; i < V::size; ++i)
            {
                CHECK(std::isnan(wray::balanced::sin(V(x))[i]) && std::isnan(wray::balanced::cos(V(x))[i]));
                CHECK(std::isnan(vc.sin[i]) && std::isnan(vc.cos[i]));
            }
        }
        const int saved_rounding = std::fegetround();
        struct power_case { T x; T y; T expected; };
        const power_case powers[] = {
            {nan, T(0), T(1)}, {T(1), nan, T(1)}, {T(-1), inf, T(1)}, {T(-1), -inf, T(1)},
            {T(-1), nan, nan}, {T(-3), T(0.5), nan}, {nan, T(2), nan},
            {-T(0), T(3), -T(0)}, {-T(0), T(2), T(0)}, {-T(0), T(-3), -inf}, {-T(0), T(-2), inf},
            {-inf, T(3), -inf}, {-inf, T(2), inf}, {-inf, T(-3), -T(0)}, {-inf, T(-2), T(0)},
            {-inf, T(0.5), inf}, {inf, T(-2), T(0)}, {T(0.5), inf, T(0)}, {T(2), -inf, T(0)},
            {T(2), inf, inf}, {T(0.5), -inf, inf},
            {T(1), std::numeric_limits<T>::max(), T(1)}, {T(1), -std::numeric_limits<T>::max(), T(1)}
        };
        struct boundary_case { T x; T y; long double normalized; };
        const auto boundaries = [] {
            if constexpr (std::is_same_v<T, float>)
                return std::array<boundary_case, 3>{{
                    {T(0x1.6a09e6p+0), T(0x1p+8), 1.99999123751235004176550776295623487007840394559754L},
                    {std::numeric_limits<T>::max(), T(1), 0x1.fffffep0L},
                    {std::nextafter(std::numeric_limits<T>::max(), T(0)), T(1), 0x1.fffffcp0L}
                }};
            else
                return std::array<boundary_case, 5>{{
                    {T(0x1.6a09e667f3bcdp+0), T(0x1.ffffffffffffep+10), 1.99999999999996478787848377585267927892907663667362L},
                    {T(0x1.6a09e667f3bccp+0), T(0x1.0000000000001p+11), 1.99999999999995209169567501532784875406877473735936L},
                    {T(0x1.6a09e667f3bcep+0), T(0x1.ffffffffffffap+10), 1.99999999999997748406129253578593582789122228331441L},
                    {std::numeric_limits<T>::max(), T(1), 0x1.fffffffffffffp0L},
                    {std::nextafter(std::numeric_limits<T>::max(), T(0)), T(1), 0x1.ffffffffffffep0L}
                }};
        }();
        for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
        {
            CHECK(std::fesetround(mode) == 0);
            volatile T infinity = inf;
            CHECK(same(wray::balanced::tanh(T(infinity)), T(1)));
            CHECK(same(wray::balanced::tanh(-T(infinity)), T(-1)));
            for (const auto& p : powers)
            {
                CHECK(same(wray::balanced::pow(p.x, p.y), p.expected));
                const V got = wray::balanced::pow(V(p.x), V(p.y));
                for (int i = 0; i < V::size; ++i)
                    CHECK(same(got[i], p.expected));
            }
            const long double bound = (mode == FE_TONEAREST ? 4.0L : 8.0L) + (WRAY_HAS_FMA ? 0.0L : 0.5L);
            for (const auto& p : boundaries)
            {
                const T got = wray::balanced::pow(p.x, p.y);
                CHECK(std::isfinite(got));
                const long double scale = std::fmax(1.0L, std::fabs(static_cast<long double>(p.y) * std::log2(static_cast<long double>(p.x))) / 4.0L);
                const long double normalized = std::ldexp(static_cast<long double>(got), 1 - std::numeric_limits<T>::max_exponent);
                const long double ulp = std::ldexp(1.0L, 1 - std::numeric_limits<T>::digits);
                CHECK(std::fabs(normalized - p.normalized) / ulp / scale <= bound);
                for (int kind = 0; kind < 3; ++kind)
                    for (int lane = 0; lane < V::size; ++lane)
                    {
                        T bases[V::size], exponents[V::size];
                        for (int i = 0; i < V::size; ++i)
                        {
                            const int ambient = (i + kind) % 3;
                            bases[i] = i == lane ? p.x : ambient == 0 ? T(1.75) : ambient == 1 ? inf : nan;
                            exponents[i] = i == lane ? p.y : ambient == 0 ? T(0.3) : ambient == 1 ? T(-1) : T(2);
                        }
                        const V result = wray::balanced::pow(V::load(bases), V::load(exponents));
                        for (int i = 0; i < V::size; ++i)
                            CHECK(same(result[i], i == lane ? got : wray::balanced::pow(bases[i], exponents[i])));
                    }
            }
        }
        CHECK(std::fesetround(saved_rounding) == 0);
    }


    template <class T, class V>
    void huge_trig()
    {
        struct row
        {
            T x;
            long double sin;
            long double cos;
            long double tan;
        };
        const row rows[] = {
            {std::numeric_limits<T>::max(), std::is_same_v<T, float> ? -0.5218765233336585405515054L : 0.004961954789184061790502671L,
             std::is_same_v<T, float> ? 0.8530210398303041580517915L : -0.9999876894265599374648701L, std::is_same_v<T, float> ? -0.611797949834248061162496L : -0.004962015874444894900500884L},
            {T(-2e7), 0.7631011174721591478089145L, 0.6462791072847256792926895L, 1.180760926464494528273596L},
            {T(0x1p+51), 0.8618395638813005325086728715L, 0.507180999376839528902991788L, 1.699274154473888050091764584L},
            {T(0x1p+52), 0.8742173026236350734799397452L, -0.4855348677422206027891489425L, -1.800524248008843576595986789L},
            {T(0x1p+53), -0.8489259648146549995612720697L, -0.5285117844130886942586601103L, 1.606257400215561186925050406L},
            {T(0x1p+85), -0.9993765291704066514261216744L, -0.03530655666177804047937499842L, 28.30569230367077050196437067L},
            {T(0x1p+86), 0.07056908810721207431363901327L, -0.9975068941133773133410783694L, -0.07074546404006220699834971604L},
            {T(0x1p+87), -0.1407863037964767777248415688L, 0.9900400076074330787494042177L, -0.1422026410192312457573581842L},
        };
        const auto close = [](T got, long double want, long double tolerance) {
            const long double ulp = std::ldexp(1.0L, std::ilogb(static_cast<T>(want)) - std::numeric_limits<T>::digits + 1);
            return std::isfinite(got) && std::fabs(static_cast<long double>(got) - want) <= tolerance * ulp;
        };
        for (const auto& r : rows)
        {
            for (const T x : {r.x, -r.x})
            {
                const long double sign = (x < 0) == (r.x < 0) ? 1.0L : -1.0L;
                const long double s = sign * r.sin;
                const long double t = sign * r.tan;
                CHECK(close(wray::sin(x), s, 1));
                CHECK(close(wray::cos(x), r.cos, 1));
                CHECK(close(wray::tan(x), t, 1));
                const auto sc = wray::sincos(x);
                CHECK(close(sc.sin, s, 1));
                CHECK(close(sc.cos, r.cos, 1));
                CHECK(close(wray::sin(V(x))[V::size - 1], s, 1));
                constexpr long double balanced_bound = WRAY_HAS_FMA ? 2.5L : 3.0L;
                const auto bsc = wray::balanced::sincos(x);
                const auto bvc = wray::balanced::sincos(V(x));
                CHECK(close(wray::balanced::sin(x), s, balanced_bound));
                CHECK(close(wray::balanced::cos(x), r.cos, balanced_bound));
                CHECK(close(bsc.sin, s, balanced_bound) && close(bsc.cos, r.cos, balanced_bound));
                CHECK(close(wray::balanced::sin(V(x))[V::size - 1], s, balanced_bound));
                CHECK(close(wray::balanced::cos(V(x))[V::size - 1], r.cos, balanced_bound));
                CHECK(close(bvc.sin[V::size - 1], s, balanced_bound) && close(bvc.cos[V::size - 1], r.cos, balanced_bound));
                if constexpr (std::is_same_v<T, float>)
                {
                    CHECK(close(wray::fast::sin(x), s, 128));
                    CHECK(close(wray::fast::cos(x), r.cos, 128));
                    CHECK(close(wray::fast::tan(x), t, 128));
                }
            }
        }
    }

    template <class T>
    void binary_specials()
    {
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T pi = T(3.141592653589793);
        const T half_pi = T(1.5707963267948966);
        CHECK(same(wray::atan2(T(0), T(1)), T(0)));
        CHECK(same(wray::atan2(-T(0), T(1)), -T(0)));
        CHECK(same(wray::atan2(T(0), -T(0)), pi));
        CHECK(same(wray::atan2(-T(0), -T(0)), -pi));
        CHECK(same(wray::atan2(T(0), T(0)), T(0)));
        CHECK(same(wray::atan2(T(1), T(0)), half_pi));
        CHECK(same(wray::atan2(T(-1), T(0)), -half_pi));
        CHECK(same(wray::atan2(inf, inf), wray::atan(T(1))));
        CHECK(same(wray::atan2(T(1), -inf), pi));
        CHECK(same(wray::atan2(nan, T(1)), nan));
        CHECK(same(wray::atan2(T(1), nan), nan));
        CHECK(same(wray::prewarp(T(12000), T(48000)), T(1)));
        CHECK(same(wray::prewarp(T(0), T(48000)), T(0)));
        CHECK(same(wray::pole_from_cutoff(T(0), T(48000)), T(1)));
        CHECK(same(wray::smoothing_coefficient(inf, T(48000)), T(0)));
        CHECK(same(wray::prewarp(nan, T(48000)), nan));
        CHECK(same(wray::pow(T(2), T(10)), T(1024)));
        CHECK(same(wray::pow(T(9), T(0.5)), T(3)));
        CHECK(same(wray::pow(T(0), T(2)), T(0)));
        CHECK(same(wray::pow(T(0), T(-2)), inf));
        CHECK(same(wray::pow(nan, T(0)), T(1)));
        CHECK(same(wray::pow(T(1), nan), T(1)));
        CHECK(same(wray::pow(T(-8), T(1.0 / 3.0)), nan));
        CHECK(same(wray::smoothing_coefficient(T(0), T(48000)), T(1)));
        CHECK(same(wray::smoothing_coefficient(std::numeric_limits<T>::denorm_min(), T(48000)), T(1)));
        CHECK(same(wray::smoothing_coefficient(std::numeric_limits<T>::min(), T(48000)), T(1)));
        CHECK(same(wray::smoothing_coefficient(inf, T(48000)), T(0)));
        CHECK(same(wray::smoothing_coefficient(std::numeric_limits<T>::max(), T(48000)), (T(1) / T(48000)) / std::numeric_limits<T>::max()));
        CHECK(same(wray::smoothing_coefficient(std::numeric_limits<T>::max() / T(4), T(8)), (T(1) / T(8)) / (std::numeric_limits<T>::max() / T(4))));
        CHECK(same(wray::pow(T(-2), T(3)), T(-8)));
        CHECK(same(wray::pow(T(-2), T(2)), T(4)));
        CHECK(same(wray::pow(T(-2), T(-1)), T(-0.5)));
        CHECK(same(wray::pow(T(-1), inf), T(1)));
        CHECK(same(wray::pow(T(-1), nan), nan));
        CHECK(same(wray::pow(-T(0), T(3)), -T(0)));
        CHECK(same(wray::pow(-T(0), T(-3)), -inf));
        CHECK(same(wray::pow(-T(0), T(2)), T(0)));
        CHECK(same(wray::pow(-T(0), T(0.5)), T(0)));
        CHECK(same(wray::pow(-inf, T(3)), -inf));
        CHECK(same(wray::pow(-inf, T(-3)), -T(0)));
        CHECK(same(wray::pow(-inf, T(2)), inf));
        CHECK(same(wray::pow(-inf, T(0.5)), inf));
        CHECK(same(wray::pow(T(-3), T(0.5)), nan));
        CHECK(same(wray::fast::pow(-2.0f, 3.0f), -8.0f) || std::fabs(wray::fast::pow(-2.0f, 3.0f) + 8.0f) < 1e-3f);
        CHECK(same(wray::pow(inf, T(2)), inf));
        CHECK(same(wray::pow(inf, T(-2)), T(0)));
        CHECK(same(wray::pow(T(0.5), inf), T(0)));
        CHECK(same(wray::pow(T(2), inf), inf));
        CHECK(same(wray::pow(T(10), T(400)), inf));
        using V = wray::simd::native<T>;
        const T huge = std::numeric_limits<T>::max();
        for (const T y : {huge, -huge, T(1e30), T(-1e30)})
        {
            CHECK(same(wray::pow(T(1), y), T(1)));
            const V v = wray::pow(V(T(1)), V(y));
            for (int i = 0; i < V::size; ++i)
                CHECK(same(v[i], T(1)));
        }
        CHECK(same(wray::pow(T(2), huge), inf));
        CHECK(same(wray::pow(T(0.5), huge), T(0)));
        for (int lane = 0; lane < V::size; ++lane)
        {
            for (int kind = 0; kind < 4; ++kind)
            {
                T bases[V::size], exponents[V::size];
                for (int i = 0; i < V::size; ++i)
                {
                    bases[i] = T(2);
                    exponents[i] = T(2);
                }
                const T base[] = {T(-2), -T(0), T(-3), -inf};
                const T exponent[] = {T(3), T(3), T(0.5), T(-3)};
                const T expected[] = {T(-8), -T(0), nan, -T(0)};
                bases[lane] = base[kind];
                exponents[lane] = exponent[kind];
                const V result = wray::pow(V::load(bases), V::load(exponents));
                for (int i = 0; i < V::size; ++i)
                    CHECK(same(result[i], i == lane ? expected[kind] : T(4)));
                if constexpr (std::is_same_v<T, float>)
                {
                    const V fast = wray::fast::pow(V::load(bases), V::load(exponents));
                    const T target = kind == 1 || kind == 3 ? -std::numeric_limits<T>::min() : expected[kind];
                    for (int i = 0; i < V::size; ++i)
                    {
                        const T want = i == lane ? target : T(4);
                        CHECK(std::isnan(want) ? std::isnan(fast[i]) :
                              std::signbit(fast[i]) == std::signbit(want) && std::abs(fast[i] - want) <= std::abs(want) * T(0x1p-16));
                    }
                }
            }
        }
    }

    template <class T>
    void octaves()
    {
        for (int k = -12; k <= 12; ++k)
        {
            const T p = std::ldexp(T(1), k);
            CHECK(same(wray::midi_to_hz(T(69 + 12 * k)), T(440) * p));
            CHECK(same(wray::hz_to_midi(T(440) * p), T(69 + 12 * k)));
            CHECK(same(wray::ratio_from_cents(T(1200 * k)), p));
        }
    }

    void ranges()
    {
        std::vector<float> in(37);
        std::vector<float> out(37);
        std::vector<double> ind(19);
        std::vector<double> outd(19);
        for (std::size_t i = 0; i < in.size(); ++i)
            in[i] = float(i) * 0.25f - 4.0f;
        for (std::size_t i = 0; i < ind.size(); ++i)
            ind[i] = double(i) * 0.5 - 3.0;
        wray::exp(in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::exp(in[i]));
        wray::fast::tanh(in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::fast::tanh(in[i]));
        wray::sin(ind, outd);
        for (std::size_t i = 0; i < ind.size(); ++i)
            CHECK(outd[i] == wray::sin(ind[i]));
        wray::atan2(in, in, out);
        for (std::size_t i = 0; i < in.size(); ++i)
            CHECK(out[i] == wray::atan2(in[i], in[i]));
        const auto sc = wray::sincos(0.5f);
        CHECK(sc.sin == wray::sin(0.5f));
        CHECK(sc.cos == wray::cos(0.5f));
    }

    struct fast_unary
    {
        const char *name;
        float (*fn)(float);
        wray::simd::native<float> (*vfn)(wray::simd::native<float>);
    };

#define WRAY_FAST_ROW(f) fast_unary{#f, [](float x) { return wray::fast::f(x); }, [](wray::simd::native<float> x) { return wray::fast::f(x); }}

    void fast_specials()
    {
        using V = wray::simd::native<float>;
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float tiny = std::numeric_limits<float>::denorm_min();
        const float small = std::numeric_limits<float>::min();
        const float huge = std::numeric_limits<float>::max();
        const fast_unary all[] = {WRAY_FAST_ROW(exp), WRAY_FAST_ROW(exp2), WRAY_FAST_ROW(log), WRAY_FAST_ROW(log2), WRAY_FAST_ROW(sin_turns),
                                  WRAY_FAST_ROW(sin), WRAY_FAST_ROW(cos_turns), WRAY_FAST_ROW(cos), WRAY_FAST_ROW(tanpi), WRAY_FAST_ROW(tan_turns),
                                  WRAY_FAST_ROW(tan), WRAY_FAST_ROW(atan), WRAY_FAST_ROW(omega), WRAY_FAST_ROW(product_log), WRAY_FAST_ROW(tanh),
                                  WRAY_FAST_ROW(sigmoid), WRAY_FAST_ROW(db_to_gain), WRAY_FAST_ROW(gain_to_db)};
        const float inputs[] = {nan, -nan, inf, -inf, 0.0f, -0.0f, tiny, -tiny, small, -small, 1.0f, -1.0f, huge, -huge};
        for (const auto &f: all)
        {
            for (const float x: inputs)
            {
                const float got = f.fn(x);
                const V v = f.vfn(V(x));
                for (int i = 0; i < V::size; ++i)
                    CHECK(same(v[i], got));
                if (std::isnan(x) && !std::isnan(got))
                    std::printf("fast %s(nan) = %g\n", f.name, static_cast<double>(got));
                CHECK(!std::isnan(x) || std::isnan(got));
            }
        }
        for (const auto &f: {all[2], all[3], all[17]})
        {
            for (const float x: {-tiny, -small, -1.0f, -huge, -inf})
                CHECK(std::isnan(f.fn(x)));
            CHECK(same(f.fn(-0.0f), f.fn(0.0f)));
            CHECK(std::isfinite(f.fn(0.0f)) && f.fn(0.0f) <= f.fn(small));
            CHECK(std::isfinite(f.fn(inf)) && f.fn(inf) >= f.fn(huge));
        }
        for (const auto &f: {all[0], all[1], all[16]})
        {
            for (const float x: {inf, -inf, huge, -huge, tiny, -tiny})
            {
                const float y = f.fn(x);
                CHECK(std::isfinite(y) && y >= small);
            }
        }
        for (const float x: {88.0f, 100.0f, huge, inf})
            CHECK(same(wray::fast::exp(x), wray::fast::exp(88.0f)));
        for (const float x: {-87.0f, -100.0f, -huge, -inf})
            CHECK(same(wray::fast::exp(x), wray::fast::exp(-87.0f)));
        for (const float x: {127.0f, 200.0f, inf})
            CHECK(same(wray::fast::exp2(x), wray::fast::exp2(127.0f)));
        for (const float x: {-126.0f, -200.0f, -inf})
            CHECK(same(wray::fast::exp2(x), wray::fast::exp2(-126.0f)));
        CHECK(same(wray::fast::pow(2.0f, 300.0f), wray::fast::exp2(127.0f)));
        CHECK(same(wray::fast::pow(2.0f, -300.0f), wray::fast::exp2(-126.0f)));
        for (const float x: {5.9f, 6.0f, 100.0f, huge, inf})
        {
            CHECK(same(wray::fast::tanh(x), 1.0f));
            CHECK(same(wray::fast::tanh(-x), -1.0f));
        }
        CHECK(wray::fast::tanh(std::nextafter(5.9f, 0.0f)) < 1.0f);
        for (const auto &f: {all[5], all[7], all[10]})
            for (const float x: {inf, -inf})
                CHECK(std::isnan(f.fn(x)));
        for (const auto &f: {all[4], all[5], all[8], all[9], all[10], all[11], all[13], all[14]})
        {
            CHECK(same(f.fn(0.0f), 0.0f));
            CHECK(same(f.fn(-0.0f), -0.0f));
        }
        for (const float y: {nan, 0.0f, 1.0f, 2.0f, -1.0f, 0.5f})
        {
            CHECK(std::isnan(wray::fast::pow(nan, y)));
            CHECK(std::isnan(wray::fast::pow(y, nan)));
            CHECK(std::isnan(wray::fast::atan2(nan, y)));
            CHECK(std::isnan(wray::fast::atan2(y, nan)));
        }
        CHECK(std::isnan(wray::fast::pow(-2.0f, 0.5f)));
        CHECK(same(wray::fast::pow(-2.0f, 3.0f), -wray::fast::pow(2.0f, 3.0f)));
        CHECK(same(wray::fast::pow(0.0f, 2.0f), wray::fast::exp2(-126.0f)));
        CHECK(same(wray::fast::pow(0.0f, -2.0f), wray::fast::exp2(127.0f)));
        CHECK(same(wray::fast::pow(inf, 2.0f), wray::fast::exp2(127.0f)));
        CHECK(same(wray::fast::pow(inf, -2.0f), wray::fast::exp2(-126.0f)));
    }

#undef WRAY_FAST_ROW

    template <class T>
    void parameter_domains()
    {
        using V = wray::simd::native<T>;
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T huge = std::numeric_limits<T>::max();
        const T edge = wray::prewarp(T(1), T(2));
        CHECK(std::isfinite(edge) && edge > T(1e6));
        const T rates[] = {T(48000), T(1), tiny, huge, inf, T(0), -T(0), T(-48000), -inf, nan};
        const T values[] = {T(1000), T(1), tiny, huge, inf, T(0), -T(0), T(-1000), -inf, nan};
        for (const T fs: rates)
        {
            for (const T v: values)
            {
                const T pole = wray::pole_from_cutoff(v, fs);
                const T warp = wray::prewarp(v, fs);
                const T coeff = wray::smoothing_coefficient(v, fs);
                CHECK(same(wray::pole_from_cutoff(V(v), V(fs))[V::size - 1], pole));
                CHECK(same(wray::prewarp(V(v), V(fs))[V::size - 1], warp));
                CHECK(same(wray::smoothing_coefficient(V(v), V(fs))[V::size - 1], coeff));
                if (std::isnan(v) || std::isnan(fs))
                {
                    CHECK(std::isnan(pole) && std::isnan(warp) && std::isnan(coeff));
                    continue;
                }
                CHECK(pole >= T(0) && pole <= T(1));
                CHECK(warp >= T(0) && warp <= edge);
                CHECK(coeff >= T(0) && coeff <= T(1));
                if (!(fs > T(0)))
                {
                    CHECK(same(pole, T(0)));
                    CHECK(same(warp, edge));
                    CHECK(same(coeff, T(1)));
                } else if (!(v > T(0)))
                {
                    CHECK(same(pole, T(1)));
                    CHECK(same(warp, T(0)));
                    CHECK(same(coeff, T(1)));
                } else if (v == inf && fs < inf)
                {
                    CHECK(same(pole, T(0)));
                    CHECK(same(warp, edge));
                    CHECK(same(coeff, T(0)));
                }
            }
        }
        for (const T fs: {T(44100), T(48000), T(192000)})
        {
            T last_pole = T(1);
            T last_warp = T(0);
            for (T ratio = T(1e-9); ratio < T(0.5); ratio *= T(1.5))
            {
                const T pole = wray::pole_from_cutoff(ratio * fs, fs);
                const T warp = wray::prewarp(ratio * fs, fs);
                CHECK(pole <= last_pole && pole > T(0));
                CHECK(warp >= last_warp && std::isfinite(warp));
                last_pole = pole;
                last_warp = warp;
            }
            CHECK(same(wray::prewarp(fs / T(2), fs), edge));
            CHECK(same(wray::prewarp(fs, fs), edge));
            CHECK(std::abs(wray::pole_from_cutoff(fs / T(2), fs) - T(0.04321391826377226)) <= T(2) * std::numeric_limits<T>::epsilon() * T(0.05));
        }
    }
}

int main()
{
    unary_specials<float>();
    unary_specials<double>();
    balanced_specials<float>();
    balanced_specials<double>();
    binary_specials<float>();
    binary_specials<double>();
    octaves<float>();
    octaves<double>();
    ranges();
    huge_trig<float, wray::f32x4>();
    huge_trig<double, wray::f64x2>();
    parameter_domains<float>();
    parameter_domains<double>();
    fast_specials();
    return wray::test::finish("special");
}
