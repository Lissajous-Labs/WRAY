#include <wray/wray.h>

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
}

int main()
{
    unary_specials<float>();
    unary_specials<double>();
    binary_specials<float>();
    binary_specials<double>();
    octaves<float>();
    octaves<double>();
    ranges();
    huge_trig<float, wray::f32x4>();
    huge_trig<double, wray::f64x2>();
    return wray::test::finish("special");
}
