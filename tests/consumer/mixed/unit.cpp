#include <wray/wray.h>

#include <bit>
#include <cstdint>
#include <random>

#define WRAY_JOIN2(a, b) a##_##b
#define WRAY_JOIN(a, b) WRAY_JOIN2(a, b)
#define WRAY_TEXT2(x) #x
#define WRAY_TEXT(x) WRAY_TEXT2(x)

extern "C" const char *WRAY_JOIN(abi, WRAY_UNIT)() { return WRAY_TEXT(WRAY_ABI); }

extern "C" std::uint64_t WRAY_JOIN(hash, WRAY_UNIT)()
{
    std::mt19937_64 g(3);
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&h](double v) { h = (h ^ std::bit_cast<std::uint64_t>(v)) * 0x100000001b3ull; };
    std::uniform_real_distribution<double> u(-8.0, 8.0);
    for (int i = 0; i < 4096; ++i)
    {
        const wray::f64x2 x(u(g));
        const wray::f32x4 y(static_cast<float>(u(g)));
        const wray::f64x2 r[] = {wray::log1p(wray::simd::abs(x)), wray::exp(x), wray::sin(x * 1000.0), wray::pow(wray::simd::abs(x), x), wray::tanh(x)};
        const wray::f32x4 q[] = {wray::atan2(y, y + 1.0f), wray::adaa::soft_clip::first(y, y * 0.5f), wray::fast::log(wray::simd::abs(y) + 1.0f), wray::erf(y)};
        for (const auto &v: r)
            mix(v[0]);
        for (const auto &v: q)
            mix(static_cast<double>(v[0]));
    }
    return h;
}
