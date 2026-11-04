#include <wray/wray.h>

#include <array>
#include <cstdio>

int main()
{
    const float k = wray::smoothing_coefficient(0.010f, 48000.0f);
    const std::array<float, 3> bases{0.5f, 2.0f, -2.0f}, exponents{2.0f, 3.0f, 3.0f};
    std::array<float, 3> powers{};
    wray::balanced::pow(bases, exponents, powers);
    const double scalar = wray::balanced::pow(1.25, 2.5);
    const auto vector = wray::balanced::pow(wray::f32x4(2.0f), wray::f32x4(3.0f));
    const bool ok = wray::floating_point_ok() && k > 0.0f && k < 1.0f && wray::exp(0.0f) == 1.0f &&
                    powers[0] > 0.24999f && powers[0] < 0.25001f && powers[1] > 7.9999f && powers[1] < 8.0001f &&
                    powers[2] > -8.0001f && powers[2] < -7.9999f && scalar > 1.74 && scalar < 1.75 &&
                    vector[0] > 7.9999f && vector[0] < 8.0001f;
    std::printf("%s %g\n", ok ? "ok" : "fail", static_cast<double>(k));
    return ok ? 0 : 1;
}
