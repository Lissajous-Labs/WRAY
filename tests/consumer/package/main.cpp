#include <wray/wray.h>

#include <array>
#include <cstdio>

int main()
{
    const std::array<float, 4> phases{0.0f, 0.25f, 0.5f, 0.75f};
    std::array<float, 4> sines{};
    wray::sin_turns(phases, sines);
    const float k = wray::smoothing_coefficient(0.010f, 48000.0f);
    const auto v = wray::exp(wray::f64x2(0.0));
    const bool ok = wray::floating_point_ok() && sines[1] == 1.0f && sines[3] == -1.0f && k > 0.0f && k < 1.0f && v[0] == 1.0;
    std::printf("%s sin_turns(0.25)=%g smoothing_coefficient=%g\n", ok ? "ok" : "fail", static_cast<double>(sines[1]), static_cast<double>(k));
    return ok ? 0 : 1;
}
