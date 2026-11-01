#include <wray/wray.h>

#include <cstdio>

int main()
{
    const float k = wray::smoothing_coefficient(0.010f, 48000.0f);
    const bool ok = wray::floating_point_ok() && k > 0.0f && k < 1.0f && wray::exp(0.0f) == 1.0f;
    std::printf("%s %g\n", ok ? "ok" : "fail", static_cast<double>(k));
    return ok ? 0 : 1;
}
