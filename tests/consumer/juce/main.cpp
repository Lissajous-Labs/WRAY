#include <juce_core/juce_core.h>
#include <wray/wray.h>

#include <cstdio>
#include <span>

int main()
{
    constexpr int n = 64;
    juce::HeapBlock<float> phases(n), sines(n);
    for (int i = 0; i < n; ++i)
        phases[i] = static_cast<float>(i) / static_cast<float>(n);
    wray::sin_turns(std::span<const float>(phases.get(), n), std::span<float>(sines.get(), n));
    const float k = wray::smoothing_coefficient(0.010f, 48000.0f);
    const double p = wray::balanced::pow(2.0, 10.0);
    const bool ok = JUCE_MODULE_AVAILABLE_wray && wray::floating_point_ok() && sines[0] <= 0.0f && sines[0] >= 0.0f &&
                    sines[n / 4] >= 1.0f && sines[n / 4] <= 1.0f && k > 0.0f && k < 1.0f && p > 1023.999 && p < 1024.001;
    std::printf("%s %s sin_turns(0.25)=%g smoothing_coefficient=%g balanced::pow(2,10)=%g\n", ok ? "ok" : "fail",
                juce::SystemStats::getJUCEVersion().toRawUTF8(), static_cast<double>(sines[n / 4]), static_cast<double>(k), p);
    return ok ? 0 : 1;
}
