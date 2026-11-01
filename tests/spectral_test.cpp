#include <wray/wray.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

#include "check.h"

namespace
{
    using cd = std::complex<double>;
    constexpr std::size_t size = 1u << 16;

    void transform(std::vector<cd> &a)
    {
        const std::size_t n = a.size();
        const int bits = std::countr_zero(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t r = 0;
            for (int b = 0; b < bits; ++b)
                r |= ((i >> b) & 1u) << (bits - 1 - b);
            if (i < r)
                std::swap(a[i], a[r]);
        }
        for (std::size_t len = 2; len <= n; len <<= 1)
        {
            const std::size_t half = len / 2;
            for (std::size_t j = 0; j < half; ++j)
            {
                const double t = -static_cast<double>(j) / static_cast<double>(len);
                const cd w(wray::cos_turns(t), wray::sin_turns(t));
                for (std::size_t i = 0; i < n; i += len)
                {
                    const cd v = a[i + j + half] * w;
                    const cd u = a[i + j];
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    std::vector<double> magnitude(const std::vector<double> &x)
    {
        std::vector<cd> a(x.begin(), x.end());
        transform(a);
        std::vector<double> m(a.size() / 2 + 1);
        for (std::size_t i = 0; i < m.size(); ++i)
            m[i] = std::abs(a[i]);
        return m;
    }

    std::string list(const std::vector<double> &v)
    {
        std::string s = "[";
        char buf[40];
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            std::snprintf(buf, sizeof buf, "%s%.6g", i ? "," : "", v[i]);
            s += buf;
        }
        return s + "]";
    }

    std::vector<double> decibels(const std::vector<double> &m, double reference, std::size_t points)
    {
        const std::size_t step = (m.size() - 1) / points;
        std::vector<double> out;
        for (std::size_t i = 0; i + step <= m.size() - 1; i += step)
        {
            const double peak = *std::max_element(m.begin() + static_cast<std::ptrdiff_t>(i), m.begin() + static_cast<std::ptrdiff_t>(i + step));
            out.push_back(20.0 * std::log10(std::max(peak / reference, 1e-30)));
        }
        return out;
    }

    struct sfdr_row
    {
        std::string name;
        double sfdr = 0.0;
        std::vector<double> spectrum;
    };

    template<class T, class F>
    sfdr_row sfdr(const char *name, F oscillator)
    {
        sfdr_row row{name, 1e9, {}};
        for (const std::size_t bin: {std::size_t(1009), std::size_t(4099), std::size_t(12289), std::size_t(30011)})
        {
            std::vector<double> x(size);
            for (std::size_t n = 0; n < size; ++n)
            {
                const T phase = T(static_cast<double>((n * bin) % size) / static_cast<double>(size));
                x[n] = static_cast<double>(oscillator(phase));
            }
            const auto m = magnitude(x);
            double spur = 0.0;
            for (std::size_t i = 0; i < m.size(); ++i)
                if (i != bin)
                    spur = std::max(spur, m[i]);
            const double db = 20.0 * std::log10(m[bin] / std::max(spur, 1e-300));
            if (db < row.sfdr)
            {
                row.sfdr = db;
                row.spectrum = decibels(m, m[bin], 1024);
            }
        }
        return row;
    }

    struct alias_row
    {
        std::string name;
        double reduction = 0.0;
        std::vector<double> naive;
        std::vector<double> smooth;
    };

    double alias_power(const std::vector<double> &m, std::size_t bin)
    {
        double total = 0.0;
        for (std::size_t i = 1; i < m.size(); ++i)
        {
            bool harmonic = false;
            for (std::size_t k = 1; k * bin < size / 2 && !harmonic; k += 2)
                harmonic = i == k * bin;
            if (!harmonic)
                total += m[i] * m[i];
        }
        return total;
    }

    template<class Shape>
    alias_row aliasing(const char *name)
    {
        constexpr std::size_t bin = 6007;
        std::vector<double> x(size), naive(size), smooth(size);
        for (std::size_t n = 0; n < size; ++n)
            x[n] = 4.0 * wray::sin_turns(static_cast<double>((n * bin) % size) / static_cast<double>(size));
        double previous = x[size - 1];
        for (std::size_t n = 0; n < size; ++n)
        {
            naive[n] = Shape::f(x[n]);
            smooth[n] = Shape::first(previous, x[n]);
            previous = x[n];
        }
        const auto a = magnitude(naive);
        const auto b = magnitude(smooth);
        return {name, 10.0 * std::log10(alias_power(a, bin) / alias_power(b, bin)), decibels(a, a[bin], 1024), decibels(b, a[bin], 1024)};
    }

    struct smoothing_row
    {
        std::vector<double> samples;
        std::vector<double> wray;
        std::vector<double> naive;
        double wray_max = 0.0;
        double naive_max = 0.0;
    };

    double ulps(float got, double ref)
    {
        const double ulp = std::ldexp(1.0, std::ilogb(static_cast<float>(ref)) - 23);
        return std::abs(static_cast<double>(got) - ref) / ulp;
    }

    smoothing_row smoothing()
    {
        smoothing_row r;
        for (int i = 0; i <= 400; ++i)
        {
            const double samples = std::pow(10.0, 9.0 * i / 400.0);
            const float fs = 48000.0f;
            const float seconds = static_cast<float>(samples / 48000.0);
            const double ref = -std::expm1(-1.0 / (static_cast<double>(seconds) * static_cast<double>(fs)));
            const double w = ulps(wray::smoothing_coefficient(seconds, fs), ref);
            const double n = ulps(1.0f - std::exp(-1.0f / (seconds * fs)), ref);
            r.samples.push_back(samples);
            r.wray.push_back(w);
            r.naive.push_back(n);
            r.wray_max = std::max(r.wray_max, w);
            r.naive_max = std::max(r.naive_max, n);
        }
        return r;
    }
}

int main(int argc, char **argv)
{
    const char *json = nullptr;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--json") == 0)
            json = argv[i + 1];

    std::vector<sfdr_row> oscillators;
    oscillators.push_back(sfdr<float>("float precise", [](float p) { return wray::sin_turns(p); }));
    oscillators.push_back(sfdr<float>("float fast", [](float p) { return wray::fast::sin_turns(p); }));
    oscillators.push_back(sfdr<double>("double precise", [](double p) { return wray::sin_turns(p); }));
    oscillators.push_back(sfdr<float>("float std::sin", [](float p) { return std::sin(2.0f * std::numbers::pi_v<float> * p); }));
    const double limits[] = {170.0, 110.0, 300.0};
    for (std::size_t i = 0; i < oscillators.size(); ++i)
    {
        std::printf("sfdr %-15s %.1f dB\n", oscillators[i].name.c_str(), oscillators[i].sfdr);
        if (i < 3)
            CHECK(oscillators[i].sfdr >= limits[i]);
    }

    std::vector<alias_row> shapes;
    shapes.push_back(aliasing<wray::adaa::hard_clip>("hard_clip"));
    shapes.push_back(aliasing<wray::adaa::soft_clip>("soft_clip"));
    shapes.push_back(aliasing<wray::adaa::tanh>("tanh"));
    shapes.push_back(aliasing<wray::adaa::atan>("atan"));
    shapes.push_back(aliasing<wray::adaa::algebraic>("algebraic"));
    shapes.push_back(aliasing<wray::adaa::radical>("radical"));
    for (const auto &s: shapes)
    {
        std::printf("aliasing %-10s reduced by %.2f dB\n", s.name.c_str(), s.reduction);
        CHECK(s.reduction >= 8.0);
    }

    const auto smooth = smoothing();
    std::printf("smoothing coefficient float: wray %.3f ulp, naive %.3g ulp\n", smooth.wray_max, smooth.naive_max);
    CHECK(smooth.wray_max <= 2.0);
    CHECK(smooth.naive_max >= 1e6);

    if (json)
    {
        std::string s = "{\"size\":" + std::to_string(size) + ",\"oscillators\":[";
        for (std::size_t i = 0; i < oscillators.size(); ++i)
            s += std::string(i ? "," : "") + "{\"name\":\"" + oscillators[i].name + "\",\"sfdr\":" + std::to_string(oscillators[i].sfdr) +
                 ",\"spectrum\":" + list(oscillators[i].spectrum) + "}";
        s += "],\"aliasing\":[";
        for (std::size_t i = 0; i < shapes.size(); ++i)
            s += std::string(i ? "," : "") + "{\"name\":\"" + shapes[i].name + "\",\"reduction\":" + std::to_string(shapes[i].reduction) +
                 ",\"naive\":" + list(shapes[i].naive) + ",\"adaa\":" + list(shapes[i].smooth) + "}";
        s += "],\"smoothing\":{\"samples\":" + list(smooth.samples) + ",\"wray\":" + list(smooth.wray) + ",\"naive\":" + list(smooth.naive) + "}}\n";
        if (std::FILE *f = std::fopen(json, "w"))
        {
            std::fputs(s.c_str(), f);
            std::fclose(f);
        }
        else
        {
            std::printf("cannot write %s\n", json);
            return 1;
        }
    }
    return wray::test::finish("spectral");
}
