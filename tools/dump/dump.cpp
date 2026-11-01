#include <wray/wray.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
    template <class T>
    std::vector<T> read_all(const char* path)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        const auto bytes = static_cast<std::size_t>(f.tellg());
        std::vector<T> v(bytes / sizeof(T));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
        return v;
    }

    template <class T>
    void write_all(const char* path, const std::vector<T>& v)
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    }

    template <class T>
    using unary = std::function<void(const std::vector<T>&, std::vector<T>&)>;

    template <class T>
    using binary = std::function<void(const std::vector<T>&, const std::vector<T>&, std::vector<T>&)>;

    template <class T, class F>
    unary<T> scalar_unary(F f)
    {
        return [f](const std::vector<T>& in, std::vector<T>& out) {
            for (std::size_t i = 0; i < in.size(); ++i)
                out[i] = f(in[i]);
        };
    }

    template <class T, class F>
    binary<T> scalar_binary(F f)
    {
        return [f](const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out) {
            for (std::size_t i = 0; i < a.size(); ++i)
                out[i] = f(a[i], b[i]);
        };
    }

    template <class T, class F>
    unary<T> vector_unary(F f)
    {
        return [f](const std::vector<T>& in, std::vector<T>& out) { wray::math::apply(in, out, f); };
    }

    template <class T, class F>
    binary<T> vector_binary(F f)
    {
        return [f](const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out) { wray::math::apply(a, b, out, f); };
    }

    template <class T, bool Fast>
    struct catalogue
    {
        std::map<std::string, std::pair<unary<T>, unary<T>>> one;
        std::map<std::string, std::pair<binary<T>, binary<T>>> two;

        template <class F>
        void add(const char* name, F f)
        {
            one[name] = {scalar_unary<T>(f), vector_unary<T>(f)};
        }

        template <class F>
        void add2(const char* name, F f)
        {
            two[name] = {scalar_binary<T>(f), vector_binary<T>(f)};
        }

        catalogue()
        {
            if constexpr (Fast)
            {
                add("exp", [](auto v) { return wray::fast::exp(v); });
                add("exp2", [](auto v) { return wray::fast::exp2(v); });
                add("log", [](auto v) { return wray::fast::log(v); });
                add("log2", [](auto v) { return wray::fast::log2(v); });
                add("sin_turns", [](auto v) { return wray::fast::sin_turns(v); });
                add("sin", [](auto v) { return wray::fast::sin(v); });
                add("cos_turns", [](auto v) { return wray::fast::cos_turns(v); });
                add("cos", [](auto v) { return wray::fast::cos(v); });
                add("tanpi", [](auto v) { return wray::fast::tanpi(v); });
                add("tan_turns", [](auto v) { return wray::fast::tan_turns(v); });
                add("tan", [](auto v) { return wray::fast::tan(v); });
                add("atan", [](auto v) { return wray::fast::atan(v); });
                add("omega", [](auto v) { return wray::fast::omega(v); });
                add("product_log", [](auto v) { return wray::fast::product_log(v); });
                add("tanh", [](auto v) { return wray::fast::tanh(v); });
                add("sigmoid", [](auto v) { return wray::fast::sigmoid(v); });
                add("db_to_gain", [](auto v) { return wray::fast::db_to_gain(v); });
                add("gain_to_db", [](auto v) { return wray::fast::gain_to_db(v); });
                add2("atan2", [](auto a, auto b) { return wray::fast::atan2(a, b); });
                add2("pow", [](auto a, auto b) { return wray::fast::pow(a, b); });
            }
            else
            {
                add("exp", [](auto v) { return wray::exp(v); });
                add("expm1", [](auto v) { return wray::expm1(v); });
                add("exp2", [](auto v) { return wray::exp2(v); });
                add("log", [](auto v) { return wray::log(v); });
                add("log1p", [](auto v) { return wray::log1p(v); });
                add("log10", [](auto v) { return wray::log10(v); });
                add("log2", [](auto v) { return wray::log2(v); });
                add("sin_turns", [](auto v) { return wray::sin_turns(v); });
                add("sin", [](auto v) { return wray::sin(v); });
                add("cos_turns", [](auto v) { return wray::cos_turns(v); });
                add("cos", [](auto v) { return wray::cos(v); });
                add("tanpi", [](auto v) { return wray::tanpi(v); });
                add("tan_turns", [](auto v) { return wray::tan_turns(v); });
                add("tan", [](auto v) { return wray::tan(v); });
                add("atan", [](auto v) { return wray::atan(v); });
                add("sinh", [](auto v) { return wray::sinh(v); });
                add("cosh", [](auto v) { return wray::cosh(v); });
                add("asinh", [](auto v) { return wray::asinh(v); });
                add("cbrt", [](auto v) { return wray::cbrt(v); });
                add("erf", [](auto v) { return wray::erf(v); });
                add("midi_to_hz", [](auto v) { return wray::midi_to_hz(v); });
                add("hz_to_midi", [](auto v) { return wray::hz_to_midi(v); });
                add("ratio_from_cents", [](auto v) { return wray::ratio_from_cents(v); });
                add("omega", [](auto v) { return wray::omega(v); });
                add("product_log", [](auto v) { return wray::product_log(v); });
                add("tanh", [](auto v) { return wray::tanh(v); });
                add("sigmoid", [](auto v) { return wray::sigmoid(v); });
                add("db_to_gain", [](auto v) { return wray::db_to_gain(v); });
                add("gain_to_db", [](auto v) { return wray::gain_to_db(v); });
                add2("atan2", [](auto a, auto b) { return wray::atan2(a, b); });
                add2("smoothing_coefficient", [](auto a, auto b) { return wray::smoothing_coefficient(a, b); });
                add2("pole_from_cutoff", [](auto a, auto b) { return wray::pole_from_cutoff(a, b); });
                add2("prewarp", [](auto a, auto b) { return wray::prewarp(a, b); });
                add("adaa_hard_clip_F1", [](auto v) { return wray::adaa::hard_clip::F1(v); });
                add2("adaa_hard_clip_first", [](auto a, auto b) { return wray::adaa::hard_clip::first(a, b); });
                add("adaa_soft_clip_F1", [](auto v) { return wray::adaa::soft_clip::F1(v); });
                add2("adaa_soft_clip_first", [](auto a, auto b) { return wray::adaa::soft_clip::first(a, b); });
                add("adaa_tanh_F1", [](auto v) { return wray::adaa::tanh::F1(v); });
                add2("adaa_tanh_first", [](auto a, auto b) { return wray::adaa::tanh::first(a, b); });
                add("adaa_atan_F1", [](auto v) { return wray::adaa::atan::F1(v); });
                add2("adaa_atan_first", [](auto a, auto b) { return wray::adaa::atan::first(a, b); });
                add("adaa_algebraic_F1", [](auto v) { return wray::adaa::algebraic::F1(v); });
                add2("adaa_algebraic_first", [](auto a, auto b) { return wray::adaa::algebraic::first(a, b); });
                add("adaa_radical_F1", [](auto v) { return wray::adaa::radical::F1(v); });
                add2("adaa_radical_first", [](auto a, auto b) { return wray::adaa::radical::first(a, b); });
                add2("pow", [](auto a, auto b) { return wray::pow(a, b); });
            }
        }
    };

    template <class T, bool Fast>
    int run(const std::string& fn, bool vectorised, const char* in_path, const char* out_path)
    {
        static const catalogue<T, Fast> table;
        auto data = read_all<T>(in_path);
        if (auto it = table.one.find(fn); it != table.one.end())
        {
            std::vector<T> out(data.size());
            (vectorised ? it->second.second : it->second.first)(data, out);
            write_all(out_path, out);
            return 0;
        }
        if (auto it = table.two.find(fn); it != table.two.end())
        {
            const std::size_t n = data.size() / 2;
            std::vector<T> a(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(n));
            std::vector<T> b(data.begin() + static_cast<std::ptrdiff_t>(n), data.end());
            std::vector<T> out(n);
            (vectorised ? it->second.second : it->second.first)(a, b, out);
            write_all(out_path, out);
            return 0;
        }
        std::fprintf(stderr, "unknown function %s\n", fn.c_str());
        return 2;
    }

    template <class T>
    std::vector<T> digest_inputs(std::uint64_t seed, std::size_t n)
    {
        using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
        std::mt19937_64 g(seed);
        std::vector<T> v(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint64_t r = g();
            v[i] = i % 2 == 0 ? std::bit_cast<T>(static_cast<U>(r)) : static_cast<T>((static_cast<double>(r >> 11) * 0x1p-53 - 0.5) * 64.0);
        }
        return v;
    }

    template <class T>
    std::uint64_t hash(const std::vector<T>& v)
    {
        using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
        std::uint64_t h = 0xcbf29ce484222325ull;
        for (T x : v)
        {
            const U raw = std::bit_cast<U>(x);
            const U magnitude = raw & (~U(0) >> 1);
            const U exponent = sizeof(T) == 4 ? U(0x7f800000u) : U(0x7ff0000000000000ull);
            const U bits = magnitude > exponent ? U(1) : raw;
            h = (h ^ std::uint64_t(bits)) * 0x100000001b3ull;
        }
        return h;
    }

    template <class T, bool Fast>
    void digest(const char* tier)
    {
        static const catalogue<T, Fast> table;
        constexpr std::size_t n = 1 << 16;
        const auto a = digest_inputs<T>(1, n);
        const auto b = digest_inputs<T>(2, n);
        std::vector<T> out(n);
        for (const auto& [name, f] : table.one)
        {
            f.second(a, out);
            std::printf("%s %s %016llx\n", tier, name.c_str(), static_cast<unsigned long long>(hash(out)));
        }
        for (const auto& [name, f] : table.two)
        {
            f.second(a, b, out);
            std::printf("%s %s %016llx\n", tier, name.c_str(), static_cast<unsigned long long>(hash(out)));
        }
    }
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "info") == 0)
    {
        std::printf("{\"fma\": %s}\n", WRAY_HAS_FMA ? "true" : "false");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "digest") == 0)
    {
        digest<float, false>("f32");
        digest<double, false>("f64");
        digest<float, true>("fast");
        return 0;
    }
    if (argc != 6)
    {
        std::fprintf(stderr, "usage: wray_dump info | wray_dump <function> <f32|f64|fast> <scalar|vector> <in> <out>\n");
        return 2;
    }
    const std::string fn = argv[1];
    const std::string tier = argv[2];
    const bool vectorised = std::strcmp(argv[3], "vector") == 0;
    if (tier == "f32")
        return run<float, false>(fn, vectorised, argv[4], argv[5]);
    if (tier == "f64")
        return run<double, false>(fn, vectorised, argv[4], argv[5]);
    if (tier == "fast")
        return run<float, true>(fn, vectorised, argv[4], argv[5]);
    std::fprintf(stderr, "unknown tier %s\n", tier.c_str());
    return 2;
}
