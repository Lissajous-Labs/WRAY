#include <wray/wray.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
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
                add("sin", [](auto v) { return wray::fast::sin(v); });
                add("cos", [](auto v) { return wray::fast::cos(v); });
                add("tan", [](auto v) { return wray::fast::tan(v); });
                add("atan", [](auto v) { return wray::fast::atan(v); });
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
                add("exp2", [](auto v) { return wray::exp2(v); });
                add("log", [](auto v) { return wray::log(v); });
                add("log2", [](auto v) { return wray::log2(v); });
                add("sin", [](auto v) { return wray::sin(v); });
                add("cos", [](auto v) { return wray::cos(v); });
                add("tan", [](auto v) { return wray::tan(v); });
                add("atan", [](auto v) { return wray::atan(v); });
                add("tanh", [](auto v) { return wray::tanh(v); });
                add("sigmoid", [](auto v) { return wray::sigmoid(v); });
                add("db_to_gain", [](auto v) { return wray::db_to_gain(v); });
                add("gain_to_db", [](auto v) { return wray::gain_to_db(v); });
                add2("atan2", [](auto a, auto b) { return wray::atan2(a, b); });
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
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::fprintf(stderr, "usage: wray_dump <function> <f32|f64|fast> <scalar|vector> <in> <out>\n");
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
