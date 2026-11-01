#include <wray/wray.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include "check.h"

namespace
{
    template <class T>
    bool same(T a, T b)
    {
        return std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t> >(a) ==
               std::bit_cast<std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t> >(b);
    }

    template <class T>
    void ranges(std::mt19937_64 &g)
    {
        constexpr std::size_t w = static_cast<std::size_t>(wray::simd::native<T>::size);
        std::uniform_real_distribution<T> u(T(-4), T(4));
        for (std::size_t n = 0; n <= 8 * w + 1; ++n)
        {
            for (std::size_t offset = 0; offset < w; ++offset)
            {
                std::vector<T> a(offset + n);
                std::vector<T> b(offset + n);
                for (std::size_t i = 0; i < a.size(); ++i)
                {
                    a[i] = u(g);
                    b[i] = u(g);
                }
                const std::span<const T> in(a.data() + offset, n);
                const std::span<const T> in2(b.data() + offset, n);

                std::vector<T> out(offset + n);
                wray::exp(in, std::span<T>(out.data() + offset, n));
                bool ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(out[offset + i], wray::exp(in[i]));
                CHECK(ok);

                wray::pow(in, in2, std::span<T>(out.data() + offset, n));
                ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(out[offset + i], wray::pow(in[i], in2[i]));
                CHECK(ok);

                if (n > 0)
                {
                    std::vector<T> shorter(n - 1, T(7));
                    wray::sin(in, shorter);
                    ok = true;
                    for (std::size_t i = 0; i + 1 < n; ++i)
                        ok = ok && same(shorter[i], wray::sin(in[i]));
                    CHECK(ok);
                    std::vector<T> longer(n + 3, T(7));
                    wray::sin(in, longer);
                    for (std::size_t i = n; i < n + 3; ++i)
                        ok = ok && longer[i] == T(7);
                    CHECK(ok);
                }

                std::vector<T> io(a);
                const std::span<T> place(io.data() + offset, n);
                wray::tanh(place, place);
                ok = true;
                for (std::size_t i = 0; i < n; ++i)
                    ok = ok && same(place[i], wray::tanh(in[i]));
                CHECK(ok);

                if constexpr (std::is_same_v<T, float>)
                {
                    wray::fast::log(in, std::span<T>(out.data() + offset, n));
                    ok = true;
                    for (std::size_t i = 0; i < n; ++i)
                        ok = ok && same(out[offset + i], wray::fast::log(in[i]));
                    CHECK(ok);
                }
            }
        }
        std::vector<T> empty;
        wray::exp(empty, empty);
        wray::pow(empty, empty, empty);
        CHECK(empty.empty());
    }
}

int main()
{
    std::mt19937_64 g(17);
    ranges<float>(g);
    ranges<double>(g);
    return wray::test::finish("range");
}
