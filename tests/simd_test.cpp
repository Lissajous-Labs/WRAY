#include <wray/simd/vec.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include "check.h"

namespace
{
    using namespace wray::simd;

    template <class T>
    bool same(T a, T b)
    {
        return std::bit_cast<typename bits_of<T>::integer>(a) == std::bit_cast<typename bits_of<T>::integer>(b);
    }

    template <class V, class F, class G>
    void lanes(const V& a, const V& b, F vector_op, G scalar_op)
    {
        using T = typename V::value_type;
        const V r = vector_op(a, b);
        for (int i = 0; i < V::size; ++i)
            CHECK(same<T>(r[i], scalar_op(a[i], b[i])));
    }

    template <class V, class M>
    void mask_lanes(const V& a, const V& b, const M& m, bool (*op)(typename V::value_type, typename V::value_type))
    {
        using T = typename V::value_type;
        const V r = select(m, V(T(1)), V(T(0)));
        for (int i = 0; i < V::size; ++i)
            CHECK((r[i] == T(1)) == op(a[i], b[i]));
    }

    template <class T>
    void floats(std::mt19937_64& g)
    {
        using V = native<T>;
        constexpr int n = V::size;
        std::uniform_real_distribution<T> d(T(-100), T(100));
        for (int iter = 0; iter < 2000; ++iter)
        {
            T a[n], b[n], c[n];
            for (int i = 0; i < n; ++i)
            {
                a[i] = d(g);
                b[i] = (iter % 7 == 0 && i == 0) ? a[i] : d(g);
                c[i] = d(g);
            }
            const V va = V::load(a), vb = V::load(b), vc = V::load(c);
            lanes(va, vb, [](V x, V y) { return x + y; }, [](T x, T y) { return T(x + y); });
            lanes(va, vb, [](V x, V y) { return x - y; }, [](T x, T y) { return T(x - y); });
            lanes(va, vb, [](V x, V y) { return x * y; }, [](T x, T y) { return T(x * y); });
            lanes(va, vb, [](V x, V y) { return x / y; }, [](T x, T y) { return T(x / y); });
            lanes(va, vb, [](V x, V) { return -x; }, [](T x, T) { return T(-x); });
            lanes(va, vb, [](V x, V y) { return min(x, y); }, [](T x, T y) { return y < x ? y : x; });
            lanes(va, vb, [](V x, V y) { return max(x, y); }, [](T x, T y) { return x < y ? y : x; });
            lanes(va, vb, [](V x, V) { return abs(x); }, [](T x, T) { return std::fabs(x); });
            lanes(va, vb, [](V x, V) { return sqrt(abs(x)); }, [](T x, T) { return std::sqrt(std::fabs(x)); });
            const V f = fma(va, vb, vc);
            for (int i = 0; i < n; ++i)
                CHECK(same<T>(f[i], wray::simd::fma(a[i], b[i], c[i])));
            mask_lanes(va, vb, va < vb, +[](T x, T y) { return x < y; });
            mask_lanes(va, vb, va <= vb, +[](T x, T y) { return x <= y; });
            mask_lanes(va, vb, va > vb, +[](T x, T y) { return x > y; });
            mask_lanes(va, vb, va >= vb, +[](T x, T y) { return x >= y; });
            mask_lanes(va, vb, va == vb, +[](T x, T y) { return x == y; });
            mask_lanes(va, vb, va != vb, +[](T x, T y) { return x != y; });
            const V both_mask = select((va < vb) & (vb > vc), V(T(1)), V(T(0)));
            for (int i = 0; i < n; ++i)
                CHECK((both_mask[i] == T(1)) == (a[i] < b[i] && b[i] > c[i]));
            const auto bits = as_bits(va);
            const V back = from_bits<V>(bits);
            for (int i = 0; i < n; ++i)
            {
                CHECK(bits[i] == std::bit_cast<typename bits_of<T>::integer>(a[i]));
                CHECK(same<T>(back[i], a[i]));
            }
        }
    }

    template <class I>
    void integers(std::mt19937_64& g)
    {
        using V = vec<I, 16 / int(sizeof(I))>;
        constexpr int n = V::size;
        using U = typename bits_of<I>::unsigned_integer;
        for (int iter = 0; iter < 2000; ++iter)
        {
            I a[n], b[n];
            for (int i = 0; i < n; ++i)
            {
                a[i] = static_cast<I>(g());
                b[i] = static_cast<I>(g());
            }
            const V va = V::load(a), vb = V::load(b);
            const int s = int(g() % (sizeof(I) * 8));
            const V sum = va + vb, diff = va - vb, conj = va & vb, disj = va | vb, ex = va ^ vb, left = va << s, right = shift_right_logical(va, s);
            for (int i = 0; i < n; ++i)
            {
                CHECK(sum[i] == static_cast<I>(static_cast<U>(a[i]) + static_cast<U>(b[i])));
                CHECK(diff[i] == static_cast<I>(static_cast<U>(a[i]) - static_cast<U>(b[i])));
                CHECK(conj[i] == static_cast<I>(a[i] & b[i]));
                CHECK(disj[i] == static_cast<I>(a[i] | b[i]));
                CHECK(ex[i] == static_cast<I>(a[i] ^ b[i]));
                CHECK(left[i] == static_cast<I>(static_cast<U>(a[i]) << s));
                CHECK(right[i] == static_cast<I>(static_cast<U>(a[i]) >> s));
            }
            I z[n];
            for (int i = 0; i < n; ++i)
                z[i] = (i % 2 == 0) ? I(0) : a[i];
            const auto m = nonzero(V::load(z));
            using F = std::conditional_t<sizeof(I) == 4, float, double>;
            const auto picked = select(m, native<F>(F(1)), native<F>(F(0)));
            for (int i = 0; i < n; ++i)
                CHECK((picked[i] == F(1)) == (z[i] != 0));
        }
    }

    void masks()
    {
        const f32x4 a(1.0f);
        const f32x4 b(2.0f);
        CHECK(all(a < b));
        CHECK(!any(a > b));
        CHECK(any(either(a < b, a > b)));
        CHECK(!any(both(a < b, a > b)));
        const f64x2 c(std::numeric_limits<double>::quiet_NaN());
        CHECK(all(c != c));
        CHECK(!any(c == c));
    }
}

int main()
{
    std::mt19937_64 g(3);
    floats<float>(g);
    floats<double>(g);
    integers<std::int32_t>(g);
    integers<std::int64_t>(g);
    masks();
    return wray::test::finish("simd");
}
