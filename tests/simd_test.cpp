#include <wray/denormals.h>
#include <wray/floating_point.h>
#include <wray/simd/vec.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

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

    template <class V>
    void comparisons(const V& va, const V& vb)
    {
        using T = typename V::value_type;
        mask_lanes(va, vb, va < vb, +[](T x, T y) { return x < y; });
        mask_lanes(va, vb, va <= vb, +[](T x, T y) { return x <= y; });
        mask_lanes(va, vb, va > vb, +[](T x, T y) { return x > y; });
        mask_lanes(va, vb, va >= vb, +[](T x, T y) { return x >= y; });
        mask_lanes(va, vb, va == vb, +[](T x, T y) { return x == y; });
        mask_lanes(va, vb, va != vb, +[](T x, T y) { return x != y; });
        mask_lanes(va, vb, !(va < vb), +[](T x, T y) { return !(x < y); });
    }

    template <class T, int n>
    void floats(std::mt19937_64& g)
    {
        using V = vec<T, n>;
        static_assert(std::is_same_v<mask_t<V>, mask_for<T, n>>);
        std::uniform_real_distribution<T> d(T(-100), T(100));
        const T nan = std::numeric_limits<T>::quiet_NaN();
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
            T h[n];
            for (int i = 0; i < n; ++i)
                h[i] = (i + iter) % 2 == 0 ? std::trunc(a[i]) + std::copysign(T(0.5), a[i]) : a[i];
            const V rounded = nearest(V::load(h));
            for (int i = 0; i < n; ++i)
                CHECK(rounded[i] == std::nearbyint(h[i]));
            CHECK(same<T>(first(va), a[0]));
            CHECK(same<T>(nearest(h[0]), rounded[0]));
            const V f = fma(va, vb, vc);
            for (int i = 0; i < n; ++i)
                CHECK(same<T>(f[i], wray::simd::fma(a[i], b[i], c[i])));
            comparisons(va, vb);
            const V both_mask = select(both(va < vb, vb > vc), V(T(1)), V(T(0)));
            const V either_mask = select(either(va < vb, vb > vc), V(T(1)), V(T(0)));
            for (int i = 0; i < n; ++i)
            {
                CHECK((both_mask[i] == T(1)) == (a[i] < b[i] && b[i] > c[i]));
                CHECK((either_mask[i] == T(1)) == (a[i] < b[i] || b[i] > c[i]));
            }
            const auto bits = as_bits(va);
            static_assert(std::is_same_v<std::remove_const_t<decltype(bits)>, integer_t<V>>);
            const V back = from_bits<V>(bits);
            for (int i = 0; i < n; ++i)
            {
                CHECK(bits[i] == std::bit_cast<typename bits_of<T>::integer>(a[i]));
                CHECK(same<T>(back[i], a[i]));
            }
            T p[n], q[n];
            for (int i = 0; i < n; ++i)
            {
                p[i] = (iter + i) % 3 == 0 ? nan : a[i];
                q[i] = (iter + i) % 5 == 0 ? nan : b[i];
            }
            comparisons(V::load(p), V::load(q));
            T out[n];
            va.store(out);
            for (int i = 0; i < n; ++i)
                CHECK(same<T>(out[i], a[i]));
        }
    }

    template <class I, int n>
    void integers(std::mt19937_64& g)
    {
        using V = vec<I, n>;
        using U = typename bits_of<I>::unsigned_integer;
        using F = std::conditional_t<sizeof(I) == 4, float, double>;
        for (int iter = 0; iter < 2000; ++iter)
        {
            I a[n], b[n];
            for (int i = 0; i < n; ++i)
            {
                a[i] = static_cast<I>(g());
                b[i] = (iter % 5 == 0 && i % 3 == 0) ? a[i] : static_cast<I>(g());
            }
            const V va = V::load(a), vb = V::load(b);
            const int s = int(g() % (sizeof(I) * 8));
            const V sum = va + vb, diff = va - vb, conj = va & vb, disj = va | vb, ex = va ^ vb, left = va << s, left2 = shift_left(va, s), right = shift_right_logical(va, s);
            for (int i = 0; i < n; ++i)
            {
                CHECK(sum[i] == static_cast<I>(static_cast<U>(a[i]) + static_cast<U>(b[i])));
                CHECK(diff[i] == static_cast<I>(static_cast<U>(a[i]) - static_cast<U>(b[i])));
                CHECK(conj[i] == static_cast<I>(a[i] & b[i]));
                CHECK(disj[i] == static_cast<I>(a[i] | b[i]));
                CHECK(ex[i] == static_cast<I>(a[i] ^ b[i]));
                CHECK(left[i] == static_cast<I>(static_cast<U>(a[i]) << s));
                CHECK(left2[i] == static_cast<I>(static_cast<U>(a[i]) << s));
                CHECK(right[i] == static_cast<I>(static_cast<U>(a[i]) >> s));
            }
            I z[n];
            for (int i = 0; i < n; ++i)
                z[i] = ((i + iter) % 2 == 0) ? I(0) : a[i];
            const auto m = nonzero(V::load(z));
            static_assert(std::is_same_v<std::remove_const_t<decltype(m)>, mask_for<F, n>>);
            const auto picked = select(m, vec<F, n>(F(1)), vec<F, n>(F(0)));
            const auto equal = select(va == vb, vec<F, n>(F(1)), vec<F, n>(F(0)));
            for (int i = 0; i < n; ++i)
            {
                CHECK((picked[i] == F(1)) == (z[i] != 0));
                CHECK((equal[i] == F(1)) == (a[i] == b[i]));
            }
            I out[n];
            va.store(out);
            for (int i = 0; i < n; ++i)
                CHECK(out[i] == a[i]);
        }
    }

    template <class T, int n>
    void reductions()
    {
        using V = vec<T, n>;
        using F = std::conditional_t<sizeof(T) == 4, float, double>;
        using R = vec<F, n>;
        T a[n];
        for (int i = 0; i < n; ++i)
            a[i] = T(i);
        const V va = V::load(a);
        const auto none = va == V(T(n));
        CHECK(!any(none));
        CHECK(all(!none));
        for (int k = 0; k < n; ++k)
        {
            const auto one = va == V(T(k));
            CHECK(any(one));
            CHECK(all(one) == (n == 1));
            CHECK(!all(!one));
            CHECK(any(!one) == (n > 1));
            CHECK(all(either(one, !one)));
            CHECK(!any(both(one, !one)));
            CHECK(any(either(one, none)));
            CHECK(!any(both(one, none)));
            const R r = select(one, R(F(1)), R(F(0)));
            for (int i = 0; i < n; ++i)
                CHECK((r[i] == F(1)) == (i == k));
        }
    }

    template <class T, int n>
    void width(std::mt19937_64& g)
    {
        using I = typename bits_of<T>::integer;
        floats<T, n>(g);
        integers<I, n>(g);
        reductions<T, n>();
        reductions<I, n>();
    }

    void denormal_guard()
    {
        using guard = wray::scoped_flush_denormals;
        volatile float tiny = 0x1p-140f;
        volatile float one = 1.0f;
#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
        CHECK(guard::flags != 0);
#endif
        const guard::state before = guard::read();
        const auto flushing = [before] {
            const guard::state now = guard::read();
            return (now & ~guard::flags) == (before & ~guard::flags) && (guard::flags == 0 || (now & guard::flags) != 0);
        };
        {
            const guard outer;
            CHECK(flushing());
            {
                const guard inner;
                CHECK(flushing());
            }
            CHECK(flushing());
            const float r = tiny * one;
            CHECK(guard::flags == 0 ? r == 0x1p-140f : r == 0.0f);
        }
        CHECK(guard::read() == before);
        CHECK(tiny * one == 0x1p-140f);
        guard::write(before | guard::flags);
        const guard::state set = guard::read();
        {
            const guard again;
            CHECK(flushing());
        }
        CHECK(guard::read() == set);
        guard::write(before);
        CHECK(guard::read() == before);
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

    template <class T, int n>
    void rounding_modes()
    {
        using V = vec<T, n>;
        std::vector<T> v{T(0.5),  T(1.5),  T(2.5),  T(-0.5), T(-1.5), T(-2.5),         T(0.3),         T(-0.3),     T(0.7),
                         T(-0.7), T(0.0),  T(-0.0), T(1e20), T(-1e20), T(4194303.5),   T(-4194303.5),  T(8388607.0), T(8388608.0),
                         std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()};
        if constexpr (std::is_same_v<T, double>)
            v.insert(v.end(), {2147483648.5, -2147483649.5, 4503599627370495.5, -4503599627370494.5, 9007199254740991.0, 6755399441055744.5});
        while (v.size() % std::size_t(n) != 0)
            v.push_back(T(3.5));
        std::vector<T> expect(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            expect[i] = std::nearbyint(v[i]);
        for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
        {
            std::fesetround(mode);
            std::vector<T> got(v.size()), scalar(v.size());
            for (std::size_t i = 0; i < v.size(); i += std::size_t(n))
                nearest(V::load(v.data() + i)).store(got.data() + i);
            for (std::size_t i = 0; i < v.size(); ++i)
                scalar[i] = nearest(v[i]);
            std::fesetround(FE_TONEAREST);
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                CHECK((std::isnan(expect[i]) && std::isnan(got[i])) || same<T>(got[i], expect[i]));
                CHECK((std::isnan(expect[i]) && std::isnan(scalar[i])) || same<T>(scalar[i], expect[i]));
            }
        }
    }
}

int main()
{
    CHECK(wray::floating_point_ok());
    static_assert(native_width<float> * int(sizeof(float)) == native_bytes);
    static_assert(native_width<double> * int(sizeof(double)) == native_bytes);
    static_assert(native<float>::size == native_width<float>);
    static_assert(native<double>::size == native_width<double>);
    std::mt19937_64 g(3);
    width<float, 4>(g);
    width<float, 8>(g);
    width<float, 16>(g);
    width<float, 32>(g);
    width<double, 2>(g);
    width<double, 4>(g);
    width<double, 8>(g);
    width<double, 16>(g);
    masks();
    denormal_guard();
    rounding_modes<float, 4>();
    rounding_modes<float, 8>();
    rounding_modes<double, 2>();
    rounding_modes<double, 4>();
    return wray::test::finish("simd");
}
