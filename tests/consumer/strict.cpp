#include <wray/wray.h>

#include <array>
#include <cstdio>
#include <span>

namespace
{
    template <class V>
    V precise(const V &x)
    {
        V s = wray::exp(x) + wray::expm1(x) + wray::exp2(x) + wray::log(x) + wray::log1p(x) + wray::log10(x) + wray::log2(x);
        s = s + wray::wrap_turns(x) + wray::sin_turns(x) + wray::cos_turns(x) + wray::tanpi(x) + wray::tan_turns(x);
        s = s + wray::sin(x) + wray::cos(x) + wray::tan(x) + wray::atan(x) + wray::sinh(x) + wray::cosh(x) + wray::asinh(x);
        s = s + wray::cbrt(x) + wray::erf(x) + wray::midi_to_hz(x) + wray::hz_to_midi(x) + wray::ratio_from_cents(x);
        s = s + wray::omega(x) + wray::product_log(x) + wray::tanh(x) + wray::sigmoid(x) + wray::db_to_gain(x) + wray::gain_to_db(x);
        s = s + wray::smoothing_coefficient(x, x) + wray::pole_from_cutoff(x, x) + wray::prewarp(x, x) + wray::atan2(x, x) + wray::pow(x, x);
        const auto sc = wray::sincos(x);
        const auto st = wray::sincos_turns(x);
        return s + sc.sin + sc.cos + st.sin + st.cos;
    }

    template <class V>
    V fast(const V &x)
    {
        V s = wray::fast::exp(x) + wray::fast::exp2(x) + wray::fast::log(x) + wray::fast::log2(x) + wray::fast::sin_turns(x);
        s = s + wray::fast::sin(x) + wray::fast::cos_turns(x) + wray::fast::cos(x) + wray::fast::tanpi(x) + wray::fast::tan_turns(x);
        s = s + wray::fast::tan(x) + wray::fast::atan(x) + wray::fast::atan2(x, x) + wray::fast::omega(x) + wray::fast::product_log(x);
        s = s + wray::fast::tanh(x) + wray::fast::sigmoid(x) + wray::fast::pow(x, x) + wray::fast::db_to_gain(x) + wray::fast::gain_to_db(x);
        const auto sc = wray::fast::sincos(x);
        const auto st = wray::fast::sincos_turns(x);
        return s + sc.sin + sc.cos + st.sin + st.cos;
    }

    template <class Shape, class V>
    V shaped(const V &x)
    {
        wray::adaa::first_order<Shape, V> stage;
        const V y = stage(x) + Shape::f(x) + Shape::F1(x) + Shape::first(x, x + x);
        stage.reset();
        return y + stage(x);
    }

    template <class V>
    V shapes(const V &x)
    {
        return shaped<wray::adaa::hard_clip>(x) + shaped<wray::adaa::soft_clip>(x) + shaped<wray::adaa::tanh>(x) + shaped<wray::adaa::atan>(x) +
               shaped<wray::adaa::algebraic>(x) + shaped<wray::adaa::radical>(x);
    }

    template <class V, class P = V>
    V filtered(const V &x)
    {
        std::array<V, 8> block{};
        block.fill(x);
        wray::filters::tpt_one_pole<V> one;
        one.prepare(48000.0);
        one.set(wray::filters::one_pole_type::lowpass, 1000.0);
        one.process(std::span<V>(block));
        wray::filters::svf<V> state;
        state.prepare(48000.0);
        state.set(wray::filters::svf_type::lowpass, 1000.0, 0.7, 0.0);
        state.process(std::span<const V>(block), std::span<V>(block));
        wray::filters::biquad<V, P> eq;
        eq.prepare(48000.0);
        eq.set(wray::filters::biquad_type::peak, 1000.0, 0.7, 3.0);
        eq.process(std::span<V>(block));
        one.reset();
        state.reset();
        eq.reset();
        return block[7] + one.tick(x) + state.tick(x) + eq.tick(x);
    }

    template <class V>
    double all(const V &x)
    {
        const V r = precise(x) + wray::balanced::pow(x, x) + shapes(x) + filtered(x);
        return static_cast<double>(wray::simd::first(r));
    }
}

int main()
{
    const wray::scoped_flush_denormals flush;
    std::array<float, 64> in{};
    std::array<float, 64> out{};
    in.fill(0.5f);
    wray::exp(in, out);
    wray::fast::exp(in, out);
    wray::balanced::pow(in, in, out);
    double sum = static_cast<double>(out[0]) + all(0.5f) + all(0.5) + all(wray::f32x4(0.5f)) + all(wray::f64x2(0.5));
    sum += all(wray::simd::native<float>(0.5f)) + all(wray::simd::native<double>(0.5));
    sum += static_cast<double>(fast(0.5f)) + static_cast<double>(wray::simd::first(fast(wray::simd::native<float>(0.5f))));
    sum += static_cast<double>(filtered<float, double>(0.5f));
    std::printf("%d %g\n", int(wray::floating_point_ok()), sum);
    return 0;
}
