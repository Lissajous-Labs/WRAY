# WRAY

(Wray & Nephew is a high-quality rum; you are highly advised to mix it with juice (JUCE))!

Fast, verified math for audio DSP: header-only C++20, SIMD (NEON, SSE2, AVX2, AVX-512, scalar), `float` and `double`.

These kernels are specifically the fast math code in our shipping plugins, Chronos and Torus. Here every core is regenerated from first principles with a rational Remez solver and re-verified against 40-digit arithmetic.

```cpp
#include <wray/wray.h>

float  a = wray::exp(0.5f);
double b = wray::atan2(1.0, -2.0);
auto   v = wray::tanh(wray::f32x4(0.25f));
wray::sin(input, output);
float  c = wray::fast::tanh(x);
float  s = wray::sin_turns(phase);
float  k = wray::smoothing_coefficient(0.010f, 48000.0f);
wray::adaa::first_order<wray::adaa::tanh, float> drive;
wray::filters::biquad<wray::f64x2> eq;
```

Every function takes a scalar, a SIMD vector, or a contiguous range:

- elementary: `exp exp2 expm1 log log2 log10 log1p pow cbrt sin cos sincos tan atan atan2 sinh cosh tanh asinh erf sigmoid`
- turns, where one turn is a full cycle: `sin_turns cos_turns sincos_turns tan_turns wrap_turns`
- audio: `db_to_gain gain_to_db midi_to_hz hz_to_midi ratio_from_cents omega product_log`
- parameter maps: `smoothing_coefficient(seconds, fs) pole_from_cutoff(hz, fs) prewarp(hz, fs)`

`wray::fast::` is the `float` tier with a 2^-16 relative-error contract for `exp exp2 log log2 sin cos sincos tan atan atan2 tanh sigmoid pow db_to_gain gain_to_db` and the turns, `omega` and `product_log` functions.

![speedup](docs/charts/speedup.png)

## Accuracy contracts

Maximum error in ulp against 40-digit references, over the full domain.

| function | float | double | function | float | double |
|---|---|---|---|---|---|
| exp | 1.0 | 1.0 | sin | 1.0 | 1.0 |
| exp2 | 1.0 | 1.0 | cos | 1.0 | 1.0 |
| expm1 | 2.0 | 1.75 | tan | 1.75 | 1.45 |
| log | 1.0 | 1.0 | atan | 1.0 | 1.1 |
| log2 | 1.5 | 1.0 | atan2 | 1.0 | 1.1 |
| log10 | 1.4 | 1.0 | sin_turns | 1.0 | 0.85 |
| log1p | 1.0 | 0.95 | cos_turns | 0.8 | 0.8 |
| pow | 2.0 | 1.5 | tan_turns | 1.2 | 1.2 |
| cbrt | 0.55 | 0.55 | sinh | 0.85 | 0.85 |
| erf | 0.9 | 0.9 | cosh | 0.9 | 0.9 |
| sigmoid | 2.0 | 1.75 | tanh | 1.35 | 1.3 |
| db_to_gain | 1.5 | 1.5 | asinh | 0.9 | 0.85 |
| gain_to_db | 1.5 | 1.5 | omega | 1.45 | 1.3 |
| midi_to_hz | 1.4 | 0.75 | product_log | 2.05 | 1.45 |
| hz_to_midi | 1.25 | 1.0 | smoothing_coefficient | 2.0 | 2.0 |
| ratio_from_cents | 0.85 | 0.75 | pole_from_cutoff | 2.0 | 2.0 |
| prewarp | 2.0 | 2.0 | | | |

Beyond the error bound, the precise tier keeps shape: every function is checked for monotonicity on its monotone intervals, exact odd or even symmetry, range bounds, and finite outputs for finite inputs, over every `float` and a dense `double` sample. The few documented exceptions are listed in `tests/accuracy_test.cpp`. `midi_to_hz(69 + 12k)` is exactly `440 * 2^k`, and the turns functions are exact at multiples of 1/8 turn.

![accuracy](docs/charts/accuracy_summary.png)

## ADAA

`wray::adaa` has first-order antiderivative antialiasing for `hard_clip`, `soft_clip` (cubic), `tanh`, `atan`, `algebraic` (x/(1+|x|)) and `radical` (x/sqrt(1+x^2)). Each shape has `f`, its antiderivative `F1`, and `first(x0, x1)`, the mean of `f` over `[x0, x1]`. `first` is accurate for every input pair, including `x0 == x1`, nearly equal inputs and huge inputs, with no ill-conditioned difference quotient.

| shape | F1 (ulp) | first (ulp) | shape | F1 (ulp) | first (ulp) |
|---|---|---|---|---|---|
| hard_clip | 0.8 | 3.0 | atan | 3.5 | 15.0 |
| soft_clip | 1.5 | 8.0 | algebraic | 6.0 | 62.0 |
| tanh | 3.0 | 10.0 | radical | 3.0 | 5.0 |

![adaa](docs/charts/adaa_aliasing.png)

## Parameter maps

`smoothing_coefficient`, `pole_from_cutoff` and `prewarp` stay within 2 ulp for every time constant and cutoff down to 1e-9 of the sample rate, where the textbook formulas lose every digit.

![smoothing](docs/charts/smoothing_precision.png)

## Filters

`wray::filters` has `tpt_one_pole`, `svf` and `biquad` (lowpass, highpass, bandpass, notch, peak, shelves, allpass, and tilt for the SVF), designed with half-angle formulas in double and run in `float`, `double` or any SIMD width. Parameters are clamped rather than trusted: cutoff to [1e-6, 0.4999] of the sample rate, Q to [1e-3, 1e3], gain to ±120 dB with NaN read as 0 dB, and a non-finite or non-positive sample rate falls back to 48 kHz. Realised coefficients are kept strictly stable in the processing precision, and state is flushed to exact zero once it falls below 2^-80 (float) or 2^-200 (double), so silence never produces subnormals.

A deep sweep checks random and edge parameters from 8 kHz to 768 kHz against the exact bilinear prototype, scaled by the condition of each response.

## Backends and determinism

NEON, SSE2, AVX2 (FMA), AVX-512 and a scalar backend. Scalar and SIMD results are bit-identical. Results depend only on whether the build has FMA: golden digests in `tests/golden` pin every function's output for the FMA class (NEON, AVX2, AVX-512) and the no-FMA class (SSE2, scalar), across compilers. `WRAY_STRICT_FP` (on by default) stops consumers' compilers from contracting the library's arithmetic.

![latency](docs/charts/latency.png)

## How it is verified

- Exhaustive: every `float` input (all 2^32 bit patterns) for each unary function, `wray_accuracy_test --exhaustive`.
- Dense: every function and tier against mpmath at 40 digits, `tools/oracle/oracle.py --check`.
- Shape: monotonicity, symmetry, bounds and finiteness, exhaustive in `float` and sampled in `double`.
- Host: every function again under flush-to-zero and in all four rounding modes.
- Determinism: golden output digests per FMA class, and bit-identical scalar and SIMD results on every backend.
- Spectral: oscillator SFDR, ADAA aliasing and parameter-map precision each assert a recorded limit (`wray_spectral_test`).

```sh
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
```

![sfdr](docs/charts/oscillator_sfdr.png)

## Where it stops working

- Builds without FMA (SSE2, scalar) add 0.5 ulp to every precise contract.
- `sin`, `cos`, `sincos` and `tan` reduce exactly for |x| ≤ 6434 (float) or 6.59e6 (double) and return NaN beyond. The fast tier has no check and returns unreduced garbage past that range.
- `exp` overflows to infinity above 88.72 (float) or 709.78 (double) and returns 0 below -103.97 or -745.13. `exp2` overflows at 128 or 1024 and returns 0 from -150 or -1075 down. `pow` overflows when y·log2|x| reaches 128 or 1024. `db_to_gain` overflows above 770.64 dB (float) or 6165.09 dB (double) and returns 0 below -903.09 dB or -6472.14 dB.
- The fast tier clamps instead: `exp` to [-87, 88], `exp2` and `pow` to [-126, 127], and `tanh` is exactly ±1 from 5.9. It never returns infinities, zeros from underflow, or subnormals.
- The turns functions are exact everywhere: the reduction drops whole turns without rounding, so `sin_turns(2^40 + 0.25)` is exactly 1.
- Filters clamp their parameters as described above and recover from NaN input after `reset()`.
- Under flush-to-zero the contracts hold for normal inputs and outputs; subnormals read and write as zero. In directed rounding modes the error grows by at most 1 ulp.
- `-ffast-math` is rejected with `#error`, since compensated arithmetic needs IEEE rounding. Define `WRAY_ALLOW_FAST_MATH` to build anyway, at your own risk.

## Rational minimax cores

Each core is the true minimax approximation on its reduced interval, so its error equioscillates. Rational or polynomial is chosen per core by cost at the target accuracy.

![remez](docs/charts/remez.png)

![pareto](docs/charts/pareto.png)

More: [float](docs/charts/accuracy_f32.png), [double](docs/charts/accuracy_f64.png), [fast](docs/charts/accuracy_fast.png), [trig range](docs/charts/trig_range.png), [float throughput](docs/charts/throughput_f32.png), [double throughput](docs/charts/throughput_f64.png).

## Regenerate

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/remez/generate.py --header include/wray/math/coefficients.h --report docs/data/cores.json
.venv/bin/python tools/oracle/oracle.py --dump build/tools/wray_dump --out docs/data/accuracy.json --samples 50000 --check
build/bench/wray_bench docs/data/bench.json
build/tests/wray_spectral_test --json docs/data/spectral.json
.venv/bin/python tools/charts/charts.py --machine "your machine"
.venv/bin/python tools/docs/check_readme.py --dump build/tools/wray_dump
```

## License

GNU General Public License v2.0 or later (GPL-2.0-or-later). See [LICENSE](LICENSE).
