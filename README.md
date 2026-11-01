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

- elementary: `exp exp2 expm1 log log2 log10 log1p pow cbrt sin cos sincos tan tanpi atan atan2 sinh cosh tanh asinh erf sigmoid`, where `tanpi(x)` is tan(πx)
- turns, where one turn is a full cycle: `sin_turns cos_turns sincos_turns tan_turns wrap_turns`, so `tan_turns(x)` is tan(2πx) and equals `sin_turns(x) / cos_turns(x)`
- audio: `db_to_gain gain_to_db midi_to_hz hz_to_midi ratio_from_cents omega product_log`
- parameter maps: `smoothing_coefficient(seconds, fs) pole_from_cutoff(hz, fs) prewarp(hz, fs)`

`wray::fast::` is the `float` tier with a 2^-16 relative-error contract for `exp exp2 log log2 sin cos sincos tan atan atan2 tanh sigmoid pow db_to_gain gain_to_db` and the turns, `omega` and `product_log` functions.

![speedup](docs/charts/speedup.png)

## Use it

Header-only, C++20. Either vendor it:

```cmake
add_subdirectory(WRAY)
target_link_libraries(your_target PRIVATE wray::wray)
```

or install it and use the CMake package:

```sh
cmake -S WRAY -B build -DCMAKE_INSTALL_PREFIX=/your/prefix && cmake --install build
```

```cmake
find_package(wray 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE wray::wray)
```

`WRAY_SIMD` (`auto`, `scalar`, `sse2`, `avx2`, `avx512`) picks the x86 instruction set and travels with the `wray::wray` target.

## Accuracy contracts

Measured maximum error in ulp. These are measured bounds, not proofs: one-argument `float` functions are checked on every input against a double-precision reference, and every function is sampled against 40-digit mpmath (50,000 points per function and tier in the published data). `double` and two-argument functions are only sampled.

| function | float | double | function | float | double |
|---|---|---|---|---|---|
| exp | 1.0 | 1.0 | sin | 1.0 | 1.0 |
| exp2 | 1.0 | 1.0 | cos | 1.0 | 1.0 |
| expm1 | 2.0 | 1.75 | tan | 1.75 | 1.45 |
| log | 1.0 | 1.0 | atan | 1.0 | 1.1 |
| log2 | 1.5 | 1.0 | atan2 | 1.0 | 1.1 |
| log10 | 1.4 | 1.0 | sin_turns | 1.0 | 0.85 |
| log1p | 1.0 | 0.95 | cos_turns | 0.8 | 0.8 |
| pow | 2.0 | 1.5 | tanpi | 1.2 | 1.2 |
| cbrt | 0.55 | 0.55 | sinh | 0.85 | 0.85 |
| erf | 0.9 | 0.9 | cosh | 0.9 | 0.9 |
| sigmoid | 2.0 | 1.75 | tanh | 1.35 | 1.3 |
| db_to_gain | 1.5 | 1.5 | asinh | 0.9 | 0.85 |
| gain_to_db | 1.5 | 1.5 | omega | 1.65 | 1.3 |
| midi_to_hz | 1.4 | 0.75 | product_log | 2.05 | 1.45 |
| hz_to_midi | 1.25 | 1.1 | smoothing_coefficient | 2.0 | 2.0 |
| ratio_from_cents | 0.85 | 0.75 | pole_from_cutoff | 2.0 | 2.0 |
| prewarp | 2.0 | 2.0 | tan_turns | 1.2 | 1.2 |

`pow` meets its bound while |y·log₂|x|| ≤ 4; past that the bound scales with the product, to contract × |y·log₂|x|| / 4 ulp, because the error of log₂|x| is multiplied by y. Negative bases follow C: integer exponents give the signed power and other exponents give NaN. The fast tier's 2^-16 bound for `pow` is likewise relative to 1 + |y·log₂|x||.

Beyond the error bound, the precise tier keeps shape: every function is checked for monotonicity on its monotone intervals, exact odd or even symmetry, range bounds, and finite outputs for finite inputs, over every `float` and a dense `double` sample. The few documented exceptions are listed in `tests/accuracy_test.cpp`. `midi_to_hz(69 + 12k)` is exactly `440 * 2^k`, and at multiples of 1/8 turn the turns functions return the correctly rounded value.

![accuracy](docs/charts/accuracy_summary.png)

## ADAA

`wray::adaa` has first-order antiderivative antialiasing for `hard_clip`, `soft_clip` (cubic), `tanh`, `atan`, `algebraic` (x/(1+|x|)) and `radical` (x/sqrt(1+x^2)). Each shape has `f`, its antiderivative `F1`, and `first(x0, x1)`, the mean of `f` over `[x0, x1]`. `first` is accurate for every input pair, including `x0 == x1`, nearly equal inputs and huge inputs, with no ill-conditioned difference quotient.

| shape | F1 (ulp) | first (ulp) | shape | F1 (ulp) | first (ulp) |
|---|---|---|---|---|---|
| hard_clip | 0.8 | 1.0 | atan | 3.5 | 15.0 |
| soft_clip | 1.5 | 3.0 | algebraic | 6.0 | 62.0 |
| tanh | 3.0 | 10.0 | radical | 3.0 | 5.0 |

![adaa](docs/charts/adaa_aliasing.png)

## Parameter maps

`smoothing_coefficient`, `pole_from_cutoff` and `prewarp` stay within 2 ulp for every time constant and cutoff down to 1e-9 of the sample rate, where the textbook formulas lose every digit.

![smoothing](docs/charts/smoothing_precision.png)

## Filters

`wray::filters` has `tpt_one_pole`, `svf` and `biquad` (lowpass, highpass, bandpass, notch, peak, shelves, allpass, and tilt for the SVF), designed with half-angle formulas in double and run in `float`, `double` or any SIMD width. Parameters are clamped rather than trusted: cutoff to [1e-6, 0.4999] of the sample rate, Q to [1e-3, 1e3], gain to ±120 dB with NaN read as 0 dB, and a non-finite or non-positive sample rate falls back to 48 kHz. Realised coefficients are kept strictly stable in the processing precision, and state is flushed to exact zero once it falls below 2^-80 (float) or 2^-200 (double), so silence never produces subnormals.

`biquad<float>` rounds its coefficients to float, which moves the poles of very low cutoffs: it holds DC and cutoff gain within 0.5% only for cutoffs of at least fs/1000, and at 768 kHz a 20 Hz lowpass has a DC gain of 0.22. For lower cutoffs use `biquad<float, double>`, which keeps double coefficients and state for float audio, or `svf<float>`; both stay within 1e-5 down to 1e-6 of the sample rate.

A deep sweep checks random and edge parameters from 8 kHz to 768 kHz against the exact bilinear prototype, scaled by the condition of each response.

## Backends and determinism

NEON, SSE2, AVX2 (FMA), AVX-512 and a scalar backend. Scalar and SIMD results are bit-identical. Results depend only on whether the build has FMA: golden digests in `tests/golden` pin every function's output for the FMA class (NEON, AVX2, AVX-512) and the no-FMA class (SSE2, scalar), across compilers. `WRAY_STRICT_FP` (on by default) stops consumers' compilers from contracting the library's arithmetic. Every header lives in an inline namespace named after its backend and FMA class (for example `wray::abi_avx2_fma`), so translation units compiled with different SIMD flags can share one program without the linker merging their code.

![latency](docs/charts/latency.png)

Over buffers the SIMD loops are several times faster than the standard library. For a single value inside a dependency chain, such as a feedback loop, the precise tier is often slower than the platform libm: its compensated steps and special-value handling lengthen the chain, and evaluating scalars without the SIMD carrier made no consistent difference. `docs/data/bench.json` records the machine, compiler, flags and backend, the fastest, median and 90th-percentile round for every row, and block timings for the filters, coefficient updates and each ADAA shape against its pointwise version.

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
- `sin`, `cos`, `sincos` and `tan` use a three-part reduction for |x| ≤ 6434 (float) or 6.59e6 (double). Larger finite inputs take a slower exact reduction against a 1280-bit table of 2/π, lane by lane, so every finite input gets an accurate result and never NaN; only infinities and NaN return NaN. The fast tier uses the same reductions, so it stays finite and within its 2^-16 bound for every finite input. For oscillators and phase accumulators, keep the phase in turns and use `sin_turns`, `cos_turns` and `sincos_turns`: their reduction is exact and cheap at any magnitude, so a growing phase never reaches the slow path.
- `exp` overflows to infinity above 88.72 (float) or 709.78 (double) and returns 0 below -103.97 (float) or -745.13 (double). `exp2` overflows at 128 (float) or 1024 (double) and returns 0 from -150 (float) or -1075 (double) down. `pow` overflows when y·log₂|x| reaches 128 (float) or 1024 (double). `db_to_gain` overflows above 770.64 dB (float) or 6165.09 dB (double) and returns 0 below -903.09 dB (float) or -6472.14 dB (double).
- The fast tier exists for `float` only, so there are no double fast clamps. It clamps where the precise tier overflows: `exp` clamps its argument to [-87, 88], `exp2`, `pow` and `db_to_gain` clamp the base-2 exponent to [-126, 127], so these four never return infinities, zeros from underflow, or subnormals. Fast `tanh` is exactly ±1 from 5.9. The other fast functions do not clamp: fast `log` of 0 returns -88.03 rather than -infinity, and fast `sigmoid` can return subnormals for large negative inputs. Every fast function returns NaN for NaN input, and fast `sin`, `cos` and `tan` return NaN for infinities. Fast `log`, `log2` and `gain_to_db` return NaN below zero, give -0 the same result as 0, and read +infinity as 2^128 (`log` gives 88.72). Fast `pow` returns NaN when either argument is NaN, including `pow(x, 0)`, and NaN for a negative base with a non-integer exponent; it reads 0 as 2^-127 and infinity as 2^128 before clamping, so `pow(0, 2)` is the smallest normal float rather than 0.
- The turns functions' reduction is exact for all inputs: whole turns are removed without rounding, and the result carries only the polynomial's error. For example, in double, `sin_turns(2^40 + 0.25)` is exactly 1; that input is not representable in float.
- Filters clamp their parameters as described above and recover from NaN input after `reset()`.
- Under flush-to-zero the measured errors stay within contract for normal inputs and outputs; subnormals read and write as zero. In the directed rounding modes the measured error grows by at most 1 ulp.
- Fast floating-point models break the compensated arithmetic. The headers stop with `#error` when the compiler announces one: `-ffast-math` or clang's `-ffp-model=fast` (`__FAST_MATH__`), MSVC `/fp:fast` (`_M_FP_FAST`), `-ffinite-math-only`, and on GCC `-fassociative-math`, `-freciprocal-math` and `-fno-signed-zeros`. Clang's individual `-fassociative-math` and `-freciprocal-math`, and FP contraction on any compiler, define no macro; Intel icx defaults to a fast model, and if it defines none of these macros only the runtime check will see it. `wray::floating_point_ok()` runs that check in the calling translation unit's floating-point model: rounding of sums and products, no reassociation, exact division, no contraction, NaN and signed zeros. On clang the headers turn contraction and reassociation off for their own code with `#pragma clang fp`, so clang's default contraction leaves wray's results unchanged; clang ignores that pragma under `-ffp-contract=fast`, and GCC has no such pragma for C++, so there the check reports contraction. Linking `wray::wray` adds `-ffp-contract=off`. Builds that evaluate float arithmetic in extended precision (`FLT_EVAL_METHOD` other than 0, as x87 on 32-bit x86) stop with `#error`. Define `WRAY_ALLOW_FAST_MATH` to build anyway, at your own risk.

## Rational minimax cores

Each core is fitted by a rational Remez exchange on its reduced interval. `docs/data/cores.json` records, per core, whether the exchange levelled the weighted error to within 1e-9 of equioscillation; a fit that did not converge is kept as the best candidate found and marked `"converged": false`. Rational or polynomial is chosen per core by cost at the target accuracy.

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
