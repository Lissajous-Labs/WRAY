<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/wray-dark.svg">
    <img alt="WRAY" src="docs/brand/wray-light.svg" width="640">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/Lissajous-Labs/WRAY/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Lissajous-Labs/WRAY/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/Lissajous-Labs/WRAY/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/Lissajous-Labs/WRAY?color=ff5c4a"></a>
  <a href="LICENSE"><img alt="License: GPL-2.0-or-later" src="https://img.shields.io/badge/license-GPL--2.0--or--later-ff5c4a"></a>
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-ff5c4a">
  <img alt="Header-only" src="https://img.shields.io/badge/header--only-yes-ff5c4a">
</p>

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

Every function takes a scalar, a SIMD vector, or a range (any length, no alignment needed, in place allowed):

- elementary: `exp exp2 expm1 log log2 log10 log1p pow cbrt sin cos sincos tan tanpi atan atan2 sinh cosh tanh asinh erf sigmoid`
- turns (one turn is a full cycle): `sin_turns cos_turns sincos_turns tan_turns wrap_turns`
- audio: `db_to_gain gain_to_db midi_to_hz hz_to_midi ratio_from_cents omega product_log`
- parameter maps: `smoothing_coefficient pole_from_cutoff prewarp`

`wray::fast::` is a quicker `float`-only tier, accurate to 2^-16 relative error.

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
find_package(wray 1.0 REQUIRED)
target_link_libraries(your_target PRIVATE wray::wray)
```

`WRAY_SIMD` (`auto`, `scalar`, `sse2`, `avx2`, `avx512`) picks the x86 instruction set and travels with the `wray::wray` target.

## Accuracy contracts

Maximum measured error in ulp (units in the last place). Every `float` input is tested for one-argument functions; the rest are densely sampled. `pow` loosens for very large results. [Details](docs/details.md#accuracy-contracts).

| function | float | double | function | float | double |
|---|---|---|---|---|---|
| exp | 1.0 | 1.0 | sin | 1.0 | 1.0 |
| exp2 | 1.0 | 1.0 | cos | 1.0 | 1.0 |
| expm1 | 2.0 | 1.85 | tan | 1.75 | 1.85 |
| log | 1.0 | 1.0 | atan | 1.0 | 1.3 |
| log2 | 1.5 | 1.15 | atan2 | 1.0 | 1.35 |
| log10 | 1.4 | 1.1 | sin_turns | 1.0 | 0.9 |
| log1p | 1.0 | 1.0 | cos_turns | 0.8 | 0.85 |
| pow | 2.0 | 2.0 | tanpi | 1.2 | 1.4 |
| cbrt | 0.55 | 0.55 | sinh | 0.85 | 0.9 |
| erf | 0.9 | 0.95 | cosh | 0.9 | 0.95 |
| sigmoid | 2.0 | 1.75 | tanh | 1.35 | 1.4 |
| db_to_gain | 1.5 | 1.5 | asinh | 0.9 | 0.95 |
| gain_to_db | 1.5 | 1.5 | omega | 1.65 | 1.4 |
| midi_to_hz | 1.4 | 0.9 | product_log | 2.05 | 1.75 |
| hz_to_midi | 1.25 | 1.2 | smoothing_coefficient | 2.0 | 2.0 |
| ratio_from_cents | 0.85 | 0.9 | pole_from_cutoff | 2.0 | 2.0 |
| prewarp | 2.0 | 2.0 | tan_turns | 1.2 | 1.4 |

![accuracy](docs/charts/accuracy_summary.png)

## ADAA

`wray::adaa` antialiases six waveshapers (`hard_clip`, `soft_clip`, `tanh`, `atan`, `algebraic`, `radical`) with first-order antiderivatives. It stays accurate for any pair of inputs, including equal ones. For mono buffers, `first_order<Shape, float>::process(in, out)` runs consecutive samples in parallel and matches the per-sample call bit for bit.

| shape | F1 (ulp) | first (ulp) | shape | F1 (ulp) | first (ulp) |
|---|---|---|---|---|---|
| hard_clip | 0.8 | 1.0 | atan | 3.5 | 15.0 |
| soft_clip | 1.5 | 3.0 | algebraic | 6.0 | 62.0 |
| tanh | 3.0 | 10.0 | radical | 3.0 | 5.0 |

![adaa](docs/charts/adaa_aliasing.png)

## Parameter maps

`smoothing_coefficient`, `pole_from_cutoff` and `prewarp` stay within 2 ulp even at tiny cutoffs where the textbook formulas lose every digit, and return safe values for bad input. [Details](docs/details.md#parameter-maps).

![smoothing](docs/charts/smoothing_precision.png)

## Filters

`wray::filters` has `tpt_one_pole`, `svf` and `biquad`, in any precision or SIMD width. Bad parameters are clamped or ignored, and silence never produces subnormals. For very low cutoffs in `float`, use `svf<float>` or `biquad<float, double>`; for per-sample modulation, use `svf`. [Details](docs/details.md#filters).

## Backends and determinism

NEON, SSE2, AVX2, AVX-512 and scalar backends. Scalar and SIMD code give bit-identical results, except for NaN sign and payload; only builds with and without FMA differ, slightly. [Details](docs/details.md#backends-and-determinism).

![latency](docs/charts/latency.png)

Over buffers the SIMD loops are several times faster than the standard library. A single value in a feedback loop can be slower than libm.

## How it is verified

Every `float` input, 10^8 `double` inputs per function, 40-digit reference checks, shape and determinism checks, and spectral tests. [Details](docs/details.md#how-it-is-verified).

```sh
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
```

![sfdr](docs/charts/oscillator_sfdr.png)

## Where it stops working

- x86 without FMA (SSE2, scalar on x86) adds 0.5 ulp to every contract.
- `sin`, `cos` and `tan` slow down for huge arguments but stay accurate. For oscillators, use the turns functions, which stay fast at any phase.
- `exp`, `exp2`, `pow` and `db_to_gain` overflow and underflow where the format runs out.
- The fast tier is `float` only and clamps instead of overflowing.
- Fast-math compiler flags stop the build with `#error`.

Exact thresholds and flag-by-flag behavior: [details](docs/details.md#where-it-stops-working).

## Rational minimax cores

Each core is fitted by a rational Remez exchange; `docs/data/cores.json` records how well each fit converged. [Details](docs/details.md#rational-minimax-cores).

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
.venv/bin/python tools/brand/logo.py --font path/to/ClashGrotesk-Semibold.ttf
```

## License

GNU General Public License v2.0 or later (GPL-2.0-or-later). See [LICENSE](LICENSE).
