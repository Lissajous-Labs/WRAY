# WRAY details

The full reference behind the [README](../README.md). `tools/docs/check_readme.py` checks the numbers on both pages against the library.

## Accuracy contracts

The contracts are measured bounds, not proofs: one-argument `float` functions are checked on every input against a double-precision reference, and every function is sampled against 40-digit mpmath (50,000 points per function and tier in the published data). `double` and two-argument functions are only sampled; the `double` bounds come from 4×10^8 inputs per function against a quad-precision reference in both FMA classes, and a weekly run with a new seed checks them again.

`pow` meets its bound while |y·log₂|x|| ≤ 4; past that the bound scales with the product, to contract × |y·log₂|x|| / 4 ulp, because the error of log₂|x| is multiplied by y. Negative bases follow C: integer exponents give the signed power and other exponents give NaN. The fast tier's 2^-16 bound for `pow` is likewise relative to 1 + |y·log₂|x||.

Beyond the error bound, the precise tier keeps shape: every function is checked for monotonicity on its monotone intervals, exact odd or even symmetry, range bounds, and finite outputs for finite inputs, over every `float` and a dense `double` sample. The few documented exceptions are listed in `tests/accuracy_test.cpp`. `midi_to_hz(69 + 12k)` is exactly `440 * 2^k`, and at multiples of 1/8 turn the turns functions return the correctly rounded value.

## Parameter maps

`smoothing_coefficient`, `pole_from_cutoff` and `prewarp` stay within 2 ulp for every time constant and cutoff down to 1e-9 of the sample rate. Outside their domain they return safe values:

| input | `smoothing_coefficient` | `pole_from_cutoff` | `prewarp` |
|---|---|---|---|
| time constant or cutoff ≤ 0 | 1 | 1 | 0 |
| sample rate ≤ 0 | 1 | 0 | largest value |
| cutoff ≥ Nyquist | | | ratio just below 0.5: 1.07e7 (float), 5.73e15 (double) |
| NaN | NaN | NaN | NaN |

No input gives a pole above 1 or a coefficient outside [0, 1].

## Filters

`tpt_one_pole`, `svf` and `biquad` cover lowpass, highpass, bandpass, notch, peak, shelves, allpass, and tilt for the SVF. They are designed with half-angle formulas in double and run in `float`, `double` or any SIMD width.

- Parameters are clamped: cutoff to [1e-6, 0.4999] of the sample rate, Q to [1e-3, 1e3], gain to ±120 dB. A non-finite or non-positive sample rate falls back to 48 kHz.
- A NaN cutoff, Q or gain keeps that parameter's previous value. A new filter starts at 1 kHz, Q 0.7071 and 0 dB.
- With 0 dB gain, inputs up to an eighth of the format's maximum keep every filter's state and output finite.
- Coefficients are kept strictly stable in the processing precision. State is flushed to exact zero below 2^-80 (float) or 2^-200 (double), so silence never produces subnormals.
- Filters recover from NaN input after `reset()`.

`biquad<float>` rounds its coefficients to float, which moves the poles of very low cutoffs: it holds DC and cutoff gain within 0.5% only for cutoffs of at least fs/1000, and at 768 kHz a 20 Hz lowpass has a DC gain of 0.22. For lower cutoffs use `biquad<float, double>`, which keeps double coefficients and state for float audio, or `svf<float>`; both stay within 1e-5 down to 1e-6 of the sample rate.

A deep sweep checks 4000 random parameter sets per filter type and precision, plus edge parameters, from 8 kHz to 768 kHz against the exact bilinear prototype, scaled by the condition of each response. With cutoff, Q and gain on a random walk across their whole ranges and changed every sample, `svf` stays below 1e8 and `tpt_one_pole` below 4 for inputs within ±1. `biquad` is a direct form: stable for fixed coefficients and slow changes, but per-sample modulation can make it grow without bound, so modulate an `svf` instead.

## Backends and determinism

- NEON covers arm64 with GCC, Clang and MSVC (Windows on ARM).
- The headers compile without warnings under `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wdouble-promotion -Wold-style-cast -Wshadow` and MSVC `/W4`.
- Scalar and SIMD results are bit-identical apart from the sign and payload of NaN results.
- Results depend only on whether the build has FMA. Golden digests in `tests/golden` pin every function's output for the FMA class (NEON, AVX2, AVX-512) and the no-FMA class (SSE2, and scalar on x86), across compilers.
- `WRAY_STRICT_FP` (on by default) stops consumers' compilers from contracting the library's arithmetic.
- Every header lives in an inline namespace named after its backend and FMA class (for example `wray::abi_avx2_fma`), so translation units compiled with different SIMD flags can share one program without the linker merging their code.

For a single value inside a dependency chain, such as a feedback loop, the precise tier is often slower than the platform libm: its compensated steps and special-value handling lengthen the chain. `docs/data/bench.json` records the machine, compiler, flags and backend, the fastest, median and 90th-percentile round for every row, and block timings for the filters, coefficient updates and each ADAA shape against its pointwise version.

## How it is verified

- Exhaustive: every `float` input (all 2^32 bit patterns) for each unary function, `wray_accuracy_test --exhaustive`, weekly on NEON, SSE2, AVX2, AVX-512 and the scalar backend; the same run repeats every fourth input under flush-to-zero and each rounding mode, after reading back that the mode took effect.
- Double: 10^8 inputs per `double` function weekly against a quad-precision reference (`tools/dense`, GCC on x86-64), with a rotating seed and the contracts as the gate; the reference is itself checked against mpmath on every run.
- Dense: every function and tier against mpmath at 40 digits, `tools/oracle/oracle.py --check`.
- Shape: monotonicity, symmetry, bounds and finiteness, exhaustive in `float` and sampled in `double`.
- Host: every function again under flush-to-zero and in all four rounding modes, after reading back that each mode took effect.
- Determinism: golden output digests per FMA class, covering every function, `sincos`, `wrap_turns`, every filter type with a parameter change mid-stream, and the stateful ADAA processors, and bit-identical scalar and SIMD results on every backend apart from NaN sign and payload.
- Spectral: oscillator SFDR, ADAA aliasing and parameter-map precision each assert a recorded limit (`wray_spectral_test`).

## Where it stops working

### Hardware and rounding

- x86 builds without FMA (SSE2, and the scalar backend on x86) add 0.5 ulp to every precise contract; arm64 always has FMA, scalar backend included.
- Under flush-to-zero the measured errors stay within contract for normal inputs and outputs; subnormals read and write as zero.
- In the directed rounding modes the measured error grows by at most 1 ulp.

### Large trig arguments

`sin`, `cos`, `sincos` and `tan` use a fast three-part reduction up to:

| | float | double |
|---|---|---|
| fast reduction for \|x\| ≤ | 6433.98 (2048π) | 6588397.3 (2^22·π/2) |

Larger finite inputs take a slower exact reduction against a 1280-bit table of 2/π, lane by lane. Every finite input still gets an accurate result; only infinities and NaN return NaN. The fast tier uses the same reductions and stays within 2^-16 for every finite input.

The turns functions reduce exactly at any magnitude: whole turns are removed without rounding, so the result carries only the polynomial's error. For example, in double, `sin_turns(2^40 + 0.25)` is exactly 1; that input is not representable in float. For oscillators and phase accumulators, keep the phase in turns and use `sin_turns`, `cos_turns` and `sincos_turns`, so a growing phase never reaches the slow path.

### Overflow and underflow

| function | finite up to | infinite from | nonzero down to | 0 from |
|---|---|---|---|---|
| `exp` float | 88.72 | 88.73 | -103.97 | -103.98 |
| `exp` double | 709.78 | 709.79 | -745.13 | -745.14 |
| `exp2` float | just below 128 | 128 | just above -150 | -150 |
| `exp2` double | just below 1024 | 1024 | just above -1075 | -1075 |
| `db_to_gain` float (dB) | 770.63 | 770.64 | -903.08 | -903.09 |
| `db_to_gain` double (dB) | 6165.09 | 6165.10 | -6472.14 | -6472.15 |

`pow` overflows when y·log₂|x| reaches 128 (float) or 1024 (double).

### Fast tier

The fast tier exists for `float` only.

- Clamped, so they never return infinities, zeros from underflow, or subnormals: `exp` clamps its argument to [-87, 88]; `exp2`, `pow` and `db_to_gain` clamp the base-2 exponent to [-126, 127].
- `tanh` is exactly ±1 from 5.9.
- Not clamped: fast `log` of 0 returns -88.03 rather than -infinity, and fast `sigmoid` can return subnormals for large negative inputs.
- NaN input gives NaN. `sin`, `cos` and `tan` of infinity give NaN.
- `log`, `log2` and `gain_to_db` return NaN below zero, treat -0 as 0, and read +infinity as 2^128 (`log` gives 88.72).
- `pow` returns NaN when either argument is NaN, including `pow(x, 0)`, and for a negative base with a non-integer exponent. It reads 0 as 2^-127 and infinity as 2^128 before clamping, so `pow(0, 2)` is the smallest normal float rather than 0.

### Compiler flags

Fast floating-point models break the compensated arithmetic.

| flag | outcome |
|---|---|
| `-ffast-math`, clang `-ffp-model=fast` (`__FAST_MATH__`), MSVC `/fp:fast` (`_M_FP_FAST`) | `#error` |
| `-ffinite-math-only` | `#error` |
| GCC `-fassociative-math`, `-freciprocal-math`, `-fno-signed-zeros` | `#error` |
| `FLT_EVAL_METHOD` other than 0 (x87 on 32-bit x86) | `#error` |
| clang `-fassociative-math`, `-freciprocal-math`, `-fno-signed-zeros`, `-fapprox-func`, default contraction | no effect on wray's results |
| MSVC `/fp:contract` | no effect on wray's results |
| clang `-ffp-contract=fast`, GCC contraction | changes results; `wray::floating_point_ok()` reports it |
| clang `-fno-honor-nans`, `-fno-honor-infinities` | unsupported |

- On clang the headers wrap their own code in `#pragma float_control(precise, on)` and `#pragma clang fp contract(off) reassociate(off)` and do their vector arithmetic with operators inside that region; MSVC contraction is switched off the same way. Clang ignores the contraction pragma under `-ffp-contract=fast`, and GCC has no such pragma for C++.
- Linking `wray::wray` adds `-ffp-contract=off`.
- `wray::floating_point_ok()` checks rounding of sums and products, no reassociation, exact division, no contraction, NaN, infinity and signed zeros, run in wray's floating-point model.
- `-fno-honor-nans` and `-fno-honor-infinities` make every NaN or infinity in the caller's own code undefined.
- `tools/ci/hostile_flags.sh` checks each outcome, and `tools/ci/header_only.sh` checks consumers that include the headers without the CMake target.
- Define `WRAY_ALLOW_FAST_MATH` to build anyway, at your own risk.

## Rational minimax cores

Each core is fitted by a rational Remez exchange on its reduced interval. `docs/data/cores.json` records, per core, whether the exchange levelled the weighted error to within 1e-9 of equioscillation; a fit that did not converge is kept as the best candidate found and marked `"converged": false`. Rational or polynomial is chosen per core by cost at the target accuracy.
