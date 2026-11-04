# Changelog

All notable changes to WRAY are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow the [stability policy](docs/details.md#stability).

## [Unreleased]

### Added

- `wray::balanced::{sin,cos,sincos}` for scalar, SIMD and range calls in `float` and `double`, plus matching `wray_balanced_<sin|cos|sincos>_<f32|f64>` C buffer calls. Round-to-nearest targets 2.5 ulp per output and directed rounding 8 ulp, plus 0.5 ulp without FMA. Every finite input is supported; paired `sincos` may differ from separate calls. Precise and fast behavior is unchanged (#93).
- `wray::balanced::tanh` for `float` and `double`, scalar/SIMD/ranges and the C ABI, with a 3-ulp nearest bound (+0.5 without FMA). Its exponential reconstruction avoids small-input cancellation and rounding-induced saturation dips. Exhaustive finite-float checks passed; M1 Max buffer throughput measured 1.084× vForce for float and 0.937× for double, while scalar latency remains slower than WRAY precise (#98).
- `wray::balanced::pow` for full-domain `float` and `double`, scalar/SIMD/ranges and `wray_balanced_pow_f32`/`wray_balanced_pow_f64` C buffer calls, using division-free ordinary-range mantissa logarithm cores and a compensated overflow-boundary path. Bounds are 4 ulp nearest and 8 ulp directed, scaled by `max(1, abs(y * log2(abs(x))) / 4)`, with 0.5 added to the base bound without FMA. Negative-base integer powers and round-to-nearest zero/infinity/NaN semantics follow precise `pow`; balanced zero/infinity special cases remain exact in directed rounding. Existing precise/fast guarantees are unchanged (#101).
- `wray_bench` measures SLEEF 3.9 (1.0- and 3.5-ulp tiers) and glibc libmvec at each backend's native vector width when they are available, checks their outputs against libm before timing, and records which were present. Native x86 CI builds SLEEF from a checksummed release and fails if a Linux run lacks either library; charts now include Linux AVX2 and AVX-512 hosts (#92).
- `adaa::first_order<Shape, T>::process(in, out)` and `process(io)` for scalar `T` run consecutive mono samples in parallel lanes and match per-sample calls bit for bit: 2.9–4.3× faster in float and 1.5–2.1× in double on an M1 Max (#49).
- `sincos(in, sin_out, cos_out)` and `sincos_turns(in, sin_out, cos_out)` produce both range outputs in one SIMD pass, including the float-only fast tier. They stop at the shortest range and support exact input aliasing with either disjoint output (#53).
- `wray_bench --baseline libm|vforce` pins its denominator, measures both on Apple, disables scalar-loop auto-vectorization without disabling fused scalar calls, and records paired ABBA samples, bootstrap intervals and a self-control (#54, #35).
- Manual CI can collect native SSE2, AVX2 and AVX-512 timings on Linux and Windows, with runtime ISA checks, per-host provenance, separate artifacts and an aggregate coverage gate. Unsupported hardware produces no timing rather than an emulated or compile-only result (#13).
- `wray_c`, an optional compiled library built with `-DWRAY_BUILD_C_API=ON` with a C ABI (`include/wray/c/wray.h`, CMake target `wray::c`): every range function in `float` and `double` plus the `float` fast tier as `wray_<name>_<f32|f64>` buffer calls, bit-identical to the C++ ranges, with no C++ symbols exported. Verified from C99, Python `ctypes`, Rust and Swift (#73).

### Changed

- `ci-pass` requires success from every job the event and change set should run. It previously failed only on `failure` or `cancelled` results, and passed #107 when both macOS arm64 test jobs were cancelled for lack of a runner (#109).
- A push to `main` runs a single job that checks the pushed commit already passed `ci` in its pull request, instead of no run at all. The README badge reads `main`'s runs, so it had stayed on a stale failure from before #79 (#106).
- Balanced `pow` accuracy and throughput are published in the dark/light figures. High-precision checks passed 250,180 pairs per precision and FMA class; worst scaled nearest error was 3.540 ulp. On M1 Max NEON, ratios of median costs show 25.5%/18.1% higher float/double ordinary-buffer throughput and 19.0%/18.2% lower scalar-chain cost than WRAY precise. It remains below vForce (0.782×/0.654×), below scalar libm in latency, and its subnormal-base edge cost rose about 33%/32% (#101).

- Balanced trig accuracy and throughput are included in the dark/light figures. Every finite float passed on NEON and non-FMA SSE2 with a worst error of 2.454 ulp. M1 Max buffer throughput remains below vForce (0.76–0.95× float, 0.87–0.99× double), and separate-call scalar latency is slower than WRAY precise (#93).
- Charts render on transparent backgrounds in a dark and a light variant, and the README picks the one matching the reader's GitHub theme. Throughput and latency use a log axis so the slowest functions no longer flatten the rest. Series and text colours keep at least 4.5:1 contrast on GitHub's default light and dark pages, and spectra are drawn as a peak-hold envelope rather than every FFT bin, which removes moiré without hiding any spur (#83).
- CI runs AVX-512 tests in three jobs (GCC, Clang, MSVC); each runs natively when its runner has AVX-512 and is compile-only otherwise, and an `avx512-coverage` gate requires at least one native run, so a runner without AVX-512 no longer fails `main` on its own (#71).
- CI no longer reruns the full matrix on pushes to `main`: `main` only fast-forwards to pull request heads that already passed, and the duplicate runs were starving the macOS arm64 runner pool (#79).
- Every public header is compiled on its own in each CI build, so a header that depends on another header's includes fails CI rather than a consumer (#77).
- Positive-base `pow` skips sign/parity work when no lane needs it and uses the bounded log mantissa to shorten an exact compensated sum. Twelve paired ABBA rounds on an M1 Max show 22% higher float throughput, 19% higher double throughput and 5–6% lower dependency-chain cost; the precise contracts and golden outputs are unchanged (#55).
- `sin`, `cos`, `sincos` and `tan` test for huge or non-finite lanes before reduction and keep the ordinary path in registers. Twelve paired ABBA rounds on an M1 Max show 5–15% higher throughput and 9–15% lower dependency-chain cost across float and double, with bit-identical results (#57).
- Regular-range `exp`, `exp2`, `pow` and `db_to_gain` scale their polynomial by adding the shifted exponent to its bits instead of multiplying by a constructed power of two. Eight paired ABBA rounds on an M1 Max show 2–5% higher precise throughput for `exp`, `exp2`, `tanh`, `sigmoid`, `pow` and `db_to_gain`, with bit-identical results (#60).
- `sin`, `cos`, `sincos` and `tan` skip the non-finite unit and apply the quadrant sign with a shift and exclusive-or when every lane is ordinary. Eight paired ABBA rounds on an M1 Max show 9–18% higher precise throughput and 20–31% higher fast-tier throughput, with bit-identical results (#61).
- Regular-path guards test that every lane qualifies rather than that any lane does not, which saves a vector NOT on NEON. Eight paired ABBA rounds on an M1 Max show 11–17% higher precise throughput for `log`, `log2`, `log10`, `log1p`, `gain_to_db` and float `hz_to_midi`, and 2–6% for `exp`, `exp2`, `db_to_gain`, `smoothing_coefficient` and double `hz_to_midi`, with bit-identical results (#63).
- `midi_to_hz` and `ratio_from_cents` skip clamping, the split exponent scale and the non-finite selects when every lane is well inside the normal range, and scale by exponent addition. Eight paired ABBA rounds on an M1 Max show 38–48% higher throughput and 17–26% lower dependency-chain cost in float and double, with bit-identical results. Inputs beyond about 100 octaves (or 1000 in double) now take the outlined general path, which runs at 0.46–0.63× its previous throughput (#65).
- `pole_from_cutoff` and `prewarp` skip their parameter selects when every lane has positive finite parameters and an in-range ratio, and `prewarp` computes the cotangent branch only when some lane is above a quarter of the sample rate. Eight paired ABBA rounds on an M1 Max show 23–25% higher `pole_from_cutoff` throughput and 52–61% higher `prewarp` throughput in float and double, with bit-identical results (#66).
- SSE2 builds without SSE4.1 round `float` vectors to nearest-even by truncating the magnitude and correcting once, instead of a magic-number add with two-sided tie fixups; the result is unchanged in every rounding mode for all 2^32 inputs. Paired native runs on hosted x86 show 4–39% higher `float` throughput across `exp`, `exp2`, trig, `tanh`, `sigmoid`, `pow`, `db_to_gain`, `midi_to_hz`, `cbrt` and `erf` under GCC, Clang and MSVC (#75).
- MSVC x64 builds with `/arch:AVX` or higher use the SSE4.1 rounding instructions, which MSVC exposes without defining `__SSE4_1__`. Paired native runs on hosted Windows show 2–67% lower scalar-call latency across trig, `exp`, `pow`, `tanh`, `db_to_gain` and `midi_to_hz` in float and double; buffer throughput moves by −10% to +5%, with bit-identical results (#76).
- The log reduction offsets the input bits by the √2 boundary before extracting exponent and mantissa, removing a compare, a multiply and two selects: 11–18% higher throughput for `log`, `log2`, `log10` and `gain_to_db` in both tiers, 8–10% for `log1p` and `hz_to_midi`, and 2–8% for `pow` on an M1 Max, with bit-identical results. MSVC on ARM64 keeps the compare form, because its optimizer miscompiles the offset form at `/O2` (#67).

## [1.0.5] - 2026-11-03

### Changed

- `sin`, `cos`, `sincos` and `tan` reduce huge arguments with word-level bit extraction and 64-bit limb products: about 110 ns to 16 ns per element on an M1 Max, with bit-identical results (#39).
- Huge trig extraction uses arithmetic shifts and masks for floor division and remainder by 64. On an M1 Max, huge-input `sin` throughput improves a further 9.8% in float and 6.0% in double; paired bootstrap intervals clear a baseline-vs-baseline control, with unchanged outputs (#51).
- Scalar multiply-adds use `__builtin_elementwise_fma` under clang, so the scalar backend and the huge-argument reduction no longer call libm `fma` (#41).
- Scalar-backend lane broadcasts no longer compile to `memset_pattern16` calls on macOS: scalar-backend float `exp` over a buffer goes from 4.15 ns to 0.87 ns per element on an M1 Max, with bit-identical results (#42).
- NEON `any` and `all` on double masks reduce with one `umaxv`/`uminv`, about 3–5% off double `exp` and `log` (#43).
- `erf` evaluates only the regions some lane needs: buffer throughput 13% faster in float and 22% in double, latency 20% lower, and inputs below 0.5 about 4.5× faster on an M1 Max, with bit-identical results (#44).
- `tanh`, `sinh`, `cosh` and `asinh` evaluate only the regions some lane needs: on an M1 Max, buffer throughput improves by 8% (float) and 25% (double) for `tanh`, 19% and 32% for `asinh`, and 9% for double `cosh`, with bit-identical results (#45).
- `omega` and `product_log` evaluate only the branches some lane needs: `product_log` buffer throughput is 2.2× (float) and 2.7× (double) faster, `omega` 1.24× and 1.59×, on an M1 Max, with bit-identical results (#46).
- ADAA computes the quadrature, the closed-form mean, the outside-range mean and the infinite and equal-input fallbacks only when some lane needs them. On an M1 Max with a 440 Hz sine at gain 0.5, `first_order` is 2.9–10× faster across the six shapes; at gain 4 it is 1.0–3.1× faster; results are bit-identical (#47).
- Scalar square roots use `__builtin_elementwise_sqrt` under clang, dropping the errno check and libm fallback that `-fmath-errno` adds; scalar-backend `asinh` is 13% (float) and 7% (double) faster under that flag (#48).

## [1.0.4] - 2026-11-03

### Changed

- `smoothing_coefficient` skips its overflow and domain handling when every lane has a positive, normal, finite product: float throughput goes from 0.9× to 1.13× of `std`, double from 0.53× to 0.68× on an M1 Max, with the same contract (#10).
- Non-FMA smoothing routes non-finite residual corrections through the edge path, preserving finite coefficients for tiny and very large positive parameter products (#52).

## [1.0.3] - 2026-11-03

### Changed

- `hz_to_midi` skips its near-zero path when no lane is within 4–16 Hz and folds the A4 offset into one exactly summed double-word step: float throughput goes from 0.6× to 1.6× of `std`, double from 0.4× to 0.86× on an M1 Max, with the same contracts (#9).

## [1.0.2] - 2026-11-03

### Added

- This changelog (#6) and a [stability policy](docs/details.md#stability) (#7).
- Pushing a `vX.Y.Z` tag publishes a GitHub release with that version's changelog entry, and fails if the tag does not match `CMakeLists.txt` (#8).
- `tools/ci/mutants.py` plants a wrong coefficient and a broken reduction step and requires the accuracy test or golden digest to catch each; CI runs it (#34).

## [1.0.1] - 2026-11-03

### Changed

- `find_package(wray 1.0)` accepts any 1.x release instead of only 1.0.x (#4).
- CI runs push builds only on `main`, reports one `ci-pass` check, pins every action to a commit SHA, fails AVX-512 jobs on runners without AVX-512F, and runs only the README check for docs-only pull requests (#1, #2, #31, #32, #33).

### Added

- README wordmark header and badges, generated by `tools/brand/logo.py` (#5).

## [1.0.0] - 2026-11-03

### Added

- Elementary functions: `exp exp2 expm1 log log2 log10 log1p pow cbrt sin cos sincos tan tanpi atan atan2 sinh cosh tanh asinh erf sigmoid`.
- Turns functions: `sin_turns cos_turns sincos_turns tan_turns wrap_turns`.
- Audio functions: `db_to_gain gain_to_db midi_to_hz hz_to_midi ratio_from_cents omega product_log`.
- Parameter maps: `smoothing_coefficient pole_from_cutoff prewarp`.
- `wray::fast::`, a `float` tier with a 2^-16 relative-error contract.
- First-order ADAA for `hard_clip soft_clip tanh atan algebraic radical`.
- Filters: `tpt_one_pole`, `svf`, `biquad`.
- Backends: NEON, SSE2, AVX2, AVX-512 and scalar, for `float` and `double`.

[Unreleased]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.5...HEAD
[1.0.5]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Lissajous-Labs/WRAY/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Lissajous-Labs/WRAY/releases/tag/v1.0.0
