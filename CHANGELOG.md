# Changelog

All notable changes to WRAY are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow the [stability policy](docs/details.md#stability).

## [Unreleased]

### Added

- `adaa::first_order<Shape, T>::process(in, out)` and `process(io)` for scalar `T` run consecutive mono samples in parallel lanes and match per-sample calls bit for bit: 2.9–4.3× faster in float and 1.5–2.1× in double on an M1 Max (#49).
- `sincos(in, sin_out, cos_out)` and `sincos_turns(in, sin_out, cos_out)` produce both range outputs in one SIMD pass, including the float-only fast tier. They stop at the shortest range and support exact input aliasing with either disjoint output (#53).
- `wray_bench --baseline libm|vforce` pins its denominator, measures both on Apple, disables scalar-loop auto-vectorization without disabling fused scalar calls, and records paired ABBA samples, bootstrap intervals and a self-control (#54, #35).

### Changed

- Positive-base `pow` skips sign/parity work when no lane needs it and uses the bounded log mantissa to shorten an exact compensated sum. Twelve paired ABBA rounds on an M1 Max show 22% higher float throughput, 19% higher double throughput and 5–6% lower dependency-chain cost; the precise contracts and golden outputs are unchanged (#55).

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
