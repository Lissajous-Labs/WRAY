# WRAY

(Wray & Nephew is a high-quality rum; you are highly advised to mix it with juice (JUCE))!

Fast, verified math for audio DSP: header-only C++23, SIMD (NEON, SSE2, scalar), `float` and `double`.

These kernels are specifically the fast math code in our shipping plugins, Chronos and Torus. Here every core is regenerated from first principles with a rational Remez solver and re-verified against 40-digit arithmetic.

```cpp
#include <wray/wray.h>

float  a = wray::exp(0.5f);
double b = wray::atan2(1.0, -2.0);
auto   v = wray::tanh(wray::f32x4(0.25f));
wray::sin(input, output);
float  c = wray::fast::tanh(x);
```

`exp exp2 log log2 sin cos sincos tan atan atan2 tanh sigmoid pow db_to_gain gain_to_db` take a scalar, a SIMD vector, or a contiguous range. `wray::fast::` is the `float` tier with a 2^-16 relative-error contract.

![speedup](docs/charts/speedup.png)

## Accuracy contracts

| function | float (ulp) | double (ulp) | function | float (ulp) | double (ulp) |
|---|---|---|---|---|---|
| exp, exp2 | 1.0 | 1.0 | atan | 2.0 | 1.5 |
| log | 1.0 | 1.0 | atan2 | 2.0 | 1.5 |
| log2 | 1.5 | 1.0 | tanh | 1.5 | 2.5 |
| sin, cos | 2.5 | 2.5 | sigmoid | 2.5 | 2.0 |
| tan | 3.5 | 4.0 | pow | 2.0 | 1.5 |
| db_to_gain | 1.5 | 1.5 | gain_to_db | 1.5 | 1.5 |

Builds without FMA (baseline SSE2) get 0.5 ulp more. Trig is reduced exactly for |x| ≤ 6434 (float) or 6.6e6 (double) and returns NaN beyond. Infinities, NaN, signed zero and subnormals are handled in the precise tier.

![accuracy](docs/charts/accuracy_summary.png)

## How it is verified

- Every `float` input (all 2^32 bit patterns) for each unary function, against a double-precision reference: `wray_accuracy_test --exhaustive`.
- Every function and tier against mpmath at 40 digits: `tools/oracle/oracle.py --check`.
- Bit-identical scalar and SIMD results, special values, and the SIMD layer itself, on NEON, SSE2 and the scalar backend.
- Polynomial reduction uses compile-time level sizes, including odd-length tails, without disabling compiler bounds diagnostics.

```sh
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
```

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
.venv/bin/python tools/charts/charts.py --machine "your machine"
```

## License

GNU General Public License v2.0 or later (GPL-2.0-or-later). See [LICENSE](LICENSE).
