#!/usr/bin/env bash
# Builds a consumer of the headers under each unsafe floating-point flag and checks the stated outcome:
# rejected (the headers stop with #error), detected (wray::floating_point_ok() returns false),
# or passes (the test suite passes with the flag added to every translation unit). A/B accepts either outcome.
# Clang's -fno-honor-nans and -fno-honor-infinities are unsupported: they make any NaN or infinity in the caller's own code undefined.
set -u
cxx=${1:?compiler}
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fused=""
if [ "$(uname -m)" = x86_64 ]; then fused="-mavx2 -mfma"; fi

if "$cxx" --version | grep -qi clang; then
    cases=(
        "rejected|-ffast-math"
        "rejected|-ffinite-math-only"
        "rejected|-ffp-model=fast"
        "passes|-fno-signed-zeros"
        "passes|-fassociative-math"
        "passes|-freciprocal-math"
        "passes|-fapprox-func"
        "detected|-ffp-contract=fast $fused"
    )
else
    cases=(
        "rejected|-ffast-math"
        "rejected|-ffinite-math-only"
        "rejected|-fno-signed-zeros"
        "rejected|-fassociative-math -fno-signed-zeros -fno-trapping-math"
        "rejected|-freciprocal-math"
        "detected|-ffp-contract=fast $fused"
    )
fi

failures=0
for entry in "${cases[@]}"; do
    want=${entry%%|*}
    flags=${entry#*|}
    got=passes
    if ! "$cxx" -std=c++20 -O2 $flags -I "$root/include" "$root/tests/consumer/strict.cpp" -o "$work/strict" 2> "$work/err"; then
        if grep -q '#error "wray' "$work/err"; then got=rejected; else got="compile-failure"; cat "$work/err"; fi
    elif ! "$work/strict" > "$work/out"; then
        got="run-failure"
    elif [ "$(cut -d' ' -f1 < "$work/out")" = 0 ]; then
        got=detected
    elif [[ "/$want/" == */passes/* ]]; then
        build="$work/build"
        rm -rf "$build"
        if ! { cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DWRAY_BUILD_BENCH=OFF -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_CXX_FLAGS="$flags" > /dev/null &&
               cmake --build "$build" > "$work/build.log" 2>&1 &&
               ctest --test-dir "$build" -j 4 -LE "exhaustive|oracle|accuracy" --output-on-failure > "$work/test.log" 2>&1; }; then
            got="suite-failure"
            tail -n 40 "$work/build.log" "$work/test.log" 2> /dev/null
        fi
    fi
    if [[ "/$want/" == */"$got"/* ]]; then
        echo "ok   $flags: $got"
    else
        echo "FAIL $flags: expected $want, got $got"
        failures=$((failures + 1))
    fi
done
exit $failures
