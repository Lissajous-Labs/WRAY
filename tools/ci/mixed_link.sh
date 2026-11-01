#!/usr/bin/env bash
# Links translation units built with different SIMD flags into one program and checks that each keeps
# its own code: the mixed program must reproduce, unit by unit, the results of single-flag programs.
set -eu
cxx=${1:?compiler}
include=${2:-$(cd "$(dirname "$0")/../.." && pwd)/include}
src=$(cd "$(dirname "$0")/../../tests/consumer/mixed" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
wide="-mavx2 -mfma"
narrow="-msse2 -mno-avx"
failures=0
for opt in -O0 -O2; do
    build() {
        "$cxx" -std=c++20 $opt -ffp-contract=off $2 -DWRAY_UNIT=a -I "$include" -c "$src/unit.cpp" -o "$work/a.o"
        "$cxx" -std=c++20 $opt -ffp-contract=off $3 -DWRAY_UNIT=b -I "$include" -c "$src/unit.cpp" -o "$work/b.o"
        "$cxx" -std=c++20 $opt -c "$src/main.cpp" -o "$work/main.o"
        "$cxx" "$work/main.o" "$work/a.o" "$work/b.o" -o "$work/$1"
        "$work/$1"
    }
    read -r _ wide_hash _ _ <<< "$(build wide "$wide" "$wide")"
    read -r _ narrow_hash _ _ <<< "$(build narrow "$narrow" "$narrow")"
    read -r abi_a got_a abi_b got_b <<< "$(build mixed "$wide" "$narrow")"
    echo "$opt: $abi_a $got_a (alone $wide_hash), $abi_b $got_b (alone $narrow_hash)"
    if [ "$wide_hash" = "$narrow_hash" ]; then echo "FAIL $opt: the two flag sets give the same results, so the check proves nothing"; failures=$((failures + 1)); fi
    if [ "$got_a" != "$wide_hash" ] || [ "$got_b" != "$narrow_hash" ]; then echo "FAIL $opt: a unit ran the other unit's code"; failures=$((failures + 1)); fi
    if [ "$abi_a" = "$abi_b" ]; then echo "FAIL $opt: both units share the namespace $abi_a"; failures=$((failures + 1)); fi
done
exit $failures
