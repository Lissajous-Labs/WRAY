#!/usr/bin/env bash
# Builds the digest program as a consumer that includes the headers without the CMake target, so none of
# the target's floating-point options apply, and checks that its output matches the golden digest for its
# FMA class, or that wray::floating_point_ok() reports the compiler's default model as unsafe.
set -eu
cxx=${1:?compiler}
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$cxx" -O2 "$@" -I "$root/include" "$root/tools/dump/dump.cpp" -o "$work/dump"
"$cxx" -O2 "$@" -I "$root/include" "$root/tests/consumer/strict.cpp" -o "$work/strict"
if "$work/dump" info | grep -q '"fma": true'; then class=fma; else class=nofma; fi
ok=$("$work/strict" | cut -d' ' -f1)
if "$work/dump" digest | diff -q - "$root/tests/golden/$class.txt" > /dev/null; then
    echo "ok   $cxx $*: digest matches tests/golden/$class.txt (floating_point_ok $ok)"
elif [ "$ok" = 0 ]; then
    echo "ok   $cxx $*: digest differs from tests/golden/$class.txt and floating_point_ok() reports the model"
else
    echo "FAIL $cxx $*: digest differs from tests/golden/$class.txt but floating_point_ok() returns true"
    "$work/dump" digest | diff - "$root/tests/golden/$class.txt" | head -n 20
    exit 1
fi
