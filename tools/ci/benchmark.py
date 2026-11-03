import argparse
import json
import os
import pathlib
import platform
import subprocess
import tempfile


def host():
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"} or os.environ.get("RUNNER_ARCH", "X64") != "X64":
        raise SystemExit(f"Native x86 measurements require an x86 host, not {machine}")
    if platform.system() == "Linux":
        text = pathlib.Path("/proc/cpuinfo").read_text()
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        flags = set(fields.get("flags", "").split())
        return {"cpu": fields.get("model name", "unknown"), "flags": sorted(flags),
                "supported": {"sse2": "sse2" in flags, "avx2": {"avx2", "fma"} <= flags,
                              "avx512": {"avx512f", "avx2", "fma"} <= flags}}
    if platform.system() == "Windows":
        command = """
        @{
            cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
            supported = @{
                sse2 = [System.Runtime.Intrinsics.X86.Sse2]::IsSupported
                avx2 = [System.Runtime.Intrinsics.X86.Avx2]::IsSupported -and [System.Runtime.Intrinsics.X86.Fma]::IsSupported
                avx512 = [System.Runtime.Intrinsics.X86.Avx512F]::IsSupported -and [System.Runtime.Intrinsics.X86.Avx512BW]::IsSupported -and [System.Runtime.Intrinsics.X86.Avx512DQ]::IsSupported -and [System.Runtime.Intrinsics.X86.Avx512CD]::IsSupported -and [System.Runtime.Intrinsics.X86.Avx2]::IsSupported -and [System.Runtime.Intrinsics.X86.Fma]::IsSupported
            }
        } | ConvertTo-Json -Compress
        """
        return json.loads(subprocess.check_output(["pwsh", "-NoProfile", "-Command", command], text=True))
    raise SystemExit("This runner collects native Linux/Windows x86 measurements, never Rosetta timings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark-results")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(args.out).resolve()
    info = host()
    info.update(platform=platform.platform(), architecture=platform.machine(),
                revision=os.environ.get("WRAY_BENCH_REVISION", os.environ.get("GITHUB_SHA", "unrecorded")),
                runner_image=os.environ.get("ImageOS", ""), runner_image_version=os.environ.get("ImageVersion", ""),
                measured=[])
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "host.json"
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    env = dict(os.environ, WRAY_BENCH_CPU=info["cpu"], WRAY_BENCH_REVISION=info["revision"])
    with tempfile.TemporaryDirectory(prefix="wray-native-bench.") as temporary:
        for backend in ("sse2", "avx2", "avx512"):
            if not info["supported"][backend]:
                print(f"UNAVAILABLE: {backend} on {info['cpu']}; no timing produced", flush=True)
                continue
            print(f"Measuring native {backend} on {info['cpu']}", flush=True)
            build = pathlib.Path(temporary) / backend
            with (out / f"build-{backend}.log").open("w") as log:
                subprocess.run(["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja",
                                "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_STANDARD=20", f"-DWRAY_SIMD={backend}",
                                "-DWRAY_BUILD_TESTS=OFF", "-DWRAY_BUILD_TOOLS=OFF", "-DWRAY_BUILD_BENCH=ON"],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run(["cmake", "--build", str(build), "--target", "wray_bench", "--parallel", "2"],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            executable = build / "bench" / ("wray_bench.exe" if os.name == "nt" else "wray_bench")
            baseline = "sleef_u10" if platform.system() == "Linux" else "libm"
            with (out / f"bench-{backend}.log").open("w") as log:
                subprocess.run([str(executable), str(out / f"bench-{backend}.json"), "--baseline", baseline],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            info["measured"].append(backend)
            manifest.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
