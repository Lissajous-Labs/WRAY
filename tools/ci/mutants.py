import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
MUTANTS = [
    ("log coefficient", "include/wray/math/coefficients.h",
     "p{ 0x1.55555cp-1f, 0x1.997c26p-2f, 0x1.2ee78ap-2f }", "p{ 0x1.55565cp-1f, 0x1.997c26p-2f, 0x1.2ee78ap-2f }"),
    ("log reduction residual", "include/wray/math/kernels.h",
     "const V lo = (f - hi) - a;", "const V lo = V(S(0));"),
]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    survivors = []
    for name, path, before, after in MUTANTS:
        with tempfile.TemporaryDirectory() as tmp:
            tree = pathlib.Path(tmp) / "wray"
            shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", ".venv", "build*", "cmake-build-*"))
            target = tree / path
            text = target.read_text(encoding="utf-8")
            if text.count(before) != 1:
                print(f"{name}: target not found exactly once in {path}")
                return 1
            target.write_text(text.replace(before, after), encoding="utf-8")
            configure = run(["cmake", "-S", ".", "-B", "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DWRAY_BUILD_BENCH=OFF",
                             f"-DWRAY_PYTHON={args.python}"], tree)
            build = run(["cmake", "--build", "build", "--target", "wray_accuracy_test", "wray_dump"], tree)
            if configure.returncode or build.returncode:
                print(f"{name}: build failed\n{configure.stderr}{build.stdout[-2000:]}{build.stderr[-2000:]}")
                return 1
            caught = [test for test in ("accuracy", "digest")
                      if run(["ctest", "--test-dir", "build", "-R", f"^{test}$"], tree).returncode != 0]
            print(f"{name}: caught by {', '.join(caught) if caught else 'nothing'}")
            if not caught:
                survivors.append(name)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
