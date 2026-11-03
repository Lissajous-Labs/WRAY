import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    tag = sys.argv[1]
    version = tag.removeprefix("v")
    declared = re.search(r"project\(wray VERSION (\S+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    if version != declared:
        print(f"tag {tag} does not match CMakeLists.txt version {declared}", file=sys.stderr)
        return 1
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[)", changelog, re.S | re.M)
    if entry is None or not entry.group(1).strip():
        print(f"CHANGELOG.md has no entry for {version}", file=sys.stderr)
        return 1
    print(entry.group(1).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
