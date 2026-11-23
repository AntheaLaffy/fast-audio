"""Check the candidate version and select its released Cargo compatibility baseline."""

import argparse
import os
import re
import subprocess
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="prior commit when checking a version change")
    args = parser.parse_args()
    package = tomllib.loads(Path("Cargo.toml").read_text())["package"]
    version = package["version"]
    locked = tomllib.loads(Path("Cargo.lock").read_text())["package"]
    assert any(p["name"] == package["name"] and p["version"] == version and "source" not in p
               for p in locked), "Update the workspace version in Cargo.lock"
    changelog = Path("CHANGELOG.md").read_text()
    if not re.search(r"^## " + re.escape(version) + r"(?:\s|$)", changelog, re.MULTILINE):
        raise SystemExit(f"Missing changelog entry for {version}")

    if args.base and args.base.strip("0"):
        parent = subprocess.run(["git", "show", f"{args.base}:Cargo.toml"],
                                capture_output=True, text=True)
        if parent.returncode == 0 and tomllib.loads(parent.stdout)["package"]["version"] != version:
            prior_notes = subprocess.run(["git", "show", f"{args.base}:CHANGELOG.md"],
                                         capture_output=True, text=True)
            if prior_notes.stdout == changelog:
                raise SystemExit("A version change needs a release note")

    major, minor, _ = version.split(".", 2)
    candidates = []
    for tag in subprocess.check_output(["git", "tag", "--merged", "HEAD"], text=True).splitlines():
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if match and match[1] == major and (major != "0" or match[2] == minor):
            candidates.append((tuple(map(int, match.groups())), tag))
    baseline = max(candidates)[1] if candidates else ""
    print(f"Candidate {version}. Stable comparison baseline: {baseline or 'none in this line'}.")
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as output:
            print(f"baseline={baseline}", file=output)


if __name__ == "__main__":
    main()
