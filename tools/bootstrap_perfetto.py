#!/usr/bin/env python3
"""Download the pinned official Perfetto C++ amalgamated SDK."""
import argparse
import base64
from pathlib import Path
import subprocess
import tempfile
import zipfile

VERSION = "v58.2"
REPOSITORY = "google/perfetto"
ASSET = "perfetto-cpp-sdk-src.zip"
DOWNLOAD_URL = f"https://github.com/{REPOSITORY}/releases/download/{VERSION}/{ASSET}"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "third_party" / f"perfetto-{VERSION}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rknpu-perfetto-") as temporary:
        subprocess.run(["gh", "release", "download", VERSION, "--repo", REPOSITORY,
                        "--pattern", ASSET, "--dir", temporary], check=True)
        with zipfile.ZipFile(Path(temporary) / ASSET) as archive:
            for filename in ("perfetto.h", "perfetto.cc"):
                (args.output / filename).write_bytes(archive.read(filename))
    encoded_license = subprocess.check_output([
        "gh", "api", f"repos/{REPOSITORY}/contents/LICENSE?ref={VERSION}", "--jq", ".content"])
    (args.output / "LICENSE").write_bytes(base64.b64decode(encoded_license))
    (args.output / "SOURCE.txt").write_text(f"Perfetto {VERSION}\n{DOWNLOAD_URL}\n")
    print(args.output)


if __name__ == "__main__":
    main()
