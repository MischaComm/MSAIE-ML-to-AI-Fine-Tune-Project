"""Download the dataset (pinned commit) and verify its checksum.

    python scripts/download_data.py

The 160 MB CSV is not committed to the repo (GitHub's 100 MB file limit).
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from malware_ml.schema import DATASET  # noqa: E402

URL = (
    "https://github.com/fabriciojoc/brazilian-malware-dataset/raw/"
    f"{DATASET['commit']}/brazilian-malware.zip"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "brazilian-malware.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    blob = urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(blob).hexdigest() != DATASET["zip_sha256"]:
        sys.exit("zip checksum mismatch — aborting")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        data = z.read("brazilian-malware.csv")
    if hashlib.sha256(data).hexdigest() != DATASET["csv_sha256"]:
        sys.exit("csv checksum mismatch — aborting")
    OUT.write_bytes(data)
    print(f"Saved {OUT} ({len(data):,} bytes), checksum verified")


if __name__ == "__main__":
    main()
