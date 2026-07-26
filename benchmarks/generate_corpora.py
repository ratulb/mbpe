#!/usr/bin/env python3
"""Generate scaled corpora from the base corpus for benchmarking.

Produces files: corpus_{size_kb}KB.txt in benchmarks/ directory.
sizes: 10KB, 100KB, 500KB, 1MB, 2MB, 5MB (by repeating the base corpus)
"""

import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CORPUS = os.path.join(SCRIPT_DIR, "corpus.txt")

# Target sizes in bytes
TARGETS = {
    "10KB": 10 * 1024,
    "100KB": 100 * 1024,
    "500KB": 500 * 1024,
    "1MB": 1 * 1024 * 1024,
    "2MB": 2 * 1024 * 1024,
    "5MB": 5 * 1024 * 1024,
}

with open(BASE_CORPUS, "r") as f:
    base = f.read()
base_bytes = len(base.encode("utf-8"))

if base_bytes == 0:
    print(f"Error: {BASE_CORPUS} is empty!")
    exit(1)

for label, target in TARGETS.items():
    path = os.path.join(SCRIPT_DIR, f"corpus_{label}.txt")
    repeats = math.ceil(target / base_bytes)
    # Repeat and truncate to exact target byte count
    data = (base * repeats).encode("utf-8")[:target]
    # Decode back to string, preserving whole characters
    data_str = data.decode("utf-8", errors="replace")
    with open(path, "w") as f:
        f.write(data_str)
    actual = len(data_str.encode("utf-8"))
    print(f"{label}: {path} ({actual:,} bytes)")

print("\nDone.")
