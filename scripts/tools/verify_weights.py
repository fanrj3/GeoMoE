#!/usr/bin/env python3
"""Verify every artifact declared by the nested weights repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a potentially large checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Validate manifest sizes and hashes; return non-zero on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    args = parser.parse_args()
    root = args.weights.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = False
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file():
            print(f"MISSING  {artifact['path']}")
            failed = True
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        ok = actual_size == artifact["size"] and actual_hash == artifact["sha256"]
        print(f"{'OK' if ok else 'FAILED':7}  {artifact['path']}")
        if not ok:
            print(f"  size:   expected={artifact['size']} actual={actual_size}")
            print(f"  sha256: expected={artifact['sha256']} actual={actual_hash}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
