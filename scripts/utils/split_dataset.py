"""
Split the large dataset CSV into 95 MB chunks for git storage.

Run this ONCE on your machine before pushing to git:
    python scripts/utils/split_dataset.py

Chunks are saved to data/chunks/ and can be committed normally (no LFS needed).
The original CSV is NOT committed (it's in .gitignore).
"""

import hashlib
from pathlib import Path

CHUNK_SIZE_MB = 95
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1_000_000  # decimal MB — GitHub limit is 100 MB (SI)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "v2v_dataset_20251017_124247.csv"
CHUNKS_DIR = ROOT / "data" / "chunks"
MANIFEST = CHUNKS_DIR / "manifest.txt"


def split() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Dataset not found: {SRC}")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    total_bytes = SRC.stat().st_size
    n_chunks = (total_bytes + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
    print(f"Splitting {SRC.name}  ({total_bytes / 1e9:.2f} GB)  →  {n_chunks} chunks of {CHUNK_SIZE_MB} MB")

    chunk_paths = []
    src_hash = hashlib.md5()

    with open(SRC, "rb") as f:
        idx = 0
        while True:
            data = f.read(CHUNK_SIZE_BYTES)
            if not data:
                break
            src_hash.update(data)
            chunk_path = CHUNKS_DIR / f"chunk_{idx:03d}.part"
            chunk_path.write_bytes(data)
            chunk_paths.append(chunk_path)
            print(f"  [{idx+1}/{n_chunks}] {chunk_path.name}  ({len(data) / 1e6:.1f} MB)")
            idx += 1

    # Write manifest for verification during assembly
    with open(MANIFEST, "w") as f:
        f.write(f"source={SRC.name}\n")
        f.write(f"total_bytes={total_bytes}\n")
        f.write(f"md5={src_hash.hexdigest()}\n")
        f.write(f"n_chunks={len(chunk_paths)}\n")
        for p in chunk_paths:
            f.write(f"{p.name}\n")

    print(f"\nDone. {len(chunk_paths)} chunks written to {CHUNKS_DIR}")
    print(f"Manifest: {MANIFEST}")
    print(f"\nNext steps:")
    print(f"  git add data/chunks/")
    print(f"  git commit -m 'feat: add dataset chunks ({n_chunks} × {CHUNK_SIZE_MB} MB)'")
    print(f"  git push")


if __name__ == "__main__":
    split()
