"""
Reassemble dataset CSV from git-committed chunks.

Called automatically by preflight_check.py when chunks are present but
the CSV is missing. Can also be run manually:

    python scripts/utils/assemble_dataset.py
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT / "data" / "chunks"
MANIFEST = CHUNKS_DIR / "manifest.txt"
DEST = ROOT / "data" / "v2v_dataset_20251017_124247.csv"


def assemble(force: bool = False) -> bool:
    """Assemble chunks into the original CSV.

    Args:
        force: Re-assemble even if the CSV already exists.

    Returns:
        True if assembly succeeded or CSV already present, False on error.
    """
    if DEST.exists() and not force:
        print(f"Dataset already exists: {DEST}  ({DEST.stat().st_size / 1e9:.2f} GB)")
        return True

    if not CHUNKS_DIR.exists() or not any(CHUNKS_DIR.glob("chunk_*.part")):
        print("ERROR: No chunks found in data/chunks/ — did you git pull?")
        return False

    if not MANIFEST.exists():
        print("ERROR: manifest.txt missing from data/chunks/ — repo may be incomplete")
        return False

    # Parse manifest
    manifest = {}
    chunk_names = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                manifest[k] = v
            elif line.endswith(".part"):
                chunk_names.append(line)

    expected_bytes = int(manifest.get("total_bytes", 0))
    expected_md5 = manifest.get("md5", "")
    n_chunks = int(manifest.get("n_chunks", len(chunk_names)))

    # Verify all chunks present
    missing = [c for c in chunk_names if not (CHUNKS_DIR / c).exists()]
    if missing:
        print(f"ERROR: Missing chunks: {missing}")
        print("Run: git lfs pull  or  git pull")
        return False

    print(f"Assembling {n_chunks} chunks → {DEST.name}")

    actual_hash = hashlib.md5()
    written = 0

    with open(DEST, "wb") as out:
        for i, name in enumerate(chunk_names):
            chunk_path = CHUNKS_DIR / name
            data = chunk_path.read_bytes()
            out.write(data)
            actual_hash.update(data)
            written += len(data)
            print(f"  [{i+1}/{n_chunks}] {name}  ({len(data) / 1e6:.1f} MB)")

    # Verify
    if expected_bytes and written != expected_bytes:
        print(f"ERROR: Size mismatch — expected {expected_bytes}, got {written}")
        DEST.unlink(missing_ok=True)
        return False

    if expected_md5 and actual_hash.hexdigest() != expected_md5:
        print(f"ERROR: MD5 mismatch — file may be corrupted")
        DEST.unlink(missing_ok=True)
        return False

    print(f"\nAssembled: {DEST}  ({written / 1e9:.2f} GB)  ✓ MD5 verified")
    return True


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ok = assemble(force=force)
    sys.exit(0 if ok else 1)
