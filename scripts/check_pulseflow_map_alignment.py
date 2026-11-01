from pathlib import Path
import numpy as np

ROOT = Path(r"V:\\GitHub\\pulseflow-ai-offline-music-recommender-main")
DATA = ROOT / "data"

FILES = {
    "embeddings": DATA / "embeddings.npy",
    "3D coordinates": DATA / "music_3d.npy",
    "raw CLAP 50-cluster labels": DATA / "music_clusters_raw_clap_50.npy",
    "mapping": DATA / "embedding_mapping.txt",
}

def inspect_array(label, path):
    if not path.exists():
        print(f"{label}: MISSING — {path}")
        return None
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        print(f"{label}: shape={arr.shape}, dtype={arr.dtype}, rows={arr.shape[0] if arr.ndim else 'n/a'}")
        return arr.shape
    except Exception as exc:
        print(f"{label}: ERROR reading {path}: {type(exc).__name__}: {exc}")
        return None

def main():
    print("PulseFlow map alignment check (read-only)")
    print(f"Data folder: {DATA}")
    print()

    shapes = {}
    for key in ("embeddings", "3D coordinates", "raw CLAP 50-cluster labels"):
        shapes[key] = inspect_array(key, FILES[key])

    mapping_path = FILES["mapping"]
    mapping_rows = None
    if mapping_path.exists():
        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping_rows = sum(1 for line in f if line.strip())
            print(f"mapping: {mapping_rows} non-empty lines")
        except Exception as exc:
            print(f"mapping: ERROR reading {mapping_path}: {type(exc).__name__}: {exc}")
    else:
        print(f"mapping: MISSING — {mapping_path}")

    print("\\nAlignment checks:")
    emb_shape = shapes.get("embeddings")
    coords_shape = shapes.get("3D coordinates")
    labels_shape = shapes.get("raw CLAP 50-cluster labels")

    if emb_shape and emb_shape[0] == mapping_rows:
        print("  OK: embeddings rows match mapping rows")
    elif emb_shape:
        print(f"  MISMATCH: embeddings rows={emb_shape[0]}, mapping rows={mapping_rows}")

    if coords_shape and coords_shape[0] == mapping_rows:
        print("  OK: 3D coordinate rows match mapping rows")
    elif coords_shape:
        print(f"  MISMATCH: 3D coordinate rows={coords_shape[0]}, mapping rows={mapping_rows}")

    if labels_shape and labels_shape[0] == mapping_rows:
        print("  OK: cluster-label rows match mapping rows")
    elif labels_shape:
        print(f"  MISMATCH: cluster-label rows={labels_shape[0]}, mapping rows={mapping_rows}")

    if coords_shape and len(coords_shape) >= 2 and coords_shape[1] != 3:
        print(f"  WARNING: coordinate array has {coords_shape[1]} columns, expected 3")
    if labels_shape and len(labels_shape) != 1:
        print(f"  WARNING: labels array is {len(labels_shape)}-D; expected a 1-D array")

    print("\\nNo files were modified.")

if __name__ == "__main__":
    main()
