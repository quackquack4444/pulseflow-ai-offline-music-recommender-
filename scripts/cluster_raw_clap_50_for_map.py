
from pathlib import Path
import json
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EMBEDDINGS_FILE = DATA / "embeddings.npy"
MAPPING_FILE = DATA / "embedding_mapping.txt"
MAP_FILE = DATA / "music_3d.npy"

LABELS_FILE = DATA / "music_clusters_raw_clap_50.npy"
INFO_FILE = DATA / "music_clusters_raw_clap_50_info.json"

TEMP_LABELS = DATA / "music_clusters_raw_clap_50.building.npy"
TEMP_INFO = DATA / "music_clusters_raw_clap_50_info.building.json"

N_CLUSTERS = 50
RANDOM_STATE = 42


def main():
    print("Loading current embeddings, mapping and coordinates...", flush=True)

    embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
    paths = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    coordinates = np.load(MAP_FILE, mmap_mode="r")

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    n = len(paths)

    if embeddings.shape[0] != n or coordinates.shape != (n, 3):
        raise ValueError(
            f"Row mismatch: embeddings={embeddings.shape}, "
            f"paths={n}, map={coordinates.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    if not np.isfinite(coordinates).all():
        raise ValueError("3D coordinates contain NaN or infinite values.")

    print(f"Current tracks: {n:,}", flush=True)
    print(f"Unique paths:   {len(set(paths)):,}", flush=True)

    print("Normalising CLAP vectors...", flush=True)
    X = normalize(np.asarray(embeddings, dtype=np.float32), norm="l2")

    print("Fitting 50 raw-CLAP clusters...", flush=True)
    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )

    labels = model.fit_predict(X).astype(np.int32)
    sizes = np.bincount(labels, minlength=N_CLUSTERS)

    if labels.shape != (n,):
        raise RuntimeError(f"Unexpected labels shape: {labels.shape}")

    if len(sizes) != N_CLUSTERS or np.any(sizes == 0):
        raise RuntimeError("Unexpected cluster count or an empty cluster.")

    info = {
        "method": "MiniBatchKMeans",
        "source": "normalised raw 512-D CLAP embeddings",
        "tracks": n,
        "unique_paths": len(set(paths)),
        "embedding_shape": list(embeddings.shape),
        "map_shape": list(coordinates.shape),
        "clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "batch_size": 4096,
        "n_init": 3,
        "max_iter": 100,
        "reassignment_ratio": 0.01,
        "cluster_sizes": sizes.tolist(),
        "labels_file": LABELS_FILE.name,
    }

    # Stage both outputs before replacing either existing file.
    try:
        print("Writing temporary outputs...", flush=True)

        with TEMP_LABELS.open("wb") as f:
            np.save(f, labels)

        TEMP_INFO.write_text(
            json.dumps(info, indent=2),
            encoding="utf-8",
        )

        # Validate the staged labels.
        check_labels = np.load(TEMP_LABELS, mmap_mode="r")
        if check_labels.shape != (n,):
            raise RuntimeError(
                f"Staged labels have incorrect shape: {check_labels.shape}"
            )

        if not np.array_equal(check_labels, labels):
            raise RuntimeError("Staged labels failed value validation.")

        # Validate the staged metadata.
        check_info = json.loads(TEMP_INFO.read_text(encoding="utf-8"))
        if (
            check_info.get("tracks") != n
            or check_info.get("clusters") != N_CLUSTERS
            or len(check_info.get("cluster_sizes", [])) != N_CLUSTERS
            or sum(check_info["cluster_sizes"]) != n
        ):
            raise RuntimeError("Staged metadata failed validation.")

        del check_labels

        print("Publishing validated outputs...", flush=True)

        # These replacements are individually atomic on the same filesystem,
        # but the two-file update as a whole is not a single atomic operation.
        os.replace(TEMP_INFO, INFO_FILE)
        os.replace(TEMP_LABELS, LABELS_FILE)

    finally:
        if TEMP_LABELS.exists():
            TEMP_LABELS.unlink()
        if TEMP_INFO.exists():
            TEMP_INFO.unlink()

    print("\nCluster sizes:", flush=True)
    for cluster_id, size in enumerate(sizes):
        print(f"  Cluster {cluster_id:02}: {size:,}", flush=True)

    print(f"\nLabels saved: {LABELS_FILE}", flush=True)
    print(f"Run details:  {INFO_FILE}", flush=True)
    print(f"Validated tracks: {n:,}", flush=True)
    print("Production embeddings, mapping and 3D coordinates were not modified.")


if __name__ == "__main__":
    main()
