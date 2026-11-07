
from pathlib import Path
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EMBEDDINGS = DATA / "embeddings.npy"
MAPPING = DATA / "embedding_mapping.txt"

PCA_OUTPUT = DATA / "music_3d_pca_direct.npy"
UMAP_OUTPUT = DATA / "music_3d_umap_direct512.npy"
INFO_OUTPUT = DATA / "projection_experiment_info.json"

RANDOM_STATE = 42



def save_checked(array, destination, expected_rows):
    temp = destination.with_name(destination.stem + "_building.npy")

    check = None

    try:
        print(f"Saving temporary file: {temp.name}", flush=True)

        with temp.open("wb") as f:
            np.save(f, array)

        # Validate the saved file without leaving its memory mapping open.
        check = np.load(temp, mmap_mode="r")

        if check.shape != (expected_rows, 3):
            raise RuntimeError(f"Invalid output shape: {check.shape}")

        if not np.isfinite(check).all():
            raise RuntimeError("Output contains non-finite values.")

        # Explicitly close the Windows memory-mapped file handle.
        check._mmap.close()
        check = None

        print(f"Publishing: {destination.name}", flush=True)
        os.replace(temp, destination)

        print("Saved and validated successfully.", flush=True)

    finally:
        if check is not None:
            check._mmap.close()

        # Only remove the temporary file if it is no longer mapped.
        if temp.exists():
            temp.unlink()


def main():
    #for path in (PCA_OUTPUT, UMAP_OUTPUT, INFO_OUTPUT):
    #    if path.exists():
    #        raise FileExistsError(f"Experimental output already exists: {path}")

    print("Loading embeddings and mapping...", flush=True)

    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    paths = json.loads(MAPPING.read_text(encoding="utf-8"))

    n = len(paths)

    if embeddings.shape != (n, 512):
        raise ValueError(
            f"Embedding/mapping mismatch: {embeddings.shape}, {n}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    print(f"Tracks: {n:,}", flush=True)

    print("\nNormalising embeddings...", flush=True)
    X = normalize(np.asarray(embeddings, dtype=np.float32), norm="l2")

    # Experiment A: direct PCA
    print("\nExperiment A: direct PCA 512D -> 3D...", flush=True)

    pca = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )

    pca_coords = pca.fit_transform(X).astype(np.float32)

    save_checked(pca_coords, PCA_OUTPUT, n)

    print(
        "PCA explained variance:",
        np.round(pca.explained_variance_ratio_, 4),
        flush=True,
    )

    # Experiment B: direct UMAP
    print("\nExperiment B: direct UMAP 512D -> 3D...", flush=True)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True,
        n_jobs=1,
        verbose=True,
    )

    umap_coords = reducer.fit_transform(X).astype(np.float32)

    save_checked(umap_coords, UMAP_OUTPUT, n)

    info = {
        "tracks": n,
        "embedding_shape": list(embeddings.shape),
        "mapping_rows": n,
        "pca_method": "Direct PCA from normalised 512D embeddings",
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "umap_method": "Direct UMAP from normalised 512D embeddings",
        "umap_metric": "cosine",
        "umap_neighbors": 30,
        "umap_min_dist": 0.1,
        "random_state": RANDOM_STATE,
        "pca_output": PCA_OUTPUT.name,
        "umap_output": UMAP_OUTPUT.name,
    }

    INFO_OUTPUT.write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENTS COMPLETED", flush=True)
    print(f"PCA coordinates:  {PCA_OUTPUT}", flush=True)
    print(f"UMAP coordinates: {UMAP_OUTPUT}", flush=True)
    print(f"Experiment info:  {INFO_OUTPUT}", flush=True)
    print(f"Tracks processed: {n:,}", flush=True)
    print("Production files were not modified.", flush=True)


if __name__ == "__main__":
    main()
