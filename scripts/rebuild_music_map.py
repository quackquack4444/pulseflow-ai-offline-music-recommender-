
import csv
import json
import os
from collections import Counter
from pathlib import Path

import hdbscan
import numpy as np
import plotly.express as px
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EMBEDDINGS = DATA / "embeddings.npy"
MAPPING = DATA / "embedding_mapping.txt"
PCA_FILE = DATA / "music_pca_50d.npy"
CLUSTERS_FILE = DATA / "music_clusters_50d.npy"
PROJECTION = DATA / "music_3d.npy"
OUTLIERS = DATA / "music_outliers.csv"
OUTPUT = DATA / "music_map.html"

HIDE_NOISE = True





def save_npy(path, array):
    temp = path.with_name(path.stem + ".building.npy")

    np.save(temp, array)

    # Validate the temporary file before replacing the destination.
    check = np.load(temp, allow_pickle=False)
    valid = check.shape == array.shape and check.dtype == array.dtype
    del check

    if not valid:
        raise ValueError(f"Temporary array validation failed: {temp}")

    # Replace the destination directly; don't rename it to a backup first.
    os.replace(temp, path)


def main():
    print("Loading CLAP embeddings and mapping...", flush=True)
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")

    with MAPPING.open("r", encoding="utf-8") as f:
        paths = json.load(f)

    n = len(embeddings)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    if len(paths) != n:
        raise ValueError(
            f"Mapping has {len(paths):,} paths but embeddings have {n:,} rows."
        )

    print(f"Tracks: {n:,}", flush=True)
    print("All tracks included; no folder exclusions.", flush=True)

    # Stage 1: PCA, 512 dimensions to 50.
    valid_pca = False
    if PCA_FILE.exists():
        cached = np.load(PCA_FILE)
        valid_pca = cached.shape == (n, 50)

    if valid_pca:
        print("Stage 1/3: reusing aligned PCA cache.", flush=True)
        reduced = np.load(PCA_FILE)
    else:
        print("Stage 1/3: calculating 50D PCA...", flush=True)
        pca = PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42,
            n_oversamples=10,
        )
        reduced = pca.fit_transform(
            np.asarray(embeddings, dtype=np.float32)
        ).astype(np.float32)

        save_npy(PCA_FILE, reduced)
        print(
            f"PCA explained variance: "
            f"{pca.explained_variance_ratio_.sum():.1%}",
            flush=True,
        )

    # Normalize the 50D vectors for clustering and UMAP.
    vectors = normalize(np.asarray(reduced, dtype=np.float32))

    # Stage 2: HDBSCAN clustering.
    valid_clusters = False
    if CLUSTERS_FILE.exists():
        cached = np.load(CLUSTERS_FILE)
        valid_clusters = cached.shape == (n,)

    if valid_clusters:
        print("Stage 2/3: reusing aligned cluster labels.", flush=True)
        labels = np.asarray(np.load(CLUSTERS_FILE), dtype=np.int32)
    else:
        print("Stage 2/3: running HDBSCAN on all tracks...", flush=True)
        print("This can take a while.", flush=True)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=3,
            metric="euclidean",
            algorithm="best",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(vectors).astype(np.int32)
        save_npy(CLUSTERS_FILE, labels)

    # Stage 3: calculate a fresh 3D UMAP projection.
    valid_projection = False
    if PROJECTION.exists():
        cached = np.load(PROJECTION)
        valid_projection = cached.shape == (n, 3)

    if valid_projection:
        print("Stage 3/3: reusing aligned 3D coordinates.", flush=True)
        coords = np.load(PROJECTION)
    else:
        print("Stage 3/3: calculating 3D UMAP coordinates...", flush=True)
        print("UMAP progress will be displayed; please be patient.", flush=True)

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
            low_memory=True,
            verbose=True,
        )
        coords = reducer.fit_transform(vectors).astype(np.float32)
        save_npy(PROJECTION, coords)

    # Validate all arrays before generating reports or HTML.
    if labels.shape != (n,) or coords.shape != (n, 3):
        raise ValueError("Final array shapes do not match the mapping.")

    if not np.isfinite(coords).all():
        raise ValueError("3D coordinates contain invalid values.")

    counts = Counter(int(x) for x in labels)
    print(f"Clusters found: {sum(x >= 0 for x in counts)}", flush=True)
    print(f"Noise tracks: {counts.get(-1, 0):,}", flush=True)

    # Write the noise report.
    temp_csv = OUTLIERS.with_name("music_outliers.building.csv")
    with temp_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "album_folder", "track", "path"])

        for i, label in enumerate(labels):
            if int(label) == -1:
                p = Path(paths[i])
                writer.writerow([
                    p.parent.parent.name,
                    p.parent.name,
                    p.stem,
                    paths[i],
                ])

    os.replace(temp_csv, OUTLIERS)

    # Create the interactive map. Noise is hidden by default.
    keep = [
        i for i, label in enumerate(labels)
        if not (HIDE_NOISE and int(label) == -1)
    ]

    df = {
        "x": np.asarray(coords[keep, 0]),
        "y": np.asarray(coords[keep, 1]),
        "z": np.asarray(coords[keep, 2]),
        "cluster": [f"Cluster {int(labels[i])}" for i in keep],
        "track": [Path(paths[i]).stem for i in keep],
        "artist": [Path(paths[i]).parent.parent.name for i in keep],
        "album_folder": [Path(paths[i]).parent.name for i in keep],
        "path": [paths[i] for i in keep],
    }

    print("Creating interactive 3D HTML map...", flush=True)
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_data={
            "track": True,
            "artist": True,
            "album_folder": True,
            "path": True,
            "x": False,
            "y": False,
            "z": False,
        },
        title=f"PulseFlow - 3D UMAP Music Map ({n:,} tracks)",
        opacity=0.75,
    )
    fig.update_traces(marker={"size": 2})
    fig.update_layout(
        template="plotly_dark",
        legend_title_text="Musical neighbourhood",
        scene={"xaxis_title": "", "yaxis_title": "", "zaxis_title": ""},
    )

    temp_html = OUTPUT.with_name("music_map.building.html")
    fig.write_html(temp_html, include_plotlyjs=True)
    os.replace(temp_html, OUTPUT)

    # Final row-count report.
    print("\nFinal validation:", flush=True)
    print(f"  Embeddings: {embeddings.shape}", flush=True)
    print(f"  Mapping paths: {len(paths):,}", flush=True)
    print(f"  50D PCA: {np.load(PCA_FILE, mmap_mode='r').shape}", flush=True)
    print(f"  Cluster labels: {np.load(CLUSTERS_FILE, mmap_mode='r').shape}", flush=True)
    print(f"  3D coordinates: {np.load(PROJECTION, mmap_mode='r').shape}", flush=True)
    print(f"  Tracks plotted: {len(keep):,}", flush=True)
    print(f"  All files aligned: {len(paths) == n == len(labels) == len(coords)}", flush=True)
    print(f"  Interactive map: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
