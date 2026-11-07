
#!/usr/bin/env python3
"""
Export PulseFlow's precomputed Sonic Neighbourhood graph to SQLite
for the Android PulseFlow app.

Run from the PulseFlow project root:
    python scripts/export_sonic_neighbourhood_android.py

Optional:
    python scripts/export_sonic_neighbourhood_android.py --output data/pulseflow_sonic_neighbourhood.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile

from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

import numpy as np


SCHEMA_VERSION = 2
EXPECTED_DIMENSIONS = 512
SCORES_FILE = Path(r"data\clap_mood_style_scores.npy")
SCORES_METADATA_FILE = Path(r"data\clap_mood_style_scores.json")
EXPECTED_CATEGORY_COUNT = 21


def load_mapping(mapping_file: Path) -> list[str]:
    with mapping_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, list) or not mapping:
        raise ValueError("Mapping must be a non-empty JSON list.")

    if any(not isinstance(p, str) or not p.strip() for p in mapping):
        raise ValueError("Mapping contains an empty or invalid path.")

    return mapping

def load_mood_style_scores(
    scores_file: Path,
    metadata_file: Path,
    mapping_file: Path,
    mapping: list[str],
):
    if not scores_file.is_file():
        raise FileNotFoundError(f"Score matrix not found: {scores_file}")

    if not metadata_file.is_file():
        raise FileNotFoundError(f"Score metadata not found: {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    scores = np.load(scores_file, allow_pickle=False)

    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported mood/style score schema version.")

    if metadata.get("track_count") != len(mapping):
        raise ValueError(
            f"Score metadata track count mismatch: "
            f"metadata={metadata.get('track_count')}, mapping={len(mapping)}"
        )

    if scores.shape != (len(mapping), EXPECTED_CATEGORY_COUNT):
        raise ValueError(
            f"Unexpected score matrix shape: {scores.shape}; "
            f"expected ({len(mapping)}, {EXPECTED_CATEGORY_COUNT})."
        )

    if metadata.get("score_shape") != list(scores.shape):
        raise ValueError("Score matrix shape does not match its metadata.")

    if scores.dtype != np.float32:
        raise ValueError(f"Expected float32 scores; found {scores.dtype}.")

    if metadata.get("score_dtype") != str(scores.dtype):
        raise ValueError("Score matrix dtype does not match its metadata.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Score matrix contains non-finite values.")

    raw_mapping_hash = hashlib.sha256(
        mapping_file.read_bytes()
    ).hexdigest()

    if metadata.get("mapping_sha256") != raw_mapping_hash:
        raise ValueError(
            "Mood/style score mapping hash does not match "
            "embedding_mapping.txt."
        )

    category_names = metadata.get("category_names", [])
    mood_names = metadata.get("mood_names", [])
    style_names = metadata.get("style_names", [])

    if len(category_names) != EXPECTED_CATEGORY_COUNT:
        raise ValueError("Unexpected number of category names.")

    if mood_names + style_names != category_names:
        raise ValueError(
            "Category order does not match the mood/style category lists."
        )

    if len(set(category_names)) != EXPECTED_CATEGORY_COUNT:
        raise ValueError("Duplicate category names found.")

    print(
        f"Validated {len(category_names)} mood/style categories "
        f"for {len(mapping):,} tracks."
    )

    return scores, metadata

def mapping_digest(mapping: list[str]) -> str:
    # Must match the hash method used by the existing playlist script.
    payload = json.dumps(
        mapping,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_relative_path(source_path: str, music_root: str) -> str:
    source = source_path.strip().replace("/", "\\")
    root = music_root.strip().replace("/", "\\").rstrip("\\")

    prefix = root + "\\"

    if not source.casefold().startswith(prefix.casefold()):
        raise ValueError(
            "Track path is outside the configured music root:\n"
            f"Track: {source_path}\n"
            f"Root:  {music_root}"
        )

    relative = source[len(prefix):].replace("\\", "/")
    pure = PurePosixPath(relative)

    if (
        not relative
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")

    return pure.as_posix()


def validate_graph(graph_file: Path, mapping: list[str]):
    with np.load(graph_file, allow_pickle=False) as graph:
        required = {
            "neighbour_ids",
            "neighbour_scores",
            "mapping_hash",
            "row_count",
            "dimensions",
            "neighbour_count",
        }

        missing = required.difference(graph.files)

        if missing:
            raise ValueError(
                f"Graph is missing required arrays: {sorted(missing)}"
            )

        ids = np.asarray(graph["neighbour_ids"], dtype=np.int32)
        scores = np.asarray(graph["neighbour_scores"], dtype=np.float32)

        stored_hash = str(graph["mapping_hash"].item())
        row_count = int(graph["row_count"].item())
        dimensions = int(graph["dimensions"].item())
        neighbour_count = int(graph["neighbour_count"].item())

        method = (
            str(graph["method"].item())
            if "method" in graph.files
            else "unspecified"
        )

    expected_hash = mapping_digest(mapping)

    if stored_hash != expected_hash:
        raise ValueError(
            "Graph mapping hash does not match embedding_mapping.txt. "
            "Export stopped; do not use a mismatched graph."
        )

    if row_count != len(mapping) or ids.shape[0] != len(mapping):
        raise ValueError(
            f"Row-count mismatch: mapping={len(mapping)}, "
            f"graph metadata={row_count}, neighbour IDs={ids.shape}"
        )

    if dimensions != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS}-D CLAP graph; "
            f"found {dimensions}."
        )

    if ids.ndim != 2 or scores.shape != ids.shape:
        raise ValueError(
            f"Invalid graph shapes: IDs={ids.shape}, scores={scores.shape}"
        )

    if ids.shape[1] != neighbour_count:
        raise ValueError(
            f"Neighbour count mismatch: metadata={neighbour_count}, "
            f"array width={ids.shape[1]}"
        )

    if np.any(ids < -1) or np.any(ids >= len(mapping)):
        raise ValueError("Graph contains out-of-range neighbour IDs.")

    valid = ids >= 0

    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("Valid neighbours contain non-finite scores.")

    if np.any(np.isfinite(scores[~valid])):
        raise ValueError("Padded neighbour slots should contain NaN scores.")

    row_ids = np.arange(len(mapping), dtype=np.int32)[:, None]

    if np.any(ids[valid] == np.broadcast_to(row_ids, ids.shape)[valid]):
        raise ValueError("Graph contains a track as its own neighbour.")

    return ids, scores, neighbour_count, method, expected_hash



def create_database(
    output_file: Path,
    relative_paths: list[str],
    ids: np.ndarray,
    scores: np.ndarray,
    category_scores: np.ndarray,
    category_metadata: dict,
    neighbour_count: int,
    method: str,
    source_hash: str,
    music_root: str,
) -> None:

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=output_file.stem + ".building.",
        suffix=".db",
        dir=str(output_file.parent),
    )
    os.close(fd)

    temp_file = Path(temp_name)

    try:
        conn = sqlite3.connect(str(temp_file))

        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")

            conn.executescript("""
                CREATE TABLE database_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE tracks (
                    row_id INTEGER PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    file_name TEXT NOT NULL
                );

                CREATE INDEX idx_tracks_relative_path
                    ON tracks(relative_path);

                CREATE TABLE neighbours (
                    source_row_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    target_row_id INTEGER NOT NULL,
                    similarity REAL NOT NULL,

                    PRIMARY KEY (source_row_id, position),

                    FOREIGN KEY (source_row_id)
                        REFERENCES tracks(row_id),

                    FOREIGN KEY (target_row_id)
                        REFERENCES tracks(row_id)
                );

                CREATE INDEX idx_neighbours_target
                    ON neighbours(target_row_id);

                CREATE TABLE categories (
                    category_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    category_type TEXT NOT NULL
                );

                CREATE TABLE track_category_scores (
                    track_row_id INTEGER NOT NULL,
                    category_id INTEGER NOT NULL,
                    score REAL NOT NULL,

                    PRIMARY KEY (track_row_id, category_id),

                    FOREIGN KEY (track_row_id)
                        REFERENCES tracks(row_id),

                    FOREIGN KEY (category_id)
                        REFERENCES categories(category_id)
                );

                CREATE INDEX idx_category_scores_category_score
                    ON track_category_scores(category_id, score DESC);
            """)

            duplicate_count = (
                len(relative_paths) - len(set(relative_paths))
            )

            info = {
                "schema_version": str(SCHEMA_VERSION),
                "data_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "track_count": str(len(relative_paths)),
                "neighbour_count": str(neighbour_count),
                "embedding_dimensions": str(EXPECTED_DIMENSIONS),
                "similarity": "cosine_similarity",
                "graph_method": method,
                "source_mapping_sha256": source_hash,
                "source_music_root": music_root,
                "path_format": "relative_posix",
                "duplicate_path_rows": str(duplicate_count),
            }

            conn.executemany(
                "INSERT INTO database_info(key, value) VALUES (?, ?)",
                sorted(info.items()),
            )

            # Insert tracks first, so category scores and neighbours
            # can safely reference their row IDs.
            track_rows = [
                (
                    row_id,
                    relative_path,
                    PurePosixPath(relative_path).name,
                )
                for row_id, relative_path in enumerate(relative_paths)
            ]

            conn.executemany(
                """
                INSERT INTO tracks(row_id, relative_path, file_name)
                VALUES (?, ?, ?)
                """,
                track_rows,
            )

            # Insert category definitions.
            category_names = category_metadata["category_names"]
            mood_names = set(category_metadata["mood_names"])
            style_names = set(category_metadata["style_names"])

            category_rows = [
                (
                    category_id,
                    name,
                    "mood" if name in mood_names else "style",
                )
                for category_id, name in enumerate(category_names)
            ]

            conn.executemany(
                """
                INSERT INTO categories
                    (category_id, name, category_type)
                VALUES (?, ?, ?)
                """,
                category_rows,
            )

            # Insert the original CLAP scores for every track/category.
            category_score_rows = (
                (
                    track_row_id,
                    category_id,
                    float(category_scores[track_row_id, category_id]),
                )
                for track_row_id in range(len(relative_paths))
                for category_id in range(len(category_names))
            )

            conn.executemany(
                """
                INSERT INTO track_category_scores
                    (track_row_id, category_id, score)
                VALUES (?, ?, ?)
                """,
                category_score_rows,
            )

            # Insert the existing sonic-neighbour relationships.
            neighbour_rows = []

            for source_id, (row_ids, row_scores) in enumerate(
                zip(ids, scores)
            ):
                position = 0

                for target_id, score in zip(row_ids, row_scores):
                    target_id = int(target_id)

                    if target_id < 0:
                        continue

                    neighbour_rows.append(
                        (
                            source_id,
                            position,
                            target_id,
                            float(score),
                        )
                    )

                    position += 1

            conn.executemany(
                """
                INSERT INTO neighbours
                    (source_row_id, position, target_row_id, similarity)
                VALUES (?, ?, ?, ?)
                """,
                neighbour_rows,
            )
            neighbour_count = len(neighbour_rows)
            conn.execute(
                "UPDATE database_info SET value = ? WHERE key = 'neighbour_count'",
                (str(neighbour_count),),
            )

            conn.commit()

            # Verify all exported tables.
            stored_tracks = conn.execute(
                "SELECT COUNT(*) FROM tracks"
            ).fetchone()[0]

            stored_edges = conn.execute(
                "SELECT COUNT(*) FROM neighbours"
            ).fetchone()[0]

            stored_categories = conn.execute(
                "SELECT COUNT(*) FROM categories"
            ).fetchone()[0]

            stored_category_scores = conn.execute(
                "SELECT COUNT(*) FROM track_category_scores"
            ).fetchone()[0]

            expected_category_scores = (
                len(relative_paths) * len(category_names)
            )

            if stored_tracks != len(relative_paths):
                raise RuntimeError("SQLite track-count verification failed.")

            if stored_edges != len(neighbour_rows):
                raise RuntimeError("SQLite neighbour-count verification failed.")

            if stored_categories != len(category_names):
                raise RuntimeError("SQLite category-count verification failed.")

            if stored_category_scores != expected_category_scores:
                raise RuntimeError(
                    "SQLite category-score count verification failed."
                )

            foreign_key_errors = conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

            if foreign_key_errors:
                raise RuntimeError(
                    f"SQLite foreign-key check failed: "
                    f"{foreign_key_errors[:5]}"
                )

            print(f"Verified tracks: {stored_tracks:,}", flush=True)
            print(f"Verified neighbours: {stored_edges:,}", flush=True)
            print(f"Verified categories: {stored_categories}", flush=True)
            print(
                f"Verified track/category scores: "
                f"{stored_category_scores:,}",
                flush=True,
            )

        finally:
            conn.close()

        # Replace the output only after successful creation and validation.
        os.replace(temp_file, output_file)

    except Exception:
        temp_file.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export PulseFlow's CLAP neighbour graph to SQLite."
    )

    parser.add_argument(
        "--music-root",
        default=r"M:\Music\Music",
        help="Windows music-library root.",
    )

    parser.add_argument(
        "--mapping",
        default=r"data\embedding_mapping.txt",
        help="Ordered JSON mapping file.",
    )

    parser.add_argument(
        "--graph",
        default=r"data\sonic_neighbourhood.npz",
        help="Precomputed neighbour graph.",
    )

    parser.add_argument(
        "--output",
        default=r"data\pulseflow_sonic_neighbourhood.db",
        help="Output SQLite database.",
    )

    args = parser.parse_args()

    mapping_file = Path(args.mapping)
    graph_file = Path(args.graph)
    output_file = Path(args.output)

    for required_file in (mapping_file, graph_file):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {required_file}"
            )

    print("Loading ordered track mapping...", flush=True)
    mapping = load_mapping(mapping_file)
    
    print("Loading and validating mood/style scores...", flush=True)
    category_scores, category_metadata = load_mood_style_scores(
        SCORES_FILE,
        SCORES_METADATA_FILE,
        mapping_file,
        mapping,
    ) 
    
    print("Converting paths to relative form...", flush=True)
    relative_paths = [
        to_relative_path(path, args.music_root)
        for path in mapping
    ]

    print("Validating neighbour graph and mapping hash...", flush=True)
    ids, scores, neighbour_count, method, source_hash = validate_graph(
        graph_file,
        mapping,
    )

    print("Writing SQLite database...", flush=True)

    create_database(
        output_file=output_file,
        relative_paths=relative_paths,
        ids=ids,
        scores=scores,
        category_scores=category_scores,
        category_metadata=category_metadata,
        neighbour_count=neighbour_count,
        method=method,
        source_hash=source_hash,
        music_root=args.music_root,
    )

    edge_count = int(np.count_nonzero(ids >= 0))
    duplicate_count = len(relative_paths) - len(set(relative_paths))

    print()
    print("PulseFlow Android database export successful.")
    print(f"Tracks:               {len(mapping):,}")
    print(f"Neighbour links:      {edge_count:,}")
    print(f"Neighbours per track: {neighbour_count}")
    print(f"Duplicate path rows:  {duplicate_count:,} (preserved)")
    print(f"Mapping SHA-256:      {source_hash}")
    print(f"SQLite database:      {output_file.resolve()}")
    print()
    print("Android path rule:")
    print("SD-card-root / syncr / relative_path")
    print()
    print("Example:")
    print("syncr/Artist/Album+/Track.flac")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
