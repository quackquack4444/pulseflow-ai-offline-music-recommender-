"""Safely reconcile PulseFlow CLAP embeddings with the current music library.

Default is a read-only dry run. Use --apply to commit changes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

AUDIO_EXTS = {
    ".mp3", ".m4a", ".flac", ".wav", ".aac", ".wma",
    ".aiff", ".aif", ".ogg", ".opus", ".alac",
}


def key_for(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def scan_tracks(root: Path) -> list[Path]:
    tracks: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        tracks.append(path)
    tracks.sort(key=lambda p: key_for(p))
    return tracks


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_json_mapping(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"Mapping must be a JSON list of path strings: {path}")
    return value


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("files"), dict):
        raise ValueError(f"Unrecognised manifest format: {path}")
    return value["files"]


def backup(path: Path, stamp: str) -> None:
    if path.exists():
        target = path.with_name(path.name + f".{stamp}.bak")
        shutil.copy2(path, target)
        print(f"Backup: {target}")


def atomic_json(path: Path, data) -> None:
    temp = path.with_name(path.name + ".maintenance.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp, path)


def atomic_npy(path: Path, data: np.ndarray) -> None:
    temp = path.with_name(path.name + ".maintenance.tmp")
    with temp.open("wb") as f:
        np.save(f, data)
    os.replace(temp, path)


def embed_tracks(paths: list[Path]) -> dict[str, np.ndarray]:
    # Reuse the tested CLAP loader/decoder from the project's embedding script.
    import embed_clap as ec

    print(f"Loading CLAP model for {len(paths)} new/changed track(s)...", flush=True)
    model = ec._load_clap()
    result: dict[str, np.ndarray] = {}
    failures: list[tuple[Path, str]] = []
    started = time.monotonic()
    for i, path in enumerate(paths, 1):
        print(f"Embedding {i}/{len(paths)}: {path}", flush=True)
        try:
            y, sr = ec.read_mono(path, target_sr=48000)
            vector = np.asarray(ec._embed_with_model(model, y, sr), dtype=np.float32).reshape(-1)
            if vector.shape != (512,):
                raise ValueError(f"Expected 512 dimensions, received {vector.shape}")
            result[key_for(path)] = vector
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
    print(f"Embedding pass took {(time.monotonic() - started) / 60:.1f} minutes.")
    if failures:
        print("ERROR: one or more tracks failed. No library index files have been changed:")
        for path, error in failures:
            print(f"  {path}\n    {error}")
        raise RuntimeError(f"{len(failures)} track(s) failed to embed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely reconcile CLAP embeddings and FAISS with the library.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--apply", action="store_true", help="Apply changes; without this option the run is read-only")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    project_root = config_path.parent

    def project_path(key: str, default: str) -> Path:
        p = Path(cfg.get(key, default))
        return p if p.is_absolute() else project_root / p

    music_root = Path(cfg.get("music_dir") or "")
    if not music_root.is_absolute():
        music_root = project_root / music_root
    if not music_root.is_dir():
        print(f"ERROR: music_dir does not exist: {music_root}")
        return 2

    embeddings_path = project_path("embedding_file", "data/embeddings.npy")
    mapping_path = project_path("mapping_file", "data/embedding_mapping.txt")
    index_path = project_path("faiss_index_file", "data/faiss.index")
    manifest_path = project_root / "data" / "clap_library_manifest.json"

    if not embeddings_path.is_file() or not mapping_path.is_file():
        print("ERROR: embeddings or mapping file is missing. Refusing to initialise or rebuild automatically.")
        print(f"  Embeddings: {embeddings_path}\n  Mapping: {mapping_path}")
        return 2

    try:
        embeddings = np.load(embeddings_path)
        mapping = load_json_mapping(mapping_path)
        old_manifest = load_manifest(manifest_path)
    except Exception as exc:
        print(f"ERROR loading current data: {type(exc).__name__}: {exc}")
        return 2

    if embeddings.ndim != 2 or embeddings.shape[1] != 512 or embeddings.shape[0] != len(mapping):
        print(f"ERROR: embeddings/mapping mismatch or wrong shape: {embeddings.shape} vs {len(mapping)} paths")
        print("No files changed. Restore a matching backup before proceeding.")
        return 2

    tracks = scan_tracks(music_root)
    current_by_key = {key_for(p): p for p in tracks}
    current_keys = set(current_by_key)
    row_keys = [key_for(p) for p in mapping]
    row_counts = Counter(row_keys)
    row_positions: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(row_keys):
        row_positions[key].append(idx)

    deleted_keys = set(row_counts) - current_keys
    new_keys = current_keys - set(row_counts)
    missing_manifest = not manifest_path.exists()
    changed_keys: set[str] = set()
    if not missing_manifest:
        for key in current_keys & set(row_counts):
            old = old_manifest.get(key)
            if old is None:
                # First time seeing this path in the manifest: cannot infer its earlier state.
                continue
            try:
                now = file_signature(current_by_key[key])
            except OSError as exc:
                print(f"ERROR statting {current_by_key[key]}: {exc}")
                return 2
            if now["size"] != old.get("size") or now["mtime_ns"] != old.get("mtime_ns"):
                changed_keys.add(key)

    stale_rows = sum(row_counts[k] for k in deleted_keys)
    changed_rows = sum(row_counts[k] for k in changed_keys)
    needs_embeddings = [current_by_key[k] for k in sorted(new_keys | changed_keys)]

    print("PulseFlow CLAP library maintenance")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (no changes will be made)'}")
    print(f"Library: {music_root}")
    print(f"Existing embedding rows: {len(mapping)}; dimensions: {embeddings.shape[1]}")
    print(f"Rows to remove for deleted/excluded paths: {stale_rows}")
    print(f"New tracks to embed: {len(new_keys)}")
    print(f"Modified tracks to re-embed: {len(changed_keys)} (replaces {changed_rows} existing row(s), preserving duplicates)")
    print(f"Total new/changed tracks to process: {len(needs_embeddings)}")
    if missing_manifest:
        print("NOTE: No baseline manifest exists yet. This run can detect new/deleted paths, but cannot identify tracks modified before the baseline was created.")
    if missing_manifest:
        print("The manifest will be established from current file sizes/timestamps after a successful apply.")

    planned_rows = len(mapping) - stale_rows - changed_rows + sum(row_counts[k] for k in changed_keys) + len(new_keys)
    # The changed rows are replaced one-for-one, so the net effect is removals + additions.
    planned_rows = len(mapping) - stale_rows + len(new_keys)
    print(f"Expected embedding rows after reconciliation: {planned_rows}")

    if not args.apply:
        print("Dry run complete. Review the counts; use --apply to perform the maintenance.")
        return 0

    if not deleted_keys and not new_keys and not changed_keys:
        # Even with no vector changes, establish/update the baseline manifest.
        manifest_files = {key_for(p): {"path": str(p), **file_signature(p)} for p in tracks}
        if missing_manifest or manifest_files != old_manifest:
            if manifest_path.exists():
                backup(manifest_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
            atomic_json(manifest_path, {"version": 1, "files": manifest_files})
            print(f"Baseline manifest saved: {manifest_path}")
        else:
            print("No library changes detected; embeddings and FAISS left untouched.")
        return 0

    try:
        new_vectors = embed_tracks(needs_embeddings) if needs_embeddings else {}
    except Exception as exc:
        print(f"STOP: {exc}")
        return 2

    # Construct reconciled rows in existing order, preserving duplicate mapping entries.
    out_vectors: list[np.ndarray] = []
    out_mapping: list[str] = []
    for i, old_path in enumerate(mapping):
        key = row_keys[i]
        if key in deleted_keys:
            continue
        if key in changed_keys:
            out_vectors.append(new_vectors[key].copy())
        else:
            out_vectors.append(embeddings[i])
        out_mapping.append(old_path)

    # Add each newly discovered path once in deterministic order.
    for key in sorted(new_keys):
        out_vectors.append(new_vectors[key])
        out_mapping.append(str(current_by_key[key]))

    updated_embeddings = np.vstack(out_vectors).astype(np.float32, copy=False) if out_vectors else np.empty((0, 512), dtype=np.float32)
    if updated_embeddings.shape[0] != len(out_mapping):
        print("ERROR: internal count mismatch; no files changed.")
        return 2

    try:
        import faiss
        index_vectors = updated_embeddings.copy()
        faiss.normalize_L2(index_vectors)
        index = faiss.IndexFlatIP(index_vectors.shape[1])
        index.add(index_vectors)
    except Exception as exc:
        print(f"ERROR preparing FAISS index: {type(exc).__name__}: {exc}")
        print("No persistent files have been changed.")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_to_backup = [embeddings_path, mapping_path, index_path, manifest_path]
    try:
        for path in paths_to_backup:
            backup(path, stamp)
        # Prepare all outputs in temporary files before replacing the originals.
        emb_temp = embeddings_path.with_name(embeddings_path.name + ".maintenance.tmp")
        map_temp = mapping_path.with_name(mapping_path.name + ".maintenance.tmp")
        index_temp = index_path.with_name(index_path.name + ".maintenance.tmp")
        manifest_temp = manifest_path.with_name(manifest_path.name + ".maintenance.tmp")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with emb_temp.open("wb") as f:
            np.save(f, updated_embeddings)
        with map_temp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(out_mapping, f, ensure_ascii=False, indent=2)
            f.write("\n")
        faiss.write_index(index, str(index_temp))
        manifest_files = {key_for(p): {"path": str(p), **file_signature(p)} for p in tracks}
        with manifest_temp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump({"version": 1, "files": manifest_files}, f, ensure_ascii=False, indent=2)
            f.write("\n")

        # Commit after every output has been prepared successfully.
        os.replace(emb_temp, embeddings_path)
        os.replace(map_temp, mapping_path)
        os.replace(index_temp, index_path)
        os.replace(manifest_temp, manifest_path)
    except Exception as exc:
        print(f"ERROR writing maintenance outputs: {type(exc).__name__}: {exc}")
        print("If the commit was interrupted, restore the timestamped .bak files listed above.")
        return 2

    # Verify persisted counts and index dimensions.
    try:
        check_emb = np.load(embeddings_path, mmap_mode="r")
        check_map = load_json_mapping(mapping_path)
        check_index = faiss.read_index(str(index_path))
        if check_emb.shape[0] != len(check_map) or check_index.ntotal != len(check_map) or check_index.d != 512:
            raise ValueError(f"post-write validation failed: embeddings={check_emb.shape}, mapping={len(check_map)}, FAISS={check_index.ntotal}x{check_index.d}")
    except Exception as exc:
        print(f"ERROR: post-write validation failed: {exc}")
        print("Timestamped backups were created; do not run playlist generation until repaired.")
        return 2

    print("Maintenance completed and validated.")
    print(f"Embedding rows: {len(check_map)}")
    print(f"FAISS vectors: {check_index.ntotal}; dimensions: {check_index.d}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
