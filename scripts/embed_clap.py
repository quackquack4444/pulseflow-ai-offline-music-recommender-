"""Compute CLAP embeddings for your music library.

This script iterates over all audio files in a given directory (recursively),
computes a 512‑dimensional CLAP embedding for each track and writes two
artefacts:

* a NumPy array of shape (n_tracks, 512) containing embeddings
* a mapping file that associates each row of the embedding array with
  the absolute path of the corresponding audio file

Embeddings are stored in the location specified by `embedding_file` in
config.yaml.  The mapping file is stored in `mapping_file` in config.yaml.

You should run this script after fingerprinting and feature extraction.  It
requires a GPU for optimal performance but will fall back to CPU if none is
available.
"""
# scripts/embed_clap.py — build CLAP embeddings (CLAP_Module + PyTorch 2.6+ safe load allowlist + NumPy 2.x polyfill)

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import yaml
import numpy as np
import soundfile as sf
import librosa
import subprocess

AUDIO_EXTS = {
    ".mp3", ".m4a", ".flac", ".wav", ".aac", ".wma",
    ".aiff", ".aif", ".ogg", ".opus", ".alac",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_tracks(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue

        yield p



def read_mono(path: Path, target_sr: int | None = None):
    """Decode audio through FFmpeg, returning mono float32 samples."""
    if target_sr is None:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype("float32"), sr

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-map", "0:a:0",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(target_sr),
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg audio decoding failed: {error}")

    y = np.frombuffer(result.stdout, dtype="<f4").copy()
    if y.size == 0:
        raise RuntimeError("FFmpeg returned no decoded audio samples.")

    return y, target_sr



def _polyfill_numpy_random_integers():
    """NumPy 2.x removed np.random.integers; some deps still call it."""
    try:
        if not hasattr(np.random, "integers"):
            np.random.integers = np.random.randint  # type: ignore[attr-defined]
    except Exception:
        pass


def _allowlist_torch_pickle_globals():
    """Work around torch.load(weights_only=True) by allowlisting NumPy types/classes."""
    try:
        from torch.serialization import add_safe_globals  # PyTorch 2.6+
    except Exception:
        try:
            from torch.serialization import safe_globals as add_safe_globals  # type: ignore
        except Exception:
            return

    to_allow = [
        np.dtype,
        getattr(np, "bool_", None),
        getattr(np, "float32", None), getattr(np, "float64", None),
        getattr(np, "int8", None), getattr(np, "int16", None),
        getattr(np, "int32", None), getattr(np, "int64", None),
        getattr(np, "uint8", None), getattr(np, "uint16", None),
        getattr(np, "uint32", None), getattr(np, "uint64", None),
        getattr(np, "complex64", None), getattr(np, "complex128", None),
    ]
    try:
        import numpy.dtypes as ndt  # NumPy 2.x classes
        for name in [
            "BoolDType",
            "Float32DType", "Float64DType",
            "Int8DType", "Int16DType", "Int32DType", "Int64DType",
            "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
            "Complex64DType", "Complex128DType",
        ]:
            cls = getattr(ndt, name, None)
            if cls is not None:
                to_allow.append(cls)
    except Exception:
        pass
    try:
        to_allow.append(np.core.multiarray.scalar)  # legacy scalar used in some ckpts
    except Exception:
        pass
    to_allow = [x for x in to_allow if x is not None]
    try:
        add_safe_globals(to_allow)  # type: ignore[misc]
    except Exception:
        pass


def _load_clap():
    """
    Use the stable pip API: laion_clap.CLAP_Module + load_ckpt().
    First run downloads weights to cache.
    """
    _polyfill_numpy_random_integers()
    _allowlist_torch_pickle_globals()
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def _embed_with_model(model, y: np.ndarray, sr: int) -> np.ndarray:
    """CLAP expects 48kHz audio and (N, T) float32. Return a 1D vector."""
    x = y.reshape(1, -1).astype("float32")
    emb = model.get_audio_embedding_from_data(x=x, use_tensor=False)
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[0]
    return emb.reshape(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                help="Maximum number of new tracks to embed (0 = all)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    music_dir = Path(cfg.get("music_dir") or "")
    if not music_dir.exists():
        print("music_dir not set or missing.")
        return

    emb_path = Path(cfg.get("embedding_file", "data/embeddings.npy"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.txt"))
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing embeddings/mapping unless --force was specified.
    existing_embs = None
    existing_mapping = []

    if emb_path.exists() and map_path.exists() and not args.force:
        try:
            existing_embs = np.load(emb_path)
            with map_path.open("r", encoding="utf-8") as f:
                existing_mapping = list(json.load(f))

            if existing_embs.ndim != 2:
                raise ValueError(
                    f"Embeddings must be 2D; got shape {existing_embs.shape}"
                )
            if existing_embs.shape[0] != len(existing_mapping):
                raise ValueError(
                    f"COUNT MISMATCH: {existing_embs.shape[0]} embedding rows "
                    f"but {len(existing_mapping)} mapping paths. "
                    "Restore a matching backup before continuing."
                )

            print(
                f"Loaded existing embeddings: "
                f"{existing_embs.shape[0]} tracks"
            )
        except Exception as e:
            print(f"Could not load existing embeddings: {e}")
            raise SystemExit(
                "STOP: existing embeddings/mapping could not be loaded. "
                "Existing data was not intentionally reset."
            )

    # Find tracks which don't already have an embedding.
    tracks = list(iter_tracks(music_dir))

    if existing_embs is not None:
        existing_paths = set(existing_mapping)
        new_tracks = [
            p for p in tracks
            if str(p) not in existing_paths
        ]

        print(f"Library contains {len(tracks)} audio files.")
        print(f"Already embedded: {len(existing_mapping)}")
        print(f"New tracks to embed: {len(new_tracks)}")
    else:
        new_tracks = tracks
        print(f"Tracks to embed: {len(new_tracks)}")

    if args.limit > 0:
        new_tracks = new_tracks[:args.limit]
        print(f"Limited to {len(new_tracks)} tracks for this run.")

    if not new_tracks:
        print("No new tracks to embed.")
        return

    print("Loading CLAP (htsat-tiny)…")
    model = _load_clap()

    new_embs = []
    new_mapping = []
    count = 0
    checkpoint_size = 100

    print(f"Starting CLAP embedding of {len(new_tracks)} new tracks...", flush=True)
    started_at = time.monotonic()

    def save_checkpoint():
        nonlocal existing_embs, existing_mapping, new_embs, new_mapping

        if not new_embs:
            return

        batch_embs = np.vstack(new_embs).astype("float32")

        if existing_embs is not None:
            existing_embs = np.vstack([existing_embs, batch_embs])
            existing_mapping.extend(new_mapping)
        else:
            existing_embs = batch_embs
            existing_mapping = list(new_mapping)

        # Write embeddings and mapping together.
        np.save(emb_path, existing_embs)

        with map_path.open("w", encoding="utf-8") as f:
            json.dump(existing_mapping, f, ensure_ascii=False, indent=2)

        print(
            f"CHECKPOINT SAVED: {len(existing_mapping)} embeddings; "
            f"mapping entries: {len(existing_mapping)}",
            flush=True,
        )

        new_embs.clear()
        new_mapping.clear()

    for attempted, p in enumerate(new_tracks, start=1):
        elapsed = time.monotonic() - started_at
        if attempted > 1 and elapsed > 0:
            seconds_per_track = elapsed / (attempted - 1)
            remaining_seconds = (len(new_tracks) - attempted + 1) * seconds_per_track
            eta_minutes = int(remaining_seconds // 60)
            eta_seconds = int(remaining_seconds % 60)
            eta_text = f"{eta_minutes}m {eta_seconds:02d}s"
        else:
            eta_text = "calculating..."
        print(
            f"PROGRESS {attempted}/{len(new_tracks)} "
            f"({attempted / len(new_tracks):.1%}) | "
            f"elapsed {int(elapsed // 60)}m {int(elapsed % 60):02d}s | "
            f"ETA {eta_text} | {p.name}",
            flush=True,
        )
        try:
            y, sr = read_mono(p, target_sr=48000)
            emb = _embed_with_model(model, y, sr).astype("float32")

            new_embs.append(emb)
            new_mapping.append(str(p))

            count += 1
            print(f"  EMBEDDED OK ({count} successful this run)", flush=True)

            if len(new_embs) >= checkpoint_size:
                save_checkpoint()

        except Exception as e:
            print(f"Embedding failed for: {p}", flush=True)
            print(f"Error type: {type(e).__name__}; details: {repr(e)}", flush=True)

    # Save anything remaining after the final batch.
    save_checkpoint()

    print(f"Wrote embeddings: {emb_path} shape={existing_embs.shape}")
    print(f"Wrote mapping: {map_path} ({len(existing_mapping)} paths)")
    
if __name__ == "__main__":
    main()
