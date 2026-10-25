"""Endâ€‘toâ€‘end playlist generation.

This script identifies the current track either from an audio snippet
(embedding match) or directly from a provided file path, selects
followâ€‘up tracks via the sequencing engine and optionally generates
naturalâ€‘language explanations via the LLM configured in `config.yaml`.

Usage:
    python generate_playlist.py --input-audio /path/to/clip.wav --length 20 --policy smooth
    python generate_playlist.py --start-path "/path/to/song.flac" --length 20 --policy contrast

Exactly one of --input-audio or --start-path must be provided.
"""
# scripts/generate_playlist.py â€” FAISS search + playlist (CLAP_Module + PyTorch 2.6+ allowlist + NumPy 2.x polyfill)

from pathlib import Path
from scripts import explanation as ex
from main import load_config

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import faiss
import soundfile as sf
import librosa

#from dejavu import Dejavu
#from dejavu.recognize import FileRecognizer

import io, os, tempfile

# --- stringify safety for LLM fields ---
def _coerce_text(val):
    if isinstance(val, (str, type(None))):
        return val or ""
    try:
        # last-resort to readable JSON
        import json as _json
        return _json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)

def _sanitize_payload(payload: dict):
    # summary strings
    s = payload.get("summary")
    if isinstance(s, dict):
        for k in ("top_pick", "overview", "reason", "note"):
            if k in s:
                s[k] = _coerce_text(s[k])
    # per-track strings
    for t in payload.get("tracks", []):
        if "title" in t: t["title"] = _coerce_text(t["title"])
        if "explanation" in t: t["explanation"] = _coerce_text(t["explanation"])



def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_mono(path: Path, target_sr: int | None = None):
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype("float32"), sr


def _polyfill_numpy_random_integers():
    try:
        if not hasattr(np.random, "integers"):
            np.random.integers = np.random.randint  # type: ignore[attr-defined]
    except Exception:
        pass


def _allowlist_torch_pickle_globals():
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
        import numpy.dtypes as ndt
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
        to_allow.append(np.core.multiarray.scalar)  # legacy helper
    except Exception:
        pass
    try:
        add_safe_globals([x for x in to_allow if x is not None])  # type: ignore[misc]
    except Exception:
        pass


def _load_clap():
    _polyfill_numpy_random_integers()
    _allowlist_torch_pickle_globals()
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def _embed_with_model(model, y: np.ndarray, sr: int) -> np.ndarray:
    x = y.reshape(1, -1).astype("float32")
    emb = model.get_audio_embedding_from_data(x=x, use_tensor=False)
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[0]
    return emb.reshape(-1)


def _embed_clip(model, clip_path: Path) -> np.ndarray:
    y, sr = read_mono(clip_path, target_sr=48000)
    emb = _embed_with_model(model, y, sr)
    return emb.astype("float32")


def _load_index_and_mapping(cfg):
    index_path = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.txt"))
    if not index_path.exists() or not map_path.exists():
        raise FileNotFoundError("Index/mapping missing. Run ingest first.")
    index = faiss.read_index(str(index_path))
        # mapping is JSON even though the file is named .txt
    try:
        with map_path.open("r", encoding="utf-8") as f:
            mapping = list(json.load(f))
    except Exception as e:
        raise RuntimeError(f"Failed to load mapping {map_path}: {e}")
    return index, mapping

def _embed_bytes(model, blob: bytes) -> np.ndarray:
    # Try direct decode (wav/flac/ogg)
    try:
        buf = io.BytesIO(blob)
        y, sr = sf.read(buf, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        # Fallback for mp3/m4a: write temp then use librosa/audioread
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(blob)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000
    return _embed_with_model(model, y.astype("float32"), sr).astype("float32")

def _load_feature_lookup(cfg):
    import os
    from pathlib import Path

    feature_file = Path(cfg.get("feature_file", "data/features_60s.json"))
    if not feature_file.exists():
        return {}

    with feature_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    def path_key(path):
        return os.path.normcase(os.path.normpath(str(path)))

    return {
        path_key(record["path"]): record
        for record in records
        if record.get("path")
    }


def _select_tracks(index, mapping, seed_emb: np.ndarray, length: int,
                   policy: str, seed_path=None, cfg=None,
                   include_plus_paths=False):
    import math
    import os

    seed = seed_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(seed)

    k = min(max(length * 30, 300), len(mapping))
    D, I = index.search(seed, k)
    cand = [
        (mapping[i], float(d))
        for i, d in zip(I[0].tolist(), D[0].tolist())
        if i >= 0
    ]
    print(f"\nFAISS candidate pool: {len(cand)} tracks before feature re-ranking")
    import re

    def in_plus_folder(path):
        normalised = str(path).replace("\\", "/")
        return re.search(r"\+/", normalised) is not None

    # Exclude tracks inside directories ending in + by default.
    # Recognises both Windows (\) and Linux (/) separators.
    seed_key = os.path.normcase(os.path.normpath(str(seed_path))) if seed_path else None

    before_filter = len(cand)

    if not include_plus_paths:
        cand = [
            (path, score) for path, score in cand
            if not in_plus_folder(path)
            or (
                seed_key is not None
                and os.path.normcase(os.path.normpath(str(path))) == seed_key
            )
        ]
        print(
            f"Excluded {before_filter - len(cand)} tracks from + folders; "
            f"{len(cand)} candidates remain"
        )
    else:
        print("Including tracks from + folders")
    
    for rank, (path, score) in enumerate(cand, 1):
        print(f"{rank:2d}. {score:.3f}  {Path(path).parent.name} â€” {Path(path).stem}")
    cand = [(path, score) for path, score in cand if os.path.normcase(os.path.normpath(str(path))) != os.path.normcase(os.path.normpath(str(seed_path)))]
    # Feature-aware ranking for a known seed track.
    if policy == "smooth" and seed_path and cfg:
        features = _load_feature_lookup(cfg)

        def path_key(path):
            return os.path.normcase(os.path.normpath(str(path)))

        seed_features = features.get(path_key(seed_path))
        seed_tempo = seed_features.get("tempo") if seed_features else None
        seed_energy = seed_features.get("energy") if seed_features else None
        print(f"Seed features: tempo={seed_tempo}, energy={seed_energy}")

        if seed_tempo is not None and seed_energy is not None and cand:
            scores = [score for _, score in cand]
            low, high = min(scores), max(scores)
            span = high - low or 1.0

            ranked = []
            missing_features = 0
            missing_tempo = 0
            missing_energy = 0
            for path, score in cand:
                clap_norm = (score - low) / span
                candidate = features.get(path_key(path))

                tempo = candidate.get("tempo") if candidate else None
                energy = candidate.get("energy") if candidate else None
                if tempo is None or energy is None:
                    missing_features += 1
                if tempo is None:
                    missing_tempo += 1
                if energy is None:
                    missing_energy += 1
                tempo_sim = None
                energy_sim = None

                if tempo is not None and energy is not None:
                    tempo_sim = math.exp(
                        -abs(math.log(
                            (float(tempo) + 1) / (float(seed_tempo) + 1)
                        ))
                    )
                    energy_sim = math.exp(
                        -abs(math.log(
                            (float(energy) + 0.001)
                            / (float(seed_energy) + 0.001)
                        ))
                    )
                    feature_sim = (tempo_sim + energy_sim) / 2
                    combined = 0.85 * clap_norm + 0.15 * feature_sim
                else:
                    # Missing features: use normalized CLAP alone.
                    combined = clap_norm

                ranked.append((
                    path, score, combined, tempo, energy,
                    tempo_sim, energy_sim
                ))
            print(
                f"\nFeature coverage: {len(cand) - missing_features}/{len(cand)} "
                f"complete; {missing_tempo} missing tempo; "
                f"{missing_energy} missing energy"
            )                
            ranked.sort(key=lambda item: item[2], reverse=True)

            print("\nRanking diagnostics (top candidates):")
            print(" #  CLAP    Tempo  Energy  T-sim  E-sim  Combined  Track")
            for i, item in enumerate(ranked[:min(length, 15)], 1):
                path, score, combined, tempo, energy, ts, es = item
                tempo_text = f"{tempo:.1f}" if tempo is not None else "-"
                energy_text = f"{energy:.3f}" if energy is not None else "-"
                ts_text = f"{ts:.3f}" if ts is not None else "-"
                es_text = f"{es:.3f}" if es is not None else "-"
                print(
                    f"{i:2d}  {score:.3f}  {tempo_text:>6}  "
                    f"{energy_text:>6}  {ts_text:>5}  {es_text:>5}  "
                    f"{combined:.3f}  {path}"
                )

            out = []
            seen = set()
            for path, score, combined, *_ in ranked:
                if path in seen:
                    continue
                seen.add(path)
                out.append((path, score))
                if len(out) >= length:
                    break
            return out

    # Original CLAP-only ranking, including contrast policy.
    out = []
    if policy == "smooth":
        seen = set()
        for path, score in sorted(cand, key=lambda item: item[1], reverse=True):
            if path in seen:
                continue
            seen.add(path)
            out.append((path, score))
            if len(out) >= length:
                break
    else:
        hi = sorted(cand, key=lambda item: item[1], reverse=True)
        lo = list(reversed(hi))
        hi_i = lo_i = 0
        seen = set()

        while len(out) < length and (hi_i < len(hi) or lo_i < len(lo)):
            if hi_i < len(hi):
                path, score = hi[hi_i]
                hi_i += 1
                if path not in seen:
                    out.append((path, score))
                    seen.add(path)
                    if len(out) >= length:
                        break

            if lo_i < len(lo):
                path, score = lo[lo_i]
                lo_i += 1
                if path not in seen:
                    out.append((path, score))
                    seen.add(path)

    return out

def _attach_llm_explanations(playlist: list[dict], policy: str, cfg: dict) -> dict[str, str]:
    """Attach per-transition explanations and return a global summary dict."""
    summary: dict[str, str] = {}
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", True):
        return summary

    llm_model = llm_cfg.get("model", "qwen2.5:3b")
    maxt = int(llm_cfg.get("max_tokens", 160))
    temp = float(llm_cfg.get("temperature", 0.7))

    # Per-transition (attach to NEXT track)
    for i in range(len(playlist) - 1):
        stats = playlist[i + 1].get("stats", {}) or {}
        try:
            txt = ex.explain_transition(playlist[i], playlist[i + 1], stats, llm_model, maxt, temp)
        except Exception:
            txt = ""
        playlist[i + 1]["explanation"] = txt

    # Global summary (top pick + overview)
    try:
        summary = ex.explain_playlist_summary(playlist, policy, llm_model, maxt, temp)
    except Exception:
        summary = {}

    return summary


def generate_from_bytes(audio_bytes: bytes, length: int = 20, policy: str = "smooth",
                        explain: bool = False, cfg_path: str = "config.yaml",
                        audio_format: str = "m4a") -> dict:
    import tempfile
    import os

    cfg = load_config(cfg_path)
    index, mapping = _load_index_and_mapping(cfg)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix="." + audio_format.lstrip("."), delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        match = _recognize_with_dejavu(Path(tmp_path))
        if not match:
            raise RuntimeError("PyDejavu could not identify the input audio.")

        seed_emb = _seed_embedding_from_match(mapping, match)
        if seed_emb is None:
            raise RuntimeError(f'No embedding found for "{match["song_name"]}".')

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    pairs = _select_tracks(index, mapping, seed_emb, length, policy)
    playlist = [{"path": p, "score": s, "title": Path(p).stem, "stats": {}} for p, s in pairs]

    summary: Dict[str, str] = {}
    if explain:
        summary = _attach_llm_explanations(playlist, policy, cfg)

    result = {
        "seed": {"source": "clip", "recognized": match["song_name"]},
        "policy": policy,
        "count": len(playlist),
        "tracks": playlist,
        "summary": summary,
    }
    _sanitize_payload(result)
    return result

def generate_from_path(start_path: str, length: int = 20, policy: str = "smooth",
                       explain: bool = False, cfg_path: str = "config.yaml") -> dict:
    cfg = load_config(cfg_path)
    index, mapping = _load_index_and_mapping(cfg)
    model = _load_clap()
    y, sr = read_mono(Path(start_path), target_sr=48000)
    seed_emb = _embed_with_model(model, y, sr).astype("float32")
    pairs = _select_tracks(
        index, mapping, seed_emb, length, policy,
        seed_path=start_path, cfg=cfg
    )
    playlist = [{"path": p, "score": s, "title": Path(p).stem, "stats": {}} for p, s in pairs]

    summary: Dict[str, str] = {}
    if explain:
        summary = _attach_llm_explanations(playlist, policy, cfg)

    result = {
        "seed": {"source": "path", "value": start_path},
        "policy": policy,
        "count": len(playlist),
        "tracks": playlist,
        "summary": summary,
    }
    _sanitize_payload(result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-audio", type=str, default="")
    ap.add_argument("--start-path", type=str, default="")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--policy", type=str, choices=["smooth", "contrast"], default="smooth")
    ap.add_argument(
        "--include-plus-paths",
        action="store_true",
        help="Include recommendation candidates from folders ending in +",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    index_path = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.json"))

    if not index_path.exists() or not map_path.exists():
        print("Index/mapping missing. Run ingest first.")
        return

    index = faiss.read_index(str(index_path))
    with map_path.open("r", encoding="utf-8") as f:
        mapping = list(json.load(f))

    # Seed embedding
    if args.input_audio:
        match = _recognize_with_dejavu(Path(args.input_audio))
        if not match:
            print("PyDejavu could not identify the input audio.")
            return

        print(f'Recognized: "{match["song_name"]}"')

        seed_emb = _embedding_for_song(mapping, match["song_name"])
        if seed_emb is None:
            print(f'No embedding found for "{match["song_name"]}".')
            return

    elif args.start_path:
        print("Loading CLAP (htsat-tiny)â€¦")
        model = _load_clap()
        seed_emb = _embed_clip(model, Path(args.start_path))
    else:
        print("Need --input-audio or --start-path.")
        return

    seed = seed_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(seed)

    # Select tracks using the shared feature-aware selector
    out = _select_tracks(
        index,
        mapping,
        seed_emb,
        args.length,
        args.policy,
        seed_path=args.start_path if args.start_path else None,
        cfg=cfg,
        include_plus_paths=args.include_plus_paths
    )

# Build playlist with the seed first, followed by recommendations.
    playlist = []

    if args.start_path:
        seed_path = str(Path(args.start_path))
        playlist.append({
            "path": seed_path,
            "score": 1.0,
            "title": Path(seed_path).stem,
            "stats": {}
        })

    seen = {
        os.path.normcase(os.path.normpath(t["path"]))
        for t in playlist
    }

    for p, s in out:
        key = os.path.normcase(os.path.normpath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        playlist.append({
            "path": str(p),
            "score": s,
            "title": Path(p).stem,
            "stats": {}
        })
        if len(playlist) >= args.length:
            break

    # LLM-forward output (per-transition + global summary)
    summary = {}
    try:
        llm_cfg = cfg.get("llm", {})
        if llm_cfg.get("enabled", True):
            model = llm_cfg.get("model", "qwen2.5:3b")
            maxt  = int(llm_cfg.get("max_tokens", 160))
            temp  = float(llm_cfg.get("temperature", 0.7))

            # Per-transition explanations (attach to NEXT track)
            for i in range(len(playlist) - 1):
                print(
                    f"LLM transition {i + 1}/{len(playlist) - 1}: "
                    f"{playlist[i]['title']} -> {playlist[i + 1]['title']}",
                    flush=True,
                )
                try:
                    stats = playlist[i + 1].get("stats", {}) or {}
                    txt = ex.explain_transition(
                        playlist[i], playlist[i + 1], stats,
                        model, maxt, temp
                    )
                    playlist[i + 1]["explanation"] = _coerce_text(txt)
                except Exception as e:
                    print(f"LLM transition failed: {e}", flush=True)
                    playlist[i + 1]["explanation"] = ""

            # Global summary (top pick + overview)
            try:
                print("LLM playlist summary: starting", flush=True)
                summary = ex.explain_playlist_summary(playlist, args.policy, model, maxt, temp)
            except Exception as e:
                print(f"LLM playlist summary failed: {e}", flush=True)
                summary = {}
    except Exception as e:
        print(f"LLM explanations skipped: {e}")

    # Pretty print to console
    if summary:
        print("\nTop pick:", summary.get("top_pick", ""))
        print("Overview:", summary.get("overview", ""))

    print("\nPlaylist:")
    for i, t in enumerate(playlist, 1):
        print(f"{i:2d}. {t['title']}  (sim={t['score']:.3f})")
        if t.get("explanation"):
            print(f"    â†’ {t['explanation']}")

    # Also write JSON payload to disk for the UI / export
    out_json = {"tracks": playlist, "summary": summary}
    out_path = Path(cfg.get("last_playlist_file", "data/last_playlist.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {out_path}")
    # Also export a standard M3U playlist for foobar2000
    m3u_path = out_path.with_suffix(".m3u8")
    with m3u_path.open("w", encoding="utf-8-sig", newline="\n") as f:
        f.write("#EXTM3U\n")
        for track in playlist:
            track_path = track.get("path")
            if track_path:
                f.write(str(track_path) + "\n")
    print(f"Saved M3U playlist: {m3u_path}")
    


def _recognize_with_dejavu(path: Path):
    from dejavu import Dejavu
    from dejavu.recognize import FileRecognizer

    config_path = Path(__file__).resolve().parents[1] / "dejavu.cnf"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    djv = Dejavu(config)
    recognizer = FileRecognizer(djv)
    return recognizer.recognize_file(str(path))
    
    
def _embedding_for_song(mapping, song_name: str):
    for i, p in enumerate(mapping):
        if Path(p).stem == song_name:
            embeddings_path = Path("data/embeddings.npy")
            embeddings = np.load(embeddings_path)
            return embeddings[i]
    return None
    
def _seed_embedding_from_match(mapping, match):
    if not match:
        return None
    return _embedding_for_song(mapping, match["song_name"])
    
    
def generate_from_path(start_path: str, length: int = 20, policy: str = "smooth",
                       explain: bool = False, cfg_path: str = "config.yaml") -> dict:
    cfg = load_config(cfg_path)
    index, mapping = _load_index_and_mapping(cfg)

    match = _recognize_with_dejavu(Path(start_path))
    if not match:
        raise RuntimeError("PyDejavu could not identify the input audio.")

    seed_emb = _seed_embedding_from_match(mapping, match)
    if seed_emb is None:
        raise RuntimeError(f'No embedding found for "{match["song_name"]}".')

    pairs = _select_tracks(index, mapping, seed_emb, length, policy)
    playlist = [{"path": p, "score": s, "title": Path(p).stem, "stats": {}} for p, s in pairs]

    summary: Dict[str, str] = {}
    if explain:
        summary = _attach_llm_explanations(playlist, policy, cfg)

    result = {
        "seed": {"source": "path", "value": start_path,
                 "recognized": match["song_name"]},
        "policy": policy,
        "count": len(playlist),
        "tracks": playlist,
        "summary": summary,
    }
    _sanitize_payload(result)
    return result

if __name__ == "__main__":
    main()


