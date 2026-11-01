r"""One-off: give every track CLAP mood / sound / voice tags by zero-shot text prompts.

Run once from the project folder (musicrec environment):
    python scripts\make_clap_tags.py

How it works: CLAP's text side and audio side share one 512-D space. Each tag is a few
short descriptions ("calm, peaceful, relaxing music", ...). We embed those descriptions
with the SAME model that made data\embeddings.npy (embed_clap.py uses
laion_clap.CLAP_Module(enable_fusion=False) + load_ckpt(), and so does this script),
then score every track by cosine similarity. Scores are converted to z-scores per tag
across the whole library, so a tag means "unusually strong for this library", which stops
one generic phrase from winning for every track.

Writes:
    data\clap_tag_scores.npy    (n_tracks x n_tags, float16 z-scores)
    data\clap_tag_names.json    (tag names and groups, same order as the columns)
The map script picks these up automatically if they are present and match the library.
Re-run this script whenever embeddings.npy or embedding_mapping.txt change.

Tags are relative descriptors, not facts: CLAP is good at coarse mood, instruments and
vocal-vs-instrumental, and poor at key, major/minor and tempo. A non-fusion CLAP model also
only 'hears' about 10 seconds of each track, so tags describe that excerpt.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMBEDDINGS_FILE = DATA / "embeddings.npy"
MAPPING_FILE = DATA / "embedding_mapping.txt"
OUT_SCORES = DATA / "clap_tag_scores.npy"
OUT_NAMES = DATA / "clap_tag_names.json"

# (group, tag name, descriptions). Edit freely: more wordings per tag = steadier scores.
TAGS = [
    ("mood", "calm", ["calm, peaceful, relaxing music", "gentle, soothing, quiet music", "a tranquil, serene piece of music"]),
    ("mood", "energetic", ["energetic, high-energy, driving music", "fast, upbeat, lively music", "an intense, powerful, energetic piece of music"]),
    ("mood", "happy", ["happy, joyful, cheerful music", "bright, uplifting, feel-good music", "a sunny, playful, upbeat piece of music"]),
    ("mood", "sad", ["sad, sorrowful, melancholic music", "a mournful, wistful, emotional piece of music", "slow, sad, heartbreaking music"]),
    ("mood", "dark", ["dark, ominous, tense music", "brooding, moody, gloomy music", "a menacing, unsettling, suspenseful piece of music"]),
    ("mood", "dreamy", ["dreamy, atmospheric, ambient music", "ethereal, floating, spacious music", "a hazy, hypnotic, atmospheric piece of music"]),
    ("mood", "romantic", ["romantic, tender, warm music", "intimate, loving, gentle music", "a soft, sentimental, romantic piece of music"]),
    ("mood", "epic", ["epic, grand, cinematic music", "triumphant, majestic, powerful music", "a dramatic, sweeping, heroic piece of music"]),
    ("mood", "aggressive", ["aggressive, angry, heavy music", "loud, distorted, intense music", "furious, harsh, powerful music"]),
    ("mood", "groovy", ["groovy, danceable, rhythmic music", "funky, laid-back music with a swinging groove", "a danceable beat with a steady rhythm"]),
    ("sound", "acoustic guitar", ["acoustic guitar music", "fingerpicked acoustic guitar", "strummed acoustic guitar with a warm tone"]),
    ("sound", "electric guitar", ["electric guitar music", "distorted electric guitar riffs", "a rock band with electric guitars"]),
    ("sound", "piano", ["solo piano music", "a piano melody", "a piece played on the piano"]),
    ("sound", "strings", ["orchestral music with strings", "a string quartet or string orchestra", "violins, violas and cellos playing"]),
    ("sound", "brass & winds", ["brass and woodwind instruments", "trumpet, saxophone and horns", "flute, clarinet and oboe"]),
    ("sound", "synths", ["synthesizer and electronic music", "electronic beats and synth pads", "synthesised electronic sounds"]),
    ("sound", "drums", ["drums and percussion", "a strong drum beat", "heavy drumming with a prominent rhythm section"]),
    ("voice", "instrumental", ["instrumental music with no vocals", "music without singing", "a purely instrumental piece"]),
    ("voice", "male vocal", ["a man singing", "male vocals", "a male singer with a deep voice"]),
    ("voice", "female vocal", ["a woman singing", "female vocals", "a female singer with a soft voice"]),
    ("voice", "choir", ["choir singing", "choral voices harmonising", "a choir singing in harmony"]),
]


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    paths = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r", allow_pickle=False)
    n = len(paths)
    if embeddings.shape != (n, 512):
        raise ValueError(f"embeddings {embeddings.shape} do not match {n} mapped tracks; rebuild embeddings first.")

    print("Loading the CLAP model (same default checkpoint as embed_clap.py)...", flush=True)
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    try:
        model.model.eval()
    except Exception:
        pass

    prompts = [p for _, _, descriptions in TAGS for p in descriptions]
    print(f"Embedding {len(prompts)} text descriptions for {len(TAGS)} tags...", flush=True)
    text = np.asarray(model.get_text_embedding(prompts, use_tensor=False), dtype=np.float32)
    if text.shape != (len(prompts), 512):
        raise ValueError(f"unexpected text embedding shape {text.shape}")
    text = l2(text)

    tag_vectors, k = [], 0
    for _, _, descriptions in TAGS:
        tag_vectors.append(text[k:k + len(descriptions)].mean(axis=0))
        k += len(descriptions)
    tag_vectors = l2(np.asarray(tag_vectors))

    print("Scoring every track against every tag...", flush=True)
    audio = l2(np.asarray(embeddings, dtype=np.float32))
    raw = audio @ tag_vectors.T
    z = (raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-6)

    np.save(OUT_SCORES, z.astype(np.float16))
    OUT_NAMES.write_text(json.dumps([{"name": name, "group": group} for group, name, _ in TAGS], indent=1), encoding="utf-8")
    print(f"Saved {OUT_SCORES} {z.shape} and {OUT_NAMES}", flush=True)

    print("\nSanity check: the three highest-scoring tracks for each tag (do they sound right?)")
    for j, (group, name, _) in enumerate(TAGS):
        top = np.argsort(-z[:, j])[:3]
        print(f"\n[{group}] {name}")
        for i in top:
            print(f"   {z[i, j]:5.2f}  {paths[int(i)]}")


if __name__ == "__main__":
    main()
