# PulseFlow — Fork Setup Notes

This is a fork of pulseflow-ai-offline-music-recommender by Philippe-Guerrier. The original README.md is unchanged; this file covers what differs here.

## What's different in this fork

- Windows double-click playlist launcher: `playlist_launcher.ps1` and `playlist_launcher.bat`.
- Separate Conda environment definitions for CLAP/FAISS and MusicNN.
- Additional experiment scripts in `scripts/experiments/`. These may contain machine-specific paths and are not required for basic playlist generation.

## Setup

### 1. Create the Conda environments

From the repository root, run:

```bat
conda env create -f environment-musicrec.yml
conda env create -f environment-musicnn.yml
```

These create the fork-specific environments:

- `musicrec-fork-test`
- `musicnn-fork-test`

The original `musicrec` and `musicnn` environments are separate and should not be modified.

### 2. Install the CLAP/FAISS Python requirements

Activate the CLAP/FAISS environment:

```bat
conda activate musicrec-fork-test
```

Install the Python requirements:

```bat
python -m pip install -r requirements.txt
```

The `requirements.txt` file should contain the additional Python dependencies for CLAP/FAISS and playlist generation. Packages already supplied by Conda, including NumPy, SciPy and FAISS, should remain managed by the Conda environment unless there is a specific reason to change them.

### 3. Configure the music library

Edit `config.yaml` so the music library path points to your own collection.

### 4. Generate embeddings

Generate embeddings for your own music collection using the project's ingestion instructions and scripts.

The `data/` directory contains derived data such as embeddings, the track mapping and FAISS index. These files are specific to a music collection and are not included in this repository.

## Generating a playlist

Activate the CLAP/FAISS environment:

```bat
conda activate musicrec-fork-test
```

Then run:

```bat
python scripts/generate_playlist.py --config config.yaml --length 20 --policy smooth
```

Options:

- `--length`: playlist length; supported values are 10, 20, 30 or 40 tracks.
- `--policy`: `smooth` or `contrast`.

Alternatively, use `playlist_launcher.bat`. Edit the folder path inside the launcher to match the location where you cloned the repository.

## Which environment is used for what?

- **`musicrec-fork-test`** — Python 3.11, CLAP embeddings, FAISS similarity search and playlist generation.
- **`musicnn-fork-test`** — Python 3.7.16, TensorFlow 2.3.4 and MusicNN tagging. This is a legacy dependency stack and must remain separate from the CLAP environment because of conflicting NumPy requirements.

Activate the MusicNN environment when running MusicNN-specific tagging scripts:

```bat
conda activate musicnn-fork-test
```

## Known limitations

- Scripts in `scripts/experiments/` may contain hard-coded paths and need editing before use on another machine.
- The launcher is Windows-specific; the Python scripts may work on other platforms.
- Model downloads and audio-decoding dependencies may be required on first use.
- MusicNN uses an older Python and TensorFlow stack. Reproducing it on another machine may require compatible package sources and builds.
- Successful environment creation does not guarantee that every experimental script will work without configuration changes.
