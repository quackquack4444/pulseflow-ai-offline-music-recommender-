# PulseFlow — Fork Setup Notes

This is a fork of `pulseflow-ai-offline-music-recommender` by Philippe-Guerrier. The original `README.md` is unchanged. This file documents the fork-specific setup, test environments, and the first successful playlist test.

## What's different in this fork

- Windows double-click playlist launcher: `playlist_launcher.ps1` and `playlist_launcher.bat`.
- Separate Conda environment definitions for CLAP/FAISS and MusicNN.
- Additional experiment scripts in `scripts/`. These may contain machine-specific paths and are not required for basic playlist generation.

## Environments

The fork uses two separate Conda environments:

| Environment | Purpose | Main stack |
|---|---|---|
| `musicrec-fork-test` | CLAP embeddings, FAISS search, playlist generation | Python 3.11 |
| `musicnn-fork-test` | MusicNN audio tagging | Python 3.7.16, TensorFlow 2.3.4 |

Keep these environments separate. MusicNN uses an older dependency stack, including older NumPy requirements. The original `musicrec` and `musicnn` environments should remain untouched.

## Setup

Run these commands from the repository root in Anaconda Prompt or another terminal where Conda is available.

### 1. Create the environments

```bat
conda env create -f environment-musicrec.yml
conda env create -f environment-musicnn.yml
```

If an environment already exists, check it first with `conda env list` rather than recreating it blindly.

### 2. Configure the local test library

Copy `config.yaml.example` to `config.yaml`, then edit `config.yaml` for your machine. Keep `config.yaml` local and do not overwrite `config.yaml.example`.

For the small fork test library, set `music_dir` to the repository's `music` folder. For example:

`V:\GitHub\pulseflow-fork\music`

The sample track used in the test below is:

`V:\GitHub\pulseflow-fork\music\05 Helpless.m4a`

The sample `music/` folder is excluded from Git by `.gitignore`. Do not accidentally point the fork configuration at the production library or production index files.

### 3. Install the CLAP/FAISS environment requirements

Activate the CLAP environment:

```bat
conda activate musicrec-fork-test
```

Install the repository requirements:

```bat
python -m pip install -r requirements.txt
```

During the initial test, these additional installs were needed in the test environment:

```bat
python -m pip install "laion-clap @ git+https://github.com/LAION-AI/CLAP.git"
python -m pip install requests pyyaml soundfile
```

The first command installs LAION-CLAP from its Git repository. The second installs dependencies that were missing when the playlist command was first tested. If the requirements file already installs a package, there is no need to install it again.

### 4. Check that the playlist command starts

```bat
python -m scripts.generate_playlist --help
```

The help text should appear without a missing-module error.

## Test embedding and playlist generation

A small test library was used to verify the fork without touching the original production environment or music library. The test library had 25 tracks, with a 25-row `(25, 512)` embedding array and a matching FAISS index and mapping.

With `musicrec-fork-test` active, run the following from the repository root:

```bat
python -m scripts.generate_playlist --config config.yaml --start-path "V:\GitHub\pulseflow-fork\music\05 Helpless.m4a" --length 10 --policy smooth
```

This test completed successfully and wrote:

- `data\last_playlist.json`
- `data\last_playlist.m3u8`

Open `data\last_playlist.m3u8` in a compatible music player to check that the tracks resolve and play. The playlist output is generated locally and depends on the paths and index configured for this fork.

### Why use `--start-path`?

`--start-path` lets the script use a specified track as the seed. It avoids the audio-fingerprint recognition path used by `--input-audio`, which currently encounters a PyDejavu compatibility problem in the Python 3.11 test environment.

## Troubleshooting

- **`No module named requests`, `yaml`, or `soundfile`:** activate `musicrec-fork-test`, then run `python -m pip install requests pyyaml soundfile`.
- **`No module named laion_clap`:** activate `musicrec-fork-test`, then run `python -m pip install "laion-clap @ git+https://github.com/LAION-AI/CLAP.git"`.
- **PyDejavu `SyntaxError` involving `print`:** the installed PyDejavu package contains Python 2 syntax and is incompatible with Python 3.11. Use `--start-path` for this test rather than `--input-audio`.
- **Missing embeddings, FAISS index, or mapping:** check the paths in `config.yaml` and confirm that the embedding array, mapping, and index belong to the fork's test library. Do not point the fork at production data by accident.
- **Model download or audio decoding fails on first run:** confirm that the active environment has the required packages and that the audio file can be read. Some models or audio decoders may require an initial download or additional system dependencies.

## MusicNN tagging

Activate the separate legacy environment before running MusicNN-specific scripts:

```bat
conda activate musicnn-fork-test
```

MusicNN inference was successfully tested on `05 Helpless.m4a` in this environment. Keep MusicNN separate from `musicrec-fork-test`; do not combine their TensorFlow/NumPy stacks.

## Other notes and limitations

- Feature extraction and feature-based ranking were verified on the 25-track test library. The latest playlist run reported seed tempo of approximately 136 BPM, energy of 0.185, and complete tempo/energy coverage for all 24 non-seed candidates.
