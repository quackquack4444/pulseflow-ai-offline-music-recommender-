# PulseFlow (fork notes)

This is a fork of [ORIGINAL-REPO](https://github.com/ORIGINAL-OWNER/ORIGINAL-REPO) by ORIGINAL-OWNER. The original `README.md` is unchanged; this file covers what differs here.

## What's different in this fork

- Windows double-click playlist launcher: `playlist_launcher.ps1` (the form) and `playlist_launcher.bat` (the launcher)
- Conda environment files: `environment-musicrec.yml` and `environment-musicnn.yml`
- Extra experiment scripts in `scripts/experiments/` (these contain the author's own paths and are not needed to generate playlists)

## Setup

1. Create the conda environments:

   ```
   conda env create -f environment-musicrec.yml
   conda env create -f environment-musicnn.yml
   ```

2. Edit `config.yaml` so the music library path points at your own music folder.

3. Generate embeddings from your own library. The `data/` folder (embeddings and FAISS index) is not included in this repo because it is derived from the author's personal music collection.

## Generating a playlist

Activate the `musicrec` environment, then run:

```
python scripts\generate_playlist.py --config config.yaml --length 20 --policy smooth
```

- `--length`: 10, 20, 30 or 40 tracks
- `--policy`: `smooth` or `contrast`

Or use the double-click launcher (`playlist_launcher.bat`), after editing the folder path inside it to match where you cloned the repo.

## Which environment is used for what

- `musicrec`: (fill in)
- `musicnn`: (fill in)

## Known limitations

- Some scripts in `scripts/experiments/` have hard-coded music paths and will need editing before they run on another machine.
- Windows only for the launcher; the Python scripts themselves may work elsewhere.
