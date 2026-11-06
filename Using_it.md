# PulseFlow – Library Maintenance and Rebuild

## A. Core library maintenance

Activate the main environment:

```bat
conda activate musicrec
```

Navigate to the PulseFlow project root.

1. Maintain the CLAP library:

```bat
python .\scripts\maintain_clap_library.py --apply
```

This ingests new tracks, updates changed tracks and removes tracks no longer present in the library.

2. Rebuild the mood/style scores:

```bat
python .\scripts\build_clap_mood_style_scores.py
```

3. Export the Android Sonic Neighbourhood database:

```bat
python .\scripts\export_sonic_neighbourhood_android.py
```

4. Validate the generated files, track counts, mapping order and database contents.

## B. Music map and visualisation (optional)

1. Rebuild the main music map:

```bat
python .\scripts\rebuild_music_map.py
```

2. Generate raw CLAP 12D clusters:

```bat
python .\scripts\cluster_raw_clap_12_for_map.py
```

3. Experiment with 3D projections:

```bat
python .\scripts\experiment_3d_projections.py
```

4. Generate the interactive PCA / genre / sonic visualisation:

```bat
python .\scripts\plot_raw_clap_12_pca_direct_genre_sonic_panel_float_clap_detail_fade_overlay_clickfix.py
```

5. Open the HTML file produced by the final script.

These steps are for visual exploration and are not required for normal library maintenance.

## C. Playlist generation (optional)

The original command-line playlist generator remains available:

```bat
python -m scripts.generate_playlist --config config.yaml --start-path "/music/xxxx.mp3"
```

## D. Android database and APK

1. Copy the validated SQLite database into the Android assets folder.
2. Verify the database update mechanism and schema version.
3. Build a new APK when distributing the updated database or app.
4. Install and test on the Android device.

## E. Important validation

- Embeddings, mapping and metadata must agree in track count and ordering.
- Mood/style scores must match the current library mapping.
- FAISS must contain the expected number of vectors.
- Android database track and neighbour counts must be verified.
- Keep backups of previous production files before replacing them.
