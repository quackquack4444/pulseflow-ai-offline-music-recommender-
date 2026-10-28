For a fresh PulseFlow library, the main datasets are the CLAP embeddings, their track mapping, the FAISS search index, and the maintenance manifest. The other datasets support playlist generation, tagging and visualisation, but are separate from the core CLAP rebuild.
# 1. Core CLAP library
All paths below are relative to V:\GitHub\pulseflow-ai-offline-music-recommender-main.
Dataset
Filename
Purpose
CLAP embeddings
data\embeddings.npy
The 512-dimensional vector for each track; the main source of musical similarity.
Track mapping
data\embedding_mapping.txt
JSON list mapping each embedding row to its music file path.
FAISS index
data\faiss.index
Fast nearest-neighbour search over the CLAP vectors, using IndexFlatIP and normalised vectors for cosine similarity.
CLAP manifest
data\clap_library_manifest.json
Tracks file signatures, including size and modification time, so maintenance can detect new, deleted and modified files.
These four files form the core library that the proposed --rebuild-all option would regenerate.
# 2. Other datasets and caches
Dataset
Filename
Purpose
Audio features
data\features_60s.json
Per-track audio features used by playlist generation, including tempo and energy.
MusicNN tags
data\musicnn_tags_progress.repaired.jsonl
MusicNN-derived tags, including MSD/MTT tag data for the tracks represented in that cache.
PCA coordinates
data\music_pca_50d.npy
Reduced-dimensional representation used in map generation.
Map labels
data\music_cluster_labels.npy
Cluster labels associated with the map data.
UMAP coordinates
data\music_umap_3d.npy
Three-dimensional coordinates for the music map.
The map-related files were generated against an earlier library state, so they should not be assumed to match the current CLAP library without checking or rebuilding them.
# 3. What a full CLAP rebuild would do
The proposed rebuild would:
Scan the configured music root, M:\Music\Music.
Generate a fresh CLAP embedding for each eligible track.
Write a fresh embeddings.npy and matching embedding_mapping.txt.
Build a new faiss.index from those embeddings.
Generate a fresh clap_library_manifest.json.
Back up the existing core files and validate the replacements before treating the rebuild as successful.
It would not automatically rebuild features_60s.json, MusicNN tags or the map datasets. Those remain separate maintenance or processing jobs.

Further notes on Datasets/Caches etc.
# 1. Audio features — data\features_60s.json
Creates/extracts: scripts\extract_features.py
Retries failed files: scripts\retry_missing_features.py
Merges recovered features: scripts\merge_retry_features.py
Uses the cache: scripts\generate_playlist.py
The extraction workflow calculates per-track audio features such as tempo and energy. The retry and merge scripts were used on 28 October 2026 to recover 439 tracks that had failed during the initial extraction, bringing the cache to 70,898 records.
# 2. MusicNN tags — data\musicnn_tags_progress.repaired.jsonl
Original tagging/backfill workflow: scripts\backfill_musicnn_tags_both.py was identified as the intended main runner.
Repairs the cache format: scripts\repair_musicnn_cache.py
Other scripts identified: scripts\backfill_musicnn_tags.py (older MSD-only workflow) and scripts\backfill_musicnn_tags_persistent.py (its exact role still needed inspection).
The repair script converts legacy tags fields to msd_tags, preserves mtt_tags, and writes the .repaired.jsonl file without overwriting the original. The repaired cache has 49,487 unique tracks with MSD and MTT tags.
# 3. Map datasets
Dataset
Filename
Creator/maintenance script
PCA-reduced coordinates
data\music_pca_50d.npy
Exact script name not confirmed in the available project history.
Cluster labels
data\music_cluster_labels.npy
Exact script name not confirmed.
3D UMAP coordinates
data\music_umap_3d.npy
Exact script name not confirmed.
These were generated as part of the music-map workflow, using PCA, clustering and UMAP. The previously identified scripts\rebuild_music_map.py is the map-rebuild script, but I can't confirm from the available record that it writes all three files directly.
The distinction matters: the PCA, cluster-label and UMAP files are derived map data, not part of the core CLAP/FAISS library. Since the core library has changed since the last recorded map build, those datasets should be checked for row alignment before being reused.



