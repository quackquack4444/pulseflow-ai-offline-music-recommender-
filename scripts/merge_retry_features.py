
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "features_60s.json"
RETRY = DATA / "features_retry_results.json"

def path_key(record):
    return os.path.normcase(os.path.normpath(record["path"]))

def main():
    with CACHE.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    if not isinstance(existing, list):
        raise TypeError("Production cache is not a JSON list; refusing to modify it.")

    with RETRY.open("r", encoding="utf-8") as f:
        retry = json.load(f)

    additions = retry.get("successes", [])
    failures = retry.get("failures", [])

    if failures:
        raise ValueError(f"Retry contains {len(failures)} failures.")
    if len(additions) != 439:
        raise ValueError(f"Expected 439 retry records, found {len(additions)}.")

    old_paths = {path_key(x) for x in existing}
    new_paths = [path_key(x) for x in additions]

    if len(set(new_paths)) != len(new_paths):
        raise ValueError("Duplicate paths in retry results.")
    overlap = old_paths.intersection(new_paths)
    if overlap:
        raise ValueError(f"{len(overlap)} retry paths already exist in cache.")

    backup = DATA / (
        "features_60s_pre_retry_merge_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    shutil.copy2(CACHE, backup)

    merged = existing + additions
    temp = CACHE.with_suffix(".merge_tmp")

    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, CACHE)
    finally:
        if temp.exists():
            temp.unlink()

    print("Merge completed successfully.")
    print(f"Previous records: {len(existing)}")
    print(f"Added records: {len(additions)}")
    print(f"New total: {len(merged)}")
    print(f"Backup: {backup}")

if __name__ == "__main__":
    main()
