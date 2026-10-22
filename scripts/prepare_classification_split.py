"""Create deduplicated, stratified classification splits without changing source data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ImageFile.LOAD_TRUNCATED_IMAGES = False
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def fingerprint(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inspect_source(path):
    """Keep decoding failures in a report; do not delete unreadable originals."""
    row = {
        "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "original_split": path.parents[2].name,
        "category": path.parents[1].name,
        "sha256": None,
        "error": None,
    }
    try:
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unsupported extension: {path.suffix}")
        with Image.open(path) as image:
            image.load()
        row["sha256"] = fingerprint(path)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def build_splits(unique_images, seed):
    # First reserve 30%; then divide that holdout equally into validation and test.
    train, holdout = train_test_split(
        unique_images, test_size=0.30, random_state=seed,
        stratify=unique_images["category"],
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, random_state=seed,
        stratify=holdout["category"],
    )
    parts = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame = frame.copy()
        frame["split"] = name
        parts.append(frame)
    manifest = pd.concat(parts, ignore_index=True)
    # No fingerprint may appear more than once, including across split boundaries.
    if manifest["sha256"].duplicated().any():
        raise RuntimeError("Duplicate image contents remain in the split manifest.")
    if set(manifest["sha256"]) != set(unique_images["sha256"]):
        raise RuntimeError("The split lost or introduced images.")
    categories = set(unique_images["category"])
    for name, frame in manifest.groupby("split"):
        if set(frame["category"]) != categories:
            raise RuntimeError(f"Not all categories are represented in {name}.")
    return manifest.sort_values(["split", "category", "sha256"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data")
    parser.add_argument("--output-dir", default="data/processed/classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Inspect and plan without writing files.")
    args = parser.parse_args()
    source = (PROJECT_ROOT / args.source_dir).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    if not source.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Source and output must stay inside this project.")
    if source == output or source.is_relative_to(output):
        raise ValueError("Output cannot be the source directory or its parent.")
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir; nothing was overwritten.")

    paths = sorted(path for path in source.glob("*/*/images/*") if path.is_file())
    if not paths:
        raise ValueError(f"No files found under {source}/<split>/<category>/images/.")
    print(f"Checking and fingerprinting {len(paths):,} source files...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        inventory = pd.DataFrame(executor.map(inspect_source, paths))
    rejected = inventory.loc[inventory["error"].notna()].copy()
    readable = inventory.loc[inventory["error"].isna()].copy()
    conflicts = readable.groupby("sha256")["category"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Identical image contents have conflicting categories. Resolve those labels before splitting.")

    # Choose one stable representative per identical file, irrespective of old Train/Val assignment.
    unique_images = readable.sort_values("source_path").drop_duplicates("sha256").sort_values("sha256")
    manifest = build_splits(unique_images, args.seed)
    # Check reproducibility before creating any output files.
    pd.testing.assert_frame_equal(manifest, build_splits(unique_images, args.seed))
    manifest["output_path"] = manifest.apply(
        lambda row: f"{row['split']}/{row['category']}/{row['sha256']}{Path(row['source_path']).suffix.lower()}",
        axis=1,
    )
    counts = pd.crosstab(manifest["category"], manifest["split"]).reindex(columns=["train", "validation", "test"])
    print(counts.to_string())
    print("Totals:", counts.sum().to_dict())
    print(f"Excluded unreadable/unsupported files: {len(rejected)}")
    print(f"Extra exact copies excluded: {len(readable) - len(unique_images):,}")
    print("PASS: all categories represented, zero shared fingerprints, reproducible assignments.")
    print("Patient/study independence is unverified: no patient identifiers were provided.")
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=False)
    print("Copying selected images into class directories...", flush=True)

    def copy_selected(row):
        target = output / row.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / row.source_path, target)
        if fingerprint(target) != row.sha256:
            raise RuntimeError(f"Copied file does not match the source fingerprint: {target}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy_selected, manifest.itertuples(index=False)))

    manifest.to_csv(output / "manifest.csv", index=False)
    inventory.to_csv(output / "source_inventory.csv", index=False)
    rejected.to_csv(output / "rejected_files.csv", index=False)
    counts.to_csv(output / "class_counts.csv")
    classes = {category: index for index, category in enumerate(sorted(manifest["category"].unique()))}
    (output / "class_to_index.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    report = {
        "seed": args.seed,
        "requested_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "actual_counts": {key: int(value) for key, value in counts.sum().items()},
        "actual_ratios": {key: float(value / len(manifest)) for key, value in counts.sum().items()},
        "source_files": len(inventory),
        "unreadable_or_unsupported": len(rejected),
        "extra_exact_copies_excluded": len(readable) - len(unique_images),
        "unique_images": len(unique_images),
        "shared_fingerprints_between_splits": 0,
        "copy_hashes_verified": True,
        "patient_independence_verified": False,
        "notes": [
            "Old Train/Val folders were pooled before creating new splits.",
            "Ratios are approximate because image counts must be integers.",
            "Exact file hashes do not detect resized or recompressed duplicates.",
            "No patient identifiers were supplied; same-patient images may cross splits.",
            "Sources and detection annotations were not changed.",
        ],
    }
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved and verified {len(manifest):,} classification images in {output}")


if __name__ == "__main__":
    main()
