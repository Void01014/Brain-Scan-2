"""Copy paired detection data, assigning box class IDs from category folders.

Folder categories are treated as the intended ground truth. Only class tokens
change: existing boxes, filenames and source files are preserved. Empty labels
stay empty. This does not deduplicate or create train/validation/test splits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import re
import shutil

CLASS_IDS = {"Glioma": 0, "Meningioma": 1, "No Tumor": 2, "Pituitary": 3}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def rewrite_label(text, class_id):
    """Validate rows and replace only their class token, retaining coordinates."""
    rewritten, changes = [], []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        fields = line.split()
        if fields:
            if len(fields) != 5:
                raise ValueError(f"line {number}: expected 5 values")
            if not fields[0].isascii() or not fields[0].isdigit():
                raise ValueError(f"line {number}: class ID must be a nonnegative integer")
            x, y, width, height = map(float, fields[1:])
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"line {number}: nonfinite coordinates")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"line {number}: invalid normalized coordinates")
            if fields[0] != str(class_id):
                changes.append((number, fields[0], class_id))
                line = re.sub(r"^(\s*)\S+", lambda match: match[1] + str(class_id), line, count=1)
        rewritten.append(line)
    return "".join(rewritten), changes


def inspect_pair(job):
    source, category, image, label = job
    try:
        with label.open(encoding="utf-8-sig", newline="") as file:
            text = file.read()
        corrected, changes = rewrite_label(text, CLASS_IDS[category])
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    return source, category, image, label, corrected, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/detection_labels_fixed"))
    parser.add_argument("--dry-run", action="store_true", help="Check and count without writing files")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    # An output inside either source tree could contaminate subsequent scans.
    for source in (root / "data/Train", root / "data/Val"):
        if output == source.resolve() or source.resolve() in output.parents:
            raise ValueError("Output must be outside the original Train/Val folders")
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Output already exists: {output}. Choose a different --output-dir.")

    jobs, skipped = [], []
    for source in ("Train", "Val"):
        source_folder = root / "data" / source
        unknown = [p.name for p in source_folder.iterdir() if p.is_dir() and p.name not in CLASS_IDS]
        if unknown:
            raise ValueError(f"Unknown category folders in {source}: {unknown}")
        for category in CLASS_IDS:
            folder = source_folder / category
            images_folder, labels_folder = folder / "images", folder / "labels"
            if not images_folder.is_dir() or not labels_folder.is_dir():
                raise FileNotFoundError(f"Missing images/labels folder: {folder}")
            images = sorted(p for p in images_folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            labels = {p.stem: p for p in labels_folder.glob("*.txt")}
            stems = [p.stem for p in images]
            if len(stems) != len(set(stems)):
                raise ValueError(f"Multiple image extensions share a basename in {folder}")
            for image in images:
                if image.stem in labels:
                    jobs.append((source, category, image, labels[image.stem]))
                else:
                    skipped.append({"path": str(image.relative_to(root)), "reason": "missing label"})
            for stem in labels.keys() - set(stems):
                skipped.append({"path": str(labels[stem].relative_to(root)), "reason": "missing image"})

    # Validate every paired label before creating any output.
    with ThreadPoolExecutor(max_workers=8) as pool:
        plans = list(pool.map(inspect_pair, jobs))
    changed_files = sum(bool(plan[5]) for plan in plans)
    changed_boxes = sum(len(plan[5]) for plan in plans)
    empty_labels = sum(not plan[4].strip() for plan in plans)
    counts = Counter(f"{plan[0]}/{plan[1]}" for plan in plans)
    summary = {"class_ids": CLASS_IDS, "paired_images": len(plans),
               "changed_label_files": changed_files, "changed_box_ids": changed_boxes,
               "empty_labels_preserved": empty_labels, "counts": dict(counts),
               "skipped": skipped, "assumption": "Folder categories determine all existing box class IDs"}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no files written.")
        return

    output.mkdir(parents=True, exist_ok=False)
    with (output / "changes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_label", "line", "old_id", "new_id"])
        for source, category, image, label, corrected, changes in plans:
            folder = output / source / category
            (folder / "images").mkdir(parents=True, exist_ok=True)
            (folder / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, folder / "images" / image.name)
            with (folder / "labels" / label.name).open("w", encoding="utf-8", newline="") as label_file:
                label_file.write(corrected)
            for number, old, new in changes:
                writer.writerow([label.relative_to(root).as_posix(), number, old, new])
    (output / "correction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Corrected copy: {output}")


if __name__ == "__main__":
    main()
