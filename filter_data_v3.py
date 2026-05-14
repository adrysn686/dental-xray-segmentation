import argparse
from pathlib import Path
import shutil

# -----------------------------
# ORIGINAL class list (old IDs)
# -----------------------------
ORIG_CLASSES = [
    'Apical Lesion', 'Calcified Canal', 'Main Root', 'Main Canal', 'Mesial Root',
    'Mesial Canal', 'Distal Root', 'Distal Canal', 'Palatal Canal', 'Palatal Root',
    'Root Canal Filling', 'crown', 'restoration', 'open apex', 'unerupted tooth',
    'separated instrument', 'root end filling', 'decay', 'root fracture',
    'interanl resorption', 'external resorption', 'fracture lesion', 'dl canal',
    'temp filling', 'pulp chamber', 'amalgam filling', 'post',
    'root canal filling palatal', 'root canal filling distal',
    'root canal filling mesial', 'root canal filling main',
    'root canal filling thermafil', 'Dental implant', 'main root 2',
    'Buccal root', 'Pulp stone', 'root canal filling ledged',
    'perforation lesion', 'root canal filling sealer',
    'root canal filling silver point'
]

# -----------------------------
# NEW class list (new IDs)
# -----------------------------
KEEP_CLASSES = [
    'Apical Lesion',
    'Main Root',
    'Main Canal',
    'Mesial Root',
    'Mesial Canal',
    'Distal Root',
    'Distal Canal',
    'Palatal Canal',
    'Palatal Root',
    'Root Canal Filling',
    'decay',
]


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="Source dataset root containing images/ and labels/")
    ap.add_argument("--dst", type=str, required=True,
                    help="Destination dataset root to write filtered data")
    ap.add_argument("--drop-empty", action="store_true",
                    help="If set, drop images that have 0 remaining objects after filtering")
    ap.add_argument("--copy-images", action="store_true",
                    help="If set, copy matching images into dst/images. If not set, only labels are written.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    src_labels = src / "labels"
    src_images = src / "images"
    if not src_labels.exists():
        raise FileNotFoundError(f"Missing labels/ at: {src_labels}")
    if args.copy_images and not src_images.exists():
        raise FileNotFoundError(f"Missing images/ at: {src_images}")

    dst_labels = dst / "labels"
    dst_images = dst / "images"
    dst_labels.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        dst_images.mkdir(parents=True, exist_ok=True)

    # old_id -> name
    oldid_to_name = {i: n for i, n in enumerate(ORIG_CLASSES)}

    # names to keep
    keep_set = set(KEEP_CLASSES)

    # name -> new_id
    name_to_newid = {n: i for i, n in enumerate(KEEP_CLASSES)}

    total_files = 0
    kept_files = 0
    removed_files = 0
    total_objs_in = 0
    total_objs_out = 0

    for lab_path in sorted(src_labels.glob("*.txt")):
        total_files += 1
        lines = lab_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) == 1 and lines[0].strip() == "":
            lines = []

        new_lines = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue

            try:
                old_id = int(float(parts[0]))
            except ValueError:
                continue

            total_objs_in += 1

            name = oldid_to_name.get(old_id, None)
            if name is None:
                continue
            if name not in keep_set:
                continue

            new_id = name_to_newid[name]
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts))
            total_objs_out += 1

        # If nothing remains and drop-empty => skip writing label (and image)
        if args.drop_empty and len(new_lines) == 0:
            removed_files += 1
            continue

        # write filtered label file
        out_lab = dst_labels / lab_path.name
        out_lab.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        kept_files += 1

        # optionally copy image
        if args.copy_images:
            img = find_matching_image(src_images, lab_path.stem)
            if img is None:
                # if you want, you can raise instead; here we just warn
                print(f"[WARN] No image found for label: {lab_path.name}")
            else:
                shutil.copy2(img, dst_images / img.name)

    # also write a dataset.yaml snippet for convenience
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(KEEP_CLASSES)}\n"
        f"names: {KEEP_CLASSES}\n",
        encoding="utf-8"
    )

    print("Done.")
    print(f"Label files scanned: {total_files}")
    print(f"Kept label files:    {kept_files}")
    print(f"Dropped (empty):     {removed_files}")
    print(f"Objects in:          {total_objs_in}")
    print(f"Objects kept:        {total_objs_out}")
    print(f"Wrote:               {dst}")


if __name__ == "__main__":
    main()
