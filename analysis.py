"""
analysis.py
-----------
  1. How many examples do we have per class?
  2. Are the classes balanced (roughly equal examples each)?
  3. Do objects tend to appear on one side of the image?
"""

from pathlib import Path
import pandas as pd

LABELS_FOLDER = "data/labels"

CLASS_NAMES = [
    'Apical Lesion',      # 0
    'Main Root',          # 1
    'Main Canal',         # 2
    'Mesial Root',        # 3
    'Mesial Canal',       # 4
    'Distal Root',        # 5
    'Distal Canal',       # 6
    'Palatal Canal',      # 7
    'Palatal Root',       # 8
    'Root Canal Filling', # 9
    'decay',              # 10
]


# ==================================================================
# STEP 1: Read all the label files
# ==================================================================
# Each .txt file = one image's annotations
# Each line in a file = one object, formatted as:
#   class_id  x1 y1 x2 y2 x3 y3 ...  (polygon points, 0.0–1.0 scale)
# We read every line and store it as a row in a table (DataFrame)
# ==================================================================

print("=" * 60)
print("STEP 1: Reading label files...")
print("=" * 60)

labels_path = Path(LABELS_FOLDER)
rows = []

for txt_file in labels_path.glob("*.txt"):
    lines = txt_file.read_text().strip().splitlines()

    for line in lines:
        numbers = line.strip().split()

        # Need at least class_id + one x,y pair to be a valid annotation
        if len(numbers) < 3:
            continue

        class_id = int(float(numbers[0]))

        # Split the remaining numbers into x and y coordinates
        coords   = list(map(float, numbers[1:]))
        x_values = coords[0::2]   # positions 0, 2, 4, ... = x coords
        y_values = coords[1::2]   # positions 1, 3, 5, ... = y coords

        # Average x/y gives us the rough center of the object
        avg_x = sum(x_values) / len(x_values)
        avg_y = sum(y_values) / len(y_values)

        # Look up the class name, or fall back to "class_N" if unknown
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

        rows.append({
            "file":       txt_file.stem,   # e.g. "img_001"
            "class_id":   class_id,
            "class_name": class_name,
            "avg_x":      avg_x,           # 0.0 = far left,  1.0 = far right
            "avg_y":      avg_y,           # 0.0 = top,        1.0 = bottom
        })

df = pd.DataFrame(rows)

if df.empty:
    print("No annotations found — check your LABELS_FOLDER path.")
    exit()

print(f"  RESULTS:\n")
print(f"  Total images with labels : {df['file'].nunique()}")
print(f"  Total annotations        : {len(df)}")
print(f"  Class IDs found          : {sorted(df['class_id'].unique())}")


# ==================================================================
# STEP 2: How many examples per class?
# ==================================================================
# More examples = model gets more practice recognizing that class.
# Too few examples for a class = model will likely miss it.
# ==================================================================

print()
print("=" * 60)
print("STEP 2: Annotation count per class")
print("=" * 60)
print()
print("  Each █ = roughly 1/30th of the most common class count.")
print("  A short bar means fewer examples — potential weak spot.\n")

class_counts = df["class_name"].value_counts()

max_count = class_counts.max()
bar_width  = 30

print(f"  {'Class':<22} {'Count':>6}   Bar")
print("  " + "-" * 58)

for name, count in class_counts.items():
    bar = "█" * int(count / max_count * bar_width)
    print(f"  {name:<22} {count:>6}   {bar}")

# Imbalance ratio = biggest class / smallest class
imbalance = class_counts.max() / class_counts.min()

print()
print(f"  Imbalance ratio: {imbalance:.1f}x")
print(f"  (How many times more examples the biggest class has vs the smallest)")
print()

if imbalance > 10:
    print("  ⚠ WARNING: Large imbalance detected!")
    print("  The model will see the rare classes very infrequently during")
    print("  training, which usually means poor detection on those classes.")
    print("  Consider: collecting more data, or using weighted loss.")
else:
    print("  ✓ Classes look reasonably balanced.")


# ==================================================================
# STEP 3: Spatial bias — do objects cluster on one side?
# ==================================================================
# Each image is split into 3 horizontal zones:
#   LEFT   = avg_x between 0.00 and 0.33
#   CENTER = avg_x between 0.33 and 0.66
#   RIGHT  = avg_x between 0.66 and 1.00
#
# If a class is >70% in one zone, the model might learn
# "this class lives on the left" instead of actually learning
# what it looks like — which would hurt real-world performance.
# ==================================================================

print()
print("=" * 60)
print("STEP 3: Spatial bias check (horizontal position)")
print("=" * 60)
print()
print("  Where do objects tend to appear in the image?")
print("  LEFT = left third | CENTER = middle | RIGHT = right third")
print("  ⚠ flag means >70% of that class is in one zone.\n")

df["zone"] = pd.cut(
    df["avg_x"], #from step 1 
    bins=[0, 0.33, 0.66, 1.0],
    labels=["LEFT", "CENTER", "RIGHT"]
)

zone_table = df.groupby(["class_name", "zone"], observed=False).size().unstack(fill_value=0)
zone_pct   = zone_table.div(zone_table.sum(axis=1), axis=0).mul(100).round(1)

print(f"  {'Class':<22} {'LEFT':>7} {'CENTER':>8} {'RIGHT':>7}  ")
print("  " + "-" * 55)

for name, row in zone_pct.iterrows():
    flag = "  ⚠ biased" if row.max() > 70 else ""
    print(f"  {name:<22} {row['LEFT']:>6}% {row['CENTER']:>7}% {row['RIGHT']:>6}%{flag}")


# ==================================================================
# FINAL SUMMARY
# ==================================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print(f"  Total images          : {df['file'].nunique()}")
print(f"  Total annotations     : {len(df)}")
print(f"  Most common class     : {class_counts.idxmax()}  ({class_counts.max()} examples)")
print(f"  Rarest class          : {class_counts.idxmin()}  ({class_counts.min()} examples)")
print(f"  Imbalance ratio       : {imbalance:.1f}x")
print()

if imbalance > 10:
    rarest = class_counts.idxmin()
    print(f"  Main concern: '{rarest}' has very few examples.")
    print(f"  The model may struggle to detect it reliably.")