import csv
import shutil, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Pass missing file path and missing dir")
parser.add_argument("--missing_path", type=str, help="Language name")
parser.add_argument("--data_path", type=str, help="Language name")
parser.add_argument("--tmp_path", type=str, help="Data path")
args = parser.parse_args()

# MFA missing-transcriptions report
CSV_FILE = Path(args.missing_path)

# Dataset root
DATA_ROOT = Path(args.data_path)

# Destination
MISSING_DIR = Path(args.tmp_path)

MISSING_DIR.mkdir(parents=True, exist_ok=True)

moved = 0
not_found = 0
already_exists = 0


with open(CSV_FILE, "r", encoding="utf-8-sig") as f:

    reader = csv.reader(f)

    for row in reader:

        if not row:
            continue

        # Usually the path is in the first column
        raw_path = row[0].strip()

        if not raw_path:
            continue

        src = Path(raw_path)

        # Skip CSV header if present
        if not src.exists():
            print(f"NOT FOUND: {src}")
            not_found += 1
            continue

        # --------------------------------------------------
        # Preserve directory structure relative to DATA_ROOT
        # --------------------------------------------------

        try:
            relative_path = src.relative_to(DATA_ROOT)
        except ValueError:
            # File is outside DATA_ROOT
            relative_path = Path(src.parent.name) / src.name

        dst = MISSING_DIR / relative_path

        dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Safety: don't overwrite existing files
        # --------------------------------------------------

        if dst.exists():
            print(f"ALREADY EXISTS: {dst}")
            already_exists += 1
            continue

        shutil.move(
            str(src),
            str(dst)
        )

        moved += 1

        if moved <= 20:
            print(f"MOVED:")
            print(f"  {src}")
            print(f"  -> {dst}")


print("\n====================================")
print("DONE")
print("====================================")

print(f"Moved          : {moved}")
print(f"Not found      : {not_found}")
print(f"Already exists : {already_exists}")

print(f"\nMissing files moved to:")
print(MISSING_DIR)
