from pathlib import Path
import shutil, argparse

parser = argparse.ArgumentParser(description="Pass oov path and missing dir")
parser.add_argument("--oov_path", type=str, help="Language name")
parser.add_argument("--data_path", type=str, help="Language name")
parser.add_argument("--tmp_path", type=str, help="Data path")
args = parser.parse_args()

# ============================================================
# PATHS
# ============================================================

OOV_FILE = Path(args.oov_path)

DATA_ROOT = Path(args.data_path)

MISSING_ROOT = Path(args.tmp_path)


# ============================================================
# COUNTERS
# ============================================================

utterances = 0

wav_moved = 0
lab_moved = 0

wav_missing = 0
lab_missing = 0

already_exists = 0


# ============================================================
# CREATE DESTINATION
# ============================================================

MISSING_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ utterance_oovs.txt
# ============================================================

with open(OOV_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Extract utterance path
        #
        # Example input:
        #
        # niaenglish_comb_h2_2854/
        # niaenglish_comb_h2_2854_2200064,
        # niaenglish_comb_h2_2854: ...
        #
        # We only need everything before first comma.
        # ----------------------------------------------------

        utterance_path = line.split(",", 1)[0].strip()

        relative = Path(utterance_path)

        utterances += 1

        # ----------------------------------------------------
        # Source files
        # ----------------------------------------------------

        wav_src = DATA_ROOT / relative.with_suffix(".wav")
        lab_src = DATA_ROOT / relative.with_suffix(".lab")

        # ----------------------------------------------------
        # Preserve speaker directory in Missing
        #
        # Missing/
        # └── niaenglish_comb_h2_2854/
        #       ├── xxx.wav
        #       └── xxx.lab
        # ----------------------------------------------------

        wav_dst = MISSING_ROOT / relative.with_suffix(".wav")
        lab_dst = MISSING_ROOT / relative.with_suffix(".lab")

        wav_dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # MOVE WAV
        # ====================================================

        if wav_src.exists():

            if wav_dst.exists():

                print(
                    f"WAV destination already exists: "
                    f"{wav_dst}"
                )

                already_exists += 1

            else:

                shutil.move(
                    str(wav_src),
                    str(wav_dst)
                )

                wav_moved += 1

        else:

            print(
                f"WAV NOT FOUND: {wav_src}"
            )

            wav_missing += 1

        # ====================================================
        # MOVE LAB
        # ====================================================

        if lab_src.exists():

            if lab_dst.exists():

                print(
                    f"LAB destination already exists: "
                    f"{lab_dst}"
                )

                already_exists += 1

            else:

                shutil.move(
                    str(lab_src),
                    str(lab_dst)
                )

                lab_moved += 1

        else:

            print(
                f"LAB NOT FOUND: {lab_src}"
            )

            lab_missing += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("MOVE COMPLETE")

print("=" * 60)

print(f"Utterances listed : {utterances}")

print()

print(f"WAV moved         : {wav_moved}")
print(f"LAB moved         : {lab_moved}")

print()

print(f"WAV not found     : {wav_missing}")
print(f"LAB not found     : {lab_missing}")

print()

print(f"Already exists    : {already_exists}")

print("\nFiles moved to:")

print(MISSING_ROOT)
