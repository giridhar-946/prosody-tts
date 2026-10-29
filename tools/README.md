# tools/

`prepare_arrow_from_mfa.py` builds the `raw.arrow` file from MFA TextGrids. It is copied
unchanged from the F5-TTS repository (`train/datasets/prepare_data_v5.py`) and needs edits
before it runs on another machine:

- `root_dirs` and the wav/lab path rules are hard-coded to the original machine's folders.
  Point them at your corpus, or rewrite the path logic so that `wav_file` matches the
  folder layout in [docs/data.md](../docs/data.md#2-phone-alignments-mfa-textgrids).
- `save_dir` is `os.getcwd() + "/data/<dataset_name>"`. Run it from the folder where you
  want the arrow files to appear.
- Known bugs (not fixed here):
  - `extract_phones_with_punctuation` calls `pdb.set_trace()` on the mismatch and error
    paths. Each such utterance stops the script until you type `c`.
  - In the duration-mismatch branch it does `skipped_files += 1` inside a function that
    never defines `skipped_files`, which raises `UnboundLocalError`.
  - `vocab.txt` is written by this script. `extract_features.py` reads it, so keep the
    file.
- Output: `raw.arrow`, `duration.json`, `vocab.txt` in `save_dir`. You still need
  `spk2id.json` (ids 0..N-1 for each speaker folder); see
  [docs/data.md](../docs/data.md#3-the-arrow-dataset).
