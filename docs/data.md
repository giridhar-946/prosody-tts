# Data requirements

This system needs three things per utterance: the **audio**, the **phone sequence**, and
the **phone durations** (a forced alignment). The data is not distributed with this
repository. This page describes what must exist and how it is laid out.

## 1. Audio

| Requirement | Value |
|---|---|
| Format | any file `soundfile` can read (wav, flac, ...) |
| Channels | mono (multi-channel input is averaged) |
| Sampling rate | any; resampled to 16 kHz for all features |
| Duration | 0.3 s to 40 s per utterance (outside this range the utterance is skipped) |
| Content | one speaker per file, clean speech with a transcript |
| Naming | the **parent folder of each wav is the speaker key** (see section 4) |

The audio is read at training time as well (for the mel spectrogram), so the wav paths in
the arrow file must stay valid.

## 2. Phone alignments (MFA TextGrids)

Each utterance needs a Praat TextGrid from the Montreal Forced Aligner with a **`phones`
tier**. Its intervals give the phone symbols and their durations. The reference script
`tools/prepare_arrow_from_mfa.py` turns these into the arrow rows; its rules are:

- An empty interval at the start becomes `ss`, at the end `es` (or merged into the previous
  phone when that phone is punctuation).
- Empty intervals next to `sp` become `pp`. Empty intervals next to punctuation are merged
  into the punctuation phone.
- Durations are measured in **20 ms frames**: `round((xmax - xmin) * 1000 / 20)`.
- Utterances whose phone durations do not add up to the TextGrid total are skipped.

**The phone inventory comes from `vocab.txt`.** `extract_features.py` turns it into
`phones.txt`, and the model can only produce those symbols. Build the arrow dataset from the
full corpus, not a small subset, otherwise common phones are missing. The development subset
(`Train_Data`, 4000 utterances, 2 speakers) has 70 symbols and cannot synthesize `oʊ` or
`ɑːɹ`, which espeak produces for ordinary English words.

The phone symbols must be IPA-style symbols. Those are the symbols the model learns, and
the same symbols must be reachable from espeak (section 6).

## 3. The arrow dataset

The training code reads one directory (passed as `--arrow_dir`) with these files:

| File | Written by | Content |
|---|---|---|
| `raw.arrow` | `tools/prepare_arrow_from_mfa.py` | one row per utterance (schema below) |
| `vocab.txt` | same script | every phone symbol in `raw.arrow`, one per line. A symbol can be a space character (`' '`) |
| `spk2id.json` | same script or by hand | `{speaker_key: id}`, ids are contiguous from 0 |
| `duration.json` | same script (optional, not read by this code) | total durations |

`raw.arrow` schema (a HuggingFace `datasets` arrow file):

| Column | Type | Meaning |
|---|---|---|
| `audio_path` | string | absolute path of the wav |
| `text` | list of string | phone symbols, in order |
| `duration` | float64 | utterance length in seconds |
| `language` | string | stored but not used by the model (see section 5) |
| `phn_durs` | list of int64 | duration of each phone in 20 ms frames; the sum is about `duration / 0.02` |

Every symbol in `text` must be in `vocab.txt`. Rows with unknown symbols are counted as
`phone` drops in `weights/config.json`.

`extract_features.py` then writes everything else into the experiment directory (`--exp_dir`),
see [training.md](training.md#1-feature-extraction).

### Example layout

```
/data/my_corpus/
  speaker_a/
    utt_0001.wav  utt_0001.TextGrid
    utt_0002.wav  utt_0002.TextGrid
  speaker_b/
    ...
```

Here the speaker keys are `speaker_a` and `speaker_b`. The arrow file for this layout is
built by `tools/prepare_arrow_from_mfa.py` (see its notes: the roots are hard-coded and
must be edited) and is stored in `/data/my_corpus_arrow/` with `raw.arrow`, `vocab.txt`,
and `spk2id.json`.

## 4. Speakers

- The speaker key is the basename of the wav's parent folder. Every utterance of a
  speaker must sit in the same folder.
- `spk2id.json` must contain every speaker key in `raw.arrow`, with ids `0..N-1`
  without gaps. `extract_features.py` stops with an error if this is not true.
- A speaker needs enough voiced audio for a meaningful F0 range. Without voiced frames the
  F0 range falls back to 50–400 Hz.
- Speaker ids enter the model as an embedding (32 dimensions, frozen). A very large speaker
  count needs a larger embedding table, which is set from the count at training time.

## 5. Accent groups

The model has a language embedding (16 dimensions, frozen). In this code the value is the
output of `data.language_of(audio_path)`, which must be listed in `exp/languages.txt`.

**Current behaviour** (`data.py`): the first `_`-separated token of the parent folder name.

| Folder | Group |
|---|---|
| `english_female_indtef` | `english` |
| `Hindi_0085` | `Hindi` |
| `IISc_IndicTIMIT_TEL_M5` | `IISc` |
| `6877` (LibriTTS speaker) | `6877` (one group per speaker) |

This is simple, but it has a problem: LibriTTS-style numeric folders each become a separate
group, so `languages.txt` can contain hundreds of entries. The docstring of `language_of`
describes the intended behaviour (two groups: Indian-accented English, and non-Indian
English). Choose one of these before training on a large corpus; the change is local to
`language_of` in `data.py`.

## 6. Phones and espeak

Training uses the phone symbols from the alignments. Inference converts text with espeak-ng
(`phonemize.py`) and maps the result onto the same inventory:

- Symbols that espeak produces but the training data does not contain are mapped by
  `ESPEAK_TO_VOCAB` (`ð→d`, `w→ʋ`, `v→ʋ`, `f→pʰ`, `ɾ→ɹ`, `θ→t`). This table was made for
  the English MFA data. Other languages need their own table.
- Stop consonants `p t k` get a closure marker `cl` before them, and the utterance gets `ss`
  and `es` markers, as in the alignments.

Check the mapping on your own data before relying on it. `phonemize.text_to_phones` raises
an error for any symbol that is not in `phones.txt`.

## 7. Size and time (measured here)

| Item | Measured |
|---|---|
| `raw.arrow` | about 1.6 KB per utterance (6.3 MB for 4000 utterances) |
| `feats.arrow` | about 89 KB per utterance (355 MB for 4000 utterances; 25 TAP values per frame, float32, 50 frames per second) |
| Feature extraction | about 5 min for 1924 utterances with 16 workers (about 18 CPU-minutes) |
| Training | about 33 s per epoch for 1924 utterances on one GPU |

Budget about 46 GB of `feats.arrow` for the full 524k-utterance Comb set. Start with
`--max_utts_per_speaker` to test.

## 8. Quality checks before training

```bash
python - <<'EOF'
from datasets import Dataset
raw = Dataset.from_file('/path/to/arrow_dir/raw.arrow')
print('rows', len(raw), 'columns', raw.column_names)
import os
missing = [p for p in raw['audio_path'][:5000] if not os.path.exists(p)]
print('missing wav files in first 5000 rows:', len(missing))
EOF
```

Also check that `sum(phn_durs)` is within about 10 frames of `duration / 0.02` for most rows.
Rows that differ more are dropped during training (`mismatch` count).
