# Training

Training has two stages: feature extraction (once per dataset), then training. Both read
the arrow dataset described in [data.md](data.md).

## 1. Feature extraction

```bash
python extract_features.py \
    --arrow_dir /path/to/arrow_dir \
    --exp_dir exp \
    --workers 16 \
    --max_utts_per_speaker 0
```

| Option | Default | Meaning |
|---|---|---|
| `--arrow_dir` | required | folder with `raw.arrow`, `vocab.txt`, `spk2id.json` |
| `--exp_dir` | `exp` | output folder (created if missing) |
| `--workers` | CPU count − 4 | parallel processes; each uses one thread |
| `--max_utts_per_speaker` | `0` (all) | random subset per speaker, for quick tests |
| `--seed` | `0` | seed for the subset |
| `--chunksize` | `8` | items per worker task |

The script refuses to run if `<exp_dir>/feats.arrow` already exists. Delete the folder or
choose a new `--exp_dir` to start again.

What it computes, for each utterance with a duration of 0.3 s to 40 s:

| Feature | Method | Frame rate | Stored as |
|---|---|---|---|
| Frame count `T` | `n_samples // 320 + 1` at 16 kHz | 20 ms | `num_frames` |
| F0 (Hz) | pYAAPT, frame space 20 ms, range 50–400 Hz | mapped to `T` | `f0` |
| TAPs (25 values) | openSMILE eGeMAPSv02 low-level descriptors (10 ms) | mapped to `T` | `taps` |

Frames of the 10 ms and 20 ms streams are mapped to the `T` mel frames proportionally over
the utterance. Unvoiced F0 is set to 1.

### Outputs in `<exp_dir>/`

| File | Content |
|---|---|
| `phones.txt` | `symbol<TAB>id`; id 0 is `<eps>` (padding). Built from `vocab.txt` |
| `speakers.txt` | speaker keys, one per line; line number is the id |
| `languages.txt` | accent group names (see [data.md](data.md#5-accent-groups)) |
| `feats.arrow` | per utterance: `raw_index`, `num_frames`, `f0`, `taps` |
| `speaker_stats.json` | per speaker `min_f0` and `max_f0` over voiced frames |
| `scalers/minmax.scaler_<speaker>` | per speaker MinMaxScaler fitted on its TAP frames |
| `extract_meta.json` | counts and settings |

Note: the TAP scalers are fitted on all extracted frames of a speaker, including the
frames that later become the validation set. The effect is small, but it is a leak.

## 2. Training

```bash
python train.py \
    --arrow_dir /path/to/arrow_dir \
    --exp_dir exp \
    --frames_per_batch 2500 \
    --max_samples 32 \
    --workers 8 \
    --epochs 300
```

Each epoch:

1. Training pass over frame-budget batches (shuffled each epoch with `--seed`).
2. Validation pass over the dev set.
3. One random dev utterance is synthesized with the current model. The output goes to
   `<exp_dir>/samples/` (see below).
4. Checkpoints are written, and the learning rate may drop.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--arrow_dir` | required | arrow folder (same as for extraction) |
| `--exp_dir` | `exp` | experiment folder with the files from step 1 |
| `--out_dir` | `<exp_dir>/weights` | where checkpoints go |
| `--frames_per_batch` | `2500` | mel frames per batch (2500 frames = 50 s of audio) |
| `--max_samples` | `32` | maximum utterances per batch |
| `--dev_fraction` | `0.005` | fraction of utterances held out for validation |
| `--max_dev` | `2000` | cap on the number of validation utterances |
| `--seed` | `0` | seed for the split, batch shuffling and sample choice |
| `--epochs` | `300` | maximum epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--patience` | `10` | early stopping: epochs without validation improvement |
| `--lr_patience` | `30` | epochs without improvement before the learning rate drops |
| `--lr_factor` | `0.1` | learning-rate multiplier at each drop |
| `--min_lr` | `1e-8` | learning-rate floor |
| `--workers` | `8` | data loader processes |
| `--log_every` | `10` | print the loss every N training steps (0 = never) |
| `--resume` | none | path to a checkpoint, usually `weights/last.pt` |
| `--device` | `cuda` if available | `cuda` or `cpu` |

### Loss

| Term | Target | Loss | Weight |
|---|---|---|---|
| `dur` | phone durations (MFA), min-max scaled to 1–50 frames | L1 | 20 |
| `pitch` | F0, min-max scaled per speaker | L1 | 10 |
| `vad` | voiced flag (F0 > 10 Hz) | binary cross-entropy | 1 |
| `mel` | log-mel (Vocos domain) | L1 | 10 |
| `taps` | eGeMAPS TAPs, scaled per speaker | MSE | 1 |
| L2 | attention weights, `1e-5` | sum of squares | 1 |

The log-mel values are negative, so the mel loss values are not directly comparable with the
paper's loss scale. The weight of 10 is a starting point.

### Logs

```
batches: train 324  dev 2
epoch 0001  train 14.7398  val 12.7177  lr 1.0e-03  best 12.7177  49s
```

(The sample run used 1,924 utterances with a 2-utterance dev set, so the dev loss is
noisy. Use the full dev set for real comparisons.)

`dropped` counts utterances skipped by the dataset checks: `no_features` (no entry in
`feats.arrow`), `duration`, `phone` (symbol not in `phones.txt`), and `mismatch` (phone
durations differ from the mel length by more than 10 frames).

## 3. Outputs

```
<exp_dir>/
  weights/
    config.json          arguments, model sizes, dropped counts
    weights-0001.pt      model weights and config for each epoch (inference)
    best.pt              lowest validation loss
    last.pt              full state (optimizer, scheduler, counters) for --resume
  samples/
    ep0001_<speaker>_<utt>.wav   synthesized dev utterance, one per epoch (16 kHz)
    original_<speaker>_<utt>.wav the source wav, copied once
```

The per-epoch `weights-NNNN.pt` files are about 21 MB each. `last.pt` is larger because it
also holds the optimizer state.

Listen to the `samples/` files to check progress. Both files for an utterance are kept, so
the same utterance can be compared across epochs.

## 4. Resuming

```bash
python train.py --arrow_dir /path/to/arrow_dir --exp_dir exp --resume exp/weights/last.pt
```

The epoch counter, best validation loss, early stopping counter, and scheduler state come
from the checkpoint. The data split is recomputed from `--seed`, so keep the same seed.

## 5. Practical notes

- Start with a subset: `extract_features.py --max_utts_per_speaker 300`, then train for a
  few epochs to check that the losses fall and the `samples/` audio is intelligible.
- Speed: about 33 s per epoch for 1924 utterances on one GPU in development. The full
  524k-utterance corpus is about 270 times larger.
- `--workers` loads the wav files and computes the mel spectrogram. Increase it if the GPU
  waits for data.
- The model has about 5.3 M trainable parameters. The phone, language and speaker
  embeddings are frozen and not counted, so changing the number of speakers or languages
  does not change the trainable size.
