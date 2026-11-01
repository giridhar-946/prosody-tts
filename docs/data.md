# Data requirements

This system needs three things per utterance: the **audio**, the **phone sequence**, and
the **phone durations** (a forced alignment).

## 1. Audio

| Requirement | Value |
|---|---|
| Format | any file `soundfile` can read (wav, flac, ...) |
| Channels | mono (multi-channel input is averaged) |
| Sampling rate | any; resampled to 16 kHz for all features |
| Duration | 0.3 s to 40 s per utterance (outside this range the utterance is skipped) |
| Naming | the **parent folder of each wav is the speaker key** (see section 4) |

The audio is read at training time as well (for the mel spectrogram), so the wav paths in
the arrow file must stay valid.

## 2. Data layout

```
/data/
  language1/
    speaker_a/
      utt_0001.wav
      utt_0001.lab
      utt_0002.wav
      utt_0002.lab
  speaker_b/
      utt_1001.wav
      utt_1001.lab
      utt_1002.wav
      utt_1002.lab
    ...
```

Also check that `sum(phn_durs)` is within about 10 frames of `duration / 0.02` for most rows.
Rows that differ more are dropped during training (`mismatch` count).
