# Inference

`synthesize.py` turns text (or phone symbols) into a 16 kHz wav file for one speaker.

## Quick use

```bash
python synthesize.py \
    --exp_dir exp \
    --speaker english_female_indtef \
    --text_file examples/sentences.txt \
    --out_dir Synthesized
```

`examples/sentences.txt`:

```
english	The weather is nice today.
english	Please call me when you reach the station.
```

Output:

```
Synthesized/english_female_indtef/
  001_english_ep0300.wav          16 kHz mono wav
  001_english_ep0300.phones.txt   the phone sequence that was used
  002_english_ep0300.wav
  ...
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--exp_dir` | `exp` | experiment folder from training |
| `--ckpt` | `<exp_dir>/weights/best.pt` | checkpoint to use (e.g. `weights/weights-0300.pt`) |
| `--speaker` | required | speaker key; must appear in `<exp_dir>/speakers.txt` |
| `--text_file` | required | input file (below) |
| `--out_dir` | `Synthesized` | output root; a speaker subfolder is created inside |
| `--voice` | `en-us` | espeak-ng voice used for the text column |
| `--phones_input` | off | the text column is already phone symbols (no espeak) |
| `--vocoder_dir` | `$MLMS_VOCODER_DIR` or the development path | Vocos folder (`config.yaml`, `pytorch_model.bin`) |
| `--device` | `cuda` if available | `cuda` or `cpu` |

## Input file

One utterance per line: `<language><TAB><text>`.

- `<language>` must be a name in `<exp_dir>/languages.txt`. It selects the accent group
  the model conditions on.
- Blank lines are ignored. The text must be separated from the language by a real tab.

Errors stop the run with a message naming the sentence number. An unknown language or an
unknown speaker is reported before any audio is produced.

## Text input (espeak)

With the default mode, the text is passed to espeak-ng (`phonemize.py`). The result is
checked against `<exp_dir>/phones.txt`.

Behind the scenes:

1. espeak-ng produces IPA phone symbols (stress removed), for example
   `h ə l oʊ h aʊ ɑːɹ j uː t ə d eɪ`.
2. Symbols that the training data does not contain are mapped (`ð→d`, `w→ʋ`, `v→ʋ`,
   `f→pʰ`, `ɾ→ɹ`, `θ→t`).
3. Markers are added: `ss` at the start, `es` at the end, and `cl` before `p`, `t`, `k`.
4. Any symbol still not in `phones.txt` raises an error listing it.

The `.phones.txt` output file shows the exact sequence that the model received. Check it
when a word sounds wrong; it tells you whether the problem is the phonemizer or the model.

Use `--voice` for a different espeak voice. The mapping table in `phonemize.py` was built
for English only, so other voices need their own check.

## Phone input

```bash
printf "english\tss h ə l oʊ cl t ə d eɪ es\n" > phones.txt.in
python synthesize.py --exp_dir exp --speaker english_female_indtef \
    --text_file phones.txt.in --phones_input
```

With `--phones_input` the text column is split on spaces, and each symbol must be in
`phones.txt`. Use this to test specific pronunciations, or to bypass espeak entirely.

## How a sentence becomes audio

| Step | What happens | Output |
|---|---|---|
| Duration | the encoder and duration head predict a length for each phone | phone lengths in 20 ms frames (rounded, clipped to 1–255) |
| F0 and VAD | the model predicts F0 and voicing for each frame; F0 is smoothed with a median filter of 3 and zeroed where VAD < 0.3 | reference F0 (Hz) |
| Mel | the model predicts a log-mel spectrogram from the phones, durations, and F0 | 100 × T log-mel |
| Vocoder | Vocos decodes the log-mel to a waveform | 16 kHz audio, peak-normalized |

The model is non-autoregressive, so the cost grows with the number of frames, not with a
sequential decoder.

## Choosing a checkpoint

- `best.pt` is the default: the lowest validation loss.
- To compare epochs, pass `--ckpt exp/weights/weights-0150.pt`. The output file name
  contains the epoch number (`ep0150`), so outputs from several checkpoints can share one
  `--out_dir` without overwriting each other.

## Vocoder

The vocoder is Vocos, used frozen. It needs:

- `config.yaml` with `sample_rate: 16000`, `n_fft: 1024`, `hop_length: 320`, and
  `n_mels: 100` (the feature extractor in `mel.py` must match).
- `pytorch_model.bin` with the generator weights. Discriminator weights in the same file
  are ignored (they show up as "unexpected keys" when the model loads, which is expected).

Set the location once:

```bash
export MLMS_VOCODER_DIR=/path/to/vocos_16khz_20ms
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `phones not in phones.txt: [...]` | espeak produced a symbol the training data does not contain. Add it to `ESPEAK_TO_VOCAB` in `phonemize.py` or pick another phrasing |
| `unknown speaker` | the key is not in `speakers.txt`. Copy it exactly, including case |
| `unknown language` | the first column does not match a line in `languages.txt` |
| `phonemizer` cannot load the espeak library | install `espeak-ng` (`apt-get install espeak-ng`) |
| Output is silent or clipped | check the `.phones.txt`; a very short or empty phone list produces little audio |
| Very long pauses or missing words | try the next checkpoint: early epochs are often unstable |
