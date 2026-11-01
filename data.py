"""Dataset, dynamic batching and collation over the F5-TTS arrow files.

Pitch and TAP targets come from feats.arrow written by extract_features.py.
The mel spectrogram is computed on the fly from the wav (as CustomDataset in F5 does),
with the Vocos 20 ms settings from mel.py.

Model inputs are built in collate_fn:
    phn_lab, phn_mask, phn_repeats, fwd_idx, rev_idx, len_mask, ref_f0, lang_lab, spkr_lab
Targets: dur, pitch, vad, mel (log-mel), taps.
"""
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from mel import SAMPLE_RATE, get_vocos_mel_spectrogram
from utils import arange_list, minmax_norm

MIN_DURATION, MAX_DURATION = 0.3, 40.0        # seconds, as in F5 CustomDataset
DUR_NORM_MIN, DUR_NORM_MAX = 1, 50            # phone duration range in 20 ms frames
MAX_PHONE_FRAMES = 255                        # fwd/rev indices are uint8 in the model
MAX_DURATION_MISMATCH = 10                    # frames, between phn_durs and the mel
N_TAPS = 25
F0_MAX_EMBED = 399
VOICED_THRESHOLD = 10.0

def language_of(audio_path):
    try:
        bname = os.path.basename(os.path.dirname(audio_path))
        return bname.split('_')[0]
    except:
        ValueError(f'no language for {audio_path}')


def speaker_of(audio_path):
    """Speaker key used by spk2id.json: the parent folder of the wav."""
    return os.path.basename(os.path.dirname(audio_path))


def read_phone_table(path):
    """phones.txt: 'symbol<TAB>id' per line. Tab-separated because the vocabulary
    contains the space character."""
    table = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line:
                sym, idx = line.rsplit('\t', 1)
                table[sym] = int(idx)
    return table


def read_name_ids(path):
    """speakers.txt / languages.txt: one name per line, id = line number."""
    names = {}
    with open(path, encoding='utf-8') as f:
        for k, line in enumerate(f):
            line = line.strip()
            if line:
                names[line] = k
    return names


def proportional_index(n_src, n_dst):
    """Index into an n_src-frame stream for each of n_dst frames (pins both edges)."""
    idx = np.floor(np.arange(n_dst) * n_src / n_dst).astype(np.int64)
    return np.clip(idx, 0, max(n_src - 1, 0))


def load_wav_16k(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(audio)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def fix_phone_durations(durs, n_frames):
    """Makes sum(durs) == n_frames by adjusting the last phone, as F5 collate_fn does.
    Returns None when that is not possible (last phone would become < 1 frame)."""
    durs = list(durs)
    diff = n_frames - sum(durs)
    if diff != 0:
        durs[-1] = durs[-1] + diff
    if durs[-1] < 1 or max(durs) > MAX_PHONE_FRAMES:
        return None
    return durs


class MLMSDataset(Dataset):
    """One item per utterance from raw.arrow, joined with feats.arrow by raw index."""

    def __init__(self, arrow_dir, feats, indices, phone_table, spk2id, lang2id, speaker_stats, scalers):
        self.raw = HFDataset.from_file(os.path.join(arrow_dir, 'raw.arrow'))
        self.feats = feats
        self.phone_table = phone_table
        self.spk2id = spk2id
        self.lang2id = lang2id
        self.speaker_stats = speaker_stats
        self.scalers = scalers

        feat_pos = {int(r): p for p, r in enumerate(feats['raw_index'])}
        feat_frames = np.asarray(feats['num_frames'])
        self.items = []     # (raw_index, feat_position, n_frames, durations)
        self.dropped = {'no_features': 0, 'duration': 0, 'phone': 0, 'mismatch': 0}
        for i in tqdm(indices, desc='Checking utterances'):
            row = self.raw[i]
            if i not in feat_pos:
                self.dropped['no_features'] += 1
                continue
            if not (MIN_DURATION <= row['duration'] <= MAX_DURATION):
                self.dropped['duration'] += 1
                continue
            if any(p not in phone_table for p in row['text']):
                self.dropped['phone'] += 1
                continue
            p = feat_pos[i]
            n_frames = int(feat_frames[p])
            durs = list(row['phn_durs'])
            if abs(n_frames - sum(durs)) > MAX_DURATION_MISMATCH:
                self.dropped['mismatch'] += 1
                continue
            durs = fix_phone_durations(durs, n_frames)
            if durs is None:
                self.dropped['mismatch'] += 1
                continue
            self.items.append((i, p, n_frames, durs))

    def get_frame_len(self, index):
        return self.items[index][2]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, p, n_frames, durs = self.items[index]
        row = self.raw[i]
        feat = self.feats[p]
        path = row['audio_path']
        spk = speaker_of(path)

        mel = get_vocos_mel_spectrogram(load_wav_16k(path)).T     # (T, 100)
        T = mel.shape[0]
        if T != n_frames:
            # feats were computed on the same wav, so this only guards against drift
            mel = mel[proportional_index(T, n_frames)]
        f0 = np.asarray(feat['f0'], dtype=np.float32)
        taps = np.asarray(feat['taps'], dtype=np.float32).reshape(-1, N_TAPS)
        stats = self.speaker_stats[spk]

        phones = [self.phone_table[s] for s in row['text']]
        return {
            'mel': mel.float(),
            'phones': np.asarray(phones, dtype=np.int64),
            'durs': np.asarray(durs, dtype=np.int64),
            'f0': f0,
            'pitch': minmax_norm(f0, stats['min_f0'], stats['max_f0']).astype(np.float32),
            'vad': (f0 > VOICED_THRESHOLD).astype(np.float32),
            'dur_norm': minmax_norm(np.asarray(durs, dtype=np.float32), DUR_NORM_MIN, DUR_NORM_MAX).astype(np.float32),
            'taps': self.scalers[spk].transform(taps).astype(np.float32),
            'lang_id': self.lang2id[language_of(path)],
            'spk_id': self.spk2id[spk],
        }


class DynamicBatchSampler(Sampler):
    """Frame-budget batches, as F5 DynamicBatchSampler: sorted by length, batches of at
    most frames_threshold mel frames and max_samples utterances, shuffled per epoch."""

    def __init__(self, dataset, frames_threshold, max_samples=0, random_seed=0, shuffle=True):
        self.dataset = dataset
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.epoch = 0

        indices = sorted(((idx, dataset.get_frame_len(idx)) for idx in range(len(dataset))), key=lambda e: e[1])
        batches, batch, batch_frames = [], [], 0
        for idx, frame_len in indices:
            fits = batch_frames + frame_len <= frames_threshold
            room = max_samples == 0 or len(batch) < max_samples
            if fits and room:
                batch.append(idx)
                batch_frames += frame_len
            else:
                if batch:
                    batches.append(batch)
                batch, batch_frames = [idx], frame_len
        if batch:
            batches.append(batch)
        self.batches = batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if not self.shuffle:
            return iter(self.batches)
        g = torch.Generator()
        g.manual_seed(self.random_seed + self.epoch)
        order = torch.randperm(len(self.batches), generator=g).tolist()
        return iter([self.batches[i] for i in order])

    def __len__(self):
        return len(self.batches)


def collate_fn(items):
    B = len(items)
    L = max(len(it['phones']) for it in items)
    Fm = max(it['mel'].shape[0] for it in items)

    def pad(arrays, length, dtype, trailing=()):
        out = np.zeros((B, length) + trailing, dtype=dtype)
        for b, a in enumerate(arrays):
            out[b, :len(a)] = a
        return torch.from_numpy(out)

    phn_lab = pad([it['phones'] for it in items], L, np.int64)
    phn_mask = pad([np.ones(len(it['phones'])) for it in items], L, np.float32).unsqueeze(-1)
    lang_lab = pad([np.full(len(it['phones']), it['lang_id']) for it in items], L, np.int64)
    spkr_lab = pad([np.full(len(it['phones']), it['spk_id']) for it in items], L, np.int64)

    repeats, fwd, rev = [], [], []
    for b, it in enumerate(items):
        frame_phone = np.repeat(np.arange(len(it['durs'])), it['durs'])
        repeats.append(np.stack((np.full(len(frame_phone), b), frame_phone), axis=-1))
        f, r = arange_list(it['durs'])
        fwd.append(f.astype(np.int64))
        rev.append(r.astype(np.int64))
    phn_repeats = pad(repeats, Fm, np.int64, (2,))
    fwd_idx = pad(fwd, Fm, np.int64)
    rev_idx = pad(rev, Fm, np.int64)
    len_mask = pad([np.ones(it['mel'].shape[0]) for it in items], Fm, np.float32).unsqueeze(-1)
    ref_f0 = pad([np.clip(it['f0'], 0, F0_MAX_EMBED).astype(np.int64) for it in items], Fm, np.int64)

    inputs = dict(phn_lab=phn_lab, phn_mask=phn_mask, phn_repeats=phn_repeats, fwd_idx=fwd_idx,
                  rev_idx=rev_idx, len_mask=len_mask, ref_f0=ref_f0, lang_lab=lang_lab, spkr_lab=spkr_lab)

    targets = dict(
        dur=pad([it['dur_norm'] for it in items], L, np.float32).unsqueeze(-1),
        pitch=pad([it['pitch'] for it in items], Fm, np.float32).unsqueeze(-1),
        vad=pad([it['vad'] for it in items], Fm, np.float32).unsqueeze(-1),
        mel=torch.nn.utils.rnn.pad_sequence([it['mel'] for it in items], batch_first=True),
        taps=pad([it['taps'] for it in items], Fm, np.float32, (N_TAPS,)),
    )
    return inputs, targets


def load_feats(exp_dir):
    return HFDataset.from_file(os.path.join(exp_dir, 'feats.arrow'))


def load_speaker_tables(exp_dir):
    """speaker_stats.json and per-speaker TAP scalers written by extract_features.py."""
    import joblib
    with open(os.path.join(exp_dir, 'speaker_stats.json'), encoding='utf-8') as f:
        stats = json.load(f)
    scalers = {}
    scal_dir = os.path.join(exp_dir, 'scalers')
    for name in os.listdir(scal_dir):
        scalers[name[len('minmax.scaler_'):]] = joblib.load(os.path.join(scal_dir, name))
    return stats, scalers
