"""Unit tests that run without the arrow data, wav files or espeak.

    pip install pytest
    pytest tests -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data  # noqa: E402
import mel  # noqa: E402
import phonemize  # noqa: E402
from model import MLMSTTS  # noqa: E402
from utils import arange_list, inv_minmax_norm, minmax_norm  # noqa: E402


def test_mel_frame_count_matches_formula():
    wav = torch.zeros(16000)                       # 1 s at 16 kHz
    m = mel.get_vocos_mel_spectrogram(wav)
    assert m.shape == (mel.N_MEL_CHANNELS, mel.mel_frames(16000)) == (100, 51)


def test_mel_is_log_domain_and_clamped():
    m = mel.get_vocos_mel_spectrogram(torch.zeros(16000))
    assert torch.allclose(m, torch.full_like(m, np.log(1e-5)))


def test_minmax_roundtrip():
    x = np.array([1.0, 10.0, 50.0])
    assert np.allclose(inv_minmax_norm(minmax_norm(x, 1, 50), 1, 50), x)


def test_arange_list_positions():
    fwd, rev = arange_list([3, 2])
    assert fwd.tolist() == [1, 2, 3, 1, 2]
    assert rev.tolist() == [3, 2, 1, 2, 1]


def test_fix_phone_durations_sums_to_frames():
    assert sum(data.fix_phone_durations([3, 4, 5], 20)) == 20
    assert data.fix_phone_durations([3, 4, 5], 1) is None      # last phone would be < 1 frame
    assert data.fix_phone_durations([3, 4, 5], 12) == [3, 4, 5]


def test_read_phone_table_keeps_space_symbol(tmp_path):
    p = tmp_path / 'phones.txt'
    p.write_text('<eps>\t0\n \t1\na\t2\n', encoding='utf-8')
    assert data.read_phone_table(str(p)) == {'<eps>': 0, ' ': 1, 'a': 2}


def test_speaker_and_language_from_path():
    path = '/x/english_female_indtef/english_female_indtef_00001.wav'
    assert data.speaker_of(path) == 'english_female_indtef'
    assert data.language_of(path) == 'english'


def test_language_of_is_folder_prefix_for_indic_and_libri():
    # Current behaviour: first token of the parent folder name.
    assert data.language_of('/x/Hindi_0085/a.wav') == 'Hindi'
    assert data.language_of('/x/6877/a.wav') == '6877'


def make_model():
    torch.manual_seed(0)
    return MLMSTTS(n_phones=20, n_langs=2, n_speakers=3)


def fake_batch(B=2, L=(5, 4), durs=((2, 1, 3, 1, 1), (2, 2, 1, 1))):
    items = []
    for b in range(B):
        d = list(durs[b])
        n = sum(d)
        items.append({
            'phones': np.arange(1, len(d) + 1, dtype=np.int64),
            'durs': np.asarray(d, dtype=np.int64),
            'f0': np.full(n, 120, dtype=np.float32),
            'pitch': np.full(n, 0.5, dtype=np.float32),
            'vad': np.ones(n, dtype=np.float32),
            'dur_norm': minmax_norm(np.asarray(d, dtype=np.float32), 1, 50).astype(np.float32),
            'taps': np.zeros((n, 25), dtype=np.float32),
            'mel': torch.zeros(n, 100),
            'lang_id': b % 2,
            'spk_id': b % 3,
        })
    return items


def test_model_forward_shapes():
    model = make_model().eval()
    inputs, targets = data.collate_fn(fake_batch())
    with torch.no_grad():
        out = model(**inputs)
    F = inputs['len_mask'].shape[1]
    assert out['dur'].shape == targets['dur'].shape
    assert out['mel'].shape == (2, F, 100) == targets['mel'].shape
    assert out['taps'].shape == (2, F, 25)
    assert out['pitch'].shape == out['vad'].shape == (2, F, 1)


def test_model_backward_runs():
    model = make_model().train()
    inputs, targets = data.collate_fn(fake_batch())
    pred = model(**inputs)
    loss = torch.nn.functional.l1_loss(pred['mel'], targets['mel']) + pred['dur'].abs().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_collate_invariants():
    items = fake_batch()
    inputs, _ = data.collate_fn(items)
    for b, it in enumerate(items):
        frames = int(inputs['len_mask'][b].sum())
        assert frames == len(it['f0']) == int(it['durs'].sum())
        assert int(inputs['phn_mask'][b].sum()) == len(it['phones'])
    assert int(inputs['phn_repeats'][..., 1].max()) < inputs['phn_lab'].shape[1]


class FakeDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def get_frame_len(self, i):
        return self.lengths[i]


def test_dynamic_batch_sampler_respects_budget():
    ds = FakeDataset([100, 300, 120, 50, 400, 90])
    sampler = data.DynamicBatchSampler(ds, frames_threshold=500, max_samples=3, random_seed=0)
    seen = sorted(i for batch in sampler.batches for i in batch)
    assert seen == list(range(len(ds)))
    for batch in sampler.batches:
        assert len(batch) <= 3
        assert sum(ds.lengths[i] for i in batch) <= 500 or len(batch) == 1


def test_espeak_tokens_are_mapped(monkeypatch):
    monkeypatch.setattr(phonemize, 'espeak_phones', lambda text, voice='en-us': ['ð', 'w', 'a', 't', 'f'])
    vocab = {'ss', 'es', 'cl', 'd', 'ʋ', 'a', 't', 'pʰ'}
    assert phonemize.text_to_phones('x', vocab) == ['ss', 'd', 'ʋ', 'a', 'cl', 't', 'pʰ', 'es']


def test_unknown_phone_raises(monkeypatch):
    monkeypatch.setattr(phonemize, 'espeak_phones', lambda text, voice='en-us': ['q'])
    with pytest.raises(ValueError):
        phonemize.text_to_phones('x', {'ss', 'es'})
