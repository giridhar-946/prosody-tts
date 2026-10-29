"""PyTorch port of the MLMS TTS graph defined in the TensorFlow train_fixed.py.

Layer-by-layer port: text encoder, duration / VAD / F0 heads, acoustic decoder,
mel head and TAPs decoder. Shapes are batch-first: (B, T, C).

Known differences from the Keras graph:
  * Keras 'same' padding and Keras BatchNormalization (momentum 0.99, eps 1e-3)
    are reproduced explicitly; weight initialisers use PyTorch defaults.
  * Keras mask propagation through the BiGRUs is not reproduced; padded
    positions are simply run through the GRU.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(position, d_model):
    """Sinusoidal table of shape (position, d_model), same as train_fixed.py."""
    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.tensor(angle_rads, dtype=torch.float32)


class SameConv1d(nn.Module):
    """Conv1d over (B, T, C) with TensorFlow 'same' padding and optional ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, relu=True):
        super().__init__()
        total = kernel_size - 1
        self.pad_before = total // 2
        self.pad_after = total - self.pad_before
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.relu = relu

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad_before, self.pad_after))
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        return x.transpose(1, 2)


class BN(nn.Module):
    """Keras BatchNormalization over the last axis of (B, T, C)."""

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class MHAttn(nn.Module):
    """Port of swan.MHAttn: windowed multi-head self-attention, no padding mask."""

    def __init__(self, in_dim, n_heads, head_size, context):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.context = context
        self.WQ = nn.Parameter(torch.empty(n_heads, in_dim, head_size))
        self.WV = nn.Parameter(torch.empty(n_heads, in_dim, head_size))
        self.WU = nn.Parameter(torch.empty(n_heads * head_size, in_dim))
        for w in (self.WQ, self.WV, self.WU):
            nn.init.xavier_uniform_(w)

    def l2_penalty(self):
        # Keras kernel_regularizer=L2(1e-5) on each weight.
        return 1e-5 * (self.WQ.pow(2).sum() + self.WV.pow(2).sum() + self.WU.pow(2).sum())

    def forward(self, x):
        B, T, _ = x.shape
        idx = torch.arange(T, device=x.device)
        bool_mask = (idx[:, None] - idx[None, :]).abs() <= self.context
        context_mask = 1e9 * (~bool_mask).float()

        Q = torch.einsum('btd,hde->bhte', x, self.WQ)
        V = torch.einsum('btd,hde->bhte', x, self.WV)
        S = torch.einsum('bhte,bhse->bhts', Q, V) / math.sqrt(self.head_size)
        W = torch.softmax(S - context_mask, dim=-1)
        C = torch.einsum('bhts,bhse->bhte', W, V)
        C = C.permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_size)
        return C @ self.WU


class TextEncoder(nn.Module):
    def __init__(self, in_dim=128, lang_dim=16, inp_dim=128):
        super().__init__()
        self.conv_bank = nn.ModuleList(
            [nn.Sequential(SameConv1d(in_dim + lang_dim, 128, k), BN(128)) for k in range(1, 9)])
        self.post_conv = nn.Sequential(SameConv1d(1024, 128, 3), BN(128))
        self.proj = nn.Sequential(SameConv1d(128, inp_dim, 3, relu=False), BN(inp_dim))
        self.gru = nn.GRU(inp_dim, 64, batch_first=True, bidirectional=True)

    def forward(self, phone_emb, lang_emb):
        x = torch.cat([phone_emb, lang_emb], dim=-1)
        x = torch.cat([layer(x) for layer in self.conv_bank], dim=-1)
        # MaxPool1D(pool_size=2, strides=1, padding='same')
        x = F.pad(x.transpose(1, 2), (0, 1), value=float('-inf'))
        x = F.max_pool1d(x, kernel_size=2, stride=1).transpose(1, 2)
        x = self.post_conv(x)
        x = self.proj(x)
        x = x + phone_emb
        x, _ = self.gru(x)
        return x


class VadF0Estimator(nn.Module):
    """Port of vad_f0_estimator (one shared MHAttn per block)."""

    def __init__(self, in_dim, n_blocks, n_heads=4, head_size=32, context=10,
                 inter_dim=128, out_dim=64):
        super().__init__()
        self.n_blocks = n_blocks
        self.inp = Dense(in_dim, inter_dim, 'relu')
        self.attn = nn.ModuleList([MHAttn(inter_dim, n_heads, head_size, context) for _ in range(n_blocks)])
        self.bn1 = nn.ModuleList([BN(inter_dim) for _ in range(n_blocks)])
        self.ffn = nn.ModuleList([nn.Sequential(Dense(inter_dim, inter_dim, 'relu'),
                                                Dense(inter_dim, inter_dim, 'relu')) for _ in range(n_blocks)])
        self.bn2 = nn.ModuleList([BN(inter_dim) for _ in range(n_blocks)])
        self.out = Dense(inter_dim, out_dim, 'relu')

    def l2_penalty(self):
        return sum(a.l2_penalty() for a in self.attn)

    def forward(self, x):
        x = self.inp(x)
        for i in range(self.n_blocks):
            cx = self.attn[i](x)
            x = self.bn1[i](x + cx)
            xe = self.ffn[i](x)
            x = self.bn2[i](xe + x)
        return self.out(x)


class NewAcousticDecoder(nn.Module):
    """Port of new_acoustic_decoder. Each outer block reuses its attention output
    for five residual sub-steps, as in the Keras source."""

    SUBSTEPS = 5

    def __init__(self, in_dim, n_blocks=1, n_heads=4, head_size=64, context=10,
                 inter_dim=128, out_dim=128):
        super().__init__()
        self.n_blocks = n_blocks
        self.inp = Dense(in_dim, inter_dim, 'relu')
        self.attn = nn.ModuleList([MHAttn(inter_dim, n_heads, head_size, context) for _ in range(n_blocks)])
        self.bn_attn = nn.ModuleList([nn.ModuleList([BN(inter_dim) for _ in range(self.SUBSTEPS)])
                                      for _ in range(n_blocks)])
        self.ffn = nn.ModuleList([nn.ModuleList([nn.Sequential(Dense(inter_dim, inter_dim, 'relu'),
                                                               Dense(inter_dim, inter_dim, 'relu'))
                                                 for _ in range(self.SUBSTEPS)]) for _ in range(n_blocks)])
        self.bn_ffn = nn.ModuleList([nn.ModuleList([BN(inter_dim) for _ in range(self.SUBSTEPS)])
                                     for _ in range(n_blocks)])
        self.out = Dense(inter_dim, out_dim, 'relu')

    def l2_penalty(self):
        return sum(a.l2_penalty() for a in self.attn)

    def forward(self, x):
        x = self.inp(x)
        for b in range(self.n_blocks):
            cx = self.attn[b](x)
            for s in range(self.SUBSTEPS):
                x = self.bn_attn[b][s](x + cx)
                xe = self.ffn[b][s](x)
                x = self.bn_ffn[b][s](xe + x)
        return self.out(x)


class AcousticDecoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_bank = nn.ModuleList(
            [nn.Sequential(SameConv1d(in_dim, 128, k + 1), BN(128)) for k in range(1, 11)])
        self.post_conv = nn.ModuleList([nn.Sequential(SameConv1d(1280 if i == 0 else 128, 128, 3), BN(128))
                                        for i in range(3)])
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = torch.cat([layer(x) for layer in self.conv_bank], dim=-1)
        x = F.pad(x.transpose(1, 2), (0, 1), value=float('-inf'))
        x = F.max_pool1d(x, kernel_size=2, stride=1).transpose(1, 2)
        for layer in self.post_conv:
            x = layer(x)
        x, _ = self.gru(x)
        return x


class TapsDecoder(nn.Module):
    def __init__(self, in_dim=100):
        super().__init__()
        self.convs = nn.Sequential(
            SameConv1d(in_dim, 128, 5), BN(128),
            SameConv1d(128, 128, 5), BN(128),
            SameConv1d(128, 128, 5), BN(128),
            SameConv1d(128, 128, 5), BN(128))

    def forward(self, x):
        return self.convs(x)


class MLMSTTS(nn.Module):
    """Multilingual, multi-speaker, prosody-aware acoustic model.

    Inputs (all long unless noted):
      phn_lab     (B, L)    phone ids of collapsed phone runs, 0 = padding
      phn_mask    (B, L, 1) float, 1 for real phones
      phn_repeats (B, F, 2) [batch index, phone-run index] per output frame
      fwd_idx     (B, F)    position of frame within its phone run (1-based)
      rev_idx     (B, F)    reverse position within its phone run
      len_mask    (B, F, 1) float, 1 for real frames
      ref_f0      (B, F)    reference F0 in Hz (int) used by the F0 embedding
      lang_lab    (B, L)    language id
      spkr_lab    (B, L)    speaker id
    Outputs: dict with dur (B,L,1), pitch (B,F,1), vad (B,F,1), mel (B,F,100) log-mel, taps (B,F,25).
    """

    N_MEL = 100          # Vocos mel bins (mel.py)
    N_TAPS = 25

    def __init__(self, n_phones, n_langs, n_speakers):
        super().__init__()
        self.phone_emb = nn.Embedding(n_phones, 128)
        self.lang_emb = nn.Embedding(n_langs, 16)
        self.spkr_emb = nn.Embedding(n_speakers, 32)
        for emb in (self.phone_emb, self.lang_emb, self.spkr_emb):
            # Frozen random embeddings, as in the Keras model (trainable=False).
            nn.init.trunc_normal_(emb.weight, std=0.5, a=-1.0, b=1.0)
            emb.weight.requires_grad_(False)
        self.register_buffer('idx_table', positional_encoding(256, 16))
        self.register_buffer('f0_table', positional_encoding(400, 32))

        self.text_encoder = TextEncoder()
        self.dur_head = nn.Sequential(Dense(160, 64, 'relu'), Dense(64, 64, 'relu'), Dense(64, 1, 'relu'))

        self.vad_estimator = VadF0Estimator(128 + 16 + 16, n_blocks=2)
        self.vad_head = nn.Sequential(Dense(64, 64, 'relu'), Dense(64, 1, 'sigmoid'))

        self.pitch_estimator = VadF0Estimator(160 + 16 + 16, n_blocks=2)
        self.pitch_head = nn.Sequential(Dense(64, 64, 'relu'), Dense(64, 1, 'relu'))

        self.acoustic_decoder = AcousticDecoder(160 + 16 + 16 + 32)
        self.new_acoustic_decoder = NewAcousticDecoder(128, n_blocks=3)
        # Linear output: the target is a log-mel (negative values), so no ReLU/sigmoid gate.
        self.mel_head = Dense(128, self.N_MEL)

        self.taps_decoder = TapsDecoder(self.N_MEL)
        self.taps_hidden = Dense(128, 64, 'relu')
        self.taps_value = Dense(64, self.N_TAPS, 'relu')
        self.taps_gate = Dense(64, self.N_TAPS, 'relu')

    @staticmethod
    def gather_frames(x, repeats):
        """Equivalent of tf.gather_nd(x, phn_repeats): (B,L,C) -> (B,F,C)."""
        return x[repeats[..., 0], repeats[..., 1]]

    def l2_penalty(self):
        return self.vad_estimator.l2_penalty() + self.pitch_estimator.l2_penalty() \
            + self.new_acoustic_decoder.l2_penalty()

    def forward_duration(self, phn_lab, phn_mask, lang_lab, spkr_lab):
        """Encoder + duration head only (used at inference for the first pass)."""
        enc = self.text_encoder(self.phone_emb(phn_lab), self.lang_emb(lang_lab))
        text_spkr = torch.cat([enc, self.spkr_emb(spkr_lab)], dim=-1)
        dur = self.dur_head(text_spkr) * phn_mask
        return dur, enc, text_spkr

    def forward(self, phn_lab, phn_mask, phn_repeats, fwd_idx, rev_idx, len_mask, ref_f0, lang_lab, spkr_lab):
        dur, enc, text_spkr = self.forward_duration(phn_lab, phn_mask, lang_lab, spkr_lab)

        fwd = self.idx_table[fwd_idx.long()]
        rev = self.idx_table[rev_idx.long()]

        up_enc = self.gather_frames(enc, phn_repeats) * len_mask
        vad_in = torch.cat([up_enc, fwd, rev], dim=-1)
        vad_feat = self.vad_estimator(vad_in)
        vad = self.vad_head(vad_feat) * len_mask

        up_text = self.gather_frames(text_spkr, phn_repeats) * len_mask
        dec_in = torch.cat([up_text, fwd, rev], dim=-1)
        pitch = self.pitch_head(self.pitch_estimator(dec_in)) * vad

        f0_emb = self.f0_table[ref_f0.long().clamp(0, 399)]
        x = torch.cat([dec_in, f0_emb], dim=-1)
        x = self.acoustic_decoder(x)
        x = self.new_acoustic_decoder(x)
        mel = self.mel_head(x) * len_mask

        taps = self.taps_decoder(mel)
        taps = self.taps_hidden(taps)
        taps = self.taps_value(taps) * self.taps_gate(taps) * len_mask

        return {'dur': dur, 'pitch': pitch, 'vad': vad, 'mel': mel, 'taps': taps}
