"""Text -> phone symbols with espeak-ng, mapped onto the arrow (MFA) phone inventory.

The training phones come from the MFA alignments stored in the F5 arrow files, so the
model's phone inventory is the arrow vocabulary (vocab.txt). Espeak produces a few
phones that never occur in that vocabulary; they are mapped here (ESPEAK_TO_VOCAB,
from a comparison of espeak and MFA output on 1500 transcripts).

Alignment markers the MFA data contains are reproduced by the rules in
prepare_data_v5.py: 'ss' at the start, 'es' at the end, and 'cl' (closure) before
voiceless stops. Word-boundary pauses ('sp') are not inserted, because the MFA data
only has them where a pause was actually detected.
"""
from phonemizer import phonemize
from phonemizer.separator import Separator

DEFAULT_VOICE = 'en-us'

ESPEAK_TO_VOCAB = {
    'ð': 'd',
    'w': 'ʋ',
    'v': 'ʋ',
    'f': 'pʰ',
    'ɾ': 'ɹ',
    'θ': 't',
}

VOICELESS_STOPS = {'p', 't', 'k'}

_SEPARATOR = Separator(phone=' ', word=' | ', syllable='')


def espeak_phones(text, voice=DEFAULT_VOICE):
    """Returns a list of espeak phone symbols, word boundaries removed."""
    text = text.strip()
    if not text:
        return []
    out = phonemize(
        text,
        language=voice,
        backend='espeak',
        with_stress=False,
        separator=_SEPARATOR,
        strip=True,
        preserve_punctuation=False,
        njobs=1,
    )
    return [t for t in out.split() if t != '|']


def text_to_phones(text, vocab, voice=DEFAULT_VOICE):
    """text -> list of phone symbols that exist in `vocab` (a set of strings).
    Raises ValueError listing any symbol that cannot be mapped."""
    raw = espeak_phones(text, voice)
    mapped = []
    for sym in raw:
        sym = ESPEAK_TO_VOCAB.get(sym, sym)
        mapped.append(sym)

    phones = ['ss']
    for i, sym in enumerate(mapped):
        if sym in VOICELESS_STOPS and (i == 0 or mapped[i - 1] != 's'):
            phones.append('cl')
        phones.append(sym)
    phones.append('es')

    unknown = sorted({p for p in phones if p not in vocab})
    if unknown:
        raise ValueError(f'phones not in the model inventory: {unknown}')
    return phones
