"""
Punctuation is turned into word tokens (COMMA, FLSTP, ...) and spaces into '_', then each
token is phonemized on its own: '_' -> 'sp', punctuation tokens -> the punctuation symbol,
words -> espeak (language picked per word: ASCII words are English) followed by the
script-specific replacement table in Replacements/. The result is wrapped in 'ss' ... 'es'.
"""
import json
import os
import re

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

_SEPARATOR = Separator(phone=' ', word='  ')
_REPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Replacements')

lang_code_map = {'hindi': 'hi', 'bengali': 'bn', 'malayalam': 'ml', 'kannada': 'kn', 'tamil': 'ta',
                 'telugu': 'te', 'assamese': 'as', 'bodo': 'hi', 'gujarati': 'gu', 'manipuri': 'bpy',
                 'marathi': 'mr', 'odia': 'or', 'rajasthani': 'hi', 'magadhi': 'hi', 'maithili': 'hi',
                 'sanskrit': 'hi', 'bhojpuri': 'hi', 'sindhi': 'sd', 'chhattisgarhi': 'hi', 'dogri': 'hi',
                 'DOI': 'hi', 'punjabi': 'pa', 'konkani': 'kok', 'nepali': 'ne', 'magahi': 'hi',
                 'english': 'en-gb', 'en': 'en-gb', 'niaenglish': 'en-gb'}

punct_map = {
    ",": " COMMA ",
    ".": " FLSTP ",
    "।": " FLSTPH ",
    "?": " QMARK ",
    "!": " EXCLM ",
    ";": " SEMICLN ",
    ":": " COLON ",
    "\"": " DQUOTE ",
    "'": " SQUOTE ",
    "“": " LDQUOTE ",
    "”": " RDQUOTE ",
    "‘": " LSQUOTE ",
    "’": " RSQUOTE ",
    "—": " DASH ",
    "–": " DASH ",
    "-": " HYPHEN ",
    "…": " ELLIPS ",
    "(": " LPAR ",
    ")": " RPAR ",
    "[": " LBRAC ",
    "]": " RBRAC ",
    "{": " LCURLY ",
    "}": " RCURLY ",
    "/": " SLASH ",
    "\\": " BSLASH "
}

punct_reverse_map = {
    "COMMA": ",",
    "FLSTP": ".",
    "FLSTPH": "।",
    "QMARK": "?",
    "EXCLM": "!",
    "SEMICLN": ";",
    "COLON": ":",
    "DQUOTE": "\"",
    "SQUOTE": "'",
    "LDQUOTE": "“",
    "RDQUOTE": "”",
    "LSQUOTE": "‘",
    "RSQUOTE": "’",
    "DASH": "—",
    "HYPHEN": "-",
    "ELLIPS": "…",
    "LPAR": "(",
    "RPAR": ")",
    "LBRAC": "[",
    "RBRAC": "]",
    "LCURLY": "{",
    "RCURLY": "}",
    "SLASH": "/",
    "BSLASH": "\\",
}

_backends = {}
_replacements = {}


def _backend(language):
    # created on first use: one espeak backend per language is slow to build
    if language not in _backends:
        if language not in lang_code_map:
            raise ValueError(f'no espeak voice for language {language}; known: {sorted(lang_code_map)}')
        _backends[language] = EspeakBackend(lang_code_map[language], preserve_punctuation=True,
                                            language_switch='remove-flags', words_mismatch='ignore')
    return _backends[language]


def _replacement_table(name):
    if name not in _replacements:
        with open(os.path.join(_REPL_DIR, f'replacements_{name}.json'), encoding='utf-8') as f:
            rep = json.load(f)
        _replacements[name] = (re.compile('|'.join(map(re.escape, rep.keys()))), rep)
    return _replacements[name]


def apply_replacements(text, pattern, replacements):
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def tokenize_gen_transcript(text, language):
    phonemized = _backend(language).phonemize([text], separator=_SEPARATOR)
    if language in ['telugu', 'kannada', 'tamil', 'malayalam']:
        pat, rep = _replacement_table('drav')
    elif language in ['english', 'en', 'niaenglish']:
        pat, rep = _replacement_table('en')
    else:
        pat, rep = _replacement_table('deva')
    result = apply_replacements(phonemized[0], pat, rep)
    normalized = re.sub(r"\s+", " ", result.strip())
    return normalized.split(" ")


def detect_language(word, lang):
    """Detects if word is English or Indic (e.g. Hindi)."""
    return 'en' if re.fullmatch(r'[a-zA-Z0-9]+', word) else lang


def tokenize_cross_transcript(text, main_lang):
    results = []
    for token in text.split(' '):
        if token in punct_reverse_map:
            results.append(punct_reverse_map[token])      # COMMA -> ","
        elif token == '_':
            results.append('sp')
        else:
            results.extend(tokenize_gen_transcript(token, language=detect_language(token, main_lang)))
    return results


def text_to_phones(text, vocab, language):
    """text -> list of phone symbols that exist in `vocab` (a set of strings).
    Raises ValueError listing any symbol not in the model inventory."""
    text = text.strip()
    text = text.replace(" ", " _ ")
    for punct, token in punct_map.items():
        text = text.replace(punct, token)
    for token in punct_map.values():
        text = text.replace(f"{token} _", token)

    # Clean multiple underscores or spaces
    text = " ".join(text.split())

    phones = ['ss'] + tokenize_cross_transcript(text, language) + ['es']

    unknown = sorted({p for p in phones if p not in vocab})
    if unknown:
        raise ValueError(f'phones not in the model inventory: {unknown}')
    return phones
