"""Convert transcripts to the tokenized .lab format used for alignment.

    No sooner was she out of sight, than Emma exclaimed, “What a picture!” I would.
 -> No _ sooner _ was _ she _ out _ of _ sight COMMA than _ Emma _ exclaimed COMMA LDQUOTE What _ a _ picture EXCLM RDQUOTE I _ would FLSTP

Words are joined by '_'; a punctuation mark becomes its token and takes the place of the
space around it. Each mark maps to exactly one token (" -> DQUOTE, “ -> LDQUOTE, ...).

Expected layout:
    <data_dir>/<language>/<speaker>/<utt>.wav  (+ <utt>.lab or <utt>.txt)

Per utterance:
    .lab already in the format -> left as is
    .lab not in the format     -> converted in place
    no .lab, .txt present      -> converted and written to .lab

Usage:
    python tools/preprocess_data.py --data_dir /data [--dry_run]
"""
import argparse
import os
import re
import sys

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
punct_map = {sym: tok for tok, sym in punct_reverse_map.items()}
punct_map["–"] = "DASH"     # en dash
PUNCT_TOKENS = set(punct_reverse_map)
_TOKEN_RE = re.compile("|".join(map(re.escape, punct_map)) + r"|[^\s" + re.escape("".join(punct_map)) + r"]+")

LAYOUT_HELP = """Expected layout:
    <data_dir>/
      <language>/
        <speaker>/
          utt_0001.wav
          utt_0001.lab   (or utt_0001.txt)
          ...
Every wav needs a .lab or .txt with the same name. Keep the data in this format."""


def convert(text):
    """Plain text -> 'word _ word PUNCT word ...'."""
    out = []
    prev_is_word = False
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok == "_":
            # word separator from a partly formatted .lab; re-inserted below where needed
            continue
        if tok in punct_map or tok in PUNCT_TOKENS:
            out.append(punct_map.get(tok, tok))
            prev_is_word = False
        else:
            if prev_is_word:
                out.append("_")
            out.append(tok)
            prev_is_word = True
    return " ".join(out)


def is_formatted(text):
    """True when text is already 'word _ word PUNCT ...' with no raw punctuation."""
    tokens = text.split()
    if not tokens:
        return False
    for i, tok in enumerate(tokens):
        if tok == "_":
            # '_' only between two words
            if i == 0 or i == len(tokens) - 1:
                return False
            left, right = tokens[i - 1], tokens[i + 1]
            if left in PUNCT_TOKENS or left == "_" or right in PUNCT_TOKENS or right == "_":
                return False
        elif tok in PUNCT_TOKENS:
            continue
        else:
            if any(ch in punct_map for ch in tok):
                return False
            # two words next to each other must have '_' between them
            if i > 0 and tokens[i - 1] != "_" and tokens[i - 1] not in PUNCT_TOKENS:
                return False
    return True


def validate_layout(data_dir):
    """Returns (problems, ignored). Problems stop the run; ignored lists loose files
    next to the language / speaker folders, which are skipped."""
    if not os.path.isdir(data_dir):
        return [f"{data_dir} is not a directory"], []
    problems, ignored = [], []
    languages = sorted(os.listdir(data_dir))
    if not any(os.path.isdir(os.path.join(data_dir, d)) for d in languages):
        problems.append(f"{data_dir}: no language folders found")
    for lang in languages:
        lang_dir = os.path.join(data_dir, lang)
        if not os.path.isdir(lang_dir):
            ignored.append(lang_dir)
            continue
        speakers = sorted(os.listdir(lang_dir))
        if not any(os.path.isdir(os.path.join(lang_dir, s)) for s in speakers):
            problems.append(f"{lang_dir}: no speaker folders found")
        for spk in speakers:
            spk_dir = os.path.join(lang_dir, spk)
            if not os.path.isdir(spk_dir):
                ignored.append(spk_dir)
                continue
            names = os.listdir(spk_dir)
            subdirs = [n for n in names if os.path.isdir(os.path.join(spk_dir, n))]
            if subdirs:
                problems.append(f"{spk_dir}: unexpected sub-folders {subdirs[:5]}")
            stems = {os.path.splitext(n)[0] for n in names if n.endswith(".wav")}
            if not stems:
                problems.append(f"{spk_dir}: no .wav files")
            texts = {os.path.splitext(n)[0] for n in names if n.endswith((".lab", ".txt"))}
            missing = sorted(stems - texts)
            if missing:
                problems.append(f"{spk_dir}: {len(missing)} wav without .lab/.txt, e.g. {missing[:3]}")
    return problems, ignored


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dry_run", action="store_true", help="report what would change, write nothing")
    args = p.parse_args()

    problems, ignored = validate_layout(args.data_dir)
    for path in ignored:
        print(f"WARNING: not a language/speaker folder, ignored: {path}")
    if problems:
        print("Data is not in the required layout:", file=sys.stderr)
        for msg in problems[:50]:
            print("  - " + msg, file=sys.stderr)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more", file=sys.stderr)
        print("\n" + LAYOUT_HELP, file=sys.stderr)
        sys.exit(1)

    kept = converted_lab = converted_txt = empty = 0
    for lang in sorted(os.listdir(args.data_dir)):
        lang_dir = os.path.join(args.data_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for spk in sorted(os.listdir(lang_dir)):
            spk_dir = os.path.join(lang_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            for name in sorted(os.listdir(spk_dir)):
                if not name.endswith(".wav"):
                    continue
                stem = os.path.join(spk_dir, name[:-4])
                lab, txt = stem + ".lab", stem + ".txt"
                src = lab if os.path.exists(lab) else txt
                with open(src, encoding="utf-8") as f:
                    text = " ".join(f.read().split())
                if not text:
                    empty += 1
                    print(f"WARNING: empty transcript {src}")
                    continue
                if src == lab and is_formatted(text):
                    kept += 1
                    continue
                new_text = convert(text)
                if src == lab:
                    converted_lab += 1
                else:
                    converted_txt += 1
                if not args.dry_run:
                    with open(lab, "w", encoding="utf-8") as f:
                        f.write(new_text + "\n")

    print(f"Already formatted .lab: {kept}")
    print(f"Converted .lab in place: {converted_lab}")
    print(f"Converted .txt -> .lab:  {converted_txt}")
    print(f"Empty transcripts:       {empty}")
    if args.dry_run:
        print("(dry run: nothing written)")


if __name__ == "__main__":
    main()
