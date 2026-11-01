import os, json, re, tqdm, argparse
from pathlib import Path
from phonemizer.backend import EspeakBackend
# backend = EspeakBackend('en-gb', preserve_punctuation=True)
from phonemizer.separator import Separator
separator = Separator(phone=' ', word='')
from phonemizer.punctuation import Punctuation

parser = argparse.ArgumentParser(description="Pass language and path")
parser.add_argument("--language", type=str, help="Language name")
parser.add_argument("--path", type=str, help="Data path")
args = parser.parse_args()

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

lang_code_map = {'hindi':'hi', 'bengali':'bn', 'malayalam':'ml', 'kannada':'kn', 'tamil':'ta', 
                 'telugu':'te', 'assamese':'as', 'bodo':'hi', 'gujarati':'gu', 'manipuri':'bpy',
                 'marathi':'mr', 'odia':'or', 'rajasthani':'hi', 'magadhi':'hi', 'maithili':'hi', 
                 'sanskrit':'hi', 'bhojpuri':'hi', 'sindhi':'sd', 'chhattisgarhi':'hi', 'dogri':'hi',
                 'DOI':'hi', 'punjabi':'pa', 'konkani':'kok', 'nepali':'ne', 'magahi':'hi',
                 'english':'en-gb', 'en':'en-gb', 'niaenglish':'en-gb', 'chattisgarhi':'hi'}
                 
deva_list = ['hindi', 'bengali', 'assamese', 'bodo', 'gujarati', 'manipuri', 'marathi', 'odia',
             'rajasthani', 'magadhi', 'maithili', 'sanskrit', 'bhojpuri', 'sindhi', 'chhattisgarhi', 
             'dogri', 'DOI', 'punjabi', 'konkani', 'nepali', 'magahi', 'chattisgarhi']
drav_list = ['malayalam', 'kannada', 'tamil', 'telugu']

bckend = lang_code_map[args.language]   
backend = EspeakBackend(bckend, preserve_punctuation=False, language_switch='remove-flags', words_mismatch='ignore')     
bck_en = EspeakBackend('en-gb', preserve_punctuation=False, language_switch='remove-flags', words_mismatch='ignore')     
output_file = f"{args.path}/{args.language}.txt"
ENGLISH_OR_NUMBER = re.compile(r'[A-Za-z0-9]')

if args.language in deva_list:
    rep_tag = 'deva'
elif args.language in drav_list:
    rep_tag = 'drav'
else:
    rep_tag = 'en'    

with open("Replacements/replacements_en.json", "r", encoding="utf-8") as f:
    replacements_en = json.load(f)
pattern_en = re.compile("|".join(map(re.escape, replacements_en.keys())))        

def apply_replacements_en(text: str) -> str:
    return pattern_en.sub(lambda m: replacements_en[m.group(0)], text) 
    
with open(f"Replacements/replacements_{rep_tag}.json", "r", encoding="utf-8") as f:
    replacements = json.load(f)
pattern = re.compile("|".join(map(re.escape, replacements.keys())))        

def apply_replacements(text: str) -> str:
    return pattern.sub(lambda m: replacements[m.group(0)], text)      

def build_pron_dict(root_dir, output_file=output_file):
    dictionary = {}

    out = open(output_file, "w", encoding="utf-8")
    # Traverse subfolders for .lab files
    for lab_file in tqdm.tqdm(Path(root_dir).rglob("*.lab")):
        with open(lab_file, "r", encoding="utf-8") as f:
            for line in f:
                # Collapse multiple spaces
                line = re.sub(r'\s+', ' ', line).strip()                
                words = line.strip().split()
                for word in words:
                    if word not in dictionary:
                        if not word.strip():
                            print("skipped: empty word")
                            continue
                        if word == '_':
                            phonemized = ['sp']
                        elif word in punct_reverse_map:  
                            punct_symbol = punct_reverse_map[word]      # COMMA → ","
                            phonemized = [punct_symbol]                                                        
                        else:
                            if args.language in deva_list or args.language in drav_list:
                                if ENGLISH_OR_NUMBER.search(word):
                                    phonemized = bck_en.phonemize([word],  separator=separator) 
                                else:
                                    phonemized = backend.phonemize([word],  separator=separator)
                            else:
                                phonemized = backend.phonemize([word],  separator=separator)
                                                        
                        if not phonemized or not phonemized[0]:
                            print (word, lab_file)
                            print("skipped: empty phonemization")
                            continue
                        if ENGLISH_OR_NUMBER.search(word):
                            result = apply_replacements_en(phonemized[0])
                        else:
                            result = apply_replacements(phonemized[0])
                        out.write(f"{word}\t{result.strip()}\n")
                        dictionary[word] = result.strip() #" ".join(list(word))            

    print(f"Dictionary saved to {output_file} with {len(dictionary)} entries.")


if __name__ == "__main__":
    build_pron_dict(args.path)

