import os, re
import sys
from textgrids import TextGrid
sys.path.append(os.getcwd())

import json
from pathlib import Path
import epitran
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import concurrent.futures
puncts = [",", ".", "।", "?", "!", ";", ":", "\"", "'","“", "”", "‘", "’", "—", "-","…", "(", ")", "[", "]", "{", "}", "/", "\\"]

def extract_phones_with_punctuation(textgrid_path, lab_file, wav_file):
    try:
        wav_path = Path(wav_file)
        if not wav_path.exists():
            print(f"{wav_path} doesnot exist")
            return None
        grids = TextGrid(textgrid_path)
        language = str(wav_path).split('/')[-1].split('_')[0]  
        phone_tier = grids.get("phones")
        tg_phones = [p.text.strip() for p in phone_tier]
        tg_durs = [int(round((p.xmax - p.xmin) * 1000 / 20)) for p in phone_tier]
        token_idx = 0
        phones=[]
        phn_durs=[]
        n = len(tg_phones)
        skip = False
        for i, label in enumerate(tg_phones):
            if skip:
                skip = False
                continue
            dur = tg_durs[i]

            # ---------- RULE 1: First phone empty -> replace with "ss"
            if i == 0 and label == "":
                phones.append("ss")
                phn_durs.append(dur)
                continue

            # ---------- RULE 2: Last phone empty
            if i == n - 1 and label == "":
                left = tg_phones[i - 1] if i > 0 else None
                if left in puncts:
                    phn_durs[-1] += dur
                    continue                
                phones.append("es")
                phn_durs.append(dur)
                continue             

            # ---------- RULE 3: Empty phone handling
            if label == "":
                left = tg_phones[i - 1] if i > 0 else None
                right = tg_phones[i + 1] if i < n - 1 else None                
                if left in puncts:
                    phn_durs[-1] += dur
                    continue
                if left == 'sp':
                    phones[-1] = 'pp'
                    phn_durs[-1] += dur
                    continue
                if right in puncts:
                    phones.append(right)
                    phn_durs.append(dur+tg_durs[i+1])
                    skip = True
                    continue
                if right == 'sp':
                    phones.append('pp')
                    phn_durs.append(dur+tg_durs[i+1])
                    skip = True
                    continue                
                phones.append("pp")
                phn_durs.append(dur)                
                continue           

            # ---------- Normal phone
            phones.append(label)
            phn_durs.append(dur)                        

        if len(phones) != len(phn_durs):
            import pdb
            pdb.set_trace()
            skipped_files +=1
            print(f"Skipping {textgrid_path} because phones ({len(phones)}) "
                f"!= durations ({len(phn_durs)})")
            return None 
        if sum(phn_durs) != sum(tg_durs):
            import pdb
            pdb.set_trace()
            skipped_files +=1
            print(f"Skipping {textgrid_path} because duration mismatch)")
            return None             

        duration = sf.info(str(wav_path)).duration
        if duration < 0.4 or duration > 40:
            print(f"{wav_path} long or short")
            return None

        return {
            "audio_path": str(wav_path),
            "text": phones,
            "duration": duration,
            "lang":language,
            "phn_durs": phn_durs,
            "vocab_set": set(phones)
        }
    except Exception as e:
        import pdb
        pdb.set_trace()
        print(f"Failed for {wav_path}: {e}")
        return None     
   

def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    skipped_files = 0
    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")    
    arrow_writer = ArrowWriter(path=f"{save_dir}/raw.arrow")
    mis =  open("missed.txt", "w")
    root_dirs = ['/data/AccentConv/Giridhar/MFA/LibriTTS_360_MFA_20', '/data/AccentConv/Giridhar/MFA/En_US', '/data/AccentConv/Giridhar/MFA/Indic_Accent']
    #if True:      
    for root_dir in root_dirs:
        align_dirs = os.listdir(root_dir)
        for align_dir1 in align_dirs:
            align_dir = os.path.join(root_dir, align_dir1)
            if not os.path.isdir(align_dir):
                continue
            for align_file in tqdm(os.listdir(align_dir)):                
                align_path = os.path.join(align_dir, align_file)
                if 'En_US' in root_dir:
                    wav_file = (os.path.join(align_dir.replace('MFA/', ''), align_file)).replace('.TextGrid', '.wav')
                    lab_file = (os.path.join(align_dir.replace('MFA/', ''), align_file)).replace('.TextGrid', '.lab')
                elif 'LibriTTS_360_MFA_20' in root_dir:
                    wav_file = (os.path.join(align_dir.replace('MFA/', '').replace('LibriTTS_360_MFA_20/', 'LibriTTS_360_MFA/'), align_file)).replace('.TextGrid', '.wav')
                    lab_file = (os.path.join(align_dir.replace('MFA/', '').replace('LibriTTS_360_MFA_20/', 'LibriTTS_360_MFA/'), align_file)).replace('.TextGrid', '.lab')
                elif 'Indic_Accent' in root_dir:
                    wav_file = (os.path.join(align_dir.replace('MFA/', ''), align_file)).replace('.TextGrid', '.wav')
                    lab_file = (os.path.join(align_dir.replace('MFA/', ''), align_file)).replace('.TextGrid', '.lab')
                    
                res = extract_phones_with_punctuation(align_path, lab_file, wav_file)

                if res:
                    arrow_writer.write({
                        "audio_path": res["audio_path"],
                        "text": res["text"],
                        "duration": res["duration"],
                        "language": res["lang"],
                        "phn_durs": res["phn_durs"]
                    })
                    duration_list.append(res["duration"])

                    new_vocab = res["vocab_set"]
                    before_size = len(text_vocab_set)
                    text_vocab_set.update(new_vocab)
                    after_size = len(text_vocab_set)

                    # if after_size > before_size:
                    #     print(f"Vocab updated by file: {wav_file} "
                    #         f"(+{after_size - before_size} new entries)")                    
                    # # text_vocab_set.update(res["vocab_set"]) 
                else:
                    mis.write(wav_file)
                    mis.write('\n')


    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(duration_list)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    print(f"Skipped {skipped_files} files due to mismatch.")


if __name__ == "__main__":
    tokenizer = "char"  # "pinyin" | "char"
    dataset_name = f"Model_V5_Comb_{tokenizer}"
    save_dir = os.getcwd() + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()
