"""
Usage: python3 uncensor_sharegpt.py --in sharegpt_html.json --out sharegpt_clean.json
"""
from better_uncensored import *
import argparse
import tqdm
import json
import numpy as np

def avg_ord(text):
  return np.mean([ord(x) for x in text])

def uncensor_conversations(content, begin=0, end=None, censor=False):
    """
    clean the input json content.
    Args:
        content: json file loaded in memory.
        begin: the beginning index of the content to be cleaned.
        end: the ending index of the content to be cleaned.
    """
    skip_cnt = 0
    missmatch_cnt = 0
    uncen_cnt = 0
    cen_cnt = 0

    content = content[begin:end]
    new_content = []

    for sample in tqdm.tqdm(content):
        skipped = False

        # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
        if any(avg_ord(msg["value"]) > 127 for msg in sample["conversations"]):
            skip_cnt += 1
            continue

        if len(sample["conversations"]) <= 1:
            # The conversation is too short
            skip_cnt += 1
            continue

        for c in sample["conversations"]:
            try:
                if c["from"] == "human":
                    continue
                original = c["value"]
                uncensored = uncensor(original, censor)
                assert len(uncensored) > 0
                c["value"] = uncensored
                # If we remove more than 10 chars then probably we uncensored it
                # Leaving some margin for error in chunking
                if len(original) - len(uncensored) > 10:
                    print(f"\nORIG: {original}\nUNCEN: {uncensored}")
                    uncen_cnt += 1
            except (AssertionError):
                skipped = True
                break
  
        if not skipped:
            new_content.append(sample)
        else:
            cen_cnt += 1

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, cen: {cen_cnt}, uncen: {uncen_cnt}")
    return new_content

def main(args):
    content = json.load(open(args['in_file'], "r"))
    content = uncensor_conversations(content, args['begin'], args['end'], args['censor'])
    json.dump(content, open(args['out_file'], "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--censor", action="store_true", help="Attempt to remove moralizing statements and recover examples. Drop them by default (faster, no ollama).")
    args = parser.parse_args()
    main(vars(args))
