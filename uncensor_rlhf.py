"""
Usage: python3 uncensor_new_dataset.py --in-file new_dataset.jsonl --out-file uncensored_dataset.jsonl

Process dataset with JSON lines format. Each line is a JSON object with "chosen" and "rejected" fields.
Focus on uncensoring the "chosen" field.
"""
from better_uncensored import *
import logging
import tqdm
import json
from datasets import load_dataset
from utils import *
from uncensor_base import *

def process_example(example, censor=False):
    ret = {"example": None, "uncen_cnt": 0, "cen_cnt": 0}
    chosen_text = example["chosen"]
    # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
    if 10 > avg_ord(chosen_text) > 127:
        return ret

    original = chosen_text
    if not censor:
        # remove human messages for better precission
        # only when we discard examples, because we modify the format
        original = "\n\n".join([piece.replace("Assistant: ","") for piece in original.split("\n\n") if piece.startswith("Assistant: ")])

    uncensored, censored = uncensor(original, censor)
    # Irrecoverable
    if len(uncensored) == 0:
        #logging.debug(f"CENSORED: {original}")
        ret["cen_cnt"] = 1
        return ret
    # Recovered
    if censored:
        example["chosen"] = uncensored
        ret["example"] = example
        ret["uncen_cnt"] = 1
        return ret
    ret["example"] = example
    return ret

def uncensor_dataset(content, censor=False):
    init_globals()
    processed_dataset = content.map(lambda e: process_example(e, censor), batched=False)
    # Prepare new content and counts
    new_content = [x["example"] for x in processed_dataset if x["example"] is not None]
    uncen_cnt = sum(x["uncen_cnt"] for x in processed_dataset)
    cen_cnt = sum(x["cen_cnt"] for x in processed_dataset)
    skip_cnt = len(processed_dataset) - len(new_content) - cen_cnt

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, censored: {cen_cnt}, uncen: {uncen_cnt}")
    return new_content

if __name__ == "__main__":
    args = uncensor_args()  # Ensure this function is defined or replaced with appropriate argument parsing
    main(vars(args), uncensor_dataset)
