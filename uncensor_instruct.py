"""
Usage: python3 uncensor_sharegpt.py --in sharegpt_html.json --out sharegpt_clean.json
"""
from better_uncensored import *
import logging
import tqdm
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil
from datasets import load_dataset
from utils import *

def process_example(example, censor=False):
    ret = {"example": None, "uncen_cnt": 0, "cen_cnt": 0}
    # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
    if 10 > avg_ord(example["output"]) > 127:
        return ret
    original = example["output"]
    uncensored, censored = uncensor(original, censor)
    # Irrecoverable
    if len(uncensored) == 0:
        logging.debug(f"CENSORED: {original}")
        ret["cen_cnt"] = 1
        return ret
    # Recovered
    if censored:
        example["output"] = uncensored
        ret["example"] = example
        ret["uncen_cnt"] = 1
        return ret
    ret["example"] = example
    return ret

def uncensor_dataset(content, censor=False, workers=None):
    init_globals()
    processed_dataset = content.map(process_example, batched=False, remove_columns=["input","instruction"])
    # partition none and not none from processed_dataset["example"]
    new_content = [x for x in processed_dataset["example"] if x is not None]
    uncen_cnt = sum(processed_dataset["uncen_cnt"])
    cen_cnt = sum(processed_dataset["cen_cnt"])
    skip_cnt = len(processed_dataset["example"]) - len(new_content) - cen_cnt

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, censored: {cen_cnt} uncen: {uncen_cnt}")
    return new_content

def main(args):
    global debug
    debug = args['debug']
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    begin = args['begin'] or 0
    end = args['end'] or ""
    content = load_dataset('json', data_files=args['in_file'], split=f"train[{begin}:{end}]")
    content = uncensor_dataset(content, args['censor'])
    json.dump(content, open(args['out_file'], "w"), indent=2)

debug = False
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = uncensor_args()
    main(vars(args))
