"""
Usage: python3 uncensor_sharegpt.py --in-file sharegpt_html.json
"""
from better_uncensored import *
import tqdm
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from utils import *

def process_conversation(conversation, censor=False):
    # The conversation is too short
    if len(conversation["conversations"]) <= 1:
        return None, 0

    if any(len(msg["value"]) < 1 for msg in conversation["conversations"]):
        return None, 0

    # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
    if any(10 > avg_ord(msg["value"]) > 127 for msg in conversation["conversations"]):
        return None, 0

    uncen_cnt = 0
    for c in conversation["conversations"]:
        # Don't censor human messages
        if c["from"] == "human":
            continue
        original = c["value"]
        uncensored, censored = uncensor(original, censor)
        # Irrecoverable
        if len(uncensored) == 0:
            return None, 0
        # Recovered
        if censored:
            c["value"] = uncensored
            uncen_cnt += 1
    return conversation, uncen_cnt

def uncensor_conversations(content, censor=False, workers=None):
    if workers is None:
        workers = get_optimal_workers()
    print(f"Starting with {workers} workers")
    skip_cnt = 0
    uncen_cnt = 0
    new_content = []

    with ProcessPoolExecutor(max_workers=workers, initializer=init_globals) as executor:
        futures = [executor.submit(process_conversation, conversation, censor) for conversation in content]
        for future in tqdm.tqdm(futures):
            result, count = future.result()
            if result is not None:
                new_content.append(result)
                uncen_cnt += count
            else:
                skip_cnt += 1

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, uncen: {uncen_cnt}")
    return new_content

def main(args):
    global debug
    debug = args['debug']
    content = json.load(open(args['in_file'], "r"))[args['begin']:args['end']]
    content = uncensor_conversations(content, args['censor'])
    json.dump(content, open(args['out_file'], "w"), indent=2)

debug = False
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = uncensor_args()
    main(vars(args))
