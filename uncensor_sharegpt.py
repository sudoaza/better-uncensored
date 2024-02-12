"""
Usage: python3 uncensor_sharegpt.py --in sharegpt_html.json --out sharegpt_clean.json
"""
from better_uncensored import *
import argparse
import tqdm
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil

def avg_ord(text):
  try:
    if len(text) < 1:
      return 0
    return np.vectorize(ord)(np.array(list(text))).mean()
  except:
     return 0

def get_optimal_workers(memory_per_worker=1600):
    """
    Estimate the optimal number of workers based on available memory and an estimated memory usage per worker.    
    Arg: memory_per_worker (int): Estimated memory usage per worker in MB.
    """
    memory_per_worker *= 1024 ** 2
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * 0.8
    else:
        available_memory = 0.95 * psutil.virtual_memory().available
    max_workers_based_on_memory = available_memory // memory_per_worker
    return max(1, int(max_workers_based_on_memory))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--censor", action="store_true", help="Attempt to remove moralizing statements and recover examples. Drop them by default (faster, no ollama).")
    args = parser.parse_args()
    main(vars(args))
