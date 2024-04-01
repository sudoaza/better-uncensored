"""
Usage: python3 uncensor_ultrackat.py --in-file new_dataset.jsonl --out-file uncensored_dataset.jsonl

Process dataset with JSON lines format. Each line is a JSON object with an array conversation at "data".
Focus on uncensoring the odd indexed elements of the "data" array.
"""
from better_uncensored import *
import logging
import tqdm
import json
from utils import *
from uncensor_base import *

convo_col = "data" ## --convo-col

def process_example(example, censor=False):
    example.update({"uncen_cnt": 0, "cen_cnt": 0})
    conversation = example[convo_col]
    example[convo_col] = []
    # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
    if 10 > avg_ord("".join(conversation)) > 127:
        return example

    if not censor:
        # remove human messages and concat all assistant messages for more speed
        # get every other element of the list
        original = "\n\n".join(conversation[1::2])
        uncensored, censored = uncensor(original, censor)
        if len(uncensored) == 0:
            #logging.debug(f"CENSORED: {original}")
            example["cen_cnt"] = 1
            return example
    else:
        new_conversation = []
        for original in conversation:
            uncensored, censored = uncensor(original, censor)
            # Irrecoverable
            if len(uncensored) == 0:
                #logging.debug(f"CENSORED: {original}")
                example["cen_cnt"] = 1
                return example
            # Recovered
            if censored:
                new_conversation.append(uncensored)
                example["uncen_cnt"] += 1
            else:
                new_conversation.append(original)
        example[convo_col] = new_conversation
    return example

def uncensor_dataset(content, censor=False):
    init_globals()
    processed_dataset = content.map(lambda e: process_example(e, censor), batched=False)
    new_content = processed_dataset.filter(lambda e: e[convo_col] not in (None, "", [])).remove_columns(["uncen_cnt", "cen_cnt"])
    uncen_cnt = sum(x["uncen_cnt"] for x in processed_dataset)
    cen_cnt = sum(x["cen_cnt"] for x in processed_dataset)
    skip_cnt = len(processed_dataset) - len(new_content) - cen_cnt

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, censored: {cen_cnt}, uncen: {uncen_cnt}")
    return new_content

if __name__ == "__main__":
    args = uncensor_args()  # Ensure this function is defined or replaced with appropriate argument parsing
    convo_col = args.convo_col or "data"
    main(vars(args), uncensor_dataset)
