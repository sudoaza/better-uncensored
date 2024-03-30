"""
Usage: python3 uncensor_sharegpt.py --in-file sharegpt_html.json
"""
from better_uncensored import *
import tqdm
import json
import numpy as np
from utils import *
from uncensor_base import *

convo_col = "conversations" ## --convo-col
output_col = "value" ## --output-col
role_col = "from" ## --role-col
human_tag = "human" ## --human-tag

def process_conversation(conversation, censor=False):
    ret = {"example": None, "uncen_cnt": 0, "cen_cnt": 0}

    # The conversation is too short
    if len(conversation[convo_col]) <= 1:
        return ret

    if any(len(msg[output_col]) < 1 for msg in conversation[convo_col]):
        return ret

    # REMOVE to keep mainly non ascii chars (chineese/korean/etc.)
    if any(10 > avg_ord(msg[output_col]) > 127 for msg in conversation[convo_col]):
        return ret

    for c in conversation[convo_col]:
        # Don't censor human messages
        if c[role_col] == human_tag:
            continue
        original = c[output_col]
        uncensored, censored = uncensor(original, censor)
        # Irrecoverable
        if len(uncensored) == 0:
            ret["cen_cnt"] += 1
            return ret
        # Recovered
        if censored:
            c[output_col] = uncensored
            ret["uncen_cnt"] += 1

    ret["example"] = conversation
    return ret

def uncensor_conversations(content, censor=False):
    init_globals()
    processed_dataset = content.map(lambda e: process_conversation(e, censor), batched=False)
    new_content = processed_dataset.filter(remove_empty_elements)

    uncen_cnt = sum(x["uncen_cnt"] for x in processed_dataset)
    cen_cnt = sum(x["cen_cnt"] for x in processed_dataset)
    skip_cnt = len(processed_dataset) - len(new_content) - cen_cnt

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, censored: {cen_cnt}, uncen: {uncen_cnt}")
    return new_content

if __name__ == "__main__":
    args = uncensor_args()
    convo_col = args.convo_col or "conversations"
    output_col = args.output_col or "value"
    role_col = args.role_col or "from"
    human_tag = args.human_tag or "human"
    main(vars(args), uncensor_conversations)
