"""
Usage: python3 uncensor_sharegpt.py --in-file instruct_data.json

Process instruction dataset like Open Instruct V1.

Example data:
```json
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        output_col: "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables..."
    },
    ...
]
```
"""
from better_uncensored import *
import logging
import tqdm
import json
from utils import *
from uncensor_base import *

output_col = "output" ## --output-col

def process_example(example, censor=False):
    try:
        if not example or output_col not in example:
            return example
        if 10 > avg_ord(example[output_col]) > 127:
            return example

        example.update({"uncen_cnt": 0, "cen_cnt": 0})
        uncensored, censored = uncensor(example[output_col], censor)
        # Irrecoverable
        if len(uncensored) == 0:
            logging.debug(f"CENSORED: {example[output_col]}")
            example["cen_cnt"] = 1
            return example
        # Recovered
        if censored:
            example[output_col] = uncensored
            example["uncen_cnt"] = 1
            return example
    except Exception as e:
        logging.error(f"Error processing example: {e}, example: {example}, ret: {ret}")
        raise e
    return example

def uncensor_dataset(content, censor=False):
    init_globals()
    processed_dataset = content.map(lambda e: process_example(e, censor), batched=False)
    new_content = processed_dataset.filter(lambda e: e[output_col] not in (None, "")).remove_columns(["uncen_cnt", "cen_cnt"])
    uncen_cnt = sum(x["uncen_cnt"] for x in processed_dataset)
    cen_cnt = sum(x["cen_cnt"] for x in processed_dataset)
    skip_cnt = len(processed_dataset) - len(new_content) - cen_cnt

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}, censored: {cen_cnt}, uncen: {uncen_cnt}")
    return new_content

if __name__ == "__main__":
    args = uncensor_args()
    output_col = args.output_col or "output"
    main(vars(args), uncensor_dataset)
