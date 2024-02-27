
import os
import json
import argparse
from better_uncensored import *
from datasets import disable_caching, load_dataset

debug = False

def get_path_extension(filepath):
    path, name = os.path.split(filepath)
    base, ext = os.path.splitext(name)
    return ext

def append_bun_to_filename(filename):
    path, name = os.path.split(filename)
    base, ext = os.path.splitext(name)
    new_name = f"{base}_bun{ext}"
    return os.path.join(path, new_name)

def uncensor_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--begin", type=str, default="0")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--split-name", type=str, default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--censor", action="store_true", help="Attempt to remove moralizing statements and recover examples. Drop them by default (faster, no ollama).")
    parser.add_argument("--no-cache", action="store_true", help="Don't use HF datasets cache.")
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = append_bun_to_filename(args.in_file)
    if args.no_cache:
        disable_caching()
    return args

def main(args, uncensor_dataset):
    global debug
    debug = args['debug']
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    content = load_dataset('json', data_files=args['in_file'], split=f"{args['split_name']}[{args['begin']}:{args['end']}]")
    content = uncensor_dataset(content, args['censor'])
    if get_path_extension(args['out_file']) == ".jsonl":
        with open(args['out_file'], 'w') as outfile:
            for entry in content:
                json.dump(entry, outfile)
                outfile.write('\n')
    else:
        # Default to JSON
        json.dump(content, open(args['out_file'], "w"), indent=2)
