
import os
import argparse
from better_uncensored import *
from datasets import disable_caching

debug = False

def append_bun_to_filename(filename):
    path, name = os.path.split(filename)
    base, ext = os.path.splitext(name)
    new_name = f"{base}_bun{ext}"
    return os.path.join(path, new_name)

def uncensor_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--censor", action="store_true", help="Attempt to remove moralizing statements and recover examples. Drop them by default (faster, no ollama).")
    parser.add_argument("--no-cache", action="store_true", help="Don't use HF datasets cache.")
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = append_bun_to_filename(args.in_file)
    if args.no_cache:
        disable_caching()

    return args

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
