import os
import numpy as np
import argparse

def avg_ord(text):
  try:
    if len(text) < 1:
      return 0
    return np.vectorize(ord)(np.array(list(text))).mean()
  except:
     return 0

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
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = append_bun_to_filename(args.in_file)
    return args