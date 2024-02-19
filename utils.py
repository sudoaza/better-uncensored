import numpy as np
import psutil
import torch

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
