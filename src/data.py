import os
import random
import numpy as np
import torch
from datasets import load_dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_samsum_data():
    dataset = load_dataset("samsum")
    return dataset