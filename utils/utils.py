import torch
import numpy as np
import random
from contextlib import contextmanager

def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@contextmanager
def eval_mode(model):
    orig_mode = model.training
    model.eval()
    yield
    model.train(orig_mode)
