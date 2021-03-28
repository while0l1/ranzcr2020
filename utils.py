from .config import CFG
import random
import numpy as np
import torch
import os
import logging
from sklearn.model_selection import GroupKFold
import pandas as pd

'''
设置随机数种子
'''
def seed_torch(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

'''
初始化logger
'''
def init_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename, mode='a')
    fh.setFormatter(logging.Formatter("[%(asctime)s]:%(message)s"))
    logger.addHandler(fh)
    return logger