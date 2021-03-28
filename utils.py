from .config import CFG
import random
import numpy as np
import torch
import os
import logging
from sklearn.model_selection import GroupKFold
import pandas as pd
import time

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
def print_log(*args):
    f = open(CFG.log_path, 'a')
    f.write(f'{time.asctime()}: ')
    for arg in args:
        print(arg, end=' ')
        f.write(f'{arg} ')
    print()
    f.write('\n')
    f.close()

def init_logger():
    print_log('='*30, f'Log begin {time.asctime()}', '='*30)