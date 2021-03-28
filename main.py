from .utils import init_logger, seed_torch
from .config import CFG

logger = init_logger(CFG.log_path) # 初始化logger

seed_torch(CFG.seed) # 设置随机数种子