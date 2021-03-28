from utils import init_logger, seed_torch
from config import CFG
from seg_train_loop import train_loop
import warnings

warnings.filterwarnings("ignore")

init_logger() # 初始化logger

seed_torch(CFG.seed) # 设置随机数种子

fold_id = 0
train_loop(fold_id, CFG.model_path, resume=False, debug=True)