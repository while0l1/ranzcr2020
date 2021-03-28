import torch.nn as nn

def seg_loss_fn():
    return nn.BCEWithLogitsLoss()