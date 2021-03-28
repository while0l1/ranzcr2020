from tqdm.auto import tqdm 
import torch
import numpy as np

def valid_fn(model, criterion, valid_loader, device):
    model.eval()
    losses = []
    bar = tqdm(valid_loader, total=len(valid_loader))
    for i, (imgs, masks) in enumerate(bar):
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(imgs)
        loss = criterion(logits, masks)
        losses.append(loss.item())
        bar.update()
    
    return np.mean(losses)