import torch
from tqdm.auto import tqdm
import numpy as np

def train_one_epoch(model, optimizer, criterion, train_loader, scaler, device):
    model.train()
    train_losses = []
    bar = tqdm(train_loader, total=len(train_loader))
    for i, (imgs, masks) in enumerate(bar):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_losses.append(loss.item())
        smooth_loss = np.mean(train_losses[-50:])
        bar.set_description(f'loss:{loss.item():.5f}, smth:{smooth_loss:.5f}')
    return np.mean(train_losses)