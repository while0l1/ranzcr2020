from scheduler_warm import GradualWarmupSchedulerV2
import numpy as np
import torch
from loss_fn import seg_loss_fn
from load_dataset import get_seg_loader
from seg_model import SegModel
from config import CFG
from seg_train import train_one_epoch
from seg_valid import valid_fn
from utils import print_log

def train_loop(fold_id, model_path, resume=False, debug=True):
    '''
    model_path: 用来保存模型的路径
    '''
    best_train_loss = np.inf
    best_valid_loss = np.inf
    best_model_path = f'{model_path}/{CFG.seg_backbone}_best_fold{fold_id}.pth'
    latest_path = f'{model_path}/{CFG.seg_backbone}_latest_fold{fold_id}.pth' # 之前模型的路径
    device = torch.device('cuda')
 
    train_loader, valid_loader = get_seg_loader(fold_id, debug=debug)

    model = SegModel(CFG.seg_backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.seg_lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.seg_epochs - CFG.seg_warm)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler)
    scaler = torch.cuda.amp.GradScaler()
    criterion = seg_loss_fn()

    if resume:
      checkpoint = torch.load(latest_path)
      model.load_state_dict(checkpoint['model'])
      try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_train_loss = checkpoint['best_train_loss']
        best_valid_loss = checkpoint['best_valid_loss']
        scheduler_warmup.load_state_dict(checkpoint['warm'])
      except:
        print('Not all keys match!')
    
    epoch_trained = scheduler_warmup.last_epoch + scheduler.last_epoch

    for epoch in range(epoch_trained+1, CFG.seg_epochs+1):
        print_log('='*30, f'Epoch {epoch + 1}, Fold {fold_id}', '='*30)
        print_log(f'best_train_loss:{best_train_loss}, lr:{scheduler.get_last_lr()}')

        if epoch_trained > CFG.seg_warm:
          scheduler_warmup.step()
        else:
          scheduler.step()
        print_log('Training...')
        print_log('~'*15)
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, scaler, device)
        print_log(f'Train loss:{train_loss}')

        if train_loss < best_train_loss:
            print_log(f'Get better train loss: {best_train_loss} -> {train_loss}')
            best_train_loss = train_loss

        print_log('Validating...')
        print_log('~'*15)
        print_log(f'best_valid_loss:{best_valid_loss}')

        valid_loss = valid_fn(model, criterion, valid_loader, device)
        print_log(f'Valid loss:{valid_loss}')

        if valid_loss < best_valid_loss:
            print_log(f'Get better valid loss: {best_valid_loss} -> {valid_loss}')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path) # 保存在验证集上最好的模型

        checkpoint = {
            'scheduler':scheduler.state_dict(),
            'warm':scheduler_warmup.state_dict(),
            'optimizer':optimizer.state_dict(),
            'model':model.state_dict(),
            'best_train_loss':best_train_loss,
            'best_valid_loss':best_valid_loss,
        }
        torch.save(checkpoint, latest_path) # 保存最新的模型

        torch.cuda.empty_cache()
        epoch_trained += 1