from scheduler_warm import GradualWarmupSchedulerV2
import numpy as np
import torch
from loss_fn import seg_loss_fn
from load_dataset import get_seg_loader
from seg_model import SegModel
from config import CFG
from seg_train import train_one_epoch
from seg_valid import valid_fn

def train_loop(fold_id, logger, model_path, resume=False, debug=True):
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
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.seg_epochs - CFG.seg_warm)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler)
    scaler = torch.cuda.amp.GradScaler()
    criterion = seg_loss_fn()

    for epoch in range(CFG.seg_epochs):
        print('='*30, 'Training Epoch', epoch, 'Fold', fold_id, '='*30)
        print(f'best_train_loss:{best_train_loss}, lr:{scheduler.get_last_lr()}')

        logger.info('='*30, 'Training Epoch', epoch, 'Fold', {fold_id}, '='*30)
        logger.info(f'best_train_loss:{best_train_loss}, lr:{scheduler.get_last_lr()}')
        scheduler_warmup.step()
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, scaler)
        logger.info(f'Train loss:{train_loss}')
        print(f'Train loss:{train_loss}')

        if train_loss < best_train_loss:
            logger.info(f'Get better loss: {best_train_loss} -> {train_loss}')
            print(f'Get better loss: {best_train_loss} -> {train_loss}')
            best_train_loss = train_loss

        print('='*30, 'Validating Epoch', scheduler.last_epoch+1, 'Fold', fold_id, '='*30)
        print(f'best_valid_loss:{best_valid_loss}')

        valid_loss = valid_fn(model, criterion, valid_loader)
        logger.info(f'Valid loss:{valid_loss}')
        print(f'Valid loss:{valid_loss}')

        if valid_loss < best_valid_loss:
            logger.info(f'**Get better **valid** loss: {best_valid_loss} -> {valid_loss}')
            print(f'Get better loss: {best_valid_loss} -> {valid_loss}')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path) # 保存在验证集上最好的模型

        checkpoint = {
            'scheduler':scheduler.state_dict(),
            'optimizer':optimizer.state_dict(),
            'model':model.state_dict(),
            'best_train_loss':best_train_loss,
            'best_valid_loss':best_valid_loss,
        }
        torch.save(checkpoint, latest_path) # 保存最新的模型