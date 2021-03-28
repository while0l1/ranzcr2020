from sklearn.model_selection import GroupKFold
import pandas as pd
import cv2
import os
import numpy as np
import ast
import torch
import albumentations
from config import CFG
from torch.utils.data import DataLoader

class RanzcrDataset(object):
    def __init__(self, root, df, mode='test', transforms=None, train_anno=None):
        self.root = root
        self.transforms = transforms
        self.filenames = (df['StudyInstanceUID']).values
        self.mode = mode
        self.train_anno = train_anno

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filenames[idx] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'test':
            img = self.transforms(image=img)['image']
            img = img.astype('float32').transpose(2, 0, 1) / 255.
            return img
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 2)).astype('float32')
            mask0 = mask[:, :, 0].copy()
            mask1 = mask[:, :, 1].copy()
            this_anno = self.train_anno.query(f'StudyInstanceUID == "{self.filenames[idx]}"')
            for _, anno in this_anno.iterrows():
                 data = np.array(ast.literal_eval(anno["data"]))
                 mask0 = cv2.polylines(mask0, np.int32([data]), isClosed=False, color=1, thickness=15, lineType=16) # 管道位置画线
                 mask1 = cv2.circle(mask1, (data[0][0], data[0][1]), radius=15, color=1, thickness=25) # 管道开始位置画圈
                 mask1 = cv2.circle(mask1, (data[-1][0], data[-1][1]), radius=15, color=1, thickness=25) # 管道结束位置画圈
            
            mask[:, :, 0] = mask0
            mask[:, :, 1] = mask1
            res = self.transforms(image=img, mask=mask)
            img = res['image']
            mask = res['mask']
            img = img.astype('float32').transpose(2, 0, 1) / 255.
            mask = mask.astype('float32').transpose(2, 0, 1)
            return torch.tensor(img), torch.tensor(mask)
        
transforms_train = albumentations.Compose([
    albumentations.Resize(CFG.image_size, CFG.image_size),                                    
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.1, p=0.75),
    # albumentations.OneOf([
    #     albumentations.GaussNoise(var_limit=[10, 50]),
    #     albumentations.MotionBlur(),
    #     albumentations.MedianBlur(),
    # ], p=0.2),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.75),
    albumentations.Cutout(max_h_size=int(CFG.image_size * 0.3), max_w_size=int(CFG.image_size * 0.3), num_holes=1, p=0.75),
#     albumentations.Normalize(),
])
transforms_valid = albumentations.Compose([
    albumentations.Resize(CFG.image_size, CFG.image_size),
#     albumentations.Normalize(),
])


'''
K-fold划分数据集
'''
def get_folds(nfolds=5):
    traindf = pd.read_csv(CFG.train_df_path)
    folds = traindf.copy()
    Fold = GroupKFold(n_splits=nfolds)
    groups = folds['PatientID'].values
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols], groups)):
        folds.loc[val_index, 'fold'] = int(n) # 添加一个fold列，将val_index对应的行设置为n

    folds['fold'] = folds['fold'].astype(int)
    return folds

'''
得到有标注信息的样本表格
'''
def get_df_with_anno():
    folds = get_folds(5) # k折
    train_anno = pd.read_csv(CFG.train_anno_path)
    unique_anno = train_anno.drop_duplicates(['StudyInstanceUID']).copy() # 去掉重复的样本名
    unique_anno['with_anno'] = True

    # 连接两个表
    train_v2 = pd.merge(folds, unique_anno[['StudyInstanceUID', 'with_anno']], left_on='StudyInstanceUID', right_on='StudyInstanceUID', how='left')

    # 将没有annotation的样本设置为False
    train_v2['with_anno'] = train_v2['with_anno'].fillna(value=False)
    sample_with_anno_df = train_v2[train_v2['with_anno'] == True].copy()
    return sample_with_anno_df

def get_seg_loader(fold_id, debug=False):
    sample_with_anno_df = get_df_with_anno()
    train_df = sample_with_anno_df[sample_with_anno_df.fold != fold_id]
    valid_df = sample_with_anno_df[sample_with_anno_df.fold == fold_id]

    # 小样本用作测试
    if debug:
        train_df = train_df.iloc[:16]
        valid_df = train_df.iloc[:16]

    train_anno = pd.read_csv(CFG.train_anno_path)

    train_data = RanzcrDataset(CFG.train_img_path, train_df, mode='train', transforms=transforms_train, train_anno=train_anno)
    valid_data = RanzcrDataset(CFG.train_img_path, valid_df, mode='valid', transforms=transforms_valid, train_anno=train_anno)

    train_loader = DataLoader(train_data, batch_size=CFG.seg_batch_size, shuffle=True, num_workers=CFG.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=CFG.seg_batch_size, shuffle=False, num_workers=CFG.num_workers)
    return train_loader, valid_loader