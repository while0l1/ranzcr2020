class CFG:
    seg_image_size = 1024
    cls_image_size = 512
    seg_backbone = 'timm-efficientnet-b1'
    cls_backbone = ''
    seg_batch_size = 4
    cls_batch_size = 32
    num_workers = 2
    seg_lr = 1e-4
    cls_lr = 1e-4
    seg_epochs = 30
    cls_epochs = 30
    seg_warm = 1
    cls_warm = 1
    seed = 42
    log_path = './train.log'
    model_path = ''
    train_anno_path = ''
    train_df_path = ''
    train_img_path = '../input/ranzcr-clip-catheter-line-classification/train/'
    target_cols = ['ETT - Abnormal', 'ETT - Borderline',
               'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
               'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
               'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

               