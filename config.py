class CFG:
    image_size = 1024
    cls_image_size = 512
    seg_backbone = 'timm-efficientnet-b1'
    cls_backbone = ''
    seg_batch_size = 6
    cls_batch_size = 32
    num_workers = 2
    seg_lr = 1e-4
    cls_lr = 1e-4
    seg_epochs = 30
    cls_epochs = 30
    seg_warm = 1
    cls_warm = 1
    seed = 42
    log_path = '/content/drive/MyDrive/ranzcr/train.log'
    model_path = '/content/drive/MyDrive/ranzcr/'
    train_anno_path = '/content/ranzcr/train_annotations.csv'
    train_df_path = '/content/ranzcr/train.csv'
    train_img_path = '/content/ranzcr/train/'
    target_cols = ['ETT - Abnormal', 'ETT - Borderline',
               'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
               'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
               'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

               