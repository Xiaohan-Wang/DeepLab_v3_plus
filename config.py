class Config:
    # mlflow
    exp_name = 'workflow_test'
    run_name = 'cityscape - signle GPU'

    # output folder
    code_test_dir = '/home/home1/xw176/work/frameworks/DeepLab_v3+/test_output'  # delete after testing the code
    working_file_dir = '/home/home1/xw176/work/frameworks/DeepLab_v3+/work_file'  # store working file for each dataset

    # dataset
    dataset = 'cityscapes'
    dataset_root = '/usr/xtmp/vision/datasets/Cityscapes'
    num_classes = 19
    ignore_index = 255

    # model
    backbone = 'resnet'
    output_stride = 16

    # train
    train_batch_size = 8
    train_num_workers = 0
    loss = 'Focal'
    lr_scheduler = 'WarmupCosine'
    lr = 3e-4
    max_epoch = 200
    warmup_epoch = 5

    # val
    val_batch_size = 4
    val_num_workers = 0

    # other
    cuda = True
    resume = None
