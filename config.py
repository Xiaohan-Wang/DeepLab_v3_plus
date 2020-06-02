class Config:
    # output folder
    code_test_dir = '/home/home1/xw176/work/frameworks/DeepLab_v3+/test_output' # delete after testing the code
    working_file_dir = '/home/home1/xw176/work/frameworks/DeepLab_v3+/work_file' # store working file for each dataset

    # model
    backbone = 'resnet'
    output_stride = 16

    # dataset
    dataset = 'cityscapes'
    dataset_root = '/usr/xtmp/vision/datasets/Cityscapes'
    num_classes = 19

    # train
    lr_scheduler = 'WarmupCosine'
    loss = 'focal'
    batch_size = 32

config = Config()




