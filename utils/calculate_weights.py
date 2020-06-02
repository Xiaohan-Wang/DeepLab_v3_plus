from tqdm import tqdm
import numpy as np
import os


def calculate_labels_weights(dataset, dataloader, num_classes, working_file_dir):
    class_pixel_num = np.zeros(num_classes)
    print("Calculating classes weights")
    for _, target in tqdm(dataloader):
        mask = (target >= 0) & (target < num_classes)
        target = target[mask]
        class_pixel_num += np.bincount(target, minlength=num_classes)
    total_pixel_num = class_pixel_num.sum()
    class_weights = 1 / (np.log(1.02 + (class_pixel_num / total_pixel_num)))
    dataset_working_file_dir = os.path.join(working_file_dir, dataset)
    if not os.path.exists(dataset_working_file_dir):
        # compared to os.mkdir, os.makedirs is used to create intermediate dirs if needed
        os.makedirs(dataset_working_file_dir)
    np.save(os.path.join(dataset_working_file_dir, 'class_weights.npy'), class_weights)
    return class_weights

