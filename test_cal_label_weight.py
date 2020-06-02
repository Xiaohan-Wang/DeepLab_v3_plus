from config import config
from datasets import build_dataset
from torch.utils.data import DataLoader
from utils.calculate_weights import calculate_labels_weights

dataset = build_dataset(config.dataset, config.dataset_root, 'train')
dataloader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
class_weights = calculate_labels_weights(config.dataset, dataloader, config.num_classes, config.working_file_dir)