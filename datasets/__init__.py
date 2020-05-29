from datasets.core.cityscapes import CityScapes

def build_dataset(dataset, root, split):
    if dataset == 'cityscapes':
        return CityScapes(root, split)